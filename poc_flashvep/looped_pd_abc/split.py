"""A: whole-prompt recurrence split. Remaining loops execute on the decode GPU."""
import os
import time
from pathlib import Path
import torch
from vllm_rlt.request import Request,RequestPhase
from vllm_rlt.pd.transport import NixlConnector,partition_segments
from .boundary import BoundaryConnector
from .common import append
from .segments import coalesce

class SplitConnector(BoundaryConnector):
 @classmethod
 def create(cls,engine,*,config,notify):
  if not engine.scheduler.config.wavefront_chunked_prefill:raise ValueError('A requires an explicit staged G1 control')
  cut=int(os.environ['ABC_SPLIT_CUT'])
  if not 0<cut<engine.model.config.total_ut_steps:raise ValueError('A adapter requires an intermediate cut. Use normal PD or colocated controls at the endpoints')
  limit=engine.scheduler.config.max_num_seqs+(config.max_receiving_requests if engine.stage_config.produces_tokens else config.max_draining_requests)
  parameter=next(engine.model.parameters());chunk=engine.scheduler.config.prefill_chunk_size
  hidden=torch.zeros((limit,chunk*engine.model.config.hidden_size),device=parameter.device,dtype=parameter.dtype)
  connector=cls(engine,hidden,NixlConnector(engine.stage_config.stage_id,engine.cache_manager,hidden,config),config,notify=notify)
  connector.cut=cut;connector.original_accept=engine.accept_import
  engine.accept_import=connector.accept_split
  return connector

 def reserve(self,tid,tokens,*args,**kwargs):
  if len(tokens)>self.engine.scheduler.config.prefill_chunk_size:
   raise ValueError('Initial A adapter transfers a complete prompt, not unfinished token chunks')
  return super().reserve(tid,tokens,*args,**kwargs)

 def prefill_submitted(self,request,*,phase,start,progress,event,stream):
  t=self.transfers[request.request_id]
  if t.compute_done:return
  n=len(request.prompt_token_ids)
  if request.prefill_depth_ranges[self.cut-1][0]<n:
   t.last_event=event
   return
  runner=self.engine.model_runner
  slot=runner._ensure_request_slot(request)
  stream=stream or torch.cuda.current_stream(self.cache.device)
  with torch.cuda.stream(stream):
   if event is not None:stream.wait_event(event)
   self.hidden[t.slot].zero_()
   state=runner.prefill_states[slot,:n]
   self.hidden[t.slot,:state.numel()].copy_(state.flatten())
   ready=torch.cuda.Event();ready.record(stream)
  g=runner.rng.peek(request.request_id)
  if g is not None:t.rng_state=g.get_state().tolist()
  t.last_event=ready
  self._queue_snapshot(t,n,ready)
  self.engine.suspend_transfer(request)
  t.compute_done=True
  t.transfer_boundary_first=t.transfer_boundary_last=(0,n);t.transfer_boundary_count=1
  self.prefill_tokens+=n
  append(Path(os.environ['ABC_RUN_DIR'])/('splits-'+str(os.getpid())+'.jsonl'),dict(time=time.monotonic(),tid=t.tid,
   cut=self.cut,depth=self.cache.max_loops,prompt_tokens=n,hidden_bytes=state.numel()*state.element_size(),
   staged_bytes=self.hidden[t.slot].numel()*self.hidden.element_size(),ownership='P_to_D',B=False))

 def _queue_snapshot(self,t,n,event):
  peer=self.transport.peers[t.peer];tables=self.cache.plane_block_tables(t.tid)
  recurrent=set(getattr(self.engine.model,'recurrent_kv_layers',range(self.cache.num_layers)))
  coda=set(getattr(self.engine.model,'coda_kv_layers',()))
  def segments(info,alltables):
   for depth,table in enumerate(alltables[:self.cut]):
    for layer in range(self.cache.num_layers):
     if layer in coda or (depth and layer not in recurrent):continue
     for p in range(0,n,self.cache.block_size):
      amount=min(self.cache.block_size,n-p)*info['token_bytes']
      for base in (info['key_ptr'],info['value_ptr']):
       yield(base+table[p//self.cache.block_size]*info['block_bytes']+layer*info['layer_bytes'],amount,info['device'])
  src=list(segments(self.transport.info,tables));dst=list(segments(peer,t.target_tables))
  size=self.transport.info['hidden_bytes']
  src.append((self.transport.info['hidden_ptr']+t.slot*size,size,self.transport.info['device']))
  dst.append((peer['hidden_ptr']+t.target_slot*size,size,peer['device']))
  src,dst=coalesce(src,dst)
  for local,remote,size in partition_segments(src,dst,self.config.transfer_chunk_bytes,self.config.max_transfer_descriptors):
   t.chunks.append((event,t.next_sequence,local,remote,size));t.next_sequence+=1

 def accept_split(self,request,hidden,*,cached_tokens=0):
  if cached_tokens:raise ValueError('A prefix-cache migration is not qualified')
  if self.engine.scheduler.requests.get(request.request_id) is not request:return False
  n=len(request.prompt_token_ids);H=self.engine.model.config.hidden_size
  allocation=self.cache._get_allocation(request.request_id)
  recurrent=set(getattr(self.engine.model,'recurrent_kv_layers',range(self.cache.num_layers)))
  coda=set(getattr(self.engine.model,'coda_kv_layers',()))
  for depth,plane in enumerate(allocation.written):
   for layer,w in enumerate(plane):
    if depth<self.cut and layer not in coda and (depth==0 or layer in recurrent):w.prefix=n
  request.num_preluded_tokens=n;request.num_prefilled_tokens=0
  request.prefill_depth_ranges[:]=Request.prefill_boundaries(n,self.cache.max_loops)
  for d in range(self.cut):
   request.prefill_depth_ranges[d][0]=n
   if d+1<self.cache.max_loops:request.prefill_depth_ranges[d+1][1]=n
  request.prefill_window_origin=request.prefill_reclaimable_tokens=0
  request.loops_done=self.cut;request.import_prompt_state=hidden[:n*H].view(n,H)
  request.import_resume_phase=RequestPhase.PREFILLING_RECURRENT
  self.engine.scheduler.queues[RequestPhase.WAITING].append(request.request_id)
  if os.environ.get('ABC_VERIFY_KV')=='1':
   from .verify import check
   check(self,request)
  return True
