"""B: immutable Ouro depth planes. Compute placement and loop count stay unchanged."""
import os
import time
from copy import deepcopy
from collections import deque
from pathlib import Path
import torch
from vllm_rlt.pd.connector import PDStageConnector
from vllm_rlt.pd.transport import kv_segments,partition_segments
from .common import append
from .boundary import BoundaryConnector
from .segments import coalesce

class LoopProgress:
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.progress_events=deque()
 def prefill_submitted(self,request,**kwargs):
  super().prefill_submitted(request,**kwargs)
  self.progress_events.append((kwargs['event'],request.request_id,deepcopy(request.prefill_depth_ranges)))
 def progress(self):
  super().progress()
  while self.progress_events:
   event,tid,frontiers=self.progress_events[0]
   if event is not None and not event.query():break
   self.progress_events.popleft()
   self.notify('loop_progress',tid=tid,frontiers=frontiers)

class ProgressConnector(LoopProgress,PDStageConnector):pass
class BoundaryProgressConnector(LoopProgress,BoundaryConnector):pass

class FrontierConnector(BoundaryConnector):
 def prefill_submitted(self,request,*,phase,start,progress,event,stream):
  t=self.transfers[request.request_id]
  if not hasattr(t,'frontiers'):t.frontiers=[t.target_cached_tokens]*self.cache.storage_depths
  peer=self.transport.peers[t.peer];tables=self.cache.plane_block_tables(t.tid)
  stream=stream or torch.cuda.current_stream(self.cache.device)
  with torch.cuda.stream(stream):
   if event is not None:stream.wait_event(event)
   if progress.prompt_complete:self.hidden[t.slot].copy_(request.hidden_state)
   ready=torch.cuda.Event();ready.record(stream)
  t.last_event=ready
  if progress.prompt_complete:
   generator=self.engine.model_runner.rng.peek(request.request_id)
   if generator is not None:t.rng_state=generator.get_state().tolist()
  boundary=getattr(self.engine.model,'requires_boundary_kv',False)
  if boundary:
   self._boundary_frontiers(t,request,ready)
   depths=()
  else:depths=range(self.cache.storage_depths)
  for depth in depths:
   end=max(request.num_prefilled_tokens,request.prefill_depth_ranges[depth][0])
   left=t.frontiers[depth]
   if left>=end:continue
   source=list(kv_segments(self.transport.info,[tables[depth]],left,end))
   target=list(kv_segments(peer,[t.target_tables[depth]],left,end))
   self._queue(t,source,target,ready)
   t.frontiers[depth]=end
   append(Path(os.environ['ABC_RUN_DIR'])/('frontiers-'+str(os.getpid())+'.jsonl'),
    dict(time=time.monotonic(),tid=t.tid,depth=depth,start=left,end=end,bytes=sum(s[1] for s in source),
     compute_owner='prefill',evidence='MEASURED_SUBMISSION_EVENT_NOT_COMPLETION',G='upstream_unchanged'))
  if progress.chunk_complete:
   self.prefill_tokens+=request.num_prefilled_tokens-start
   t.transfer_boundary_count+=1
   boundary=(start,request.num_prefilled_tokens)
   if t.transfer_boundary_first is None:t.transfer_boundary_first=boundary
   t.transfer_boundary_last=boundary
  if progress.prompt_complete:
   size=self.transport.info['hidden_bytes']
   self._queue(t,[(self.transport.info['hidden_ptr']+t.slot*size,size,self.transport.info['device'])],
    [(peer['hidden_ptr']+t.target_slot*size,size,peer['device'])],ready)
   t.compute_done=True
  t.transfer_overlaps+=int(self.transport.busy(t.tid))

 def _boundary_frontiers(self,t,request,event):
  peer=self.transport.peers[t.peer];tables=self.cache.plane_block_tables(t.tid)
  if not hasattr(t,'layer_frontiers'):t.layer_frontiers={}
  recurrent=set(self.engine.model.recurrent_kv_layers);coda=set(self.engine.model.coda_kv_layers)
  source=[];target=[]
  for depth,table in enumerate(tables):
   for layer in range(self.cache.num_layers):
    if layer in recurrent:end=max(request.num_prefilled_tokens,request.prefill_depth_ranges[depth][0])
    elif depth:continue
    elif layer in coda:end=min(request.num_prefilled_tokens,len(request.prompt_token_ids)-1)
    else:end=request.num_preluded_tokens
    left=t.layer_frontiers.get((depth,layer),0)
    if left>=end:continue
    def segments(info,blocktable):
     p=left
     while p<end:
      block=blocktable[p//self.cache.block_size];offset=p%self.cache.block_size
      count=min(end-p,self.cache.block_size-offset)
      for base in (info['key_ptr'],info['value_ptr']):
       yield(base+block*info['block_bytes']+layer*info['layer_bytes']+offset*info['token_bytes'],count*info['token_bytes'],info['device'])
      p+=count
    source.extend(segments(self.transport.info,table));target.extend(segments(peer,t.target_tables[depth]))
    t.layer_frontiers[depth,layer]=end
  self._queue(t,source,target,event)

 def _queue(self,t,source,target,event):
  source,target=coalesce(source,target)
  if os.environ.get('ABC_REVERSE_FRAGMENTS')=='1':source.reverse();target.reverse()
  for local,remote,size in partition_segments(source,target,self.config.transfer_chunk_bytes,self.config.max_transfer_descriptors):
   t.chunks.append((event,t.next_sequence,local,remote,size));t.next_sequence+=1
