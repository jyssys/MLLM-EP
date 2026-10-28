"""Minimal Huginn/Dory full-prefill PD adapter, with boundary KV and RNG continuity."""
import torch
import os
from vllm_rlt.pd.connector import PDStageConnector
from vllm_rlt.pd.transport import partition_segments
from .segments import coalesce

def valid_segments(info,tables,start,end,recurrent_layers,coda_layers,prompt_length):
 for depth,table in enumerate(tables):
  for layer in range(info['num_layers']):
   if depth and layer not in recurrent_layers:continue
   limit=min(end,prompt_length-1) if layer in coda_layers else end
   position=start
   while position<limit:
    page=position//info['block_size'];offset=position%info['block_size']
    count=min(limit-position,info['block_size']-offset)
    for base in (info['key_ptr'],info['value_ptr']):
     yield (base+table[page]*info['block_bytes']+layer*info['layer_bytes']+offset*info['token_bytes'],count*info['token_bytes'],info['device'])
    position+=count

class BoundaryConnector(PDStageConnector):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  original=self.notify
  def notify(kind,**fields):
   if kind=='commit':
    fields['rng_state']=getattr(self.transfers[fields['tid']],'rng_state',None)
    if os.environ.get('ABC_VERIFY_KV')=='1':
     from .verify import fingerprint
     fields['fingerprint']=fingerprint(self,self.transfers[fields['tid']])
   original(kind,**fields)
  self.notify=notify
  original_accept=self.engine.accept_import
  def accept(request,hidden,**kwargs):
   result=original_accept(request,hidden,**kwargs)
   if result and os.environ.get('ABC_VERIFY_KV')=='1':
    from .verify import check
    check(self,request)
   return result
  self.engine.accept_import=accept
 def command(self,command):
  if command['kind']=='commit' and os.environ.get('ABC_VERIFY_KV')=='1':
   self.transfers[command['tid']].expected_fingerprint=command.get('fingerprint')
  if command['kind']=='commit' and command.get('rng_state') is not None:
   t=self.transfers[command['tid']]
   generator=self.engine.model_runner.rng.acquire(t.tid,t.request.sampling_params,self.cache.device)
   generator.set_state(torch.tensor(command['rng_state'],dtype=torch.uint8))
  return super().command(command)
 def prefill_submitted(self,request,**kwargs):
  if kwargs['progress'].prompt_complete:
   generator=self.engine.model_runner.rng.peek(request.request_id)
   if generator is not None:self.transfers[request.request_id].rng_state=generator.get_state().tolist()
  super().prefill_submitted(request,**kwargs)
 def queue_chunk(self,t,start,end,event,final):
  peer=self.transport.peers[t.peer];tables=self.cache.plane_block_tables(t.tid)
  start=max(start,t.target_cached_tokens);model=self.engine.model
  recurrent=getattr(model,'recurrent_kv_layers',range(self.cache.num_layers));coda=getattr(model,'coda_kv_layers',())
  src=list(valid_segments(self.transport.info,tables,start,end,recurrent,coda,len(t.request.prompt_token_ids)))
  dst=list(valid_segments(peer,t.target_tables,start,end,recurrent,coda,len(t.request.prompt_token_ids)))
  if start<end:
   t.transfer_boundary_count+=1;t.transfer_boundary_last=(start,end)
   if t.transfer_boundary_first is None:t.transfer_boundary_first=(start,end)
  if final:
   size=self.transport.info['hidden_bytes']
   src.append((self.transport.info['hidden_ptr']+t.slot*size,size,self.transport.info['device']))
   dst.append((peer['hidden_ptr']+t.target_slot*size,size,peer['device']))
  src,dst=coalesce(src,dst)
  for local,remote,size in partition_segments(src,dst,self.config.transfer_chunk_bytes,self.config.max_transfer_descriptors):
   t.chunks.append((event,t.next_sequence,local,remote,size));t.next_sequence+=1
