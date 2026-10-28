"""A: exact recurrent continuation across two GPUs using native RLT model operators.

This is an integrated request probe, not the optimized PDEngine serving path.
It copies valid KV only. It also transfers full prompt state at intermediate cuts.
"""
import argparse
from copy import deepcopy
import os
import time
import torch
from vllm_rlt.models import AutoModelForCausalLM
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.pd.config import PDConfig
from vllm_rlt.pd.transport import NixlConnector,partition_segments
from .common import ROOT,MODELS,MODEL_PINS,write,append
from .worker import prompts
from .segments import coalesce

class Native:
 def __init__(self,name,device):
  self.name=name;self.device=torch.device(device)
  if name=='dory':
   from .dory_adapter import DoryForCausalLM
   self.model=DoryForCausalLM.from_pretrained(MODELS[name],device=device,dtype=torch.bfloat16)
  else:self.model=AutoModelForCausalLM.from_pretrained(MODELS[name],device=device,dtype=torch.bfloat16)
  self.c=self.model.config;self.R=MODEL_PINS[name][2]
 def cache(self,n):
  return KVCacheManager(self.c.num_hidden_layers,self.c.num_key_value_heads,self.c.head_dim,
   ((n+15)//16)*self.R+8,16,self.R,device=self.device,dtype=torch.bfloat16,backend='flash_attn_4',
   recurrent_layers=getattr(self.model,'recurrent_kv_layers',None))
 def prelude(self,tokens,cache,generator):
  positions=list(range(len(tokens))) if tokens.ndim==1 else []
  batch=cache._prepare_batch(['r']*len(tokens),[0]*len(tokens),positions,packed_prefill=True)
  if self.name in ('huginn','dory'):return self.model.prelude_prepared(tokens,batch,cache,generators=[generator]*len(tokens))
  return self.model.prelude(tokens)
 def core(self,h,cache,d,positions):
  batch=cache._prepare_batch(['r']*len(positions),[d]*len(positions),positions,packed_prefill=len(positions)>1)
  return self.model.recurrent_prepared(h,batch,cache,compute_gate=False)[0]
 def coda(self,h,cache,positions):
  if self.name in ('huginn','dory'):
   batch=cache._prepare_batch(['r']*len(positions),[0]*len(positions),positions,packed_prefill=len(positions)>1)
   return self.model.coda_prepared(h,batch,cache)
  return self.model.coda(h)
 def decode_prelude(self,token,position,cache,generator):
  ids=torch.tensor([token],device=self.device)
  if self.name in ('huginn','dory'):
   batch=cache._prepare_batch(['r'],[0],[position])
   return self.model.prelude_prepared(ids,batch,cache,generators=[generator])
  return self.model.prelude(ids)

def kv_descriptors(cache,allocation,info):
 result=[]
 for depth,plane in enumerate(allocation.written):
  table=allocation.block_tables[depth]
  for layer,w in enumerate(plane):
   if w.pending:raise RuntimeError('The snapshot has holes in a KV frontier')
   for start in range(0,w.prefix,cache.block_size):
    count=min(cache.block_size,w.prefix-start)
    block=table[start//cache.block_size]
    for base in (info['key_ptr'],info['value_ptr']):
     result.append((base+block*info['block_bytes']+layer*info['layer_bytes'],count*info['token_bytes'],info['device']))
 return result

def handoff(source,target,ca,cb,h,generator):
 from nixl._api import nixl_agent,nixl_agent_config
 hidden_a=h.contiguous();hidden_b=torch.empty_like(h,device=target.device)
 cfg=PDConfig();a=NixlConnector('a-'+str(os.getpid()),ca,hidden_a,cfg)
 b=NixlConnector('b-'+str(os.getpid()),cb,hidden_b,cfg)
 a.connect(b.info);b.connect(a.info)
 src=ca._get_allocation('r');dst=cb._get_allocation('r')
 # Destination frontiers remain invalid until every transfer completes.
 layout=deepcopy(src.written)
 local=kv_descriptors(ca,src,a.info)
 placeholder=deepcopy(dst);placeholder.written=layout
 remote=kv_descriptors(cb,placeholder,b.info)
 local.append((hidden_a.data_ptr(),hidden_a.numel()*hidden_a.element_size(),source.device.index))
 remote.append((hidden_b.data_ptr(),hidden_b.numel()*hidden_b.element_size(),target.device.index))
 torch.cuda.synchronize(source.device);torch.cuda.synchronize(target.device)
 local,remote=coalesce(local,remote)
 begin=time.perf_counter();parts=list(partition_segments(local,remote,cfg.transfer_chunk_bytes,cfg.max_transfer_descriptors))
 try:
  for seq,(s,d,size) in enumerate(parts):
   while not a.submit('r',seq,b.info['agent'],s,d,size):a.poll();b.poll()
  while a.pending:
   a.poll();b.poll()
   if time.perf_counter()-begin>60:raise TimeoutError('State handoff timed out')
  torch.cuda.synchronize(target.device)
  elapsed=(time.perf_counter()-begin)*1000
  dst.written=layout
  transferred=sum(s[1] for s in local)
 finally:a.close();b.close()
 other=torch.Generator(device=target.device);other.set_state(generator.get_state())
 return hidden_b,other,dict(bytes=transferred,transfer_ms=elapsed,NIXL_transfers=len(parts),
  hidden_bytes=hidden_a.numel()*hidden_a.element_size(),valid_KV_bytes=transferred-hidden_a.numel()*hidden_a.element_size())

@torch.inference_mode()
def generate(adapters,tokens,cap,cut,seed,coda_mode='full'):
 a,b=adapters;R=a.R
 ca=a.cache(len(tokens)+cap);assert ca.allocate('r',len(tokens)+cap)
 cb=None;owner=a;cache=ca
 g=torch.Generator(device=a.device).manual_seed(seed)
 trace=[];transfer=None;start=time.perf_counter()
 if cut==0:
  owner=b;cache=cb=b.cache(len(tokens)+cap);assert cache.allocate('r',len(tokens)+cap)
  g=torch.Generator(device=b.device).manual_seed(seed)
 h=owner.prelude(torch.tensor(tokens,device=owner.device),cache,g)
 positions=list(range(len(tokens)))
 for d in range(R):
  before=torch.cuda.Event(enable_timing=True);after=torch.cuda.Event(enable_timing=True)
  before.record(torch.cuda.current_stream(owner.device));h=owner.core(h,cache,d,positions);after.record(torch.cuda.current_stream(owner.device))
  trace.append((d,owner.device.index,before,after))
  if cut is not None and cut>0 and d+1==cut:
   cb=b.cache(len(tokens)+cap);assert cb.allocate('r',len(tokens)+cap)
   h,g,transfer=handoff(a,b,ca,cb,h,g);owner=b;cache=cb
 if coda_mode=='split' and len(positions)>1:
  owner.coda(h[:-1],cache,positions[:-1])
  logits=owner.coda(h[-1:],cache,positions[-1:])[-1].float()
 else:logits=owner.coda(h,cache,positions)[-1].float()
 first_logits=logits.cpu()
 out=[];events=[]
 for t in range(cap):
  token=int(logits.argmax());out.append(token);events.append(time.perf_counter())
  eos=owner.c.eos_token_id
  if owner.name=='huginn' and os.environ.get('ABC_HUGINN_REFERENCE')=='1':eos=(65505,65508)
  if token in (eos if isinstance(eos,(tuple,list)) else (eos,)):break
  if t==cap-1:break
  position=len(tokens)+t
  h=owner.decode_prelude(token,position,cache,g)
  for d in range(R):h=owner.core(h,cache,d,[position])
  logits=owner.coda(h,cache,[position])[-1].float()
 torch.cuda.synchronize(owner.device)
 elapsed=(time.perf_counter()-start)*1000
 measurements=[dict(loop=d+1,gpu=gpu,gpu_ms=x.elapsed_time(y)) for d,gpu,x,y in trace]
 # Keep the final cache for a separate correctness check, outside timed generation.
 return dict(output_ids=out,e2e_ms=elapsed,ttft_ms=(events[0]-start)*1000,
  tpot_ms=1000*(events[-1]-events[0])/max(1,len(events)-1),transfer=transfer,loops=measurements),first_logits,cache

def compare_cache(a,b):
 aa=a._get_allocation('r');bb=b._get_allocation('r');maximum=0.;equal=True;valid=0
 for d,plane in enumerate(aa.written):
  for layer,w in enumerate(plane):
   other=bb.written[d][layer]
   if w.prefix!=other.prefix or w.pending!=other.pending:return dict(equal=False,reason='frontier mismatch')
   for pos in range(w.prefix):
    x=aa.block_tables[d][pos//a.block_size];y=bb.block_tables[d][pos//b.block_size];off=pos%a.block_size
    for ka,kb in ((a.key_cache,b.key_cache),(a.value_cache,b.value_cache)):
     left=ka[x,layer,off].cpu();right=kb[y,layer,off].cpu()
     equal=equal and torch.equal(left,right);maximum=max(maximum,float((left.float()-right.float()).abs().max()));valid+=left.numel()*left.element_size()
 return dict(equal=equal,max_abs=maximum,valid_bytes=valid)

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=('ouro','huginn','dory'),required=True)
 p.add_argument('--name',required=True);p.add_argument('--requests',type=int,default=2)
 p.add_argument('--length',type=int,default=128);p.add_argument('--cap',type=int,default=8)
 p.add_argument('--cuts',default='1');p.add_argument('--enable-recurrence-split',action='store_true')
 args=p.parse_args();dest=ROOT/'raw'/args.name;write(dest/'status.json',dict(status='RUNNING',pid=os.getpid()))
 torch.set_num_threads(1)
 try:
  adapters=[Native(args.model,'cuda:0'),Native(args.model,'cuda:1')]
  inputs=prompts(args.model,args.requests,args.length)
  cuts=[int(x) for x in args.cuts.split(',')] if args.enable_recurrence_split else [adapters[0].R]
  for i,tokens in enumerate(inputs):
   baseline,logits,cache=generate(adapters,tokens,args.cap,None,20261029+i)
   for cut in cuts:
    row,other,transferred=generate(adapters,tokens,args.cap,cut,20261029+i)
    row.update(model=args.model,sample_id=i,cut=cut,baseline=baseline,logit_max_abs=float((logits-other).abs().max()),
     logit_relative_l2=float(torch.linalg.vector_norm(logits-other)/torch.linalg.vector_norm(logits)),
     output_equal=baseline['output_ids']==row['output_ids'],KV=compare_cache(cache,transferred),
     evidence='MEASURED_NATIVE_OPERATOR_REQUEST_PROBE_NOT_PDEngine_SERVING',gpus=os.environ['CUDA_VISIBLE_DEVICES'],
     seed=20261029+i,depth=adapters[0].R,cap=args.cap,input_ids=tokens)
    append(dest/'samples.jsonl',row)
    print('STATE',args.model,i,cut,row['output_equal'],row['KV'],flush=True)
  write(dest/'status.json',dict(status='COMPLETE'))
 except BaseException as e:
  write(dest/'status.json',dict(status='FAILED',error=repr(e)));raise

if __name__=='__main__':main()
