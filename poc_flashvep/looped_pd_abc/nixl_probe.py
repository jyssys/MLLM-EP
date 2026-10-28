"""Actual NIXL GPU-to-GPU copies. This is not a model-serving benchmark."""
import argparse
import os
import time
import torch
from nixl._api import nixl_agent,nixl_agent_config
from .common import ROOT,setup,write,append

def main():
 setup();p=argparse.ArgumentParser();p.add_argument('--name',default='nixl_pair01');args=p.parse_args()
 source=nixl_agent('abc-source-'+str(os.getpid()),nixl_agent_config(backends=['UCX']))
 target=nixl_agent('abc-target-'+str(os.getpid()),nixl_agent_config(backends=['UCX']))
 a=torch.full((64*1024*1024,),73,dtype=torch.uint8,device='cuda:0')
 b=torch.zeros_like(a,device='cuda:1')
 registrations=[(source,source.register_memory(a)),(target,target.register_memory(b))]
 name=source.add_remote_agent(target.get_agent_metadata());name=name.decode() if isinstance(name,bytes) else name
 samples=[]
 try:
  for size in (1024,1024**2,16*1024**2,64*1024**2):
   for repeat in range(13):
    local=source.get_xfer_descs([(a.data_ptr(),size,0)],'VRAM')
    remote=source.get_xfer_descs([(b.data_ptr(),size,1)],'VRAM')
    handle=source.initialize_xfer('WRITE',local,remote,name,backends=['UCX'])
    torch.cuda.synchronize(0);torch.cuda.synchronize(1);start=time.perf_counter()
    status=source.transfer(handle)
    while status!='DONE':
     status=source.check_xfer_state(handle)
     if status=='ERR':raise RuntimeError('NIXL WRITE failed')
     if time.perf_counter()-start>30:raise TimeoutError('NIXL copy deadline')
    elapsed=1000*(time.perf_counter()-start)
    source.release_xfer_handle(handle)
    assert bool((b[:size]==73).all())
    if repeat>=3:
     row=dict(bytes=size,repeat=repeat,wall_ms=elapsed,GB_per_second=size/elapsed/1e6,
      evidence='MEASURED_NIXL_OPERATOR_NOT_SERVING',visible=os.environ['CUDA_VISIBLE_DEVICES'],correct=True)
     samples.append(row);append(ROOT/'raw'/args.name/'samples.jsonl',row)
  write(ROOT/'raw'/args.name/'status.json',dict(status='COMPLETE',samples=len(samples)))
 finally:
  source.remove_remote_agent(name)
  for agent,registration in registrations:agent.deregister_memory(registration)

if __name__=='__main__':main()
