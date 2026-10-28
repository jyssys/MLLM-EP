"""Two-process NIXL cost probe with persistent GPU registration and byte checks."""
import argparse
import os
import time
import traceback
import torch
import torch.multiprocessing as mp
from .common import ROOT,UUIDS,append,write

def lane(rank,pipe):
 try:
  torch.set_num_threads(1);torch.cuda.set_device(rank)
  from nixl._api import nixl_agent,nixl_agent_config
  data=torch.zeros(64*1024*1024,device=f'cuda:{rank}',dtype=torch.uint8)
  agent=nixl_agent(f'abc-ipc-{os.getpid()}',nixl_agent_config(backends=['UCX']))
  registration=agent.register_memory(data)
  pipe.send(dict(metadata=agent.get_agent_metadata(),pointer=data.data_ptr(),device=rank,
   uuid=str(torch.cuda.get_device_properties(rank).uuid),pid=os.getpid()))
  peer=pipe.recv();name=agent.add_remote_agent(peer['metadata'])
  name=name.decode() if isinstance(name,bytes) else name
  if rank==0:
   for size in (1024,1024**2,16*1024**2,64*1024**2):
    for repeat in range(13):
     expected=(repeat+1)%251;data[:size].fill_(expected);torch.cuda.synchronize(rank)
     local=agent.get_xfer_descs([(data.data_ptr(),size,rank)],'VRAM')
     remote=agent.get_xfer_descs([(peer['pointer'],size,peer['device'])],'VRAM')
     handle=agent.initialize_xfer('WRITE',local,remote,name,notif_msg=b'copy',backends=['UCX'])
     begin=time.perf_counter();status=agent.transfer(handle)
     while status!='DONE':
      status=agent.check_xfer_state(handle)
      if status=='ERR':raise RuntimeError('NIXL WRITE failed')
      if time.perf_counter()-begin>30:raise TimeoutError('NIXL WRITE timed out')
     elapsed=1000*(time.perf_counter()-begin);agent.release_xfer_handle(handle)
     pipe.send(dict(kind='sample',bytes=size,repeat=repeat,expected=expected,wall_ms=elapsed))
     if pipe.recv()!='verified':raise RuntimeError('Destination byte check failed')
   pipe.send(dict(kind='done'))
  else:
   while True:
    agent.get_new_notifs()
    if not pipe.poll(.0001):continue
    command=pipe.recv()
    if command['kind']=='stop':break
    torch.cuda.synchronize(rank)
    pipe.send(bool((data[:command['bytes']]==command['expected']).all()))
  agent.remove_remote_agent(name);agent.deregister_memory(registration)
 except BaseException as error:
  pipe.send(dict(kind='error',error=repr(error),traceback=traceback.format_exc()));raise

def receive(pipe,timeout=180):
 if not pipe.poll(timeout):raise TimeoutError('NIXL child control deadline')
 value=pipe.recv()
 if isinstance(value,dict) and value.get('kind')=='error':raise RuntimeError(value)
 return value

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--name',default='nixl_two_process_pair01');args=parser.parse_args()
 folder=ROOT/'raw'/args.name;write(folder/'status.json',dict(status='RUNNING',pid=os.getpid()))
 context=mp.get_context('spawn');parents=[];children=[]
 try:
  for rank in (0,1):
   parent,child=context.Pipe();process=context.Process(target=lane,args=(rank,child));process.start()
   parents.append(parent);children.append(process)
  metadata=[receive(p) for p in parents]
  physical=list(map(int,os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
  for rank,row in enumerate(metadata):
   assert row['uuid'].removeprefix('GPU-')==UUIDS[physical[rank]].removeprefix('GPU-')
  write(folder/'workers.json',[{k:v for k,v in row.items() if k!='metadata'} for row in metadata])
  parents[0].send(metadata[1]);parents[1].send(metadata[0]);count=0
  while True:
   sample=receive(parents[0])
   if sample['kind']=='done':break
   parents[1].send(dict(kind='verify',bytes=sample['bytes'],expected=sample['expected']))
   passed=receive(parents[1]);parents[0].send('verified' if passed is True else 'failed')
   if passed is not True:raise RuntimeError('Actual NIXL payload did not match')
   if sample['repeat']>=3:
    sample.update(correct=True,visible=os.environ['CUDA_VISIBLE_DEVICES'],
     GB_per_second=sample['bytes']/sample['wall_ms']/1e6,
     evidence='MEASURED_PERSISTENT_TWO_PROCESS_NIXL_OPERATOR_NOT_SERVING')
    append(folder/'samples.jsonl',sample);count+=1
  parents[1].send(dict(kind='stop'))
  for process in children:
   process.join(60)
   if process.exitcode!=0:raise RuntimeError('NIXL child did not exit normally')
  write(folder/'status.json',dict(status='COMPLETE',samples=count))
 except BaseException as error:
  write(folder/'status.json',dict(status='FAILED',error=repr(error)));raise
 finally:
  for process in children:
   if process.is_alive():process.terminate();process.join(10)

if __name__=='__main__':main()
