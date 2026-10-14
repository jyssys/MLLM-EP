"""New physical baseline and every-occurrence timelines on the same inputs."""
import argparse
import contextlib
import gzip
import os
import signal
import socket
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from poc_dllm_multigpu_e2e_atlas.contract import *
from poc_dllm_multigpu_e2e_atlas.observer import AtlasObserver
from poc_dllm_multigpu_e2e_atlas.runtime import setup


def worker(rank,port,args):
    torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=False
    model,runner,dllm,tokenizer=setup(rank,port,args)
    selected=rows(args.split)[:args.limit]
    def expired(signum,frame):raise TimeoutError('Native request exceeded180 seconds')
    signal.signal(signal.SIGALRM,expired)
    def generate(row,mode,repeat):
        ids=torch.tensor([token_ids(row,args.prefix)],device=rank,dtype=torch.long)
        torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        dist.barrier();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        observer=None if mode=='clean' else AtlasObserver(runner,dllm,mode)
        context=contextlib.nullcontext() if observer is None else observer
        probe=None
        before=dllm.num_forwards
        with context:
            signal.alarm(180);start=time.perf_counter_ns()
            if mode=='profile':torch.cuda.nvtx.range_push('ATLAS_REQUEST|'+row['request_id']+'|P'+str(args.prefix))
            try:
                request_scope=contextlib.nullcontext() if observer is None or mode=='parity' else observer.scope('request_gpu')
                with request_scope:
                    if mode=='profile' and args.eager_diagnostic:
                        from poc_dllm_multigpu_e2e_atlas.diagnostic import EagerProbe
                        with EagerProbe(runner,model,args.output,rank) as probe:
                            output=dllm.generate(ids,gen_length=OUTPUT,block_length=BLOCK)
                    else:output=dllm.generate(ids,gen_length=OUTPUT,block_length=BLOCK)
            finally:signal.alarm(0)
            returned=time.perf_counter_ns();torch.cuda.synchronize();end=time.perf_counter_ns()
            if mode=='profile':torch.cuda.nvtx.range_pop()
        values=output[0].cpu().tolist()[args.prefix:]
        if probe is not None:probe.finish()
        result=dict(request_id=row['request_id'],domain=row['domain'],split=row['split'],prefix=args.prefix,
            rank=rank,physical_gpu=GPUS[rank],mode=mode,repeat=repeat,eager_prefill=args.eager_prefill,
            start_host_ns=start,return_host_ns=returned,completion_host_ns=end,
            wall_ms=(end-start)/1e6,host_return_ms=(returned-start)/1e6,final_wait_ms=(end-returned)/1e6,
            output_ids=values,output_tokens=len(values),NFE=dllm.num_forwards-before,
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            observation=None if observer is None else observer.finish())
        dest=Path(args.output)/f"{row['request_id']}.p{args.prefix}.{mode}.{repeat}.rank{rank}.json.gz"
        with gzip.open(dest,'wt') as stream:json.dump(result,stream)
        if rank==0:print(row['request_id'],args.prefix,mode,repeat,'NFE',result['NFE'],'ms',round(result['wall_ms'],3),flush=True)
        return result
    for repeat in range(3):generate(selected[0],'clean',-repeat-1)
    for index,row in enumerate(selected):
        if args.profile:
            for repeat in range(3):generate(row,'clean',repeat)
            dist.barrier();torch.cuda.synchronize()
            if args.torch_profile:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False) as prof:
                    generate(row,'profile',0)
                events=[]
                for event in prof.events():
                    events.append(dict(name=event.name,start_us=event.time_range.start,end_us=event.time_range.end,
                        device_type=str(event.device_type),device_index=event.device_index,thread=event.thread,
                        id=event.id,cpu_parent=None if event.cpu_parent is None else event.cpu_parent.id))
                with gzip.open(Path(args.output)/f'torch_events.rank{rank}.json.gz','wt') as stream:json.dump(events,stream)
            else:
                torch.cuda.profiler.start();generate(row,'profile',0)
                dist.barrier();torch.cuda.synchronize();torch.cuda.profiler.stop()
            break
        modes=['clean','coarse'] if index%2==0 else ['coarse','clean']
        for repeat in range(args.repeats):
            for mode in modes[repeat%2:]+modes[:repeat%2]:generate(row,mode,repeat)
        generate(row,'parity',0)
    torch.cuda.synchronize();dist.barrier()
    write(Path(args.output)/f'worker_complete.rank{rank}.json',dict(success=True,mask=MASK,
        cleanup='All measurements persisted. Final four-rank barrier passed. Bounded worker exit avoids legacy graph destructor stalls.'))
    os._exit(0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--prefix',type=int,required=True,choices=PREFIXES)
    p.add_argument('--limit',type=int,default=8);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--split',choices=['calibration','held_out'],default='calibration')
    p.add_argument('--eager-prefill',action='store_true');p.add_argument('--profile',action='store_true')
    p.add_argument('--eager-diagnostic',action='store_true')
    p.add_argument('--torch-profile',action='store_true')
    args=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==MASK
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    mp.spawn(worker,args=(port,args),nprocs=4,join=True)
    write(Path(args.output)/'status.json',dict(success=True,GPUS=GPUS,phase='NEW_NATIVE_BASELINE',profile=args.profile))


if __name__=='__main__':main()
