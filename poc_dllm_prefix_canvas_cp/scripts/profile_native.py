"""Stage A only: native weights, cache, attention and diffusion schedule."""
import argparse
import gzip
import os
import signal
import socket
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from poc_dllm_prefix_canvas_cp.contract import *
from poc_dllm_prefix_canvas_cp.runtime import setup
from poc_dllm_prefix_canvas_cp.observer import Observer


def worker(rank,port,args):
    torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=False
    model,runner,dllm,tokenizer=setup(rank,port,args)
    with gzip.open(ROOT/'results/frozen_tokens.json.gz','rt') as inp:rows=json.load(inp)
    rows=[r for r in rows if r['split']=='calibration'][:args.limit]
    def watchdog(signum,frame):raise TimeoutError('180 seconds per native request')
    signal.signal(signal.SIGALRM,watchdog)
    def generate(row,prefix,mode,rep):
        ids=row['head']+row['background'][:prefix-len(row['head'])-len(row['tail'])]+row['tail']
        assert len(ids)==prefix and prefix+OUTPUT<=MAX_CONTEXT
        ids=torch.tensor([ids],device=rank,dtype=torch.long)
        torch.manual_seed(19411);torch.cuda.manual_seed_all(19411)
        dist.barrier();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        active_runner,active_dllm=runner,dllm
        bank=None
        if mode!='clean':active_runner,active_dllm,bank=runner._poc_marked
        before=active_dllm.num_forwards
        with Observer(model,active_dllm,rank,mode,prefix,runner=active_runner) as observer:
            start=time.perf_counter();signal.alarm(180)
            try:out=active_dllm.generate(ids,gen_length=OUTPUT,block_length=32)
            finally:signal.alarm(0)
            torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000
        output=out[0].cpu().tolist()
        observation=observer.finish()
        value=dict(request_id=row['request_id'],domain=row['domain'],prefix=prefix,mode=mode,repeat=rep,
            rank=rank,physical_gpu=GPUS[rank],wall_ms=wall,NFE=active_dllm.num_forwards-before,
            output_ids=output[prefix:],output_tokens=len(output)-prefix,
            max_memory_allocated=torch.cuda.max_memory_allocated(),max_memory_reserved=torch.cuda.max_memory_reserved(),
            model_context=MAX_CONTEXT,requested_output=OUTPUT,block=32,**observation,
            graph_samples=[] if bank is None else bank.read(observer.graph_keys),
            graph_samples_scope='Last native replay per graph key. Multiplication by call count is a stationarity estimate, not full-request direct timing.')
        path=Path(args.output)/f"{row['request_id']}.p{prefix}.{mode}.{rep}.rank{rank}.json.gz"
        with gzip.open(path,'wt') as sink:json.dump(value,sink)
        if rank==0:print(row['request_id'],prefix,mode,rep,'NFE',value['NFE'],'WALL',round(wall,2),flush=True)
        return value
    factory=runner._poc_factory
    for pindex,prefix in enumerate(args.prefixes):
        runner,dllm=factory(prefix)
        write(Path(args.output)/f'configuration.p{prefix}.rank{rank}.json',dict(prefix=prefix,
            static_KV_capacity=runner.max_length,model_position_limit=model.config.max_position_embeddings,
            native_graphs=[list(k) for k in runner.graph_runner.graphs],
            attention_event_nodes='external CUDA-event nodes in a separate copy of the native graphs; unchanged math'))
        for rep in range(3):generate(rows[0],prefix,'clean',-1-rep)
        for index,row in enumerate(rows):
            order=['clean','coarse','clean','coarse','clean']
            if (index+pindex)%2:order.reverse()
            seen={}
            for mode in order:
                rep=seen.get(mode,0);seen[mode]=rep+1
                generate(row,prefix,mode,rep)
            if index==0:
                generate(row,prefix,'trace',0)
                generate(row,prefix,'collectives',0)
                if args.profiler and prefix==max(args.prefixes):
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False) as prof:
                        generate(row,prefix,'clean',99)
                    cuda=[]
                    for e in prof.events():
                        if str(e.device_type).endswith('CUDA'):
                            cuda.append(dict(name=e.name,start_us=e.time_range.start,end_us=e.time_range.end,
                                device=e.device_index,thread=e.thread))
                    with gzip.open(Path(args.output)/f'profiler_cuda.rank{rank}.json.gz','wt') as f:json.dump(cuda,f)
                    write(Path(args.output)/f'profiler_scope.rank{rank}.json',dict(
                        request_id=row['request_id'],prefix=prefix,repeat=99,kernel_events=len(cuda),
                        label='HEAVY_PROFILER_DIAGNOSTIC_NOT_HEADLINE',no_candidate=True))
    torch.cuda.synchronize();dist.barrier()
    write(Path(args.output)/f'worker_complete.rank{rank}.json',dict(success=True,
        scope='All native measurements persisted; GPU work synchronized; all four ranks reached final barrier',
        teardown='Bounded subprocess exits before legacy SGLang graph/IPC destructors'))
    # Legacy SGLang graph/IPC teardown stalled after the completed v6 campaign.
    # This is experiment-process cleanup, not a changed inference operator.
    # The parent guard still restores burners and handles every error path.
    os._exit(0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--backend',default='tp4',choices=['tp4','ep4'])
    p.add_argument('--limit',type=int,default=8);p.add_argument('--prefixes',nargs='+',type=int,default=PREFIXES)
    p.add_argument('--profiler',action='store_true')
    p.add_argument('--eager-prefill',action='store_true',help='Native large-prefill fallback; refinement graphs remain enabled')
    a=p.parse_args();a.discovery=True
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==MASK
    assert all(p in PREFIXES for p in a.prefixes)
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    mp.spawn(worker,args=(port,a),nprocs=4,join=True)
    write(Path(a.output)/'status.json',dict(success=True,GPUS=GPUS,phase='STAGE_A_NATIVE_ONLY',prefixes=a.prefixes))


if __name__=='__main__':main()
