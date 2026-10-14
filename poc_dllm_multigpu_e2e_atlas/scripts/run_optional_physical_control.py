"""One paired full-model exact storage control. No architecture change."""
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
from poc_dllm_multigpu_e2e_atlas.runtime import setup,build
from poc_dllm_multigpu_e2e_atlas.observer import AtlasObserver
from poc_dllm_multigpu_e2e_atlas.compact_scatter import CompactKV


def worker(rank,port,args):
    torch.set_grad_enabled(False);torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory=False
    model,baseline,base_dllm,tokenizer=setup(rank,port,args)
    with CompactKV(model) as patch:candidate,cand_dllm=build(model,baseline.server_args,args.prefix,args.eager_prefill)
    write(Path(args.output)/f'compact_capture.rank{rank}.json',dict(records=patch.records,
        physical_gpu=GPUS[rank],scope='Candidate capture and native graph warmup only. No source-cache in-place mutation.',
        co_resident_allocated=torch.cuda.memory_allocated(),co_resident_reserved=torch.cuda.memory_reserved()))
    arms={'native':(baseline,base_dllm),'compact':(candidate,cand_dllm)}
    selected=rows(args.split)[:args.limit]
    def expired(signum,frame):raise TimeoutError('180 seconds per request')
    signal.signal(signal.SIGALRM,expired)
    def generate(row,arm,mode,repeat):
        runner,dllm=arms[arm]
        ids=torch.tensor([token_ids(row,args.prefix)],device=rank,dtype=torch.long)
        torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        observer=None if mode=='clean' else AtlasObserver(runner,dllm,mode)
        dist.barrier();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        before=dllm.num_forwards
        with contextlib.nullcontext() if observer is None else observer:
            start=time.perf_counter_ns();signal.alarm(180)
            try:
                scope=contextlib.nullcontext() if observer is None or mode=='parity' else observer.scope('request_gpu')
                with scope:output=dllm.generate(ids,gen_length=OUTPUT,block_length=BLOCK)
            finally:signal.alarm(0)
            torch.cuda.synchronize();end=time.perf_counter_ns()
        result=dict(request_id=row['request_id'],domain=row['domain'],split=row['split'],arm=arm,mode=mode,repeat=repeat,
            prefix=args.prefix,rank=rank,physical_gpu=GPUS[rank],NFE=dllm.num_forwards-before,
            output_ids=output[0].cpu().tolist()[args.prefix:],wall_ms=(end-start)/1e6,
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            observation=None if observer is None else observer.finish())
        result['output_tokens']=len(result['output_ids'])
        path=Path(args.output)/f"{row['request_id']}.{arm}.{mode}.{repeat}.rank{rank}.json.gz"
        with gzip.open(path,'wt') as stream:json.dump(result,stream)
        if rank==0:print(row['request_id'],args.prefix,arm,mode,repeat,'NFE',result['NFE'],'ms',round(result['wall_ms'],3),flush=True)
        return result
    for repeat in range(3):
        for arm in arms:generate(selected[0],arm,'clean',-repeat-1)
    parity=[]
    for index,row in enumerate(selected):
        truth={arm:generate(row,arm,'parity',0) for arm in arms}
        assert truth['native']['output_ids']==truth['compact']['output_ids']
        assert truth['native']['NFE']==truth['compact']['NFE']
        assert truth['native']['observation']['updates']==truth['compact']['observation']['updates']
        key=(1,True,32,2*args.prefix)
        left=baseline.graph_runner.output_buffers[key];right=candidate.graph_runner.output_buffers[key]
        independent=[a.untyped_storage().data_ptr()!=b.untyped_storage().data_ptr()
            for a,b in zip(left.past_key_values,right.past_key_values)]
        assert (all(independent) and
            left.logits.untyped_storage().data_ptr()!=right.logits.untyped_storage().data_ptr()), 'Output buffers must not alias'
        kv_checks=[torch.equal(a,b) for a,b in zip(left.past_key_values,right.past_key_values)]
        logits=torch.equal(left.logits,right.logits)
        assert len(kv_checks)==40 and all(kv_checks) and logits,'Frozen exact KV/logit parity failed'
        parity.append(dict(request_id=row['request_id'],rank=rank,output_acceptance_NFE_exact=True,
            regular_graph_KV_exact=kv_checks,regular_graph_logits_exact=logits,independent_output_storage=independent))
        for repeat in range(3):
            order=['native','compact'] if (index+repeat)%2==0 else ['compact','native']
            for arm in order:
                value=generate(row,arm,'clean',repeat)
                assert value['output_ids']==truth[arm]['output_ids'] and value['NFE']==truth[arm]['NFE']
        for arm in (['native','compact'] if index%2==0 else ['compact','native']):
            value=generate(row,arm,'coarse',0)
            assert value['output_ids']==truth[arm]['output_ids'] and value['NFE']==truth[arm]['NFE']
            assert value['observation']['updates']==truth[arm]['observation']['updates']
        if index==0 and args.profile:
            for arm in arms:
                dist.barrier();torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False) as prof:
                    value=generate(row,arm,'profile',0)
                assert value['output_ids']==truth[arm]['output_ids'] and value['NFE']==truth[arm]['NFE']
                events=[]
                for event in prof.events():
                    events.append(dict(name=event.name,start_us=event.time_range.start,end_us=event.time_range.end,
                        device_type=str(event.device_type),device_index=event.device_index,thread=event.thread,
                        id=event.id,cpu_parent=None if event.cpu_parent is None else event.cpu_parent.id))
                with gzip.open(Path(args.output)/f'torch_events.{arm}.rank{rank}.json.gz','wt') as stream:json.dump(events,stream)
        write(Path(args.output)/f'parity.rank{rank}.json',parity)
    torch.cuda.synchronize();dist.barrier()
    write(Path(args.output)/f'worker_complete.rank{rank}.json',dict(success=True,GPUS=GPUS))
    os._exit(0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--prefix',type=int,choices=PREFIXES,default=16384)
    p.add_argument('--limit',type=int,default=8);p.add_argument('--split',choices=['calibration','held_out'],default='calibration')
    p.add_argument('--eager-prefill',action='store_true');p.add_argument('--profile',action='store_true');args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==MASK
    assert (ROOT/'manifests/frozen_control.json').exists()
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    mp.spawn(worker,args=(port,args),nprocs=4,join=True)
    write(Path(args.output)/'status.json',dict(success=True,GPUS=GPUS,candidate='EXACT_COMPACT_OUT_OF_PLACE_SCATTER'))


if __name__=='__main__':main()
