"""Bounded metadata-only port diagnostic. Not MoE/EP performance evidence."""
import argparse
import json
import multiprocessing as mp
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from gpu_scope import allowed_devices, physical_gpu


def worker(rank,args):
    import torch
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    started=datetime.now(timezone.utc).isoformat()
    tm=json.loads(args.token_ids.read_text())
    special=torch.tensor(tm['special_ids']+[-1],device='cuda')
    records=[]
    for m in (1,4,16,128,512,2048):
        generator=torch.Generator(device='cuda').manual_seed(7381)
        ids=torch.randint(0,160000,(m,),device='cuda',generator=generator)
        ids[::3]=special[0]
        ops={'isin':lambda:torch.isin(ids,special),
             'broadcast_membership':lambda:(ids[:,None]==special[None,:]).any(-1)}
        assert torch.equal(ops['isin'](),ops['broadcast_membership']())
        for _ in range(20):
            for fn in ops.values():fn()
        torch.cuda.synchronize()
        rng=random.Random(812+m)
        for repeat in range(30):
            order=list(ops);rng.shuffle(order)
            for name in order:
                torch.cuda.synchronize()
                events=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                t0=time.perf_counter();events[0].record()
                for _ in range(48):ops[name]()
                events[1].record();events[1].synchronize()
                records.append(dict(rank=rank,physical_gpu=physical_gpu(rank),M=m,repeat=repeat,
                    implementation=name,cuda_ms=float(events[0].elapsed_time(events[1])),
                    wall_ms=1000*(time.perf_counter()-t0),calls=48,
                    scope='METADATA_ONLY_NOT_MOE_OR_EP_TIMING'))
        if m==1:
            for name,fn in ops.items():
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA]) as prof:fn()
                prof.export_chrome_trace(str(args.out/f'profile_gpu{physical_gpu(rank)}_{name}.json'))
                (args.out/f'ops_gpu{physical_gpu(rank)}_{name}.json').write_text(json.dumps([
                    dict(key=e.key,count=e.count,cpu_time_total_us=e.cpu_time_total,
                         device_time_total_us=e.device_time_total) for e in prof.key_averages()],indent=2))
    (args.out/f'rank{rank}.json').write_text(json.dumps(records))
    (args.out/f'completed_{rank}.json').write_text(json.dumps(dict(started_utc=started,
        finished_utc=datetime.now(timezone.utc).isoformat(),physical_gpu=physical_gpu(rank),
        status='COMPLETE',records=len(records),torch=torch.__version__,
        scope='PORT_METADATA_DIAGNOSTIC_NOT_MOE_BENCHMARK'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token-ids',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();allowed_devices()
    assert not args.out.exists(),args.out;args.out.mkdir(parents=True)
    ctx=mp.get_context('spawn');processes=[ctx.Process(target=worker,args=(r,args)) for r in range(4)]
    for proc in processes:proc.start()
    for proc in processes:proc.join()
    assert all(proc.exitcode==0 for proc in processes),[p.exitcode for p in processes]
