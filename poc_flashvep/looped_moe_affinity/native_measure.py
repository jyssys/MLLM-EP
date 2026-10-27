"""Matched one-GPU timing and greedy outputs. No retained KV or forced token IDs."""
import time
import gzip
import json
import numpy as np
import torch
from .common import append,write
from .workloads import inputs
from .traces import RouterTrace
from .full_routes import FullRoutes
from .native_resume import inherit

def generate(a,ids,cap,trace=None):
    prefix=ids.clone();outputs=[];times=[]
    torch.cuda.synchronize();start=time.perf_counter()
    for step in range(cap):
        if trace is not None:
            trace.step=step;trace.phase='prefill' if step==0 else 'decode'
        logits=a.forward(prefix);token=logits[:,-1].argmax(-1,keepdim=True)
        torch.cuda.synchronize();outputs.append(token.flatten().cpu().tolist())
        times.append((time.perf_counter()-start)*1000)
        prefix=torch.cat([prefix,token],dim=-1);del logits
    return dict(e2e_ms=times[-1],ttft_ms=times[0],tpot_ms=(times[-1]-times[0])/(cap-1),
        timestamps_ms=times,generated_ids=outputs,cap=cap,eos_enabled=False,
        scope='LIVE_NATIVE_FULL_PREFIX_GREEDY_HARDWARE_GENERATION_NOT_NATURAL_QUALITY')

def measure(a,target,config):
    route_output=gzip.open(target/'decode_router_corpus.jsonl.gz','wt')
    completed=set()
    if config.get('reuse_native_from'):
        from .common import ROOT
        completed=inherit(ROOT/'raw'/config['reuse_native_from'],target,route_output)
    for b in [1,2,4,8,16]:
        for length in [128,512,2048]:
            if (b,length) in completed or [b,length] in config.get('skip_cells',[]):continue
            ids,requests=inputs(a,b,length)
            for _ in range(10):result=a.forward(ids);del result
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            duration=[]
            for _ in range(5):
                torch.cuda.synchronize();start=time.perf_counter();result=a.forward(ids)
                torch.cuda.synchronize();duration.append((time.perf_counter()-start)*1000);del result
            write(target/f'native_forward_b{b}_p{length}.rank0.json',dict(batch=b,prefix=length,clean_ms=duration,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),requests=[r['id'] for r in requests],evidence='LIVE_NATIVE_ONE_GPU'))
            cap=32;warm=generate(a,ids,cap)
            for repeat in range(3):
                result=generate(a,ids,cap)
                append(target/f'native_generation_b{b}_p{length}.rank0.jsonl',dict(result,batch=b,prefix=length,repeat=repeat,
                    output_tokens=b*cap,warmup_exact=result['generated_ids']==warm['generated_ids']))
                assert result['generated_ids']==warm['generated_ids']
            # Separate functional observers. They never enter the clean timing above.
            for split in ['calibration','heldout']:
                source,items=inputs(a,b,length,split=split)
                trace=RouterTrace(a);trace.frontier_only=True
                trace.request_ids=[r['id'] for r in items];trace.domain=[r['domain'] for r in items]
                generated=generate(a,source,cap,trace=trace);trace.close()
                if split=='heldout':assert generated['generated_ids']==warm['generated_ids'],'Observer changed native generation'
                for row in trace.records:
                    row.update(model=a.name,split=split,original_prefix_tokens=length)
                    route_output.write(json.dumps(row)+'\n')
                route_output.flush()
                full=FullRoutes(a);probe=a.forward(source);full.close()
                assert probe[:,-1].argmax(-1).cpu().tolist()==generated['generated_ids'][0]
                del probe
                np.savez_compressed(target/f'full_routes_b{b}_p{length}_{split}.npz',**full.values)
                write(target/f'full_routes_b{b}_p{length}_{split}.json',dict(batch=b,prefix=length,split=split,
                    requests=[r['id'] for r in items],domains=[r['domain'] for r in items],full_position_coverage=True,
                    scope='Separate native observer. First-token output unchanged. No timing claim.'))
            print('NATIVE_MEASURED',a.name,b,length,sum(duration)/len(duration),flush=True)
    route_output.close()
