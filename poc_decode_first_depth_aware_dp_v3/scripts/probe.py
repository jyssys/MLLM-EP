"""Run the pinned native CDB engine. Do not modify its gate or cache policy."""
import argparse
import collections
import os
import time
import torch
from poc_decode_first_depth_aware_dp.runtime import load_engine, stats_dict
from poc_decode_first_depth_aware_dp_v3.contract import *

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--graphs', action='store_true')
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--part',type=int,default=0)
    p.add_argument('--parts',type=int,default=1)
    p.add_argument('--cap',type=int,default=256)
    p.add_argument('--quality',action='store_true')
    p.add_argument('--trace',action='store_true')
    p.add_argument('--fixed',action='store_true')
    p.add_argument('--no-refill',action='store_true')
    args=p.parse_args()
    out=Path(os.environ['LOOPED_KV_OUTPUT'])
    model,tok,engine=load_engine(q=None if args.fixed else .5,refill=not args.no_refill,
        graphs=args.graphs,batch=args.batch,blocks=2048 if args.batch>1 else 640,
        max_model_len=10240 if args.cap==8192 else 8192)
    if args.smoke:
        rows=[dict(id=f'smoke_{i}',dataset='sanity',prompt='Q: What is 2 + 3?\nA:',stop_strings=[])
              for i in range(args.batch)]
        args.cap=16
    elif args.quality:
        rows=[r for r in read(OLD_ROOT/'artifacts/quality_protocol_v3_pilot32.json')['rows'] if r['split']=='pilot']
        rows=rows[args.part::args.parts]
    else:
        rows=read(args.manifest)['rows'][args.part::args.parts]
    engine.generate_batch([tok.encode('Q: What is 1 + 1?\nA:')],max_new_tokens=4,
        eos_token_id=tok.eos_token_id,warmup=True)
    depths=[]
    launches=[]
    original=engine._record_exit
    def record(item, depth):
        depths.append(dict(request_id=item.state.request_id,token_position=item.token_position,
            depth=depth+1,tick=engine.last_stats.scheduler_ticks,host_s=time.perf_counter()))
        return original(item,depth)
    engine._record_exit=record
    methods=['_run_mixed_recurrent_batch','_run_prefill_batch','_stage_coda_batch']
    if args.trace:
        for method in methods:
            call=getattr(engine,method)
            def timed(*a,_call=call,_method=method,**kw):
                stream=engine.inputs_and_outputs.compute_stream
                begin=torch.cuda.Event(enable_timing=True)
                end=torch.cuda.Event(enable_timing=True)
                start=time.perf_counter()
                begin.record(stream)
                result=_call(*a,**kw)
                end.record(stream)
                launches.append((_method,start,time.perf_counter(),begin,end,len(a[0]) if isinstance(a[0],list) else None))
                return result
            setattr(engine,method,timed)
    progress=[]
    for offset in range(0,len(rows),args.batch):
        group=rows[offset:offset+args.batch]
        inputs=[tok.encode(r['prompt'],add_special_tokens=False) for r in group]
        assert all(len(ids)+args.cap<=engine.cdb_config.max_model_len for ids in inputs)
        depths.clear()
        launches.clear()
        # Mixed discovery domains share only native EOS. Quality uses its frozen stops.
        stops=[tok.encode(s,add_special_tokens=False) for s in group[0].get('stop_strings',[])] if args.quality else None
        torch.cuda.synchronize()
        start=time.perf_counter()
        values=engine.generate_batch(inputs,max_new_tokens=args.cap,eos_token_id=tok.eos_token_id,
            stop_sequences=stops,warmup=False)
        torch.cuda.synchronize()
        wall=(time.perf_counter()-start)*1000
        stats=stats_dict(engine.last_stats)
        assert sum(e['depth'] for e in depths)==stats['recurrent_steps']
        timing=[dict(stage=m,host_start_s=s-start,host_ms=(e-s)*1000,gpu_ms=b.elapsed_time(z),batch=n)
                for m,s,e,b,z,n in launches]
        write(out/f'batch_{offset:04d}.json',dict(wall_ms=wall,stats=stats,trace=args.trace,
            graphs=args.graphs,timings=timing,physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),
            memory_allocated=torch.cuda.memory_allocated(),peak_memory=torch.cuda.max_memory_allocated()))
        for row,value in zip(group,values):
            events=[e for e in depths if e['request_id']==value.request_id]
            tokens=value.generated_tokens
            assert len(events)==len(tokens)-1
            result=dict(id=row['id'],dataset=row['dataset'],prompt_tokens=len(value.prompt_ids),
                output_ids=tokens,text=tok.decode(tokens,skip_special_tokens=False),output_length=len(tokens),
                finish_reason='eos' if tokens[-1]==tok.eos_token_id else 'length_or_stop',
                depth_events=events,mean_depth=sum(e['depth'] for e in events)/len(events) if events else None,
                TTFT_ms=(value.first_token_time-value.created_time)*1000,
                E2E_ms=(value.lifespan[1]-value.created_time)*1000,
                decode_ms=(value.lifespan[1]-value.first_token_time)*1000,
                cap=args.cap,graphs=args.graphs,batch=args.batch,label='MEASURED_ONLINE',
                physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']))
            write(out/(row['id'].replace('/','_')+'.json'),result)
            progress.append(dict(id=row['id'],tokens=len(tokens),mean_depth=result['mean_depth']))
            print('REQUEST',progress[-1],flush=True)
        write(out/'progress.json',progress)
    write(out/'status.json',dict(success=True,requests=len(rows),online=True,graphs=args.graphs))

if __name__=='__main__':main()
