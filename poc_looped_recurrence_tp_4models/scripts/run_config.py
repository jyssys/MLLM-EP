"""One native engine/config. All clean measurements use frozen teacher tokens."""
import argparse
import dataclasses
import gzip
import os
import signal
import statistics
import time
from poc_looped_recurrence_tp_4models.contract import *


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=MODELS,required=True);parser.add_argument('--family',choices=['native','tied'],default='native')
    parser.add_argument('--r',type=int,choices=[1,2,4],default=1);parser.add_argument('--tp',type=int,choices=[1,2,4],required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    assert os.environ['CUDA_VISIBLE_DEVICES']==','.join(map(str,GPUS[:args.tp]))
    assert args.model=='ouro' or (args.r==1 and args.family=='native')
    os.environ.update(POC4_MODEL=args.model,POC4_FAMILY=args.family,POC4_R=str(args.r))
    from vllm import LLM,SamplingParams
    def timeout(signum,frame):raise TimeoutError('Per-request300s watchdog')
    signal.signal(signal.SIGALRM,timeout)
    kwargs=dict(model=str(model_path(args.model)),trust_remote_code=True,dtype='bfloat16',tensor_parallel_size=args.tp,
        seed=41015,max_model_len=2048 if args.smoke else 32768,max_num_seqs=8,
        max_num_batched_tokens=2048 if args.smoke else 32768,
        enable_prefix_caching=False,enable_chunked_prefill=False,enable_expert_parallel=False,
        gpu_memory_utilization=.85,enforce_eager=args.smoke,disable_log_stats=False,async_scheduling=False,
        distributed_executor_backend='mp',worker_extension_cls='poc_looped_recurrence_tp_4models.worker_probe.FourModelProbe',
        compilation_config={'cudagraph_capture_sizes':[1,2,4,8]})
    if args.model=='ouro':kwargs['hf_overrides']={'total_ut_steps':args.r}
    if args.smoke:kwargs['kv_cache_memory_bytes']=8*1024**3
    write(output/'engine_contract.json',kwargs)
    llm=LLM(**kwargs)
    owned=llm.collective_rpc('inspect_four',timeout=120);write(output/'ownership.json',owned)
    layers=48*args.r if args.model=='ouro' else (28 if args.model=='qwen3' else 36)
    assert len(owned)==args.tp and all(r['cache_slots']==layers for r in owned)
    assert {r['GPU_uuid'].removeprefix('GPU-') for r in owned}==set(UUIDS[:args.tp])
    assert all('vllm.v1.worker.gpu.model_runner'==r['model_runner'] for r in owned),'Do not silently switch model runner'
    with gzip.open(ROOT/'results/frozen_tokens.json.gz','rt') as stream:rows=[r for r in json.load(stream) if r['model']==args.model]
    def generate(row,prefix,count=64,teacher=True):
        llm.collective_rpc('teacher',timeout=120,args=(row['teacher_ids'][:count] if teacher else None,))
        signal.alarm(300);start=time.perf_counter_ns()
        try:value=llm.generate([{'prompt_token_ids':row['prompt_ids'][str(prefix)]}],
            SamplingParams(temperature=0,max_tokens=count,ignore_eos=True),use_tqdm=False)[0]
        finally:signal.alarm(0)
        elapsed=(time.perf_counter_ns()-start)/1e6;ids=list(value.outputs[0].token_ids)
        if teacher:assert ids==row['teacher_ids'][:count]
        return dict(output_ids=ids,wall_ms=elapsed,metrics=dataclasses.asdict(value.metrics),
            cached_tokens=value.num_cached_tokens,finish_reason=value.outputs[0].finish_reason)
    # Bounded full-logit and written-KV probes precede candidate timing.
    for prefix in ([1024] if args.smoke else [1024,16384]):
        llm.collective_rpc('start_parity_capture',timeout=120)
        value=generate(rows[0],prefix,4)
        evidence=llm.collective_rpc('finish_parity_capture',timeout=120,args=(f'parity_P{prefix}',prefix))
        write(output/f'parity_P{prefix}.json',dict(result=value,evidence=evidence))
        if args.family=='tied':
            import numpy as np
            reference=output.parent/output.name.replace('_tied_','_native_')
            assert (reference/f'parity_P{prefix}.json').exists(),'Native matched reference must precede this arm'
            checks=[]
            for rank in range(args.tp):
                a=np.load(reference/f'parity_P{prefix}.rank{rank}.npz');b=np.load(output/f'parity_P{prefix}.rank{rank}.npz')
                for field in ['logits','hidden','KV']:
                    same=np.array_equal(a[field],b[field]);delta=np.abs(a[field].astype(float)-b[field].astype(float))
                    checks.append(dict(rank=rank,field=field,bitwise_equal=same,max_abs=float(delta.max()),
                        relative_L2=float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(a[field].astype(float).ravel()),1e-30))))
            write(output/f'parity_check_P{prefix}.json',checks)
            assert all(c['bitwise_equal'] for c in checks),'Frozen exact control gate failed'
    if args.smoke:
        write(output/'status.json',dict(success=True,scope='Eager feasibility only',**vars(args)));os._exit(0)
    for prefix in PREFIXES:
        chosen=rows if prefix!=32704 else [r for r in rows if r['request_id'] in ['coding_00','math_00','document_00']]
        for _ in range(3):generate(chosen[0],prefix)
        for row in chosen:
            for repeat in range(3):
                llm.collective_rpc('inspect_memory',timeout=120,args=(True,))
                value=generate(row,prefix)
                write(output/f'P{prefix}.{row["request_id"]}.repeat{repeat}.json',dict(
                    model=args.model,family=args.family,R=args.r,TP=args.tp,prefix=prefix,
                    request_id=row['request_id'],domain=row['domain'],repeat=repeat,**value))
                write(output/f'P{prefix}.{row["request_id"]}.repeat{repeat}.memory.json',
                    llm.collective_rpc('inspect_memory',timeout=120))
            print(args.model,args.family,args.r,args.tp,prefix,row['request_id'],round(value['wall_ms'],3),flush=True)
    # Actual all-forward counts and low-overhead outer intervals, separate from clean.
    llm.collective_rpc('runtime_counts_start',timeout=120);llm.collective_rpc('start_outer_events',timeout=120)
    measured=generate(rows[0],16384)
    outer=llm.collective_rpc('finish_outer_events',timeout=120);counts=llm.collective_rpc('runtime_counts_finish',timeout=120)
    write(output/'outer_and_counts.json',dict(result=measured,outer=outer,counts=counts,
        scope='64 actual model/sampler calls. Diagnostic observer lane, not primary clean wall.'))
    # One prefill plus three cached forwards. Kernel service is not an E2E share.
    clean=[generate(rows[0],16384,4)['wall_ms'] for _ in range(3)]
    llm.collective_rpc('start_kernel_profile',timeout=120)
    profiled=generate(rows[0],16384,4)
    traces=llm.collective_rpc('finish_kernel_profile',timeout=120)
    write(output/'kernel_profile.json',dict(clean_ms=clean,profile_ms=profiled['wall_ms'],traces=traces,
        observer_percent=(profiled['wall_ms']/statistics.mean(clean)-1)*100,
        limitation='Diagnostic kernel service. No globally aligned rank clocks or pure non-overlap claim.'))
    write(output/'status.json',dict(success=True,**vars(args)))
    os._exit(0)


if __name__=='__main__':main()
