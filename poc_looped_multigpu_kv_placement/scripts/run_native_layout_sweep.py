"""Official vLLM Ouro model. No custom attention/cache/model execution engine."""
import argparse
import dataclasses
import gzip
import os
import signal
import time
from poc_looped_multigpu_kv_placement.contract import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--tp',type=int,choices=[1,2,4],required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--eager',action='store_true')
    parser.add_argument('--prefixes',type=int,nargs='+',default=[1024,4096,8192,16384,32704])
    parser.add_argument('--limit',type=int,default=8);parser.add_argument('--concurrency',type=int,nargs='+',default=[1])
    parser.add_argument('--capacity-prefixes',type=int,nargs='+',default=[1024,32704])
    parser.add_argument('--profile-only',action='store_true');parser.add_argument('--trace-prefix',type=int,default=16384)
    parser.add_argument('--events-only',action='store_true')
    parser.add_argument('--quality-only',action='store_true')
    args=parser.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    expected=','.join(map(str,GPUS[:args.tp]));assert os.environ['CUDA_VISIBLE_DEVICES']==expected
    from vllm import LLM,SamplingParams
    def expired(signum,frame):raise TimeoutError('Bounded native request watchdog')
    signal.signal(signal.SIGALRM,expired)
    kwargs=dict(model=str(model_path('ouro')),trust_remote_code=True,dtype='bfloat16',
        tensor_parallel_size=args.tp,seed=41015,max_model_len=2048 if args.smoke else 32768,
        max_num_seqs=8,max_num_batched_tokens=2048 if args.smoke else 32768,
        enable_prefix_caching=False,enable_chunked_prefill=False,enable_expert_parallel=False,
        gpu_memory_utilization=.85,enforce_eager=args.eager or args.smoke,disable_log_stats=False,
        async_scheduling=False,distributed_executor_backend='mp',
        worker_extension_cls='poc_looped_multigpu_kv_placement.worker_probe.KVWorkerProbe',
        compilation_config={'cudagraph_capture_sizes':[1,2,4,8]})
    if args.smoke:kwargs['kv_cache_memory_bytes']=8*1024**3
    write(output/'engine_contract.json',kwargs)
    llm=LLM(**kwargs)
    write(output/'loaded_ownership.json',llm.collective_rpc('inspect_ownership',timeout=120,args=(True,)))
    if args.quality_only:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(model_path('ouro'),trust_remote_code=True)
        frozen=json.loads((ROOT/'results/frozen_quality.json').read_text())
        for row in [r for r in frozen['cases'] if r['model']=='ouro']:
            signal.alarm(300)
            value=llm.generate([{'prompt_token_ids':row['prompt_token_ids']}],
                SamplingParams(temperature=0,max_tokens=512,ignore_eos=False),use_tqdm=False)[0]
            signal.alarm(0)
            write(output/f'{row["request_id"]}.json',dict(request_id=row['request_id'],
                output_ids=list(value.outputs[0].token_ids),text=value.outputs[0].text,
                finish_reason=value.outputs[0].finish_reason,output_length=len(value.outputs[0].token_ids),
                scope='Natural calibration sanity, not benchmark accuracy. No code execution.'))
    elif args.events_only:
        with gzip.open(ROOT/'results/frozen_tokens.json.gz','rt') as stream:rows=[r for r in json.load(stream) if r['model']=='ouro'][:args.limit]
        params=SamplingParams(temperature=0,max_tokens=64,ignore_eos=True)
        prompt={'prompt_token_ids':rows[0]['prefix_token_ids'][str(args.trace_prefix)]}
        for _ in range(3):llm.generate([prompt],params,use_tqdm=False)
        for row in rows:
            prompt={'prompt_token_ids':row['prefix_token_ids'][str(args.trace_prefix)]};truth=None
            for repeat in range(3):
                for mode in (['clean','events'] if repeat%2==0 else ['events','clean']):
                    if mode=='events':llm.collective_rpc('start_outer_events',timeout=120)
                    start=time.perf_counter_ns();value=llm.generate([prompt],params,use_tqdm=False)[0];elapsed=(time.perf_counter_ns()-start)/1e6
                    record=llm.collective_rpc('finish_outer_events',timeout=120) if mode=='events' else None
                    current=list(value.outputs[0].token_ids)
                    if truth is None:truth=current
                    assert truth==current
                    write(output/f'{row["request_id"]}.{mode}.{repeat}.json',dict(request_id=row['request_id'],mode=mode,
                        prefix=args.trace_prefix,repeat=repeat,wall_ms=elapsed,output_ids=current,trace=record,parity=True))
    elif args.profile_only:
        with gzip.open(ROOT/'results/frozen_tokens.json.gz','rt') as stream:row=next(r for r in json.load(stream) if r['model']=='ouro')
        prompt={'prompt_token_ids':row['prefix_token_ids'][str(args.trace_prefix)]}
        params=SamplingParams(temperature=0,max_tokens=4,ignore_eos=True)
        for _ in range(3):llm.generate([prompt],params,use_tqdm=False)
        clean=[];truth=None
        for _ in range(3):
            start=time.perf_counter_ns();value=llm.generate([prompt],params,use_tqdm=False)[0];elapsed=(time.perf_counter_ns()-start)/1e6
            clean.append(elapsed);truth=list(value.outputs[0].token_ids)
        method='enable_loop_trace' if args.eager else 'start_kernel_profile'
        llm.collective_rpc(method,timeout=120)
        start=time.perf_counter_ns();value=llm.generate([prompt],params,use_tqdm=False)[0];elapsed=(time.perf_counter_ns()-start)/1e6
        observed=llm.collective_rpc('finish_loop_trace' if args.eager else 'finish_kernel_profile',timeout=120)
        assert truth==list(value.outputs[0].token_ids)
        write(output/'profile_observation.json',dict(clean_ms=clean,profile_ms=elapsed,output_parity=True,
            tokens=truth,prefix=args.trace_prefix,eager=args.eager,trace=observed,
            interpretation='Same-lane observer comparison. Eager source tracing is not optimized-native critical-path time.'))
    elif args.smoke:
        prompt={'prompt_token_ids':[1,102,211,17,38,61]}
        params=SamplingParams(temperature=0,max_tokens=4,ignore_eos=True,logprobs=5)
        result=llm.generate([prompt],params,use_tqdm=False)[0]
        write(output/'smoke.json',dict(tokens=list(result.outputs[0].token_ids),text=result.outputs[0].text,
            metrics=None if result.metrics is None else dataclasses.asdict(result.metrics),
            logprobs=[{str(k):dict(logprob=v.logprob,rank=v.rank) for k,v in row.items()} for row in result.outputs[0].logprobs]))
        write(output/'after_request_ownership.json',llm.collective_rpc('inspect_ownership',timeout=120))
    else:
        with gzip.open(ROOT/'results/frozen_tokens.json.gz','rt') as stream:rows=[r for r in json.load(stream) if r['model']=='ouro'][:args.limit]
        params=SamplingParams(temperature=0,max_tokens=64,ignore_eos=True)
        for p in args.prefixes:
            for warmup in range(3):llm.generate([{'prompt_token_ids':rows[0]['prefix_token_ids'][str(p)]}],params,use_tqdm=False)
            for concurrency in args.concurrency:
                if concurrency>1 and p not in args.capacity_prefixes:continue
                assert concurrency<=len(rows)
                for index in range(0,len(rows),concurrency):
                    chosen=rows[index:index+concurrency]
                    if len(chosen)!=concurrency:continue
                    prompts=[{'prompt_token_ids':r['prefix_token_ids'][str(p)]} for r in chosen]
                    for repeat in range(3):
                        llm.collective_rpc('inspect_memory',timeout=120,args=(True,))
                        signal.alarm(300);start=time.perf_counter_ns()
                        try:results=llm.generate(prompts,params,use_tqdm=False)
                        finally:signal.alarm(0)
                        end=time.perf_counter_ns()
                        records=[]
                        for row,value in zip(chosen,results):
                            assert len(value.outputs[0].token_ids)==64
                            records.append(dict(request_id=row['request_id'],domain=row['domain'],output_ids=list(value.outputs[0].token_ids),
                                text=value.outputs[0].text,metrics=None if value.metrics is None else dataclasses.asdict(value.metrics),
                                num_cached_tokens=value.num_cached_tokens,finish_reason=value.outputs[0].finish_reason))
                        label=f'P{p}.B{concurrency}.batch{index}.repeat{repeat}'
                        write(output/f'{label}.json',dict(tp=args.tp,prefix=p,concurrency=concurrency,repeat=repeat,
                            wall_ms=(end-start)/1e6,requests=records))
                        write(output/f'{label}.memory.json',llm.collective_rpc('inspect_memory',timeout=120))
                        print(label,round((end-start)/1e6,3),flush=True)
    write(output/'status.json',dict(success=True,tp=args.tp))
    # Guard owns process cleanup. Avoid framework teardown deadlocks after evidence flush.
    os._exit(0)


if __name__=='__main__':main()
