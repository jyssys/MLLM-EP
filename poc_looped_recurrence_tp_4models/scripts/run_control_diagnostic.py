"""Retained graph audit, sampler overhead, and bounded natural continuations."""
import argparse
import dataclasses
import gzip
import os
import statistics
import time
from poc_looped_recurrence_tp_4models.contract import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--family',choices=['native','tied'],default='native');parser.add_argument('--tp',type=int,required=True)
    parser.add_argument('--r',type=int,choices=[1,2,4],default=4);parser.add_argument('--repeat-source-grid',action='store_true')
    parser.add_argument('--output',required=True);args=parser.parse_args();output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    if args.model!='ouro':args.r=1
    os.environ.update(POC4_MODEL=args.model,POC4_FAMILY=args.family,POC4_R=str(args.r))
    assert os.environ['CUDA_VISIBLE_DEVICES']==','.join(map(str,GPUS[:args.tp]))
    from vllm import LLM,SamplingParams
    from transformers import AutoTokenizer
    kwargs=dict(model=str(model_path(args.model)),trust_remote_code=True,dtype='bfloat16',tensor_parallel_size=args.tp,
        max_model_len=32768,max_num_seqs=8,max_num_batched_tokens=32768,gpu_memory_utilization=.85,
        seed=41015,enable_prefix_caching=False,enable_chunked_prefill=False,enable_expert_parallel=False,
        enforce_eager=False,disable_log_stats=False,
        async_scheduling=False,distributed_executor_backend='mp',
        worker_extension_cls='poc_looped_recurrence_tp_4models.graph_probe.GraphProbe',
        compilation_config={'cudagraph_capture_sizes':[1,2,4,8]})
    if args.model=='ouro':kwargs['hf_overrides']={'total_ut_steps':args.r}
    write(output/'engine_contract.json',kwargs);llm=LLM(**kwargs)
    write(output/'graphs.json',llm.collective_rpc('graph_nodes',timeout=120))
    with gzip.open(ROOT/'results/frozen_tokens.json.gz','rt') as stream:
        rows=[r for r in json.load(stream) if r['model']==args.model]
    def generate(ids,count,teacher=None,eos=False,bypass=False):
        llm.collective_rpc('teacher',timeout=120,args=(teacher,))
        llm.collective_rpc('sampler_bypass',timeout=120,args=(bypass,))
        start=time.perf_counter_ns()
        result=llm.generate([{'prompt_token_ids':ids}],SamplingParams(temperature=0,max_tokens=count,ignore_eos=not eos),use_tqdm=False)[0]
        return dict(wall_ms=(time.perf_counter_ns()-start)/1e6,ids=list(result.outputs[0].token_ids),
            text=result.outputs[0].text,finish_reason=result.outputs[0].finish_reason,metrics=dataclasses.asdict(result.metrics))
    # Natural tokens establish a common trajectory for a same-input hook-cost control.
    ids=rows[0]['prompt_ids']['1024'];baseline=generate(ids,64)
    for _ in range(3):generate(ids,64,baseline['ids'])
    pairs=[]
    for repeat in range(5):
        arms=['native_sample','forced_copy'] if repeat%2==0 else ['forced_copy','native_sample']
        for arm in arms:
            value=generate(ids,64,baseline['ids'] if arm=='forced_copy' else None,bypass=arm=='native_sample')
            assert value['ids']==baseline['ids'],'Sampler observer must preserve the identical natural trajectory'
            pairs.append(dict(repeat=repeat,arm=arm,**value))
    write(output/'teacher_hook_overhead.json',dict(pairs=pairs,
        scope='Five technical pairs on one request. Same teacher IDs as the natural greedy trajectory. Not independent requests.'))
    if args.repeat_source_grid:
        # Independent model sessions test an observed short-prefill discrepancy.
        # This repeats calibration requests. It is not untouched confirmation.
        for _ in range(3):generate(rows[0]['prompt_ids']['1024'],64,rows[0]['teacher_ids'])
        repeated=[]
        for row in rows:
            for repeat in range(3):repeated.append(dict(request_id=row['request_id'],repeat=repeat,
                **generate(row['prompt_ids']['1024'],64,row['teacher_ids'])))
        write(output/'source_grid_repeat.json',dict(results=repeated,
            scope='Post-hoc independent-session diagnostic. Same eight frozen calibration requests. Retained graph policy. Not held-out.'))
    # The main kernel lane uses 16K. This lane checks the distinct 1K prefill kernel schedule.
    clean_results=[generate(ids,4,rows[0]['teacher_ids'][:4]) for _ in range(3)]
    clean=[r['wall_ms'] for r in clean_results]
    llm.collective_rpc('start_kernel_profile',timeout=120)
    profiled=generate(ids,4,rows[0]['teacher_ids'][:4])
    traces=llm.collective_rpc('finish_kernel_profile',timeout=120)
    write(output/'P1024_kernel_profile.json',dict(clean_ms=clean,profile_ms=profiled['wall_ms'],traces=traces,
        clean_metrics=[r['metrics'] for r in clean_results],profiled_metrics=profiled['metrics'],
        observer_percent=(profiled['wall_ms']/statistics.mean(clean)-1)*100))
    # Metadata-selected domains and prompts freeze before any model response.
    natural=[]
    tokenizer=AutoTokenizer.from_pretrained(model_path(args.model),trust_remote_code=True)
    probes=read_quality()
    write(output/'natural_prompt_manifest.json',probes)
    for row in probes:
        prompt=row['prompt']
        if tokenizer.chat_template:prompt=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
        encoded=tokenizer.encode(prompt,add_special_tokens=False)
        for repeat in range(2):natural.append(dict(request_id=row['id'],repeat=repeat,prompt_ids=encoded,
            chat_template_used=bool(tokenizer.chat_template),**generate(encoded,256,eos=True)))
    write(output/'natural_generation.json',dict(results=natural,
        scope=f'Bounded natural sanity only. No benchmark accuracy. No generated code execution. Ouro diagnostic R={args.r}.'))
    write(output/'status.json',dict(success=True,**vars(args)));os._exit(0)


def read_quality():
    return [dict(id='math',prompt='A box contains 12 pencils. Three students each take 2 pencils. How many pencils remain? Explain briefly.'),
        dict(id='coding',prompt='Write a Python function that returns the maximum value in a non-empty list. Include a brief explanation.'),
        dict(id='document',prompt='Explain why a scientific comparison needs a control group. Use one short paragraph.')]


if __name__=='__main__':main()
