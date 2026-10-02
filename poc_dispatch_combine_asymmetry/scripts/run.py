"""Bounded native EP4 prefill arms, using validated external-DP wave protocol."""
import argparse
import copy
import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path

from poc_dispatch_combine_asymmetry.contract import MASK, MLLM, LLM, ROOT, write
from poc_native_lp_ep4.scripts.network import choose_dp_port
from poc_mllm_ep_communication_motivation.scripts.run_q1_q2 import wave, image_prompt
from poc_mllm_ep_communication_motivation.scripts.run_q3_interference import matched_text


def worker(rank, port, args, barrier, progress, schedule):
    try:
        os.environ.update(VLLM_DP_RANK=str(rank), VLLM_DP_RANK_LOCAL=str(rank), VLLM_DP_SIZE='2',
            VLLM_DP_MASTER_IP='127.0.0.1', VLLM_DP_MASTER_PORT=str(port),
            FLASHVEP_CONFIGURED_ALL2ALL_BACKEND='deepep_high_throughput', FLASHVEP_CONFIGURED_DBO='false')
        from transformers import AutoProcessor, AutoTokenizer
        from vllm import LLM as Engine, SamplingParams
        model = MLLM if args.model == 'mllm' else LLM
        processor = AutoProcessor.from_pretrained(MLLM, trust_remote_code=True)
        tokenizer = processor.tokenizer if args.model=='mllm' else AutoTokenizer.from_pretrained(model)
        options = dict(model=model, dtype='bfloat16', trust_remote_code=True, tensor_parallel_size=2,
            enable_expert_parallel=True, expert_placement_strategy='linear', all2all_backend='deepep_high_throughput',
            enable_ep_weight_filter=True, enable_dbo=False, gpu_memory_utilization=.90,
            kv_cache_memory_bytes=2<<30, max_model_len=20000, max_num_batched_tokens=20000,
            max_num_seqs=1, enable_chunked_prefill=False, enable_prefix_caching=False,
            enable_flashinfer_autotune=False, enforce_eager=True, compilation_config={'mode':0},
            attention_config={'backend':'FLASH_ATTN'}, async_scheduling=False, disable_log_stats=True)
        if args.model=='mllm':
            options.update(limit_mm_per_prompt={'image':1}, skip_mm_profiling=True, mm_processor_cache_gb=0)
        llm = Engine(**options)
        progress.put((rank,'MODEL_READY'))
        prepared = {}
        for entry in schedule:
            key = entry['arm']+'_'+entry['size']
            if key not in prepared:
                if entry['arm']=='visual':
                    prompt = image_prompt(processor, entry['request'])
                    count = entry['request']['prompt_token_count']
                else:
                    # Same frozen rendered text string for VL/LLM; count independently.
                    prompt, _ = matched_text(processor,entry['request']['prompt_token_count'])
                    count = len(tokenizer.encode(prompt,add_special_tokens=False))
                prepared[key] = prompt,count
        records = []
        for wave_id, entry in enumerate(schedule):
            prompt, count = prepared[entry['arm']+'_'+entry['size']]
            c = {k:v for k,v in entry.items() if k!='request'}
            c.update(wave=wave_id, prompt_tokens=count, physical_gpus=[0,1,4,5], topology='TP2-DP2-EP4-PP1',
                     model=args.model, capture_routes=entry['lane']=='internal' and entry['repeat']==0)
            if rank==0: write(args.control,c)
            barrier.wait(timeout=90)
            progress.put((rank,f'WAVE_{wave_id}_START'))
            start = time.perf_counter_ns()
            out = wave(llm,[copy.deepcopy(prompt)] if rank==0 else [],
                [SamplingParams(max_tokens=1,temperature=0.,seed=3407)] if rank==0 else [],barrier,wave_id)
            ms = (time.perf_counter_ns()-start)/1e6
            records.append({**c,'dp_rank':rank,'host_wave_ms':ms,
                            'first_tokens':[list(o.outputs[0].token_ids) for o in out]})
            write(args.output/f'worker.dp{rank}.json',{'ok':True,'records':records})
            progress.put((rank,f'WAVE_{wave_id}_COMPLETE'))
        llm.llm_engine.engine_core.shutdown()
    except BaseException:
        write(args.output/f'worker.dp{rank}.failure.json',{'traceback':traceback.format_exc()})
        progress.put((rank,'FAILED'))
        raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['mllm','llm'],default='mllm')
    p.add_argument('--arms',default='visual,text')
    p.add_argument('--sizes',default='high,low,medium')
    p.add_argument('--repeats',type=int,default=10)
    p.add_argument('--lane',choices=['clean','internal','bypass','overhead'],default='clean')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--control',type=Path,required=True)
    p.add_argument('--watchdog',type=int,default=420)
    p.add_argument('--append-internal',type=int,default=0)
    p.add_argument('--append-overhead',type=int,default=0)
    args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==MASK
    assert os.environ.get('ASYMMETRY_HOOK_IMPORTED')=='1'
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    rows=json.loads(Path('poc_mllm_ep_communication_motivation/artifacts/manifests/frozen_manifest.json').read_text())['q2_visual_sweep']
    schedule=[]
    for rep in range(-3,args.repeats):
        arms=args.arms.split(',')
        if rep>=0 and rep%2:arms=arms[::-1]
        for size in args.sizes.split(','):
            row=next(r for r in rows if r['size_class']==size)
            for arm in arms:
                lanes=['bypass','clean'] if args.lane=='overhead' else [args.lane]
                if rep%2:lanes=lanes[::-1]
                for lane in lanes:
                    schedule.append({'repeat':rep,'arm':arm,'size':size,'lane':lane,'request':row})
    for rep in range(args.append_internal):
        for size in args.sizes.split(','):
            row=next(r for r in rows if r['size_class']==size)
            for arm in args.arms.split(','):
                schedule.append({'repeat':rep,'arm':arm,'size':size,'lane':'internal','request':row})
    for rep in range(args.append_overhead):
        row=next(r for r in rows if r['size_class']=='high')
        for arm in args.arms.split(','):
            for lane in (['bypass','clean'] if rep%2==0 else ['clean','bypass']):
                schedule.append({'repeat':rep,'arm':arm,'size':'high','lane':lane,'request':row,'overhead_pair':True})
    write(args.output/'manifest.json',{'schedule':schedule,'runtime':{'EPLB':False,'redundant_experts':0,'DBO':False,
        'prefix_caching':False,'chunked_prefill':False,'eager':True,'TP':2,'DP':2,'EP':4,'PP':1}})
    write(args.control,{'lane':'bypass'})
    ctx=mp.get_context('spawn'); barrier=ctx.Barrier(2); progress=ctx.Queue(); port=choose_dp_port()
    workers=[ctx.Process(target=worker,args=(rank,port,args,barrier,progress,schedule)) for rank in (0,1)]
    for proc in workers:proc.start()
    last=time.monotonic();failure=None
    while any(proc.is_alive() for proc in workers):
        try:
            rank,status=progress.get(timeout=5);last=time.monotonic()
            print(rank,status,flush=True)
            if status=='FAILED':failure='worker failure';break
        except queue.Empty:pass
        if time.monotonic()-last>args.watchdog:failure='TIMEOUT no progress';break
    if failure:
        for proc in workers:
            if proc.is_alive():proc.terminate()
    for proc in workers:proc.join(timeout=20)
    ok=not failure and all(proc.exitcode==0 for proc in workers)
    write(args.output/'status.json',{'ok':ok,'failure':failure,'exitcodes':[p.exitcode for p in workers]})
    if not ok:raise RuntimeError(failure)


if __name__=='__main__':main()
