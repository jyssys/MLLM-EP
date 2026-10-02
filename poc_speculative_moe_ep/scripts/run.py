"""Native exact greedy SD, external DP, bounded request-level waves."""
import argparse
import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path

from poc_speculative_moe_ep.contract import ROOT,MASK,TARGET,DRAFT,write
from poc_native_lp_ep4.scripts.network import choose_dp_port
from poc_mllm_ep_communication_motivation.scripts.run_q1_q2 import wave


def worker(rank,port,args,barrier,progress,schedule):
    try:
        os.environ.update(VLLM_DP_RANK=str(rank),VLLM_DP_RANK_LOCAL=str(rank),VLLM_DP_SIZE='2',
            VLLM_DP_MASTER_IP='127.0.0.1',VLLM_DP_MASTER_PORT=str(port),
            FLASHVEP_CONFIGURED_ALL2ALL_BACKEND='deepep_high_throughput',FLASHVEP_CONFIGURED_DBO='false')
        from vllm import LLM,SamplingParams
        options=dict(model=TARGET,dtype='bfloat16',tensor_parallel_size=2,enable_expert_parallel=True,
            expert_placement_strategy='linear',all2all_backend='deepep_high_throughput',enable_ep_weight_filter=True,
            enable_dbo=False,enable_eplb=False,eplb_config={'num_redundant_experts':0},gpu_memory_utilization=.90,
            kv_cache_memory_bytes=2<<30,max_model_len=4096,max_num_batched_tokens=4096,max_num_seqs=1,
            enable_chunked_prefill=False,enable_prefix_caching=False,enable_flashinfer_autotune=False,
            enforce_eager=True,compilation_config={'mode':0},attention_config={'backend':'FLASH_ATTN'},
            async_scheduling=False,disable_log_stats=True)
        if args.gamma:
            options['speculative_config']=dict(method='draft_model',model=DRAFT,
                num_speculative_tokens=args.gamma,draft_tensor_parallel_size=2,enforce_eager=True)
        llm=LLM(**options)
        original_step=llm.llm_engine.step
        completion_clock={'ns':None}
        def measured_step(*a,**kw):
            results=original_step(*a,**kw)
            if any(getattr(r,'finished',False) for r in results):
                completion_clock['ns']=time.perf_counter_ns()
            return results
        llm.llm_engine.step=measured_step
        progress.put((rank,'MODEL_READY'))
        records=[]
        for wi,c in enumerate(schedule):
            control={k:v for k,v in c.items() if k!='row'}
            control.update(wave=wi,gamma=args.gamma,lane=args.lane if c['repeat']>=0 else args.lane,
                           request_id=c['row']['request_id'],output_cap=args.tokens,
                           prompt_tokens=c['row']['prompt_tokens'])
            if rank==0:write(args.control,control)
            barrier.wait(timeout=180)
            progress.put((rank,f'WAVE_{wi}_START'))
            completion_clock['ns']=None
            start=time.perf_counter_ns()
            outputs=wave(llm,[{'prompt_token_ids':c['row']['prompt_token_ids']}] if rank==0 else [],
                [SamplingParams(temperature=0.,seed=3407,max_tokens=args.tokens)] if rank==0 else [],barrier,wi)
            ms=(time.perf_counter_ns()-start)/1e6
            # Flush is outside host-wave headline timer, using deferred GPU events.
            drained=llm.collective_rpc('sd_ep_flush',timeout=90)
            barrier.wait(timeout=90)
            records.append({**control,'dp_rank':rank,'host_wave_ms':ms,'drain':drained,
                'actual_request_completion_ms':(completion_clock['ns']-start)/1e6 if completion_clock['ns'] is not None else None,
                'request_boundary':'wave-ready before admission -> native finished RequestOutput; excludes DP drain/RPC',
                'outputs':[dict(tokens=list(o.outputs[0].token_ids),text=o.outputs[0].text,
                                finish_reason=o.outputs[0].finish_reason) for o in outputs]})
            write(args.output/f'worker.dp{rank}.json',dict(ok=True,records=records))
            progress.put((rank,f'WAVE_{wi}_COMPLETE'))
        llm.llm_engine.engine_core.shutdown()
    except BaseException:
        write(args.output/f'worker.dp{rank}.failure.json',dict(traceback=traceback.format_exc()))
        progress.put((rank,'FAILED'));raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gamma',type=int,choices=[0,2,4,8],required=True)
    p.add_argument('--lane',choices=['trace','clean','bypass'],default='trace')
    p.add_argument('--requests',type=int,default=2);p.add_argument('--start',type=int,default=0)
    p.add_argument('--repeats',type=int,default=1);p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--tokens',type=int,default=64)
    p.add_argument('--watchdog',type=int,default=240)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--control',type=Path,required=True)
    args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==MASK
    assert os.environ.get('SD_EP_HOOK_IMPORTED')=='1'
    assert 1<=args.tokens<=128 and args.requests>0
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);write(args.control,dict(lane='bypass'))
    rows=json.loads((ROOT/'artifacts/frozen_pool.json').read_text())['rows'][args.start:args.start+args.requests]
    schedule=[dict(repeat=-i-1,row=rows[i%len(rows)]) for i in range(args.warmup)]
    schedule += [dict(repeat=r,row=row) for r in range(args.repeats) for row in (rows if r%2==0 else rows[::-1])]
    write(args.output/'manifest.json',dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        schedule=schedule,gpu_mask=MASK,topology='TP2-DP2-EP4-PP1'))
    ctx=mp.get_context('spawn');barrier=ctx.Barrier(2);progress=ctx.Queue();port=choose_dp_port()
    workers=[ctx.Process(target=worker,args=(rank,port,args,barrier,progress,schedule)) for rank in (0,1)]
    for proc in workers:proc.start()
    last=time.monotonic();failure=None
    while any(proc.is_alive() for proc in workers):
        try:
            rank,status=progress.get(timeout=5);last=time.monotonic();print(rank,status,flush=True)
            if status=='FAILED':failure='SD_RUNTIME_NOT_VALIDATED worker failure';break
        except queue.Empty:pass
        if time.monotonic()-last>args.watchdog:failure='TIMEOUT bounded wave watchdog';break
    if failure:
        for proc in workers:
            if proc.is_alive():proc.terminate()
    for proc in workers:proc.join(timeout=15)
    ok=not failure and all(p.exitcode==0 for p in workers)
    write(args.output/'status.json',dict(ok=ok,failure=failure,exitcodes=[p.exitcode for p in workers]))
    if not ok:raise RuntimeError(failure)


if __name__=='__main__':main()
