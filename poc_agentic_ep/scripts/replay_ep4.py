"""TRUE EP4 replay of frozen semantic histories, with prefix caching on."""
import argparse
import json
import multiprocessing as mp
import os
import queue
import subprocess
import time
import traceback
from pathlib import Path
from poc_agentic_ep.contract import ROOT, MASK, TARGET, write
from poc_native_lp_ep4.scripts.network import choose_dp_port
from poc_mllm_ep_communication_motivation.scripts.run_q1_q2 import wave


def worker(rank,port,args,barrier,progress,rows):
    try:
        os.environ.update(VLLM_DP_RANK=str(rank),VLLM_DP_RANK_LOCAL=str(rank),VLLM_DP_SIZE='2',
            VLLM_DP_MASTER_IP='127.0.0.1',VLLM_DP_MASTER_PORT=str(port),
            FLASHVEP_CONFIGURED_ALL2ALL_BACKEND='deepep_high_throughput',FLASHVEP_CONFIGURED_DBO='false')
        from vllm import LLM, SamplingParams
        llm=LLM(model=TARGET,dtype='bfloat16',tensor_parallel_size=2,enable_expert_parallel=True,
            expert_placement_strategy='linear',all2all_backend='deepep_high_throughput',enable_ep_weight_filter=True,
            enable_dbo=False,enable_eplb=False,eplb_config={'num_redundant_experts':0},gpu_memory_utilization=.90,
            kv_cache_memory_bytes=8<<30,max_model_len=16384,max_num_batched_tokens=16384,max_num_seqs=1,
            enable_chunked_prefill=False,enable_prefix_caching=True,enable_flashinfer_autotune=False,
            enforce_eager=True,compilation_config={'mode':0},attention_config={'backend':'FLASH_ATTN'},
            async_scheduling=False,disable_log_stats=True)
        progress.put((rank,'MODEL_READY'));records=[]
        # Stage0 small diagnostic is separate from the frozen semantic cohort.
        schedule=[dict(request_id='sanity',trajectory_id='SANITY_ONLY',turn=-1,split='diagnostic',
            input_token_ids=[9707,11,576,649,498,3465,30],prompt_tokens=7,decode_tokens=0,
            exclusion_start=0)] + rows
        for wi,row in enumerate(schedule):
            c={k:v for k,v in row.items() if k not in ('input_token_ids','prompt_token_ids','teacher_forced_ids')}
            c.update(wave=wi,lane='trace',gamma=0)
            if rank==0:write(args.control,c)
            barrier.wait(timeout=240);progress.put((rank,f'WAVE_{wi}_START'))
            start=time.perf_counter()
            outputs=wave(llm,[{'prompt_token_ids':row['input_token_ids']}] if rank==0 else [],
                [SamplingParams(temperature=0.,seed=3407,max_tokens=1)] if rank==0 else [],barrier,wi)
            ms=1000*(time.perf_counter()-start)
            drained=llm.collective_rpc('sd_ep_flush',timeout=120)
            barrier.wait(timeout=120)
            records.append({**c,'dp_rank':rank,'diagnostic_instrumented_wave_wall_ms':ms,'drain':drained,
                'outputs':[dict(tokens=list(o.outputs[0].token_ids),finish_reason=o.outputs[0].finish_reason,
                    num_cached_tokens=getattr(o,'num_cached_tokens',None)) for o in outputs]})
            write(args.output/f'worker.dp{rank}.json',dict(ok=True,records=records))
            progress.put((rank,f'WAVE_{wi}_COMPLETE'))
        llm.llm_engine.engine_core.shutdown()
    except BaseException:
        write(args.output/f'worker.dp{rank}.failure.json',dict(traceback=traceback.format_exc()))
        progress.put((rank,'FAILED'));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--control',type=Path,required=True);p.add_argument('--watchdog',type=int,default=480)
    p.add_argument('--trajectory-limit',type=int,default=12);p.add_argument('--turn-limit',type=int,default=9)
    args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==MASK and os.environ.get('AGENT_EP_HOOK_IMPORTED')=='1'
    if args.output.exists():raise FileExistsError('new run ID required')
    args.output.mkdir(parents=True);write(args.control,dict(lane='bypass'))
    manifest=json.loads((ROOT/'results/frozen_manifest_v2.json').read_text())
    selected={t['trajectory_id'] for t in manifest['trajectories'][:args.trajectory_limit]}
    rows=[t for t in manifest['rows'] if t['trajectory_id'] in selected and t['turn']<args.turn_limit]
    write(args.output/'manifest.json',dict(mask=MASK,topology='TP2-DP2-EP4-PP1',prefix_caching=True,
        target=TARGET,rows=[t['request_id'] for t in rows],trace='new suffix + teacher-forced32 only',
        timing='DIAGNOSTIC_INSTRUMENTED_NOT_HEADLINE_LATENCY'))
    # Stage0 feasibility measurement precedes model launch, no concurrent load.
    subprocess.run([os.sys.executable,'-m','poc_agentic_ep.scripts.benchmark_p2p',str(args.output/'p2p.json')],check=True,timeout=90)
    ctx=mp.get_context('spawn');barrier=ctx.Barrier(2);progress=ctx.Queue();port=choose_dp_port()
    workers=[ctx.Process(target=worker,args=(rank,port,args,barrier,progress,rows)) for rank in (0,1)]
    for proc in workers:proc.start()
    last=time.monotonic();failure=None
    while any(proc.is_alive() for proc in workers):
        try:
            rank,status=progress.get(timeout=5);last=time.monotonic();print(rank,status,flush=True)
            if status=='FAILED':failure='RUNTIME_UNAVAILABLE worker failure';break
        except queue.Empty:pass
        if time.monotonic()-last>args.watchdog:failure='TIMEOUT bounded wave watchdog';break
    if failure:
        for proc in workers:
            if proc.is_alive():proc.terminate()
    for proc in workers:proc.join(timeout=15)
    ok=not failure and all(p.exitcode==0 for p in workers)
    write(args.output/'status.json',dict(ok=ok,failure=failure,exitcodes=[p.exitcode for p in workers]))
    if not ok:raise RuntimeError(failure or 'worker exit failure')


if __name__=='__main__':main()
