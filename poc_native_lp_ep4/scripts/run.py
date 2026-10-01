#!/usr/bin/env python3
"""Paired homogeneous serving waves with native LP on TRUE EP4."""
from __future__ import annotations

import argparse
import copy
import hashlib
import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path

from poc_native_lp_ep4.contract import MODEL, MODEL_REVISION, GPU_MASK, atomic_json, read_json, groups_for
from poc_mllm_ep_collective_granularity.scripts.run_serving import free_port, make_injection
from poc_mllm_ep_communication_motivation.scripts.run_q3_interference import image_prompt


def worker(rank, port, args, barrier, progress, manifest, waves):
    try:
        os.environ["VLLM_NO_USAGE_STATS"]="1"
        os.environ.update({"VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": "2", "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false"})
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import RequestOutputKind
        processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
        llm = LLM(model=MODEL, dtype="bfloat16", trust_remote_code=True,
            tensor_parallel_size=2, enable_expert_parallel=True,
            expert_placement_strategy="linear", all2all_backend="deepep_high_throughput",
            enable_ep_weight_filter=True, enable_dbo=False,
            gpu_memory_utilization=0.90, kv_cache_memory_bytes=4 << 30,
            max_model_len=20000,
            max_num_batched_tokens=20000 if args.chunk == 0 else args.chunk+4,
            max_num_seqs=5, enable_chunked_prefill=args.chunk>0,
            limit_mm_per_prompt={"image":1}, skip_mm_profiling=True,
            enable_prefix_caching=False, mm_processor_cache_gb=0,
            enable_flashinfer_autotune=False, enforce_eager=True,
            compilation_config={"mode":0}, attention_config={"backend":"FLASH_ATTN"},
            async_scheduling=False, disable_log_stats=True,
            logits_processors=["poc_mllm_vision_dominant_captioning.force_tokens:ForceTokensLogitsProcessor"])
        # LLM._add_request forces FINAL_ONLY. Use the same prepared engine
        # input while retaining DELTA output for per-iteration token evidence.
        def add_streaming(prompt, params, lora_request=None, priority=0):
            return llm.llm_engine.add_request(str(next(llm.request_counter)), prompt,
                params, lora_request=lora_request, priority=priority)
        llm._add_request = add_streaming
        # Engine timestamp is recorded after physical model execution, before
        # external-DP client polling/barriers. Client receive time can contain
        # output-queue backlog and is retained only as a secondary diagnostic.
        original_get=llm.llm_engine.engine_core.get_output
        completion_timestamp=[None]
        def get_output_timestamp():
            out=original_get()
            completion_timestamp[0]=out.timestamp
            return out
        llm.llm_engine.engine_core.get_output=get_output_timestamp
        progress.put((rank,"MODEL_READY"))
        residents = manifest["q3_decode_pool"][:4]
        prompts = [image_prompt(processor,row) for row in residents]
        records = []
        for wave_id, wave in enumerate(waves):
            progress.put((rank,f'WAVE_{wave_id}_START'))
            if args.atomic_wave_admission:
                # External-DP engines run independently of client polling.
                # Queue all four residents before either rank starts a wave;
                # otherwise serial image rendering admits 1/2/3-request
                # batches and can race idle/dummy transitions. This is the
                # installed pause/resume API, outside all measured windows.
                barrier.wait(timeout=300)
                llm.llm_engine.engine_core.call_utility('pause_scheduler','keep',False)
                barrier.wait(timeout=300)
            arm, modality, repeat = wave["arm"],wave["modality"],wave["repeat"]
            instrumented=args.trace and wave.get('instrumented',True)
            control = {"wave":wave_id,"arm":"FULL" if args.uniform_resident_prefill else arm,"condition":modality,
                "replicate":repeat,"outer_iteration":0,
                "capture_stage":instrumented and repeat>=0,"route_stats":instrumented and args.route_stats and repeat>=0,
                "special_token_ids":processor.tokenizer.all_special_ids,
                "injection_request_ids":[],"instrumentation_bypass":not instrumented}
            if rank==0:
                atomic_json(args.control,control)
            barrier.wait(timeout=300)
            params=[]
            for row in residents:
                tokens=row["forced_token_ids"][:args.decode_tokens]
                if len(tokens)!=args.decode_tokens:
                    raise RuntimeError("frozen resident continuation too short")
                params.append(SamplingParams(max_tokens=len(tokens),temperature=0.,
                    seed=3407,output_kind=RequestOutputKind.DELTA,
                    extra_args={"mllm_vds_forced_token_ids":tokens}))
            resident_ids=llm._add_completion_requests(copy.deepcopy(prompts),params,use_tqdm=False)
            resident_caller_ids=[str(r).split('-')[0] for r in resident_ids]
            injected_prompt,n_tokens,preprocess=make_injection(processor,modality,manifest["q3_large_visual"])
            # Diagnostic model-ready arrival: exactly the installed CPU
            # renderer, moved out of the client-controlled decode loop. It
            # does NOT cache/skip the GPU vision encoder or change pixels.
            prepared=None
            if args.preprocess_before_injection and injected_prompt is not None:
                prepare_begin=time.perf_counter_ns()
                prepared=llm._preprocess_cmpl_one(injected_prompt)
                preprocess['renderer_cpu_ms']=(time.perf_counter_ns()-prepare_begin)/1e6
            if args.atomic_wave_admission:
                barrier.wait(timeout=300)
                llm.llm_engine.engine_core.call_utility('resume_scheduler')
                barrier.wait(timeout=300)
            injections=[]
            injection_begin=None
            injection_finish=None
            injection_engine_finish=None
            injection_enqueue_ms=None
            steps=[]
            emitted={}
            finished=set()
            iteration=0
            start=time.monotonic()
            while llm.llm_engine.has_unfinished_requests():
                if time.monotonic()-start>args.wave_timeout:
                    raise TimeoutError("wave watchdog")
                resident_progress=min(len(emitted.get(r,[])) for r in resident_caller_ids)
                ready = resident_progress>=args.inject_at
                if args.uniform_resident_prefill:
                    # Every arm starts from the same resident KV state. CP
                    # applies only to the injected prefill, not to residents'
                    # earlier image prefills outside the measured window.
                    control['arm']=arm if ready else 'FULL'
                if rank==0 and injected_prompt is not None and not injections and ready:
                    injection_begin=time.perf_counter_ns()
                    injection_params=SamplingParams(max_tokens=8 if args.correctness else 1,
                            temperature=0.,seed=3407, output_kind=RequestOutputKind.DELTA,
                            extra_args={"native_lp_groups":groups_for(arm,n_tokens),
                                        "native_lp_capture":args.correctness})
                    if prepared is not None:
                        injections=[llm._add_request(copy.copy(prepared),injection_params)]
                    else:
                        injections=llm._add_completion_requests([copy.deepcopy(injected_prompt)],
                            [injection_params],use_tqdm=False)
                    injection_enqueue_ms=(time.perf_counter_ns()-injection_begin)/1e6
                control.update({"outer_iteration":iteration,
                    "injection_request_ids":[str(x) for x in injections],
                    "capture_stage":instrumented and repeat>=0 and resident_progress>=args.inject_at-2
                        and (injection_finish is None or iteration<=finish_iteration+2),
                    "route_stats":instrumented and args.route_stats and repeat>=0 and resident_progress>=args.inject_at-2
                        and (injection_finish is None or iteration<=finish_iteration+2)})
                baseline_target=wave.get('paired_condition','FULL')
                trace_end=args.inject_at+max(12,groups_for(baseline_target,2029)+8)
                if arm=='DECODE_ONLY' and resident_progress>trace_end:
                    control['capture_stage']=control['route_stats']=False
                if rank==0:
                    atomic_json(args.control,control)
                barrier.wait(timeout=300)
                begin=time.perf_counter_ns()
                outputs=llm.llm_engine.step()
                end=time.perf_counter_ns()
                engine_end=int(completion_timestamp[0]*1e9) if completion_timestamp[0] is not None else None
                barrier.wait(timeout=300)
                out_rows=[]
                for output in outputs:
                    rid=str(output.request_id)
                    tokens=list(output.outputs[0].token_ids) if output.outputs else []
                    emitted.setdefault(rid,[]).extend(tokens)
                    out_rows.append({"id":rid,"tokens":tokens,"finished":output.finished})
                    if output.finished:
                        finished.add(rid)
                    # Runtime may translate caller IDs, so injection can be
                    # recognized at completion by the unique short length.
                    if output.finished and len(emitted[rid]) in (1,8):
                        injection_finish=end
                        injection_engine_finish=engine_end
                        finish_iteration=iteration
                steps.append({"iteration":iteration,"step_ms":(end-begin)/1e6,
                    "end_ns":end,"engine_end_ns":engine_end,"outputs":out_rows})
                iteration+=1
                if iteration>args.decode_tokens+64:
                    raise RuntimeError("bounded iteration limit")
            resident_outputs=[v for v in emitted.values() if len(v)==args.decode_tokens]
            expected=[r["forced_token_ids"][:args.decode_tokens] for r in residents]
            if sorted(resident_outputs)!=sorted(expected):
                raise RuntimeError("resident forced continuation parity")
            record={**wave,"wave":wave_id,"dp_rank":rank,"steps":steps,
                "resident_ids":[str(x) for x in resident_ids],"emitted":emitted,
                "injection_ids":[str(x) for x in injections],"prompt_tokens":n_tokens,
                "injection_begin_ns":injection_begin,"injection_finish_ns":injection_finish,
                "injection_engine_finish_ns":injection_engine_finish,
                "injection_enqueue_ms":injection_enqueue_ms,
                "ttft_ms":None if injection_engine_finish is None else (injection_engine_finish-injection_begin)/1e6,
                "resident_parity":True,"instrumented":instrumented,**preprocess}
            records.append(record)
            atomic_json(args.output/f"worker.dp{rank}.json",{"ok":True,"waves":records})
            from poc_native_lp_ep4.native import flush
            flush()
            progress.put((rank,f"WAVE_{wave_id}_COMPLETE"))
        if args.trace:
            from poc_mllm_ep_communication_motivation.instrumentation import _drain
            from poc_mllm_ep_collective_granularity.instrumentation import _drain_events
            _drain(); _drain_events()
        from poc_native_lp_ep4.native import flush
        flush()
        llm.llm_engine.engine_core.shutdown()
    except BaseException:
        atomic_json(args.output/f"worker.dp{rank}.failure.json",{"traceback":traceback.format_exc()})
        progress.put((rank,"FAILED"))
        raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--manifest",type=Path,default=Path("poc_mllm_ep_communication_motivation/artifacts/manifests/frozen_manifest.json"))
    p.add_argument("--control",type=Path,default=Path("poc_native_lp_ep4/artifacts/control.json"))
    p.add_argument("--arms",default="DECODE_ONLY,FULL,LP_AUTO,LP_G4,LP_G8,LP_G16")
    p.add_argument("--modalities",default="VISUAL_PREFILL")
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--decode-tokens",type=int,default=64)
    p.add_argument("--inject-at",type=int,default=16)
    p.add_argument("--chunk",type=int,default=0)
    p.add_argument("--trace",action="store_true")
    p.add_argument("--route-stats",action="store_true")
    p.add_argument("--correctness",action="store_true")
    p.add_argument('--preprocess-before-injection',action='store_true',
                   help='diagnostic GPU/LM-ready arrival, CPU renderer outside victim loop')
    p.add_argument('--pair-baseline',action='store_true',help='fresh decode-only wave before each measured arm')
    p.add_argument('--interleave-instrumentation',action='store_true',
                   help='component overhead diagnostic: paired hook-bypass/trace waves in one engine')
    p.add_argument("--wave-timeout",type=int,default=90)
    p.add_argument("--watchdog",type=int,default=420)
    p.add_argument('--atomic-wave-admission',action='store_true')
    p.add_argument('--repeat-offset',type=int,default=0)
    p.add_argument('--uniform-resident-prefill',action='store_true')
    args=p.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES")==GPU_MASK
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    waves=[]
    arms=args.arms.split(",")
    for local_repeat in range(-1,args.repeats):
        repeat=local_repeat if local_repeat<0 else local_repeat+args.repeat_offset
        rotated=arms[repeat%len(arms):]+arms[:repeat%len(arms)] if repeat>=0 else arms
        for arm in rotated:
            if args.pair_baseline and arm=='DECODE_ONLY':
                continue
            modalities=args.modalities.split(',')
            if repeat>=0 and repeat%2:modalities=modalities[::-1]
            for modality in (['DECODE_ONLY'] if arm=='DECODE_ONLY' else modalities):
                anchor=None
                if args.pair_baseline:
                    anchor=len(waves)
                    waves.append({'arm':'DECODE_ONLY','modality':'DECODE_ONLY','repeat':repeat,
                                  'paired_condition':arm,'paired_modality':modality})
                waves.append({"arm":arm,"modality":"DECODE_ONLY" if arm=="DECODE_ONLY" else modality,
                              "repeat":repeat,'baseline_wave':anchor})
                if args.interleave_instrumentation and arm!='DECODE_ONLY':
                    waves[-1]['instrumented']=False
                    waves.append({"arm":arm,"modality":modality,"repeat":repeat,
                                  'baseline_wave':anchor,'instrumented':True,
                                  'overhead_control_wave':len(waves)-1})
                    if repeat>=0 and repeat%2:
                        waves[-2]['instrumented']=True
                        waves[-1]['instrumented']=False
                        waves[-2]['overhead_control_wave']=len(waves)-1
                        waves[-1].pop('overhead_control_wave',None)
    atomic_json(args.output/"manifest.json",{**vars(args),"output":str(args.output),
        "manifest":str(args.manifest),"control":str(args.control),"waves":waves,
        "physical_gpus":[0,1,4,5],"topology":"TP2-DP2-EP4-PP1",
        "model_revision":MODEL_REVISION,"vision_cached_control":os.environ.get("NATIVE_LP_VISION_CACHE")=="1",
        "component_trace_lane":"LOW_OVERHEAD" if os.environ.get('NATIVE_LP_LIGHT_TRACE')=='1' else 'ROUTE_DIAGNOSTIC' if args.trace else None,
        "injection_rule":"all four DP0 residents have emitted inject_at tokens",
        "primary_token_timestamp":"EngineCoreOutputs.timestamp monotonic completion, not client polling",
        "worker_token_capacity":20000,"chunk_budget_per_wave":"CP size plus four decode rows"})
    atomic_json(args.output/'source_signature.json',{str(path):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path('poc_native_lp_ep4/native.py'),Path('poc_native_lp_ep4/light_trace.py'),
                     Path('poc_native_lp_ep4/scripts/run.py'),Path('poc_native_lp_ep4/contract.py')]})
    atomic_json(args.control,{"condition":"STARTUP","capture_stage":False,"route_stats":False})
    ctx=mp.get_context("spawn")
    barrier=ctx.Barrier(2); progress=ctx.Queue()
    manifest=read_json(args.manifest); port=free_port()
    workers=[ctx.Process(target=worker,args=(rank,port,args,barrier,progress,manifest,waves)) for rank in (0,1)]
    for proc in workers: proc.start()
    last=time.monotonic();failure=None;model_ready=set()
    while any(proc.is_alive() for proc in workers):
        try:
            rank,status=progress.get(timeout=5)
            print(rank,status,flush=True);last=time.monotonic()
            if status=='MODEL_READY':model_ready.add(rank)
            if status=="FAILED": failure=f"dp{rank} failed";break
        except queue.Empty:
            pass
        active_watchdog=args.wave_timeout if model_ready=={0,1} else args.watchdog
        if time.monotonic()-last>active_watchdog:
            failure="TIMEOUT";break
        if any(proc.exitcode not in (None,0) for proc in workers):
            failure="worker error";break
    if failure:
        for proc in workers:
            if proc.is_alive(): proc.terminate()
    for proc in workers:
        proc.join(timeout=15)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
    atomic_json(args.output/"status.json",{"ok":failure is None,"failure":failure,
        "exitcodes":[proc.exitcode for proc in workers]})
    progress.cancel_join_thread()
    progress.close()
    if failure:
        # Failed children can strand multiprocessing finalizers. The scoped
        # GPU-session parent owns process-group teardown and burner restoration.
        import sys
        sys.stdout.flush();sys.stderr.flush()
        os._exit(1)


if __name__=="__main__": main()
