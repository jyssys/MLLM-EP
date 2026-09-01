#!/usr/bin/env python3
"""Bounded mixed prefill/decode serving run for the straggler forensic.

One long-lived text request generates a small decode stream while image
prefills are submitted in the same real vLLM scheduler invocation.  This is
diagnostic only: routes, placement, and model execution are untouched.
"""

from __future__ import annotations

import copy
import json
import multiprocessing as mp
import os
import socket
import time
import traceback
from pathlib import Path
from typing import Any

from poc_flashvep.ep4_serving_straggler_regime.run_serving import (
    PREVIOUS, _prepare_requests, _port, _write,
)


def _mixed_schedule(warmups: int, iterations: int, prefill_count: int) -> list[dict[str, Any]]:
    rows = []
    for i in range(warmups + iterations):
        rows.append({
            "wave": i,
            "batch_id": f"mixed_prefill_decode_c{prefill_count}_i{i}",
            "condition": "mixed_prefill_decode",
            "modality": "mixed",
            "concurrency": prefill_count + 1,
            "decode_requests": 1,
            "prefill_requests": prefill_count,
            "request_ids": ["text_21_fast_gptq"] + ["fast_gptq"] * prefill_count,
            "phase": "mixed",
            "instrument": True,
            "measured": i >= warmups,
            "iteration": i - warmups,
            "source_dp_rank": 0,
        })
    return rows


def _generate_mixed(llm: Any, prompts: list[dict[str, Any]], params: list[Any],
                    barrier: Any, wave: int) -> list[Any]:
    from vllm.outputs import RequestOutput
    from vllm.v1.engine import EngineCoreRequestType

    if prompts:
        barrier.wait(timeout=1800)
        llm._add_completion_requests(prompts, params, use_tqdm=False)
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)
    else:
        llm.llm_engine.engine_core._send_input(
            EngineCoreRequestType.START_DP_WAVE, (wave, -1)
        )
        barrier.wait(timeout=1800)
        outputs = []
    barrier.wait(timeout=1800)
    return outputs


def _run_rank(rank: int, port: int, args: Any, barrier: Any,
              schedule: list[dict[str, Any]]) -> None:
    output = args.output_dir / f"driver.dp_rank{rank}.json"
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": "2", "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_MATRIX_CONTROL": str((args.output_dir / "control.json").resolve()),
            "FLASHVEP_MATRIX_RAW_DIR": str((args.output_dir / "raw_live").resolve()),
            "FLASHVEP_SCHEDULER_TRACE_DIR": str((args.output_dir / "scheduler_trace").resolve()),
            "FLASHVEP_DEEPEP_PROOF_DIR": str((args.output_dir / "backend_proof").resolve()),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false", "FLASHVEP_SERVING_PROBE": "1",
        })
        from poc_flashvep.deepep_revalidation.backend_probe import install_backend_probe
        from poc_flashvep.ep4_serving_straggler_regime.live_instrumentation import install
        from poc_flashvep.ep4_serving_straggler_regime.serving_probe import install as install_scheduler
        install_backend_probe(); install(); install_scheduler()

        from vllm import LLM, SamplingParams
        requests, metadata = _prepare_requests(args.model_path)
        llm = LLM(
            model=args.model_path, dtype="bfloat16", tensor_parallel_size=2,
            enable_expert_parallel=True, expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput", enable_dbo=False,
            enable_return_routed_experts=False, enable_ep_weight_filter=True,
            trust_remote_code=True, gpu_memory_utilization=0.90,
            kv_cache_memory_bytes=8 << 30, max_model_len=16384,
            max_num_batched_tokens=int(args.max_num_batched_tokens), max_num_seqs=16,
            limit_mm_per_prompt={"image": 16}, skip_mm_profiling=True,
            enable_prefix_caching=False, enable_flashinfer_autotune=False,
            enforce_eager=True, disable_log_stats=True,
        )
        schedule_records = []
        for entry in schedule:
            if rank == 0:
                _write(args.output_dir / "control.tmp.json", entry)
                (args.output_dir / "control.tmp.json").replace(args.output_dir / "control.json")
                ids = entry["request_ids"]
                prompts = [copy.deepcopy(requests[ids[0]])]
                prompts += [copy.deepcopy(requests[ids[i]]) for i in range(1, len(ids))]
                params = [SamplingParams(max_tokens=64, temperature=0.0)]
                params += [SamplingParams(max_tokens=1, temperature=0.0)] * int(entry["prefill_requests"])
            else:
                prompts, params = [], []
            barrier.wait(timeout=1800)
            started = time.perf_counter_ns()
            outputs = _generate_mixed(llm, prompts, params, barrier, int(entry["wave"]))
            wall_ms = (time.perf_counter_ns() - started) / 1e6
            schedule_records.append({
                **entry, "driver_dp_rank": rank, "wall_ms": wall_ms,
                "output_count": len(outputs),
                "output_token_counts": [[int(t) for t in out.outputs[0].token_ids] for out in outputs],
                "prompt_metadata": [metadata[rid] for rid in entry["request_ids"]] if rank == 0 else [],
            })
        flush = {**schedule[-1], "wave": len(schedule), "batch_id": "flush",
                 "flush": True, "instrument": False, "measured": False}
        if rank == 0:
            _write(args.output_dir / "control.tmp.json", flush)
            (args.output_dir / "control.tmp.json").replace(args.output_dir / "control.json")
        barrier.wait(timeout=1800)
        _generate_mixed(llm, [copy.deepcopy(requests["fast_gptq"])] if rank == 0 else [],
                        [SamplingParams(max_tokens=1, temperature=0.0)] if rank == 0 else [],
                        barrier, int(flush["wave"]))
        _write(output, {"ok": True, "records": schedule_records})
    except BaseException:
        _write(output, {"ok": False, "traceback": traceback.format_exc()})
        raise


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefill-count", type=int, default=3)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule = _mixed_schedule(args.warmups, args.iterations, args.prefill_count)
    _write(args.output_dir / "schedule.json", schedule)
    _write(args.output_dir / "run_metadata.json", {
        "model_path": args.model_path, "configuration": {
            "dtype": "BF16", "tp": 2, "dp": 2, "ep": 4, "pp": 1,
            "all2all": "deepep_high_throughput", "dbo": False,
            "prefix_cache": False, "expert_placement": "linear",
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_model_len": 16384, "max_num_seqs": 16, "physical_gpus": [1,2,3,4],
        }, "warmups": args.warmups, "iterations": args.iterations,
        "scheduling": "real vLLM mixed prefill/decode batch",
        "workload_source": str(PREVIOUS), "prefill_count": args.prefill_count,
    })
    context = mp.get_context("spawn"); barrier = context.Barrier(2); port = _port()
    ps = [context.Process(target=_run_rank, args=(r, port, args, barrier, schedule)) for r in range(2)]
    for p in ps: p.start()
    for p in ps: p.join(5400)
    codes = [p.exitcode for p in ps]
    if codes != [0, 0]: raise RuntimeError(f"mixed serving run failed: {codes}")
    print(args.output_dir)


if __name__ == "__main__":
    main()
