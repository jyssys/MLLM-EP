#!/usr/bin/env python3
"""Bounded real-vLLM serving run for the two preregistered topologies."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import socket
import time
import traceback
from pathlib import Path
from typing import Any

from poc_flashvep.ep4_serving_straggler_regime.run_serving import _prepare_requests


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _request_catalog(model_path: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    return _prepare_requests(model_path)


def _ids(kind: str) -> list[str]:
    if kind == "vision":
        return ["coins", "coffee", "histology", "coffee_rocket", "fast_gptq", "method"]
    if kind == "text":
        return ["text_00_coins", "text_05_coffee", "text_08_histology", "text_14_coffee_rocket", "text_21_fast_gptq", "text_23_method"]
    if kind == "long":
        return ["long_6img_natural_fine", "long_10img_chart_mixed"]
    raise ValueError(kind)


def _distribution(dp_size: int, concurrency: int, kind: str, mode: str) -> dict[str, list[str]]:
    pool = _ids(kind)
    out = {str(i): [] for i in range(dp_size)}
    if mode == "balanced":
        for i in range(concurrency):
            out[str(i % dp_size)].append(pool[i % len(pool)])
    else:
        # This distribution is fixed before the run and deliberately mixes
        # short and long real requests; no latency-based selection is used.
        order = [pool[-1], pool[-2], pool[-3] if len(pool) > 2 else pool[0], pool[0],
                 pool[-1], pool[-2], pool[0], pool[min(1, len(pool) - 1)],
                 pool[-1], pool[-2], pool[-3] if len(pool) > 2 else pool[0],
                 pool[min(1, len(pool) - 1)], pool[-1], pool[-2],
                 pool[-3] if len(pool) > 2 else pool[0], pool[0]]
        if dp_size == 2:
            for i in range(concurrency):
                rank = 0 if i < (concurrency + 1) // 2 else 1
                out[str(rank)].append(order[i % len(order)])
        else:
            for i in range(concurrency):
                out[str(i % dp_size)].append(order[i % len(order)])
    return out


def build_schedule(dp_size: int, warmups: int, iterations: int, scope: str = "primary") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    kinds = (("vision", "vision"), ("text", "text"), ("long", "vision"))
    if scope == "smoke":
        kinds = (("vision", "vision"),)
    for kind, modality in kinds:
        concs = (2, 4, 8, 16) if kind != "long" else (2, 4, 8)
        if scope == "smoke":
            concs = (2, 4)
        elif scope == "stress":
            concs = (8, 16) if kind != "long" else (8,)
            kinds = kinds
        if scope == "stress" and kind == "text":
            # Keep a matched text control at the same high concurrency.
            concs = (8, 16)
        for mode in ("balanced", "heterogeneous"):
            for concurrency in concs:
                by_dp = _distribution(dp_size, concurrency, kind, mode)
                for it in range(warmups + iterations):
                    rows.append({
                        "wave": len(rows), "batch_id": f"{kind}_{mode}_c{concurrency}_i{it}",
                        "condition": f"{kind}_{mode}", "modality": modality,
                        "workload_kind": kind, "mode": mode, "concurrency": concurrency,
                        "request_ids_by_dp": by_dp,
                        "request_ids": [x for xs in by_dp.values() for x in xs],
                        "phase": "prefill", "instrument": True,
                        "measured": it >= warmups, "iteration": it - warmups,
                    })
    return rows


def _run_engine(llm: Any, prompts: list[dict[str, Any]], sampling: Any, barrier: Any, wave: int) -> list[Any]:
    from vllm.outputs import RequestOutput
    from vllm.v1.engine import EngineCoreRequestType
    barrier.wait(timeout=1800)
    if prompts:
        llm._add_completion_requests(prompts, sampling, use_tqdm=False)
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)
    else:
        llm.llm_engine.engine_core._send_input(EngineCoreRequestType.START_DP_WAVE, (wave, -1))
        outputs = []
    barrier.wait(timeout=1800)
    return outputs


def _worker(dp_rank: int, port: int, args: argparse.Namespace, barrier: Any, schedule: list[dict[str, Any]]) -> None:
    out = args.output_dir / f"driver.dp_rank{dp_rank}.json"
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(dp_rank), "VLLM_DP_RANK_LOCAL": str(dp_rank),
            "VLLM_DP_SIZE": str(args.dp), "VLLM_DP_MASTER_IP": "127.0.0.1", "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_MATRIX_CONTROL": str((args.output_dir / "control.json").resolve()),
            "FLASHVEP_MATRIX_RAW_DIR": str((args.output_dir / "raw_live").resolve()),
            "FLASHVEP_SCHEDULER_TRACE_DIR": str((args.output_dir / "scheduler_trace").resolve()),
            "FLASHVEP_DEEPEP_PROOF_DIR": str((args.output_dir / "backend_proof").resolve()),
            "FLASHVEP_TOPOLOGY_PROOF_DIR": str((args.output_dir / "topology_proof").resolve()),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false", "FLASHVEP_SERVING_PROBE": "1",
        })
        from poc_flashvep.deepep_revalidation.backend_probe import install_backend_probe
        from poc_flashvep.ep4_serving_straggler_regime.serving_probe import install as install_scheduler
        from poc_flashvep.dp_ep_arrival_skew_two_topologies.arrival_instrumentation import install
        install_backend_probe(); install(); install_scheduler()
        from vllm import LLM, SamplingParams
        requests, metadata = _request_catalog(args.model_path)
        llm = LLM(
            model=args.model_path, dtype="bfloat16", tensor_parallel_size=args.tp,
            enable_expert_parallel=True,
            expert_placement_strategy="linear", all2all_backend="deepep_high_throughput",
            enable_dbo=False, enable_return_routed_experts=False, enable_ep_weight_filter=True,
            trust_remote_code=True, gpu_memory_utilization=0.90, kv_cache_memory_bytes=8 << 30,
            max_model_len=16384, max_num_batched_tokens=args.max_num_batched_tokens, max_num_seqs=16,
            limit_mm_per_prompt={"image": 16}, skip_mm_profiling=True, enable_prefix_caching=False,
            enable_flashinfer_autotune=False, enforce_eager=True, disable_log_stats=True,
        )
        sampling = SamplingParams(max_tokens=1, temperature=0.0)
        records: list[dict[str, Any]] = []
        for entry in schedule:
            if dp_rank == 0:
                tmp = args.output_dir / "control.tmp.json"; _write(tmp, entry); tmp.replace(args.output_dir / "control.json")
            barrier.wait(timeout=1800)
            ids = entry["request_ids_by_dp"].get(str(dp_rank), [])
            prompts = [copy.deepcopy(requests[rid]) for rid in ids]
            started = time.perf_counter_ns()
            outputs = _run_engine(llm, prompts, sampling, barrier, int(entry["wave"]))
            wall = (time.perf_counter_ns() - started) / 1e6
            records.append({**entry, "driver_dp_rank": dp_rank, "wall_ms": wall,
                            "output_tokens": [[int(t) for t in o.outputs[0].token_ids] for o in outputs],
                            "prompt_metadata": [metadata[rid] for rid in ids]})
        flush = {**schedule[-1], "wave": len(schedule), "batch_id": "flush", "flush": True, "instrument": False, "measured": False}
        if dp_rank == 0:
            tmp = args.output_dir / "control.tmp.json"; _write(tmp, flush); tmp.replace(args.output_dir / "control.json")
        barrier.wait(timeout=1800)
        _run_engine(llm, [copy.deepcopy(requests["coins"])] if dp_rank == 0 else [], sampling, barrier, int(flush["wave"]))
        _write(out, {"ok": True, "dp_rank": dp_rank, "records": records})
    except BaseException:
        _write(out, {"ok": False, "dp_rank": dp_rank, "traceback": traceback.format_exc()})
        raise


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--model-path", required=True); ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--topology", choices=("A", "B"), required=True); ap.add_argument("--max-num-batched-tokens", type=int, default=16384)
    ap.add_argument("--warmups", type=int, default=1); ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--scope", choices=("smoke", "primary", "stress"), default="primary")
    args = ap.parse_args(); args.tp, args.dp = ((2, 2) if args.topology == "A" else (1, 4))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule = build_schedule(args.dp, args.warmups, args.iterations, args.scope); _write(args.output_dir / "schedule.json", schedule)
    _write(args.output_dir / "run_metadata.json", {"topology": args.topology, "tp": args.tp, "dp": args.dp, "ep": 4, "pp": 1, "dtype": "BF16", "dbo": False, "all2all": "deepep_high_throughput", "visible_devices": "1,2,3,4", "max_num_batched_tokens": args.max_num_batched_tokens, "warmups": args.warmups, "iterations": args.iterations})
    ctx = mp.get_context("spawn"); barrier = ctx.Barrier(args.dp); port = _port()
    procs = [ctx.Process(target=_worker, args=(r, port, args, barrier, schedule)) for r in range(args.dp)]
    for p in procs: p.start()
    for p in procs: p.join(7200)
    codes = [p.exitcode for p in procs]
    if codes != [0] * args.dp: raise RuntimeError(f"serving run failed: {codes}")
    print(args.output_dir)


if __name__ == "__main__":
    main()
