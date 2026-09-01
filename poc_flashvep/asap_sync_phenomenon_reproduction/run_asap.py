#!/usr/bin/env python3
"""Bounded real-vLLM ASAP synchronization reproduction.

The driver uses one host owner per DP engine and a barrier between waves; no
Python thread runs model forward.  Text prompts are exact token-id sequences,
so balanced/heterogeneous sequence composition is controlled without images.
"""

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


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0)); return int(s.getsockname()[1])


def _seq(tokenizer: Any, n: int) -> list[int]:
    base = tokenizer.encode(" the", add_special_tokens=False)
    if not base:
        base = [1]
    ids = (base * ((n + len(base) - 1) // len(base)))[:n]
    if tokenizer.bos_token_id is not None and n > 1:
        ids[0] = int(tokenizer.bos_token_id)
    return [int(x) for x in ids]


def _schedule(dp: int, mode: str, scale: int, delay: float, warmups: int, iterations: int) -> list[dict[str, Any]]:
    # Same token volume per DP group.  Heterogeneous composition is fixed
    # before measurements: one long request versus many short requests.
    if mode == "balanced":
        per = [[scale // 4] * 4 for _ in range(dp)]
    else:
        per = [[scale]] + [[scale // 8] * 8 for _ in range(dp - 1)]
    rows = []
    for it in range(warmups + iterations):
        rows.append({"wave": len(rows), "condition": f"{mode}_{scale}", "mode": mode, "scale": scale,
                     "delay_ms": delay, "measured": it >= warmups, "iteration": it - warmups,
                     "request_lengths_by_dp": per, "instrument": True, "phase": "prefill"})
    return rows


def _run_engine(llm: Any, prompts: list[Any], sampling: Any, barrier: Any, wave: int, dp: int) -> list[Any]:
    from vllm.outputs import RequestOutput
    from vllm.v1.engine import EngineCoreRequestType
    if prompts:
        barrier.wait(timeout=1800)
        llm._add_completion_requests(prompts, sampling, use_tqdm=False)
        out = llm._run_engine(RequestOutput, use_tqdm=False)
    else:
        llm.llm_engine.engine_core._send_input(EngineCoreRequestType.START_DP_WAVE, (wave, -1))
        out = []
    barrier.wait(timeout=1800)
    return out


def _worker(dp_rank: int, args: argparse.Namespace, barrier: Any, schedule: list[dict[str, Any]], port: int) -> None:
    output = args.output_dir / f"driver.dp{dp_rank}.json"
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(dp_rank), "VLLM_DP_RANK_LOCAL": str(dp_rank),
            "VLLM_DP_SIZE": str(args.dp), "VLLM_DP_MASTER_IP": "127.0.0.1", "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_MATRIX_CONTROL": str((args.output_dir / "control.json").resolve()),
            "FLASHVEP_MATRIX_RAW_DIR": str((args.output_dir / "raw_live").resolve()),
            "FLASHVEP_ASAP_RAW_DIR": str((args.output_dir / "asap_raw").resolve()),
            "FLASHVEP_SCHEDULER_TRACE_DIR": str((args.output_dir / "scheduler_trace").resolve()),
            "FLASHVEP_DEEPEP_PROOF_DIR": str((args.output_dir / "backend_proof").resolve()),
            "FLASHVEP_TOPOLOGY_PROOF_DIR": str((args.output_dir / "topology_proof").resolve()),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput", "FLASHVEP_CONFIGURED_DBO": "false",
            "FLASHVEP_SERVING_PROBE": "1", "FLASHVEP_INJECT_DP_RANK": str(args.inject_dp),
            "FLASHVEP_INJECT_LAYER": str(args.inject_layer), "FLASHVEP_INJECT_DELAY_MS": str(args.delay_ms),
        })
        from poc_flashvep.deepep_revalidation.backend_probe import install_backend_probe
        from poc_flashvep.ep4_serving_straggler_regime.serving_probe import install as install_scheduler
        from poc_flashvep.asap_sync_phenomenon_reproduction.asap_instrumentation import install
        install_backend_probe(); install(); install_scheduler()
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        llm = LLM(model=args.model_path, dtype="bfloat16", tensor_parallel_size=args.tp,
                   enable_expert_parallel=True, expert_placement_strategy="linear",
                   all2all_backend="deepep_high_throughput", enable_dbo=False,
                   enable_return_routed_experts=False, enable_ep_weight_filter=True,
                   trust_remote_code=True, gpu_memory_utilization=0.90, kv_cache_memory_bytes=4 << 30,
                   max_model_len=args.max_model_len, max_num_batched_tokens=args.max_num_batched_tokens,
                   max_num_seqs=32, enable_chunked_prefill=args.chunked_prefill,
                   enable_prefix_caching=False, enable_flashinfer_autotune=False,
                   enforce_eager=True, disable_log_stats=True)
        sampling = SamplingParams(max_tokens=1, temperature=0.0)
        records = []
        for entry in schedule:
            if dp_rank == 0:
                _write(args.output_dir / "control.tmp.json", entry); (args.output_dir / "control.tmp.json").replace(args.output_dir / "control.json")
            barrier.wait(timeout=1800)
            prompts = [{"prompt_token_ids": _seq(tokenizer, int(n))} for n in entry["request_lengths_by_dp"][dp_rank]]
            started = time.perf_counter_ns()
            outputs = _run_engine(llm, prompts, sampling, barrier, int(entry["wave"]), args.dp)
            wall = (time.perf_counter_ns() - started) / 1e6
            records.append({**entry, "dp_rank": dp_rank, "wall_ms": wall,
                            "prompt_lengths": [len(x["prompt_token_ids"]) for x in prompts],
                            "output_tokens": [[int(t) for t in o.outputs[0].token_ids] for o in outputs]})
        flush = {**schedule[-1], "wave": len(schedule), "flush": True, "instrument": False, "measured": False}
        if dp_rank == 0:
            _write(args.output_dir / "control.tmp.json", flush); (args.output_dir / "control.tmp.json").replace(args.output_dir / "control.json")
        barrier.wait(timeout=1800)
        _run_engine(llm, [{"prompt_token_ids": _seq(tokenizer, 32)}] if dp_rank == 0 else [], sampling, barrier, int(flush["wave"]), args.dp)
        _write(output, {"ok": True, "dp_rank": dp_rank, "records": records})
    except BaseException:
        _write(output, {"ok": False, "dp_rank": dp_rank, "traceback": traceback.format_exc()}); raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True); ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--topology", choices=("A", "B"), required=True); ap.add_argument("--mode", choices=("balanced", "heterogeneous"), required=True)
    ap.add_argument("--scale", type=int, default=8192); ap.add_argument("--delay-ms", type=float, default=0.0)
    ap.add_argument("--inject-dp", type=int, default=0); ap.add_argument("--inject-layer", type=int, default=24)
    ap.add_argument("--chunked-prefill", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192); ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--warmups", type=int, default=1); ap.add_argument("--iterations", type=int, default=1)
    args = ap.parse_args(); args.tp, args.dp = ((2, 2) if args.topology == "A" else (1, 4))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule = _schedule(args.dp, args.mode, args.scale, args.delay_ms, args.warmups, args.iterations)
    _write(args.output_dir / "schedule.json", schedule)
    _write(args.output_dir / "run_metadata.json", {"model_path": args.model_path, "topology": args.topology, "tp": args.tp, "dp": args.dp, "ep": 4, "pp": 1, "dtype": "BF16", "dbo": False, "chunked_prefill": args.chunked_prefill, "max_num_batched_tokens": args.max_num_batched_tokens, "scale": args.scale, "mode": args.mode, "delay_ms": args.delay_ms, "visible_devices": "1,2,3,4", "warmups": args.warmups, "iterations": args.iterations})
    ctx = mp.get_context("spawn"); barrier = ctx.Barrier(args.dp); port = _port()
    ps = [ctx.Process(target=_worker, args=(r, args, barrier, schedule, port)) for r in range(args.dp)]
    for p in ps: p.start()
    for p in ps: p.join(5400)
    codes = [p.exitcode for p in ps]
    if codes != [0] * args.dp: raise RuntimeError(f"run failed {codes}")
    print(args.output_dir)


if __name__ == "__main__": main()
