#!/usr/bin/env python3
"""Bounded real-vLLM Qwen3-30B-A3B TP2/DP4/EP8 trace.

Four ordinary vLLM DP drivers are synchronized per wave. Each wave submits one
real text prompt to every DP domain, so the underlying EP8 workers execute the
production routing and TritonExperts path. Instrumentation is read-only.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import time
import traceback
from pathlib import Path
from typing import Any

os.environ["PATH"] = (
    "/home/esjung/anaconda3/envs/flashvep-poc/bin:"
    "/home/esjung/.venvs/flashvep-deepep-v020/bin:"
    + os.environ.get("PATH", "")
)


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


SEEDS = {
    "code": "Implement a robust distributed systems component with interfaces, failure modes, tests, and complexity analysis. ",
    "math": "Solve this mathematical problem rigorously, showing derivations, intermediate checks, and edge cases. ",
    "reasoning": "Analyze the scenario step by step, compare alternatives, and justify the final conclusion precisely. ",
    "factual": "Answer this factual question accurately with definitions, evidence, and practical qualifications. ",
    "chat": "Provide a clear helpful answer to this general question and discuss practical considerations. ",
}


def _prompt(domain: str, target_tokens: int) -> str:
    seed = SEEDS[domain]
    # Repeated natural language controls prompt length/domain only; routes are
    # produced by the actual Qwen router.
    words = max(1, len(seed.split()))
    text = seed * max(1, target_tokens // words + 2)
    return text[: max(128, target_tokens * 5)]


def _schedule(reps: int) -> list[dict[str, Any]]:
    # The schedule is fixed before execution. It intentionally includes
    # balanced and heterogeneous per-DP token volumes and repeated domains.
    fixed = [
        ("balanced_2k", ["code", "math", "reasoning", "factual"], [2048] * 4),
        ("hetero_512_1k_2k_4k", ["code", "math", "reasoning", "chat"], [512, 1024, 2048, 4096]),
        ("vision_proxy_long", ["reasoning", "reasoning", "reasoning", "reasoning"], [1024, 2048, 3072, 4096]),
        ("long_balanced", ["code", "code", "math", "math"], [3072] * 4),
        ("short_mixed", ["chat", "factual", "chat", "factual"], [512] * 4),
        ("long_math", ["math", "math", "reasoning", "reasoning"], [4096] * 4),
    ]
    out: list[dict[str, Any]] = []
    wave = 0
    for rep in range(reps):
        for condition_idx, (condition, domains, tokens) in enumerate(fixed):
            out.append({
                "wave": wave, "step": wave, "rep": rep,
                "condition": condition, "batch_id": f"{condition}_rep{rep}",
                "domains": domains, "target_tokens": tokens,
                "measured": rep >= 1, "instrument": True,
                # The validated Qwen route hook consumes the same request/wave
                # metadata as its multimodal capture path.  Keep this metadata
                # explicit at the wave boundary; it is not read in the model
                # hot path and does not alter routing or scheduling.
                "request_id": f"qwen_ep8_{condition}_rep{rep}",
                "modality": "text",
                "pair_id": condition_idx,
                "token_bucket": "stage0",
                "phase": "natural_prefill",
                "iteration": wave,
                "source_dp_rank": -1,
            })
            wave += 1
    return out


def _generate(llm: Any, prompts: list[dict[str, Any]], sampling: Any,
              barrier: Any, wave: int) -> list[Any]:
    from vllm.outputs import RequestOutput
    from vllm.v1.engine import EngineCoreRequestType

    if prompts:
        barrier.wait(timeout=1800)
        llm._add_completion_requests(prompts, sampling, use_tqdm=False)
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)
    else:
        # Tell V1 DP scheduler that this rank participates in the wave without
        # owning a request; this keeps the EP8 collective wave aligned.
        llm.llm_engine.engine_core._send_input(
            EngineCoreRequestType.START_DP_WAVE, (wave, -1))
        barrier.wait(timeout=1800)
        outputs = []
    barrier.wait(timeout=1800)
    return outputs


def _run_rank(dp_rank: int, port: int, args: argparse.Namespace,
              barrier: Any, schedule: list[dict[str, Any]]) -> None:
    out = args.output / f"driver.dp_rank{dp_rank}.json"
    try:
        raw = args.output / "raw_live"
        os.environ.update({
            "VLLM_DP_RANK": str(dp_rank), "VLLM_DP_RANK_LOCAL": str(dp_rank),
            "VLLM_DP_SIZE": "4", "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_MATRIX_CONTROL": str((args.output / "control.json").resolve()),
            "FLASHVEP_MATRIX_RAW_DIR": str(raw.resolve()),
            "FLASHVEP_DEEPEP_PROOF_DIR": str((args.output / "backend_proof").resolve()),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false",
            "FLASHVEP_MATRIX_ENABLE": "1",
            "FLASHVEP_ACTION": str(args.action),
            "FLASHVEP_ACTION_RAW_DIR": str((args.output / "action_raw").resolve()),
            "FLASHVEP_ACTION_ARM_FILE": str((args.output / "action_armed").resolve()),
            "FLASHVEP_MIGRATION_BENCH": "1" if args.migration_bench else "0",
            "FLASHVEP_MIGRATION_RAW_DIR": str((args.output / "migration_raw").resolve()),
            "FLASHVEP_MIGRATION_ARM_FILE": str((args.output / "migration_armed").resolve()),
        })
        from poc_flashvep.deepep_revalidation.backend_probe import install_backend_probe
        from poc_flashvep.live_traffic_matrix_validation.instrumentation import install
        install_backend_probe(); install()
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model, dtype="bfloat16", tensor_parallel_size=2,
            enable_expert_parallel=True, expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput", enable_dbo=False,
            enable_return_routed_experts=False, enable_ep_weight_filter=True,
            trust_remote_code=True, gpu_memory_utilization=0.90,
            kv_cache_memory_bytes=6 << 30, max_model_len=8192,
            max_num_batched_tokens=16384, max_num_seqs=4,
            enable_prefix_caching=False, enable_flashinfer_autotune=False,
            moe_backend="auto", enforce_eager=True, disable_log_stats=True,
        )
        pc = llm.llm_engine.vllm_config.parallel_config
        proof = {
            "pid": os.getpid(), "dp_rank": dp_rank,
            "physical_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "parallel_config_repr": str(pc),
            "tensor_parallel_size": int(getattr(pc, "tensor_parallel_size", -1)),
            "data_parallel_size": int(getattr(pc, "data_parallel_size", -1)),
            "enable_expert_parallel": bool(getattr(pc, "enable_expert_parallel", False)),
            "world_size": int(getattr(pc, "world_size", -1)),
            "ep_size": 8, "experts_per_ep_rank": 16,
            "max_num_batched_tokens": 16384, "max_model_len": 8192,
            "dtype": "bfloat16", "all2all_backend": "deepep_high_throughput",
            "dbo": False, "prefix_cache": False, "expert_placement": "linear",
            "action": str(args.action),
        }
        _write(args.output / f"runtime_proof.dp_rank{dp_rank}.json", proof)
        sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
        # Synchronize all driver processes after their child workers have
        # initialized model/NCCL/DeepEP.  TEMP is armed only for real request
        # waves, never for vLLM's dummy profile run.
        barrier.wait(timeout=1800)
        if dp_rank == 0 and args.action != "KEEP":
            (args.output / "action_armed").touch()
        if dp_rank == 0 and args.migration_bench:
            (args.output / "migration_armed").touch()
        barrier.wait(timeout=1800)
        records: list[dict[str, Any]] = []
        for entry in schedule:
            if dp_rank == 0:
                _write(args.output / "control.tmp.json", entry)
                (args.output / "control.tmp.json").replace(args.output / "control.json")
            barrier.wait(timeout=1800)
            prompt = [{"prompt": _prompt(entry["domains"][dp_rank],
                                         int(entry["target_tokens"][dp_rank]))}]
            start = time.perf_counter_ns()
            outputs = _generate(llm, prompt, sampling, barrier, int(entry["wave"]))
            wall = (time.perf_counter_ns() - start) / 1e6
            records.append({**entry, "dp_rank": dp_rank,
                            "domain": entry["domains"][dp_rank],
                            "prompt_chars": len(prompt[0]["prompt"]),
                            "wall_ms": wall,
                            "output_tokens": [[int(t) for t in o.outputs[0].token_ids]
                                              for o in outputs]})
        flush = {"wave": len(schedule), "batch_id": "flush", "flush": True,
                 "instrument": False, "measured": False}
        if dp_rank == 0:
            _write(args.output / "control.tmp.json", flush)
            (args.output / "control.tmp.json").replace(args.output / "control.json")
        barrier.wait(timeout=1800)
        _generate(llm, [{"prompt": "flush"}], sampling, barrier, int(flush["wave"]))
        _write(out, {"ok": True, "records": records})
    except BaseException:
        _write(out, {"ok": False, "traceback": traceback.format_exc()})
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--action", default="KEEP",
                    choices=["KEEP", "TEMP_BALANCE", "CAPACITY_MILD", "CAPACITY_STRONG"],
                    help="experiment-only router action; KEEP is route preserving")
    ap.add_argument("--migration-bench", action="store_true",
                    help="one-shot actual Qwen expert-weight broadcast diagnostic")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    schedule = _schedule(args.reps)
    _write(args.output / "schedule.json", schedule)
    _write(args.output / "run_metadata.json", {
        "model": args.model,
        "configuration": {"dtype": "BF16", "tp": 2, "dp": 4, "ep": 8,
                           "pp": 1, "backend": "deepep_high_throughput",
                           "placement": "linear", "dbo": False,
                           "prefix_cache": False, "max_num_batched_tokens": 16384,
                           "max_model_len": 8192, "physical_gpus": list(range(8))},
        "reps": args.reps,
        "note": "Real text prompts; four synchronized ordinary vLLM DP drivers; largest prefill call selected during analysis.",
        "action": args.action,
        "migration_bench": bool(args.migration_bench),
    })
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(4); port = _port()
    ps = [ctx.Process(target=_run_rank, args=(r, port, args, barrier, schedule))
          for r in range(4)]
    for p in ps: p.start()
    for p in ps: p.join(10800)
    for p in ps:
        if p.is_alive(): p.terminate(); p.join(30)
    codes = [p.exitcode for p in ps]
    _write(args.output / "run_status.json", {"exitcodes": codes, "ok": codes == [0] * 4})
    if codes != [0] * 4:
        raise SystemExit(f"driver failure: {codes}")


if __name__ == "__main__":
    main()
