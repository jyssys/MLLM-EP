#!/usr/bin/env python3
"""Bounded real-vLLM DeepSeek-V2-Lite EP4 route/timing capture.

Two driver processes (DP=2, TP=2) submit ordinary requests to the vLLM V1
engine.  The experiment is intentionally read-only: the hook only observes
the route histogram and wraps the existing fused expert call in CUDA events.
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

# FlashInfer's first-use rotary extension is built by the worker subprocess.
# Keep both validated environments on PATH for spawned vLLM workers.
os.environ["PATH"] = "/home/esjung/anaconda3/envs/flashvep-poc/bin:/home/esjung/.venvs/flashvep-deepep-v020/bin:" + os.environ.get("PATH", "")


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return int(s.getsockname()[1])


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def _domain_prompt(domain: str, target_tokens: int) -> str:
    seeds = {
        "code": "Implement and explain a robust Python algorithm for a streaming parser with error recovery. Include complexity and edge cases. ",
        "math": "Solve this problem carefully and show the derivation, checking every assumption and intermediate calculation. ",
        "reasoning": "Analyze the following scenario step by step, compare alternatives, and provide a precise justified conclusion. ",
        "factual": "Answer the question with accurate facts, definitions, and concise supporting context. ",
        "chat": "Please provide a helpful, clear response to the following general question and mention practical considerations. ",
        "long_code": "Design a production-quality distributed systems component. Discuss interfaces, failure modes, testing, and implementation details. ",
        "long_math": "Give a rigorous solution to a sequence of mathematical and logical subproblems, retaining all definitions and verifying the final result. ",
    }
    seed = seeds[domain]
    # Repetition is only a length/domain control; no route is synthesized.
    n = max(1, target_tokens // max(1, len(seed.split())))
    return (seed * n)[: max(64, target_tokens * 5)]


def _schedule(reps: int) -> list[dict[str, Any]]:
    fixed = [
        ("code", "math", 384, 384),
        ("math", "reasoning", 512, 256),
        ("reasoning", "factual", 768, 320),
        ("long_code", "chat", 1536, 256),
        ("long_math", "factual", 2048, 384),
        ("chat", "chat", 256, 256),
    ]
    out: list[dict[str, Any]] = []
    wave = 0
    for rep in range(reps):
        for d0, d1, n0, n1 in fixed:
            out.append({"wave": wave, "rep": rep, "condition": f"{d0}_vs_{d1}",
                        "batch_id": f"{d0}_vs_{d1}_rep{rep}",
                        "step": wave,
                        "domains": [d0, d1], "target_tokens": [n0, n1],
                        "measured": rep >= 1, "instrument": True})
            wave += 1
    return out


def _generate(llm: Any, prompt: str, sampling: Any, barrier: Any,
              request_tag: str) -> list[Any]:
    from vllm.outputs import RequestOutput
    barrier.wait(timeout=1800)
    llm._add_completion_requests([{"prompt": prompt}], sampling, use_tqdm=False)
    outputs = llm._run_engine(RequestOutput, use_tqdm=False)
    barrier.wait(timeout=1800)
    return outputs


def _run_rank(dp_rank: int, port: int, args: argparse.Namespace,
              barrier: Any, schedule: list[dict[str, Any]]) -> None:
    path = args.output / f"driver.dp_rank{dp_rank}.json"
    try:
        raw = args.output / "raw_live"
        os.environ.update({
            "VLLM_DP_RANK": str(dp_rank), "VLLM_DP_RANK_LOCAL": str(dp_rank),
            "VLLM_DP_SIZE": "2", "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_RL_RAW_DIR": str(raw.resolve()),
            "FLASHVEP_RL_CONTROL": str((args.output / "control.json").resolve()),
            "FLASHVEP_DEEPEP_PROOF_DIR": str((args.output / "backend_proof").resolve()),
            "FLASHVEP_RL_EXPERIMENT": "deepseekv2_ep4_straggler",
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false",
        })
        from poc_flashvep.deepep_revalidation.backend_probe import install_backend_probe
        from .instrumentation import install
        install_backend_probe(); install()
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model, dtype="bfloat16", tensor_parallel_size=2,
            enable_expert_parallel=True, expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput", enable_dbo=False,
            enable_return_routed_experts=False, enable_ep_weight_filter=True,
            trust_remote_code=True, gpu_memory_utilization=0.88,
            kv_cache_memory_bytes=6 << 30, max_model_len=4096,
            max_num_batched_tokens=8192, max_num_seqs=4,
            enable_prefix_caching=False, enable_flashinfer_autotune=False,
            moe_backend="auto", enforce_eager=True, disable_log_stats=True,
        )
        pc = llm.llm_engine.vllm_config.parallel_config
        proof = {
            "pid": os.getpid(), "dp_rank": dp_rank,
            "physical_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "model": args.model, "parallel_config_repr": str(pc),
            "tensor_parallel_size": int(getattr(pc, "tensor_parallel_size", -1)),
            "data_parallel_size": int(getattr(pc, "data_parallel_size", -1)),
            "enable_expert_parallel": bool(getattr(pc, "enable_expert_parallel", False)),
            "world_size": int(getattr(pc, "world_size", -1)),
            "max_num_batched_tokens": 8192, "max_model_len": 4096,
            "dtype": "bfloat16", "all2all_backend": "deepep_high_throughput",
            "dbo": False, "prefix_cache": False, "expert_placement": "linear",
        }
        _write(args.output / f"runtime_proof.dp_rank{dp_rank}.json", proof)
        sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
        records: list[dict[str, Any]] = []
        for entry in schedule:
            if dp_rank == 0:
                _write(args.output / "control.tmp.json", entry)
                (args.output / "control.tmp.json").replace(args.output / "control.json")
            barrier.wait(timeout=1800)
            dom = entry["domains"][dp_rank]
            prompt = _domain_prompt(dom, int(entry["target_tokens"][dp_rank]))
            start = time.perf_counter_ns()
            outs = _generate(llm, prompt, sampling, barrier, f"{entry['wave']}-dp{dp_rank}")
            wall = (time.perf_counter_ns() - start) / 1e6
            records.append({**entry, "dp_rank": dp_rank, "domain": dom,
                            "prompt_chars": len(prompt), "wall_ms": wall,
                            "output_tokens": [[int(t) for t in o.outputs[0].token_ids]
                                              for o in outs]})
        # Trigger event resolution only after all measured forwards.
        flush = {"wave": len(schedule), "batch_id": "flush", "flush": True,
                 "instrument": False, "measured": False}
        if dp_rank == 0:
            _write(args.output / "control.tmp.json", flush)
            (args.output / "control.tmp.json").replace(args.output / "control.json")
        barrier.wait(timeout=1800)
        _generate(llm, "flush", sampling, barrier, "flush")
        _write(path, {"ok": True, "records": records})
    except BaseException:
        _write(path, {"ok": False, "traceback": traceback.format_exc()})
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=1)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    schedule = _schedule(args.reps)
    _write(args.output / "schedule.json", schedule)
    _write(args.output / "run_metadata.json", {
        "model": args.model, "configuration": {
            "dtype": "BF16", "tp": 2, "dp": 2, "ep": 4, "pp": 1,
            "backend": "deepep_high_throughput", "placement": "linear",
            "dbo": False, "prefix_cache": False, "max_num_batched_tokens": 8192,
            "max_model_len": 4096, "physical_gpus": [1, 2, 3, 4],
        }, "reps": args.reps,
        "note": "Real text prompts; each DP driver owns its own ordinary vLLM requests.",
    })
    ctx = mp.get_context("spawn"); barrier = ctx.Barrier(2); port = _port()
    ps = [ctx.Process(target=_run_rank, args=(r, port, args, barrier, schedule))
          for r in (0, 1)]
    for p in ps: p.start()
    for p in ps: p.join(7200)
    for p in ps:
        if p.is_alive(): p.terminate(); p.join(30)
    codes = [p.exitcode for p in ps]
    _write(args.output / "run_status.json", {"exitcodes": codes, "ok": codes == [0, 0]})
    if codes != [0, 0]:
        raise SystemExit(f"driver failure: {codes}")


if __name__ == "__main__":
    main()
