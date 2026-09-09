#!/usr/bin/env python3
"""Faithful vLLM DCP/spec request-level matrix for a cached MLA MoE model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def percentile(xs: list[float], p: float) -> float:
    ys = sorted(xs)
    if not ys:
        return float("nan")
    at = (len(ys) - 1) * p / 100
    lo = int(at)
    hi = min(lo + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (at - lo)


def prompt(index: int, tokens: int) -> str:
    # Recurrent phrases make prompt-lookup speculation useful without changing
    # the target model or its exact greedy verification semantics.
    phrases = [
        "A distributed system processes requests in repeated stages. Analyze latency, correctness, and capacity carefully. ",
        "For every step verify the invariant, then explain the same invariant with one concrete example. ",
        "Context parallel attention and mixture-of-experts execution interact through memory and communication. ",
        "Write a precise technical assessment, repeat the key evidence, and finish with a concise conclusion. ",
    ]
    seed = phrases[index % len(phrases)]
    return (seed * max(1, tokens // 15 + 1))[: tokens * 5]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dcp", type=int, choices=[1, 2, 4], required=True)
    parser.add_argument("--spec", action="store_true")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4,5,6,7"
    args.out.mkdir(parents=True, exist_ok=False)

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    spec = None
    if args.spec:
        spec = {
            "method": "ngram",
            "num_speculative_tokens": 3,
            "prompt_lookup_min": 1,
            "prompt_lookup_max": 4,
        }
    started = time.time()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=4,
        decode_context_parallel_size=args.dcp,
        speculative_config=spec,
        trust_remote_code=True,
        gpu_memory_utilization=0.86,
        kv_cache_memory_bytes=4 << 30,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        max_num_seqs=max(16, args.requests),
        enable_prefix_caching=False,
        enable_flashinfer_autotune=False,
        enforce_eager=True,
        disable_log_stats=False,
    )
    engine = llm.llm_engine
    pc = engine.vllm_config.parallel_config
    proof = {
        "visible": os.environ["CUDA_VISIBLE_DEVICES"],
        "vllm_version": __import__("vllm").__version__,
        "model": args.model,
        "tp": pc.tensor_parallel_size,
        "dcp": pc.decode_context_parallel_size,
        "pcp": pc.prefill_context_parallel_size,
        "spec": spec,
        "parallel_config": str(pc),
        "engine_init_s": time.time() - started,
    }
    # vLLM enriches the supplied speculative dict with runtime config objects.
    (args.out / "runtime_proof.json").write_text(json.dumps(proof, indent=2, default=str))

    sampling = SamplingParams(
        temperature=0,
        max_tokens=args.output_tokens,
        ignore_eos=True,
        output_kind=RequestOutputKind.CUMULATIVE,
    )

    def cohort(label: str, count: int, warmup: bool) -> list[dict]:
        rows: dict[str, dict] = {}
        for i in range(count):
            rid = f"{label}-{i}"
            text = prompt(i, args.prompt_tokens)
            # RequestStateStats uses the process monotonic clock.
            submitted = time.monotonic()
            rendered = llm._preprocess_cmpl_one({"prompt": text})
            engine.add_request(rid, rendered, sampling)
            rows[rid] = {"request_id": rid, "submitted": submitted,
                         "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                         "token_times": [], "warmup": warmup}
        cohort_start = time.perf_counter()
        while engine.get_num_unfinished_requests():
            for output in engine.step():
                if output.request_id not in rows or not output.outputs:
                    continue
                row = rows[output.request_id]
                ids = list(output.outputs[0].token_ids)
                previous = len(row["token_times"])
                metrics = output.metrics
                if metrics is not None and ids:
                    ready = float(metrics.last_token_ts)
                    row["token_times"].extend([ready] * max(0, len(ids) - previous))
                    if previous == 0:
                        row["token_times"][0] = float(metrics.first_token_ts)
                if output.finished:
                    row["output_ids"] = ids
                    row["output_text"] = output.outputs[0].text
                    row["prompt_tokens_actual"] = len(output.prompt_token_ids or [])
                    row["finished"] = float(metrics.last_token_ts) if metrics else time.time()
        cohort_wall = time.perf_counter() - cohort_start
        for row in rows.values():
            times = row["token_times"]
            row["e2e_s"] = row["finished"] - row["submitted"]
            row["ttft_s"] = times[0] - row["submitted"]
            row["tpot_s"] = ((times[-1] - times[0]) / (len(times) - 1)
                             if len(times) > 1 else None)
            row["cohort_wall_s"] = cohort_wall
            row["tokens_per_s"] = sum(len(x.get("output_ids", [])) for x in rows.values()) / cohort_wall
        return list(rows.values())

    all_rows: list[dict] = []
    for i in range(args.warmup):
        all_rows += cohort(f"warmup{i}", min(2, args.requests), True)
    all_rows += cohort("measured", args.requests, False)
    with (args.out / "requests.jsonl").open("w") as sink:
        for row in all_rows:
            sink.write(json.dumps(row) + "\n")
    measured = [row for row in all_rows if not row["warmup"]]
    summary = {"count": len(measured)}
    for metric in ("e2e_s", "ttft_s", "tpot_s"):
        xs = [float(row[metric]) for row in measured if row.get(metric) is not None]
        summary[metric] = {"p50": percentile(xs, 50), "p90": percentile(xs, 90),
                           "p99": percentile(xs, 99), "mean": sum(xs) / len(xs)}
    summary["throughput_tokens_s"] = measured[0]["tokens_per_s"]
    summary["output_ids"] = [row["output_ids"] for row in measured]
    summary["prompt_tokens_actual"] = [row["prompt_tokens_actual"] for row in measured]
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
