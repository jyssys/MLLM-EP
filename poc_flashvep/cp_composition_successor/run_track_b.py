#!/usr/bin/env python3
"""Native PCP/DCP transition screen on a four-GPU MLA MoE model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def make_ids(tokenizer, length: int) -> list[int]:
    seed = tokenizer.encode(
        "Analyze this exact context-parallel KV ownership test. ",
        add_special_tokens=False,
    )
    filler = tokenizer.encode(
        "Every global token position has one canonical cache owner and future keys remain invisible. ",
        add_special_tokens=False,
    )
    ids = list(seed)
    while len(ids) < length:
        ids.extend(filler)
    return ids[:length]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--topology", choices=("pcp_only", "dcp_only", "pcp_dcp"), required=True)
    parser.add_argument("--prompt-tokens", type=int, default=16384)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4,5,6,7"
    args.out.mkdir(parents=True, exist_ok=False)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    configs = {
        "pcp_only": dict(tensor_parallel_size=1, prefill_context_parallel_size=4,
                         decode_context_parallel_size=1),
        "dcp_only": dict(tensor_parallel_size=4, prefill_context_parallel_size=1,
                         decode_context_parallel_size=4),
        "pcp_dcp": dict(tensor_parallel_size=1, prefill_context_parallel_size=4,
                         decode_context_parallel_size=4),
    }
    cfg = configs[args.topology]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_ids = make_ids(tokenizer, args.prompt_tokens)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        device_ids=[0, 1, 2, 3],
        enforce_eager=True,
        kv_cache_memory_bytes=8 << 30,
        max_model_len=32768,
        max_num_batched_tokens=32768,
        max_num_seqs=4,
        enable_prefix_caching=False,
        enable_expert_parallel=False,
        enable_flashinfer_autotune=False,
        disable_log_stats=False,
        **cfg,
    )
    engine = llm.llm_engine
    pc = engine.vllm_config.parallel_config
    proof = {
        "vllm": __import__("vllm").__version__,
        "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "model": args.model,
        "topology": args.topology,
        "parallel_config": str(pc),
        "prompt_tokens": len(prompt_ids),
        "prompt_sha256": hashlib.sha256(bytes(str(prompt_ids), "utf-8")).hexdigest(),
    }
    (args.out / "runtime_proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    params = SamplingParams(
        temperature=0,
        max_tokens=args.output_tokens,
        ignore_eos=True,
        output_kind=RequestOutputKind.CUMULATIVE,
    )
    rows = []
    for iteration in range(args.warmups + args.repetitions):
        request_id = f"{args.topology}-{iteration}"
        rendered = llm._preprocess_cmpl_one({"prompt_token_ids": prompt_ids})
        submitted = time.monotonic()
        engine.add_request(request_id, rendered, params)
        token_times: list[float] = []
        output_ids: list[int] = []
        while engine.get_num_unfinished_requests():
            for output in engine.step():
                if output.request_id != request_id or not output.outputs:
                    continue
                ids = [int(x) for x in output.outputs[0].token_ids]
                metrics = output.metrics
                assert metrics is not None
                old = len(token_times)
                if len(ids) > old:
                    token_times.extend([float(metrics.last_token_ts)] * (len(ids) - old))
                    if old == 0:
                        token_times[0] = float(metrics.first_token_ts)
                output_ids = ids
        assert len(token_times) == args.output_tokens
        ttft_ms = 1000 * (token_times[0] - submitted)
        itls_ms = [1000 * (b - a) for a, b in zip(token_times, token_times[1:])]
        first_itl = itls_ms[0]
        steady = itls_ms[1:]
        steady_median = statistics.median(steady)
        transition_excess = max(0.0, first_itl - steady_median)
        e2e_ms = 1000 * (token_times[-1] - submitted)
        rows.append(
            {
                "iteration": iteration,
                "warmup": iteration < args.warmups,
                "ttft_ms": ttft_ms,
                "e2e_ms": e2e_ms,
                "first_itl_ms": first_itl,
                "steady_itl_p50_ms": steady_median,
                "steady_itl_p90_ms": percentile(steady, 0.9),
                "transition_excess_ms": transition_excess,
                "transition_direct_request_share_pct": 100 * transition_excess / e2e_ms,
                "output_ids": output_ids,
            }
        )
    measured = [row for row in rows if not row["warmup"]]
    summary = {
        "proof": proof,
        "rows": rows,
        "median": {
            key: statistics.median(float(row[key]) for row in measured)
            for key in (
                "ttft_ms", "e2e_ms", "first_itl_ms", "steady_itl_p50_ms",
                "steady_itl_p90_ms", "transition_excess_ms",
                "transition_direct_request_share_pct",
            )
        },
        "measured_output_ids": [row["output_ids"] for row in measured],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    engine.engine_core.shutdown()
    print(args.out / "summary.json")


if __name__ == "__main__":
    main()
