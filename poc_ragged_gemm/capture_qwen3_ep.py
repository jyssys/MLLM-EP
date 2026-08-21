#!/usr/bin/env python3
"""Capture bounded text-only Qwen3 EP4 live expert histograms and CUDA times."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def corpus() -> str:
    roots = [
        Path(__file__).parents[1] / "README.md",
        Path(__file__).parents[1] / "docs" / "flashvep_poc_spec.md",
        Path(__file__).parents[1] / "poc_flashvep" / "STATUS.md",
    ]
    chunks = [p.read_text(errors="ignore") for p in roots if p.exists()]
    base = "\n\n".join(chunks)
    if not base:
        base = "Mixture of experts systems route each token to a sparse subset of expert networks."
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4,5,6,7":
        raise RuntimeError("live capture requires exactly physical GPUs 4,5,6,7")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    control = args.output_dir / "control.json"; raw = args.output_dir / "raw_live"
    os.environ.update({"RAGGED_GEMM_CONTROL": str(control.resolve()), "RAGGED_GEMM_RAW_DIR": str(raw.resolve())})
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = tokenizer.encode(corpus(), add_special_tokens=False)
    max_len = 1536
    while len(ids) < max_len:
        ids += ids[: max_len - len(ids)]
    lengths = [128, 512, 1536]
    batches = [1, 4, 8]
    prompts: dict[int, str] = {}
    for length in lengths:
        prompts[length] = tokenizer.decode(ids[:length])
    llm = LLM(
        model=args.model, dtype="bfloat16", tensor_parallel_size=4,
        enable_expert_parallel=True, expert_placement_strategy="linear",
        all2all_backend="deepep_high_throughput", enable_dbo=False,
        enable_ep_weight_filter=True, enable_return_routed_experts=False,
        trust_remote_code=True, gpu_memory_utilization=0.90,
        kv_cache_memory_bytes=1 << 30, max_model_len=2048,
        max_num_batched_tokens=16384, max_num_seqs=8,
        enable_prefix_caching=False, enable_flashinfer_autotune=False,
        enforce_eager=True,
    )
    sampling = SamplingParams(max_tokens=1, temperature=0.0)
    schedule = []
    wave = 0
    # One untimed compile/warmup for each shape, followed by preregistered repeats.
    for length in lengths:
        for batch in batches:
            for repeat in range(-1, args.repeats):
                schedule.append({
                    "wave": wave, "workload": f"L{length}_B{batch}",
                    "prefill_tokens": length, "batch_size": batch, "repeat": repeat,
                    "instrument": repeat >= 0,
                }); wave += 1
    driver = []
    for entry in schedule:
        write_json(control, entry)
        request_prompts = [prompts[entry["prefill_tokens"]] + f"\nRequest {i}:" for i in range(entry["batch_size"])]
        outputs = llm.generate(request_prompts, sampling, use_tqdm=False)
        driver.append({**entry, "output_tokens": [int(o.outputs[0].token_ids[0]) for o in outputs]})
    flush = {**schedule[-1], "wave": wave, "flush": True, "instrument": False}
    write_json(control, flush)
    llm.generate([prompts[128]], sampling, use_tqdm=False)
    write_json(args.output_dir / "driver.json", {"schedule": schedule, "records": driver})
    print(args.output_dir)


if __name__ == "__main__":
    main()
