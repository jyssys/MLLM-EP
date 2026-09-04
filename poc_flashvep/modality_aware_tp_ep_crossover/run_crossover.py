"""Bounded real Qwen3-VL TP-only versus EP4 crossover runner.

One model instance is used per topology, with a fixed real-image prompt set,
two warmups, and paired repeated requests.  The worker hook records CUDA event
durations for each MoE layer without changing model execution.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoProcessor

MODEL = "/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
IMAGE = "/home/esjung/anaconda3/lib/python3.14/site-packages/skimage/data/astronaut.png"


def _stats(values: list[float]) -> dict[str, float]:
    values = [float(v) for v in values]
    ordered = sorted(values)
    def pct(q: float) -> float:
        if not ordered:
            return 0.0
        pos = (len(ordered) - 1) * q
        lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    return {"n": len(values), "median": statistics.median(values),
            "p25": pct(.25), "p75": pct(.75), "p90": pct(.90),
            "mean": statistics.fmean(values) if values else 0.0,
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0}


def _make_requests(processor: Any, images: list[Image.Image]) -> list[dict[str, Any]]:
    # The three populations have deliberately similar prompt lengths but
    # different real multimodal composition.  Image placeholders are expanded
    # by the Qwen3-VL processor/vLLM path, and final counts are recorded.
    text_fill = ("Explain the following topic carefully with concrete details. "
                 "Discuss assumptions, mechanisms, and edge cases. ")
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": text_fill * 165}]}],
        tokenize=False, add_generation_prompt=True)
    large_images = [im.resize((896, 896)) for im in images[:4]]
    vision = processor.apply_chat_template(
        [{"role": "user", "content": [
            *[{"type": "image", "image": im} for im in large_images],
            {"type": "text", "text": "Compare these images and explain their visual structure in detail."},
        ]}], tokenize=False, add_generation_prompt=True)
    # One real image plus a longer text prefix gives a mixed population.
    mixed = processor.apply_chat_template(
        [{"role": "user", "content": [
            *[{"type": "image", "image": im} for im in large_images[:2]],
            {"type": "text", "text": text_fill * 80},
        ]}], tokenize=False, add_generation_prompt=True)
    return [
        {"workload": "text_heavy", "modality": "text", "prompt": text},
        {"workload": "vision_heavy", "modality": "vision", "prompt": vision,
         "multi_modal_data": {"image": large_images}},
        {"workload": "mixed", "modality": "mixed", "prompt": mixed,
         "multi_modal_data": {"image": large_images[:2]}},
    ]


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--topology", choices=("tp_only", "ep4"), required=True)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--decode-tokens", type=int, default=1)
    args = ap.parse_args()
    out = args.output
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "worker_raw"; raw.mkdir()
    control = out / "control.json"
    os.environ["FLASHVEP_CROSSOVER_CONTROL"] = str(control.resolve())
    os.environ["FLASHVEP_CROSSOVER_RAW_DIR"] = str(raw.resolve())

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    image_paths = [
        IMAGE,
        "/home/esjung/anaconda3/lib/python3.14/site-packages/skimage/data/camera.png",
        "/home/esjung/anaconda3/lib/python3.14/site-packages/skimage/data/coffee.png",
        "/home/esjung/anaconda3/lib/python3.14/site-packages/skimage/data/chelsea.png",
    ]
    images = [Image.open(p).convert("RGB") for p in image_paths]
    requests = _make_requests(processor, images)
    # Store the exact chat templates, but not binary image data, in the manifest.
    manifest = {
        "model": args.model, "topology": args.topology,
        "physical_gpus": [1, 2, 3, 4], "visible_devices": "1,2,3,4",
        "dtype": "bfloat16", "tp": 4, "dp": 1, "pp": 1,
        "ep": 4 if args.topology == "ep4" else 0,
        "expert_parallel_enabled": args.topology == "ep4",
        "all2all_backend": "deepep_high_throughput" if args.topology == "ep4" else "not_applicable",
        "expert_placement": "linear", "moe_backend": "triton",
        "dbo": False, "prefix_caching": False, "enforce_eager": True,
        "warmups": args.warmups, "iterations": args.iterations,
        "decode_tokens": args.decode_tokens, "images": image_paths,
        "workloads": [{"workload": r["workload"], "modality": r["modality"],
                        "prompt": r["prompt"],
                        "token_count_template": len(processor.tokenizer.encode(r["prompt"]))}
                       for r in requests],
    }
    (out / "workload_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=args.model, dtype="bfloat16", tensor_parallel_size=4,
        enable_expert_parallel=args.topology == "ep4",
        expert_placement_strategy="linear", trust_remote_code=True,
        enable_ep_weight_filter=args.topology == "ep4",
        enable_return_routed_experts=False, gpu_memory_utilization=0.90,
        kv_cache_memory_bytes=1 << 30, max_model_len=4096,
        max_num_batched_tokens=4096, max_num_seqs=1,
        skip_mm_profiling=True, mm_processor_cache_gb=0,
        enable_prefix_caching=False, enable_flashinfer_autotune=False,
        enforce_eager=True, disable_log_stats=False, moe_backend="triton",
    )
    if args.topology == "ep4":
        kwargs["all2all_backend"] = "deepep_high_throughput"
    llm = LLM(**kwargs)
    pc = llm.llm_engine.vllm_config.parallel_config
    proof = {
        "requested": manifest,
        "parallel_config": {k: _jsonable(getattr(pc, k)) for k in (
            "tensor_parallel_size", "data_parallel_size", "pipeline_parallel_size",
            "enable_expert_parallel", "expert_placement_strategy", "all2all_backend",
            "use_sequence_parallel_moe", "enable_dbo") if hasattr(pc, k)},
        "world_size": _jsonable(getattr(pc, "world_size", None)),
        # vLLM 0.20 deliberately gates all-to-all MoE kernels on DP>1.  Keep
        # this derived field in the proof so a requested EP flag cannot be
        # mistaken for a live DeepEP communication path.
        "effective_moe_parallel": {
            "use_ep": _jsonable(getattr(pc, "enable_expert_parallel", False)),
            "use_all2all_kernels": bool(
                getattr(pc, "data_parallel_size", 1) > 1
                and getattr(pc, "enable_expert_parallel", False)
            ),
            "deep_ep_high_throughput_active": bool(
                getattr(pc, "data_parallel_size", 1) > 1
                and getattr(pc, "enable_expert_parallel", False)
                and args.topology == "ep4"
            ),
        },
        "hook": "poc_flashvep.modality_aware_tp_ep_crossover.worker_hook",
    }
    (out / "runtime_proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    sampling = SamplingParams(max_tokens=args.decode_tokens, temperature=0.0)

    rows: list[dict[str, Any]] = []
    reference_tokens: dict[str, list[int]] = {}
    for req in requests:
        for i in range(args.warmups):
            control.write_text(json.dumps({"wave": f"warmup-{req['workload']}-{i}",
                                           "workload": req["workload"],
                                           "modality": req["modality"], "measured": False}))
            llm.generate([req], sampling, use_tqdm=False)
    for iteration in range(args.iterations):
        for req in requests:
            control.write_text(json.dumps({"wave": f"measure-{iteration}-{req['workload']}",
                                           "workload": req["workload"],
                                           "modality": req["modality"],
                                           "iteration": iteration, "measured": True}))
            t0 = time.perf_counter_ns()
            result = llm.generate([req], sampling, use_tqdm=False)[0]
            wall = (time.perf_counter_ns() - t0) / 1e6
            token_ids = [int(x) for x in (result.prompt_token_ids or [])]
            if req["workload"] not in reference_tokens:
                reference_tokens[req["workload"]] = token_ids
            elif token_ids != reference_tokens[req["workload"]]:
                raise AssertionError(f"prompt tokenization changed: {req['workload']}")
            rows.append({"topology": args.topology, "workload": req["workload"],
                         "modality": req["modality"], "iteration": iteration,
                         "wall_ms": wall, "prompt_tokens": len(token_ids),
                         "output_token_ids": [int(x) for x in result.outputs[0].token_ids]})
    with (out / "request_wall.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["topology", "workload", "modality", "iteration", "wall_ms", "prompt_tokens", "output_token_ids"])
        writer.writeheader(); writer.writerows(rows)
    # Trigger a final event flush after the measured waves.  The hook flushes
    # at worker exit; this marker helps post-run analysis distinguish complete
    # from interrupted runs.
    (out / "run_complete.json").write_text(json.dumps({"rows": len(rows), "status": "complete"}, indent=2) + "\n")
    print(json.dumps({"topology": args.topology, "rows": len(rows), "wall_ms": _stats([r["wall_ms"] for r in rows])}, indent=2))


if __name__ == "__main__":
    main()
