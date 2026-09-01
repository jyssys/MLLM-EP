#!/usr/bin/env python3
"""Bounded live workload for Text/single/repeated/diverse image comparison.

This runner only changes the request list.  The existing read-only
DeepEP/vLLM instrumentation records the production dispatch, expert and
combine calls; routing, placement and weights are untouched.
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

from poc_flashvep.live_traffic_matrix_validation.run_live import _generate
from poc_flashvep.vision_tile_motivation.profile_vision_tile_motivation import (
    _prepare_sample,
    expanded_sample_suite,
    sample_suite,
)

PREVIOUS = Path(
    "poc_flashvep/deepep_revalidation/results/"
    "live_prefill_execution_regime_20260821_111609"
)


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _image_catalog() -> dict[str, dict[str, Any]]:
    rows = sample_suite() + expanded_sample_suite()
    return {str(row["sample_id"]): row for row in rows}


def _rows(model_path: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Create a fixed, pre-registered workload; no row is selected by latency."""
    catalog = _image_catalog()
    # Small images keep repeated 8-image prompts below the validated 4K limit.
    image = lambda *names: [str(p) for name in names for p in catalog[name]["image_paths"]]
    question = "Compare the images and summarize their visible content briefly."
    image_rows = [
        {"request_id": "single_coins", "condition": "single_image", "category": "natural", "image_count": 1,
         "diversity": "single", "image_paths": image("coins"), "question": "Describe this image briefly."},
        {"request_id": "single_grass", "condition": "single_image", "category": "fine_grained", "image_count": 1,
         "diversity": "single", "image_paths": image("grass"), "question": "Describe this image briefly."},
        {"request_id": "repeat2_coins", "condition": "repeated_multi_image", "category": "natural", "image_count": 2,
         "diversity": "repeated", "image_paths": image("coins", "coins"), "question": question},
        {"request_id": "repeat4_coins", "condition": "repeated_multi_image", "category": "natural", "image_count": 4,
         "diversity": "repeated", "image_paths": image("coins", "coins", "coins", "coins"), "question": question},
        {"request_id": "repeat8_coins", "condition": "repeated_multi_image", "category": "natural", "image_count": 8,
         "diversity": "repeated", "image_paths": image("coins", "coins", "coins", "coins", "coins", "coins", "coins", "coins"), "question": question},
        {"request_id": "diverse2_small", "condition": "diverse_multi_image", "category": "natural", "image_count": 2,
         "diversity": "diverse", "image_paths": image("coins", "cat"), "question": question},
        {"request_id": "diverse4_small", "condition": "diverse_multi_image", "category": "natural", "image_count": 4,
         "diversity": "diverse", "image_paths": image("coins", "cat", "grass", "moon"), "question": question},
        {"request_id": "diverse8_small", "condition": "diverse_multi_image", "category": "natural", "image_count": 8,
         "diversity": "diverse", "image_paths": image("coins", "cat", "grass", "moon", "camera", "horse", "rocket", "astronaut"), "question": question},
    ]
    text_rows = json.loads((PREVIOUS / "text_prompts.json").read_text())
    by_id = {row["request_id"]: row for row in text_rows}
    # Length-matched controls for 1/2/4/8 image scales where locally available.
    controls = [
        ("text_control_small", "text_00_coins"),
        ("text_control_medium", "text_08_histology"),
        ("text_control_large", "text_21_fast_gptq"),
    ]
    rows = image_rows + [
        {"request_id": rid, "condition": "text_only", "image_count": 0,
         "diversity": "none", "image_paths": [], "question": "", "prompt": by_id[src]["prompt"]}
        for rid, src in controls
    ]
    # Prepare only to record processor token counts/spans in the manifest.  The
    # children repeat this deterministic preprocessing when submitting.
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    metadata: dict[str, dict[str, Any]] = {}
    for row in image_rows:
        _, meta = _prepare_sample(processor, row)
        metadata[row["request_id"]] = meta
    for row in rows:
        if row["condition"] == "text_only":
            metadata[row["request_id"]] = {**row, "processor_prompt_tokens": len(row["prompt"].split()), "processor_vision_tokens": 0, "images": []}
    return rows, metadata


def _run_rank(rank: int, port: int, args: argparse.Namespace, barrier: Any,
              schedule: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    out = args.output_dir / f"driver.dp_rank{rank}.json"
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": "2", "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_MATRIX_CONTROL": str((args.output_dir / "control.json").resolve()),
            "FLASHVEP_MATRIX_RAW_DIR": str((args.output_dir / "raw_live").resolve()),
            "FLASHVEP_DEEPEP_PROOF_DIR": str((args.output_dir / "backend_proof").resolve()),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false", "FLASHVEP_MATRIX_ENABLE": "1",
        })
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        requests: dict[str, dict[str, Any]] = {}
        for row in rows:
            if row["condition"] == "text_only":
                requests[row["request_id"]] = {"prompt": row["prompt"]}
            else:
                requests[row["request_id"]] = _prepare_sample(processor, row)[0]
        llm = LLM(
            model=args.model_path, dtype="bfloat16", tensor_parallel_size=2,
            enable_expert_parallel=True, expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput", enable_dbo=False,
            enable_return_routed_experts=False, enable_ep_weight_filter=True,
            trust_remote_code=True, gpu_memory_utilization=0.90,
            kv_cache_memory_bytes=1 << 30, max_model_len=4096,
            max_num_batched_tokens=16384, max_num_seqs=2,
            limit_mm_per_prompt={"image": 16}, skip_mm_profiling=True,
            enable_prefix_caching=False, enable_flashinfer_autotune=False,
            enforce_eager=True, disable_log_stats=True,
        )
        sampling = SamplingParams(max_tokens=1, temperature=0.0)
        records = []
        for entry in schedule:
            if rank == 0:
                tmp = args.output_dir / "control.tmp.json"
                _write(tmp, entry); tmp.replace(args.output_dir / "control.json")
            barrier.wait(timeout=1800)
            prompt = [copy.deepcopy(requests[entry["request_id"]])] if rank == entry["source_dp_rank"] else []
            start = time.perf_counter_ns()
            outputs = _generate(llm, prompt, sampling, barrier, int(entry["wave"]))
            wall_ms = (time.perf_counter_ns() - start) / 1_000_000
            tokens = [int(t) for output in outputs for t in output.outputs[0].token_ids]
            records.append({**entry, "driver_dp_rank": rank, "wall_ms": wall_ms, "output_tokens": tokens})
        flush = {**schedule[-1], "wave": len(schedule), "flush": True, "instrument": False, "measured": False}
        if rank == 0:
            tmp = args.output_dir / "control.tmp.json"; _write(tmp, flush); tmp.replace(args.output_dir / "control.json")
        barrier.wait(timeout=1800)
        prompt = [copy.deepcopy(requests[flush["request_id"]])] if rank == flush["source_dp_rank"] else []
        _generate(llm, prompt, sampling, barrier, int(flush["wave"]))
        _write(out, {"ok": True, "records": records})
    except BaseException:
        _write(out, {"ok": False, "traceback": traceback.format_exc()})
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, type=str)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=2)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    rows, metadata = _rows(args.model_path)
    # Fixed order, fixed seed-free workload: all conditions are scheduled
    # before measurements and no latency-based selection is possible.
    schedule = []
    for iteration in range(args.warmups + args.iterations):
        for index, row in enumerate(rows):
            schedule.append({
                "request_id": row["request_id"], "condition": row["condition"],
                "modality": "vision" if row["image_count"] else "text",
                "pair_id": index,
                "image_count": int(row["image_count"]), "diversity": row["diversity"],
                "token_bucket": "live_bounded", "prompt_tokens": int(metadata[row["request_id"]].get("processor_prompt_tokens", 0)),
                "source_dp_rank": index % 2, "phase": "main", "instrument": True,
                "measured": iteration >= args.warmups, "iteration": iteration - args.warmups,
            })
    for wave, entry in enumerate(schedule):
        entry["wave"] = wave
    _write(args.output_dir / "workload_rows.json", rows)
    _write(args.output_dir / "workload_metadata.json", metadata)
    _write(args.output_dir / "schedule.json", schedule)
    _write(args.output_dir / "run_metadata.json", {
        "model_path": args.model_path, "warmups": args.warmups, "iterations": args.iterations,
        "rows": len(rows), "configuration": {"dtype": "BF16", "tp": 2, "dp": 2, "ep": 4,
        "pp": 1, "all2all": "deepep_high_throughput", "dbo": False,
        "prefix_cache": False, "expert_placement": "linear", "physical_gpus": [1, 2, 3, 4]},
        "instrumentation": "existing live_traffic_matrix_validation CUDA-event hook",
    })
    context = mp.get_context("spawn"); barrier = context.Barrier(2); port = _port()
    ps = [context.Process(target=_run_rank, args=(rank, port, args, barrier, schedule, rows)) for rank in range(2)]
    for p in ps: p.start()
    for p in ps: p.join(3600)
    codes = [p.exitcode for p in ps]
    if codes != [0, 0]:
        raise RuntimeError(f"live run failed: {codes}")
    print(args.output_dir)


if __name__ == "__main__":
    main()
