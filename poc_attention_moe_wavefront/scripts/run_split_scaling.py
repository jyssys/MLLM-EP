#!/usr/bin/env python3
"""Measure exact sequential split stage costs over the registered cut grid."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from poc_attention_moe_wavefront.wavefront.oracle import candidate_splits
from poc_flashvep.non_dbo_causal_wavefront.run_stage0 import _port, _rank


REQUEST_MAP = {
    "W2_standard": "coffee",
    "W3_vision_heavy": "method",
    "W4_multi_image": "coffee_rocket",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--baseline-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    args.variant = "S"
    args.output_dir.mkdir(parents=True, exist_ok=False)

    source_manifest = json.loads((args.previous / "workload_manifest.json").read_text())
    source_items = {
        pair["vision"]["request_id"]: pair["vision"]
        for pair in source_manifest["pairs"]
    }
    baseline = json.loads(args.baseline_manifest.read_text())
    rows: list[dict] = []
    for workload_id, request_id in REQUEST_MAP.items():
        with np.load(args.previous / source_items[request_id]["route_file"]) as archive:
            total = int(len(archive["prompt_token_ids"]))
        if total != int(baseline[workload_id]["processor_prompt_tokens"]):
            raise AssertionError((workload_id, total, baseline[workload_id]))
        boundaries = [int(value) for value in baseline[workload_id]["modality_boundaries"]]
        # One untimed warmup at the global-static split.
        split = round(total / 3)
        rows.append({
            "request_id": request_id, "workload_id": workload_id,
            "variant": "S", "prompt_tokens": total,
            "prefix_tokens": split, "tail_tokens": total - split,
            "split_fraction": split / total, "phase": "warmup",
            "measured": False, "iteration": 0, "stage_profile": False,
            "flush_after": False,
        })
        for split_index, split in enumerate(candidate_splits(total, boundaries)):
            for repetition in range(args.repetitions):
                rows.append({
                    "request_id": request_id, "workload_id": workload_id,
                    "variant": "S", "prompt_tokens": total,
                    "prefix_tokens": int(split), "tail_tokens": total - int(split),
                    "split_fraction": int(split) / total,
                    "phase": "measured", "measured": True,
                    "iteration": split_index * args.repetitions + repetition,
                    "split_index": split_index, "repetition": repetition,
                    "stage_profile": True, "flush_after": False,
                })
    for wave, row in enumerate(rows):
        row["wave"] = wave
    rows[-1]["flush_after"] = True
    (args.output_dir / "schedule.json").write_text(json.dumps(rows, indent=2) + "\n")

    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    port = _port()
    processes = [
        context.Process(target=_rank, args=(rank, port, args, barrier, rows))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    exits = [process.exitcode for process in processes]
    if exits != [0, 0]:
        raise RuntimeError(f"split scaling failed: {exits}")


if __name__ == "__main__":
    main()
