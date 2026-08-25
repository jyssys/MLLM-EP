"""Capture fixed-layer hidden states and raw per-expert outputs."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Any

from poc_flashvep.cross_modal_routing_imprint.capture import MODEL, _prepare
from poc_flashvep.visual_expert_functional_redundancy.capture import (
    _json, _port, _run_rank, _suite,
)

LAYERS = [4, 12, 24, 36, 44, 47]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", default=MODEL)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _json(args.output_dir / "control.json", {"capture": False, "capture_id": "warmup"})

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    prepared_list = [_prepare(processor, row) for row in _suite()]
    prepared = {item[1]["sample_id"]: item for item in prepared_list}
    schedule = [{"wave": index, "sample_id": item[1]["sample_id"],
                 "capture_id": item[1]["sample_id"], "capture": True,
                 "source_dp_rank": index % 2}
                for index, item in enumerate(prepared_list)]
    policy: dict[str, Any] = {
        "layers": LAYERS,
        "regions": {"early": [4, 12], "middle": [24], "late": [36, 44, 47]},
        "samples_per_category": 8,
        "fixed_edge": 448,
        "fixed_prompt": "Describe the image briefly.",
        "hash_sample_fraction": 0.25,
        "matched_group_min": 8,
        "matched_group_cap": 32,
        "representative_ratios": [0.25, 0.5, 0.75, 1.0],
        "quality_threshold": {"cosine": 0.99, "relative_l2": 0.10},
        "selection": {
            "oracle": "greedy output-space cosine k-medoids",
            "practical": "greedy pre-expert-hidden cosine k-medoids",
        },
        "go": "at <=50% representatives, median visual cosine >=0.99 and relative-L2 <=0.10; visual required ratio >=20pp below text; practical hidden selection repeats the direction",
        "hold": "practical 50% visual cosine >=0.95 and relative-L2 <=0.20 with a positive required-ratio gap and >=2/4 diversity directions, or the oracle meets GO quality while practical selection retains a positive modality direction",
        "no_go": "little modality difference, visual diversity remains high, or only output-space oracle succeeds",
    }
    manifest = {
        "model": args.model_path,
        "configuration": {"dtype": "BF16", "tp": 2, "dp": 2, "ep": 4,
                          "pp": 1, "all2all": "deepep_high_throughput",
                          "physical_gpus": [1, 2, 3, 4], "dbo": False},
        "policy": policy,
        "schedule": schedule,
        "samples": [item[1] for item in prepared_list],
    }
    _json(args.output_dir / "manifest.json", manifest)
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    port = _port()
    processes = [ctx.Process(target=_run_rank,
                             args=(rank, port, args, prepared, schedule, barrier))
                 for rank in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    codes = [process.exitcode for process in processes]
    if codes != [0, 0]:
        raise RuntimeError(f"capture failed: {codes}")
    print(args.output_dir)


if __name__ == "__main__":
    main()
