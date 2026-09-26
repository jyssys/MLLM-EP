from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from poc_mllm_ep_foundation.io import atomic_json


RATIO = {"R0": 1.0, "R05": 0.5, "R025": 0.25}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--discovery-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="materialize all nine cells before discovery policy freeze",
    )
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    manifest = json.loads(args.discovery_manifest.read_text(encoding="utf-8"))
    base = {str(row["request_id"]): row for row in manifest["requests"]}
    baseline_rows = [
        row for row in analysis["scored_rows"] if row["config"] == "R0_K8"
    ]
    baseline_rows.sort(key=lambda row: int(row["vision_token_count"]))
    selected = []
    for quantile in (0.1, 0.5, 0.75, 0.9):
        target = float(np.quantile(
            [row["vision_token_count"] for row in baseline_rows], quantile
        ))
        chosen = min(
            (row for row in baseline_rows if row["base_request_id"] not in selected),
            key=lambda row: abs(float(row["vision_token_count"]) - target),
        )
        selected.append(str(chosen["base_request_id"]))

    if args.all_configs:
        configs = [
            f"{resolution}_K{k}"
            for resolution in ("R0", "R05", "R025")
            for k in (8, 6, 4)
        ]
    else:
        configs = ["R0_K8"]
        configs.extend(
            row["config"] for row in analysis["frozen_candidates"]
            if row["config"] not in configs
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for config in configs:
        resolution, kval = config.split("_K", 1)
        rows = []
        for base_id in selected:
            source = dict(base[base_id])
            source.update({
                "base_request_id": base_id,
                "request_id": f"timing_{base_id}__{resolution}__K{kval}",
                "resolution_label": resolution,
                "resolution_ratio_target": RATIO[resolution],
                "k_visual": int(kval),
                "k_text": 8,
                "k_decode": 8,
            })
            rows.append(source)
        target = args.output_dir / f"timing_{config.lower()}.json"
        atomic_json(target, {
            "schema_version": "mllm_resolution_expert_tradeoff.timing_manifest.v1",
            "config": config,
            "visual_bins": ["low", "mid", "mid_high", "high"],
            "requests": rows,
        })
        outputs.append(str(target.resolve()))
    print(json.dumps({"base_ids": selected, "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
