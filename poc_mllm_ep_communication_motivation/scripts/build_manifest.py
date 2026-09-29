#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from poc_mllm_ep_communication_motivation.contract import atomic_json, read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--decode-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cohort = {row["request_id"]: row for row in read_json(args.cohort)["requests"]}
    observed = {}
    with gzip.open(args.metadata, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            observed[row["request_id"]] = row
    candidates = []
    for request_id, source in cohort.items():
        if request_id not in observed or not Path(source["image_path"]).exists():
            continue
        measured = observed[request_id]
        candidates.append(
            {
                **source,
                "visual_token_count": int(measured["vision_token_count"]),
                "prompt_token_count": int(measured["prompt_token_count"]),
                "image_grid_thw": measured["image_grid_thw"],
            }
        )
    candidates.sort(key=lambda row: row["visual_token_count"])
    targets = (128, 672, 2048)
    selected = []
    used = set()
    for label, target in zip(("low", "medium", "high"), targets, strict=True):
        row = min(
            (r for r in candidates if r["request_id"] not in used),
            key=lambda r: abs(r["visual_token_count"] - target),
        )
        row = dict(row)
        row["size_class"] = label
        selected.append(row)
        used.add(row["request_id"])
    decode = read_json(args.decode_pool)["requests"][:8]
    atomic_json(
        args.output,
        {
            "schema_version": "comm_motivation_manifest.v1",
            "selection_frozen_before_new_routing": True,
            "q2_visual_sweep": selected,
            "q3_decode_pool": decode,
            "q3_large_visual": selected[-1],
            "source_assets": {
                "cohort": str(args.cohort.resolve()),
                "observed_metadata": str(args.metadata.resolve()),
                "decode_pool": str(args.decode_pool.resolve()),
            },
        },
    )


if __name__ == "__main__":
    main()
