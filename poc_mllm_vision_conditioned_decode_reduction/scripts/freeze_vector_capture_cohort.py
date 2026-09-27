#!/usr/bin/env python3
"""Freeze the routing-blind discovery capture cohort and its causal tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--reference-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    tokens = {}
    for line in args.reference_results.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            tokens[str(row["pair_id"])] = [int(value) for value in row["output_token_ids"]]
    eligible = [
        row for row in manifest["anchors"]
        if row["analysis_split"] == "discovery" and len(tokens.get(row["pair_id"], [])) >= 128
    ][:args.limit]
    if len(eligible) != args.limit:
        raise RuntimeError(f"only {len(eligible)} eligible anchors")
    frozen = []
    for row in eligible:
        frozen.append({
            key: row[key]
            for key in (
                "pair_id", "analysis_split", "dataset", "dataset_sample_id", "question",
                "original_image_path", "original_image_sha256", "vision_token_count",
                "natural_output_token_count", "natural_output_sha256",
            )
        } | {"forced_token_ids": tokens[row["pair_id"]][:128]})
    payload = {
        "schema_version": 1,
        "selection": "first four discovery anchors with natural output >=128; routing blind",
        "routing_oracle_used": False,
        "forced_length": 128,
        "anchors": frozen,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
