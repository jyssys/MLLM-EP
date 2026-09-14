#!/usr/bin/env python3
"""Extract chronological real LLaDA2 compacted EP4 route streams."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True); ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--builder-root", type=Path, required=True); args = ap.parse_args()
    sys.path.insert(0, str(args.builder_root))
    from build_shape_atlas import load_raw, shape_class, summarize

    cases = load_raw(args.raw_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        serial = 0
        for key, case in sorted(cases.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
            if case["layer"] != args.layer:
                continue
            metric = summarize(case)
            if not metric["fresh_m"]:
                continue
            row = {
                "case_id": f"stream_{serial:03d}_{case['dataset']}_w{case['wave']}_l{case['layer']}",
                "dataset": case["dataset"], "request_template": case["dataset"],
                "wave": case["wave"], "layer": case["layer"], "phase": case["phase"],
                "shape_class": shape_class(metric["fresh_m"]),
                "physical_dense_m": case["physical_m"], "fresh_m": metric["fresh_m"],
                "global_m": metric["fresh_m"],
                "source_fresh_m": [len(x) for x in case["source_topk_ids"]],
                "topk": 8, "hidden": 4096, "experts": 256, "experts_per_rank": 64,
                "source_topk_ids": case["source_topk_ids"],
                "rank_load": metric["rank_load"], "rank_load_cv": metric["rank_load_cv"],
                "remote_assignments": metric["remote_assignments"],
                "remote_bytes_one_way_bf16": metric["remote_bytes_one_way_bf16"],
                "mean_fanout": metric["destination_fanout_mean"],
                "active_experts": metric["active_experts"],
                "epoch_status": "future-known compaction sensitivity, not measured Epoch",
                "provenance": "measured LLaDA2 true-EP4 routes + oracle live-row filter",
            }
            serial += 1; stream.write(json.dumps(row) + "\n")


if __name__ == "__main__": main()
