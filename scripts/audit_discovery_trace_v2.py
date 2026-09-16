#!/usr/bin/env python3
"""Audit aggregate-v2 fields against retained full-row heavy requests."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from virtual_ep.discovery_trace import DiscoveryTrace
from virtual_ep.schema import TraceBundle, invocation_slices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heavy", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--prompt-records", type=Path, required=True)
    parser.add_argument("--requests", type=int, nargs="+", required=True)
    parser.add_argument("--samples-per-request", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    requested = set(args.requests)
    heavy = TraceBundle.load(args.heavy).select_requests(requested)
    aggregate = DiscoveryTrace.load(args.aggregate)
    prompt_rows = [json.loads(line) for line in args.prompt_records.read_text().splitlines() if line]
    prompts = {int(row["sample_id"]): int(row["prompt_tokens"]) for row in prompt_rows}
    invocations = list(invocation_slices(heavy.rows))
    blocks = defaultdict(set)
    by_request = defaultdict(list)
    max_layer = int(heavy.rows["layer_id"].max())
    mask_states = {}
    for key, selected in invocations:
        blocks[key[0]].add(key[1]); by_request[key[0]].append((key, selected))
        if key[4] == max_layer:
            mask_states[(key[0], key[1], key[2])] = heavy.rows["is_masked"][selected][-32:].copy()
    block_maps = {request: {block: i for i, block in enumerate(sorted(values))}
                  for request, values in blocks.items()}
    lookup = {(int(aggregate.arrays["request_id"][i]), int(aggregate.arrays["block_id"][i]),
               int(aggregate.arrays["iteration_id"][i]), int(aggregate.arrays["layer_id"][i])): i
              for i in range(len(aggregate.arrays["request_id"]))}
    rng = np.random.default_rng(20260917)
    checks = []
    for request in sorted(requested):
        candidates = by_request[request]
        picks = rng.choice(len(candidates), size=min(args.samples_per_request, len(candidates)), replace=False)
        for pick in picks:
            key, selected = candidates[int(pick)]
            request_id, old_block, iteration, _nfe, layer = key
            new_block = block_maps[request_id][old_block]
            index = lookup[(request_id, new_block, iteration, layer)]
            ids = heavy.rows["expert_ids"][selected]
            physical = len(ids); prompt = prompts[request_id]; current_start = physical - 32
            is_masked = heavy.rows["is_masked"][selected]
            classes = np.full(physical, 1, dtype=np.int8); classes[:prompt] = 0
            positions = np.arange(current_start, physical); generated = positions >= prompt
            current_mask = is_masked[current_start:] & generated
            prior = mask_states.get((request_id, old_block, iteration - 1))
            newly = ((prior & ~is_masked[current_start:] & generated)
                     if prior is not None else np.zeros(32, dtype=bool))
            current = classes[current_start:]
            current[current_mask] = 2; current[newly] = 3
            current[generated & ~current_mask & ~newly] = 4
            hist = np.zeros((5, 256), dtype=np.uint16)
            for class_id in range(5):
                chosen = classes == class_id
                if chosen.any():
                    hist[class_id] = np.bincount(ids[chosen].reshape(-1), minlength=256)
            checks.append({
                "request": request_id, "block": new_block, "iteration": iteration, "layer": layer,
                "current_routes_exact": bool(np.array_equal(ids[-32:], aggregate.arrays["current_expert_ids"][index])),
                "position_classes_exact": bool(np.array_equal(classes[-32:], aggregate.arrays["current_position_class"][index])),
                "expert_histograms_exact": bool(np.array_equal(hist, aggregate.arrays["expert_counts_by_class"][index])),
                "ep2_rank_load_exact": bool(np.array_equal(
                    np.bincount((ids // 128).reshape(-1), minlength=2), aggregate.arrays["rank_load_ep2"][index])),
                "ep4_rank_load_exact": bool(np.array_equal(
                    np.bincount((ids // 64).reshape(-1), minlength=4), aggregate.arrays["rank_load_ep4"][index])),
            })
    fields = ["current_routes_exact", "position_classes_exact", "expert_histograms_exact",
              "ep2_rank_load_exact", "ep4_rank_load_exact"]
    summary = {
        "heavy_requests": sorted(requested), "sampled_invocations": len(checks),
        "field_pass_rate": {field: float(np.mean([row[field] for row in checks])) for field in fields},
        "all_checks_pass": all(all(row[field] for field in fields) for row in checks),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "checks"}, indent=2))


if __name__ == "__main__":
    main()
