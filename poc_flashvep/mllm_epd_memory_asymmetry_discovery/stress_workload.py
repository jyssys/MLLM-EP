"""Freeze two-wave capacity controls with the existing client arrival algorithm."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from .cache_replay import BLOCK, GIB, KV_BYTES, prepare, replay
from .workload import file_digest, make_manifest, ordered_images, write_frozen


ARRIVAL_SEED = 2026112402


def wave_rows(rows, add_replays):
    groups = defaultdict(list)
    for row in rows:
        groups[ordered_images(row)].append(row)
    first = [{**group[0], "cache_wave": 1} for group in groups.values()]
    second = []
    for group in groups.values():
        if len(group) > 1:
            for row in group[1:2] if add_replays else group[1:]:
                second.append({**row, "cache_wave": 2, "simulated_replay": False})
        elif add_replays:
            request = group[0]
            second.append({**request, "request_id": request["request_id"] + "__controlled_replay",
                           "cache_wave": 2, "simulated_replay": True,
                           "replayed_request_id": request["request_id"]})
    return first, second


def arrivals(rows, seed, qps):
    rng = random.Random(seed)
    offset = 0
    trace = []
    for index, row in enumerate(rows):
        if index:
            offset += rng.expovariate(qps)
        trace.append({"index": index, "request_id": row["request_id"], "offset_seconds": offset, "replica": 0})
    return {"seed": seed, "QPS": qps, "serial": False, "trace": trace,
            "definition": "Open-loop Poisson. Arrival is the planned timestamp. Dispatch lateness remains in TTFT."}


def measure_waves(first, second, assets, sizes):
    first_records = prepare(first, assets)
    all_records = prepare(first + second, assets)
    first_unique = len({block for _, _, _, keys in first_records for block in keys})
    all_unique = len({block for _, _, _, keys in all_records for block in keys})
    last_visual_row = [row["images"][-1]["visual_span"][1] for row in first]
    summary = []
    for capacity in sizes:
        cold = replay(first_records, capacity, 0)
        total = replay(all_records, capacity, 0)
        summary.append({"P_KV_GiB": capacity,
                        "first_wave_prefix_hit_requests": cold["prefix_hit_requests"],
                        "second_wave_prefix_hit_requests": total["prefix_hit_requests"] - cold["prefix_hit_requests"],
                        "second_wave_reusable_prompt_rows": total["prefill_avoided_rows"] - cold["prefill_avoided_rows"],
                        "total_prefix_block_evictions": total["prefix_block_evictions"]})
    return {"first_wave_requests": len(first), "second_wave_requests": len(second),
            "second_wave_exact_question_replays": sum(row["simulated_replay"] for row in second),
            "first_wave_full_prompt_KV_unique_GiB": first_unique * BLOCK * KV_BYTES / GIB,
            "all_full_prompt_KV_unique_GiB": all_unique * BLOCK * KV_BYTES / GIB,
            "first_wave_image_prefix_rounded_up_KV_upper_bound_GiB": sum(
                ((rows + 1 + BLOCK - 1) // BLOCK) * BLOCK * KV_BYTES for rows in last_visual_row) / GIB,
            "max_input_rows": max(row["total_input_tokens"] for row in first + second), "simulation": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural149", type=Path, required=True)
    parser.add_argument("--asset-repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=ARRIVAL_SEED)
    parser.add_argument("--qps", type=float, nargs="+", default=[1.0, 2.5, 5.0])
    args = parser.parse_args()
    if any(qps <= 0 for qps in args.qps):
        parser.error("QPS must be positive")
    parent = json.loads(args.natural149.read_bytes())
    rows = parent["requests"]
    sizes = (4, 8, 12, 16, 32, 42, 64, 80)
    results = {"evidence": "OFFLINE_ORACLE", "parent_sha256": file_digest(args.natural149),
               "scope": "Controlled source-order replay. Generic byte-bounded LRU and causal key lookup, "
                        "complete-before-next assumption. Active pages, NIXL transfer, scheduling and vLLM's "
                        "free-page policy are absent. Actual native eviction and hits require online counters.",
               "cohorts": {}}
    for name, add_replays in (("w5_natural_two_wave149", False), ("w5_eviction188", True)):
        first, second = wave_rows(rows, add_replays)
        combined = first + second
        if len({row["request_id"] for row in combined}) != len(combined):
            raise ValueError(f"Request IDs are not unique for {name}")
        manifest = args.run_root / f"{name}_manifest.json"
        write_frozen(manifest, make_manifest(name, combined, parent))
        trace_rows = []
        for qps in args.qps:
            rate = format(qps, "g").replace(".", "p")
            arrival = args.run_root / f"{name}_q{rate}_arrival.json"
            trace = arrivals(combined, args.seed, qps)
            write_frozen(arrival, trace)
            trace_rows.append({"qps": qps, "seed": args.seed, "file": arrival.name, "sha256": file_digest(arrival),
                               "first_replay_at_seconds": trace["trace"][len(first)]["offset_seconds"],
                               "arrival_span_seconds": trace["trace"][-1]["offset_seconds"]})
        results["cohorts"][name] = {"manifest_filename": manifest.name, "manifest_sha256": file_digest(manifest),
                                    "traces": trace_rows, **measure_waves(first, second, args.asset_repo.absolute(), sizes)}
    args.results.mkdir(parents=True, exist_ok=True)
    write_frozen(args.results / "capacity_stress_workloads.json", results)
    print(json.dumps(results["cohorts"], indent=2))


if __name__ == "__main__":
    main()
