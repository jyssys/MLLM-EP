"""Native eager layer/group and expert-use diagnostic; no traffic-to-E2E claim."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import statistics


def phase(requests):
    pre = any("PREFILL" in r["status"] for r in requests)
    dec = any("DECOD" in r["status"] for r in requests)
    return "mixed" if pre and dec else "prefill" if pre else "decode"


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    args = p.parse_args()
    stages, routes, layers = defaultdict(list), [], defaultdict(list)
    counts = Counter()
    sampled, seen = set(), set()
    for path in (args.run/"mechanism").glob("*.jsonl"):
        for line in path.open():
            row = json.loads(line)
            counts[row["kind"]] += 1
            reqs = row.get("requests", [])
            if not reqs or not all(":measure" in r["request_id"] for r in reqs):
                continue
            current_phase = phase(reqs)
            seen.update(r["request_id"] for r in reqs)
            key = row["local_invocation_id"]
            if row["kind"] == "cuda_stage" and row["stage"] == "full_layer_stack":
                stages[key].append({"rank": row["rank"], "phase": current_phase,
                                    "cuda_ms": row["cuda_ms"], "sampled": row["route_sampled"],
                                    "requests": tuple(r["request_id"] for r in reqs),
                                    "group_caps_selected": sorted({r["num_stages"] for r in reqs})})
            elif row["kind"] == "cuda_stage" and row["stage"].startswith("layer:") and not row["route_sampled"]:
                layers[row["rank"], current_phase, row["stage"], row["prefill_tokens"] > 0].append(row["cuda_ms"])
            elif row["kind"] == "route_histogram":
                sampled.add(key)
                assert sum(row["expert_histogram"]) == row["total_assignments"] == row["M"]*row["top_k"]
                routes.append({"invocation": key, "phase": current_phase,
                               "layer": row["layer"], "M": row["M"],
                               "prefill_tokens": row["prefill_tokens"],
                               "decode_tokens": row["M"]-row["prefill_tokens"],
                               **{field: row[field] for field in (
                                   "active_experts", "prefill_active_experts", "decode_active_experts",
                                   "experts_needed_only_by_prefill", "routed_weight_bytes_proxy_per_tp_rank")}})
    joined, coverage = [], Counter()
    for key, group in stages.items():
        coverage[len(group)] += 1
        assert len(group) == len({r["rank"] for r in group}), "Duplicate rank row"
        if {r["rank"] for r in group} != {0, 1}:
            continue
        assert len({r["requests"] for r in group}) == 1
        joined.append({"invocation": key, "phase": group[0]["phase"],
                       "route_sampled": group[0]["sampled"],
                       "full_stack_critical_cuda_ms": max(r["cuda_ms"] for r in group),
                       "requests": len(group[0]["requests"]),
                       "selected_stage_counts": json.dumps(group[0]["group_caps_selected"])})
    layer_rows = [{"rank": k[0], "phase": k[1], "layer": k[2], "prefill_active": k[3],
                   "samples": len(v), "layer_median_ms": statistics.median(v)}
                  for k, v in layers.items()]
    summary = {"scope": "NATIVE_EAGER_MEASUREMENT_PATCH_NOT_CLEAN_GRAPH_PERFORMANCE",
               "record_counts": dict(counts), "full_stack_rank_coverage": dict(coverage),
               "observed_measured_request_ids": len(seen),
               "sampled_route_invocations": len(sampled), "by_phase": {},
               "limitations": ["routed-weight bytes are compulsory logical bytes, not measured DRAM traffic",
                               "route-copy steps excluded from clean layer profiles",
                               "bounded capture may omit the end of a workload",
                               "cross-rank maxima use same-device durations, never absolute timestamp subtraction"]}
    for key in sorted({r["phase"] for r in joined+routes}):
        spans = [r["full_stack_critical_cuda_ms"] for r in joined if r["phase"] == key and not r["route_sampled"]]
        hist = [r for r in routes if r["phase"] == key]
        active_prefill = [r for r in hist if r["prefill_tokens"] > 0]
        summary["by_phase"][key] = {"unsampled_stack_observations": len(spans),
            "full_stack_median_ms": statistics.median(spans) if spans else None,
            "route_samples": len(hist),
            "active_experts_median": statistics.median(r["active_experts"] for r in hist) if hist else None,
            "prefill_only_experts_median": statistics.median(r["experts_needed_only_by_prefill"] for r in hist) if hist else None,
            "prefill_active_route_samples": len(active_prefill),
            "prefill_active_all_experts_median": statistics.median(r["active_experts"] for r in active_prefill) if active_prefill else None,
            "prefill_active_decode_experts_median": statistics.median(r["decode_active_experts"] for r in active_prefill) if active_prefill else None,
            "prefill_active_prefill_only_experts_median": statistics.median(r["experts_needed_only_by_prefill"] for r in active_prefill) if active_prefill else None}
    for name, rows in (("joined_stacks", joined), ("sampled_expert_use", routes), ("layer_profiles", layer_rows)):
        if rows:
            with (args.run/(name+".csv")).open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    (args.run/"mechanism_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
