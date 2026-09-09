"""Validate cross-stage identity before interpreting native PP cost profiles.

All CUDA durations are same-device. Makespan proxies are deliberately NOT
converted into request gains: PP stages overlap and host enqueues are not GPU
start timestamps.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"COMMON"))
from analyze_transfer_cpu import balanced_partition, percentile


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for path in (args.run/"mechanism").glob("*.jsonl")
               for line in path.open()]
    stages, layers, associations = defaultdict(list), defaultdict(list), {}
    for row in records:
        if row["kind"] == "run_batch_host":
            associations[row["pp_rank"], row["local_invocation_id"]] = row
        if row["kind"] != "stage" or not any(":measure" in rid for rid in row["request_ids"]):
            continue
        identity = row.get("request_shape_key")
        if not identity:
            continue
        if row["stage"] == "stage_compute":
            stages[identity].append(row)
        elif row["stage"].startswith("layer:"):
            layers[identity].append(row)
    coverage, joined, duplicates, partitions = Counter(), [], [], []
    for identity, rows in stages.items():
        coverage[len({r["pp_rank"] for r in rows})] += 1
        if len(rows) != len({r["pp_rank"] for r in rows}):
            duplicates.append(identity)
            continue
        if {r["pp_rank"] for r in rows} != {0, 1, 2, 3}:
            continue
        assert len({(r["phase"], r["M"]) for r in rows}) == 1
        costs = {r["pp_rank"]: r["cuda_ms"] for r in rows}
        mean = statistics.mean(costs.values())
        joined.append({"request_shape_key": identity, "phase": rows[0]["phase"],
                       "M": rows[0]["M"], "requests": len(rows[0]["request_ids"]),
                       "stage_ms": json.dumps(costs), "max_mean": max(costs.values())/mean,
                       "perfect_equal_stage_makespan_proxy_pct": 100*(1-mean/max(costs.values()))})
        layer_rows = layers[identity]
        layer_costs = {}
        for row in layer_rows:
            index = int(row["stage"].rsplit(".", 1)[-1])
            assert index not in layer_costs, (identity, index)
            layer_costs[index] = row["cuda_ms"]
        if set(layer_costs) == set(range(48)):
            ordered = [layer_costs[i] for i in range(48)]
            current = max(sum(ordered[i:i+12]) for i in range(0, 48, 12))
            optimum, cuts = balanced_partition(ordered, 4)
            partitions.append({"request_shape_key": identity, "phase": rows[0]["phase"],
                               "M": rows[0]["M"], "current_group_max_ms": current,
                               "optimal_group_max_ms": optimum, "cuts": json.dumps(cuts),
                               "layer_partition_proxy_pct": 100*(1-optimum/current)})
    predictions = []
    for row in records:
        if row["kind"] != "alp_learning_sample" or row["predicted_before_update_host_s"] is None:
            continue
        context = associations.get((row["pp_rank"], row["after_local_invocation_id"]))
        if not context or not any(":measure" in rid for rid in context["request_ids"]):
            continue
        actual = row["observed_host_s"]
        if actual <= 0:
            continue
        predictions.append({"phase": context["phase"], "M": context["M"],
                            "pp_rank": row["pp_rank"], "chunk": row["chunk"],
                            "actual_host_s": actual,
                            "predicted_before_update_host_s": row["predicted_before_update_host_s"],
                            "absolute_error_pct": 100*abs(row["predicted_before_update_host_s"]-actual)/actual})
    summary = {"scope": "INSTRUMENTED_NATIVE_PP_DIAGNOSTIC_NOT_REQUEST_ORACLE",
               "records": len(records), "stage_key_rank_coverage": dict(coverage),
               "duplicate_stage_keys_excluded": len(duplicates), "complete_keys": len(joined),
               "complete_48_layer_profiles": len(partitions), "by_phase": {}}
    for phase in sorted({r["phase"] for r in joined+predictions}):
        group = [r for r in joined if r["phase"] == phase]
        errors = [r["absolute_error_pct"] for r in predictions if r["phase"] == phase]
        proxy = [r["layer_partition_proxy_pct"] for r in partitions if r["phase"] == phase]
        summary["by_phase"][phase] = {
            "complete_keys": len(group),
            "stage_max_mean_median": statistics.median(r["max_mean"] for r in group) if group else None,
            "alp_host_samples": len(errors),
            "alp_host_ape_p50": statistics.median(errors) if errors else None,
            "alp_host_ape_p90": percentile(errors, .9) if errors else None,
            "partition_proxy_p50_pct": statistics.median(proxy) if proxy else None}
    write_csv(args.run/"joined_stages.csv", joined)
    write_csv(args.run/"layer_partition_proxy.csv", partitions)
    write_csv(args.run/"alp_prediction_error.csv", predictions)
    (args.run/"mechanism_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
