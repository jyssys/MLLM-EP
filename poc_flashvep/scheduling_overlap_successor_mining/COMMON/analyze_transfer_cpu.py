"""CPU-only request joins, observer controls and labelled layer-cost diagnostics.

No cross-device absolute CUDA timestamps; no sum of TP-duplicate request rows.
Group balancing is a stage-cost proxy, NOT a direct request-latency oracle.
"""
import argparse
from collections import defaultdict, Counter
import csv
import json
import math
import os
from pathlib import Path
import statistics


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered)-1)*fraction
    low = int(position)
    return ordered[low] + (ordered[min(low+1, len(ordered)-1)]-ordered[low])*(position-low)


def balanced_partition(costs, groups):
    """Exact minimax contiguous partition with nonempty groups."""
    n = len(costs)
    assert 1 <= groups <= n and all(c >= 0 for c in costs)
    prefix = [0.0]
    for c in costs:
        prefix.append(prefix[-1]+c)
    dp = [[math.inf]*(n+1) for _ in range(groups+1)]
    cuts = {}
    dp[0][0] = 0.0
    for g in range(1, groups+1):
        for end in range(g, n+1):
            for start in range(g-1, end):
                value = max(dp[g-1][start], prefix[end]-prefix[start])
                if value < dp[g][end]:
                    dp[g][end] = value
                    cuts[g, end] = start
    bounds, end = [n], n
    for g in range(groups, 0, -1):
        end = cuts[g, end]
        bounds.append(end)
    return dp[groups][n], list(reversed(bounds))


def save_csv(path, rows):
    if rows:
        with path.open("w") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def read_requests(root):
    return [json.loads(s) for p in root.glob("requests_dp*.jsonl") for s in p.read_text().splitlines()]


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only analysis required"
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrumented", type=Path, required=True)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    observed, clean = read_requests(args.instrumented), read_requests(args.clean)
    request_map = {r["internal_request_id"]: r for r in observed}
    def key(r):
        return r["label"], r["dp_rank"], r["source_request_id"], r["family"]
    a = {key(r): r for r in observed if not r["warmup"]}
    b = {key(r): r for r in clean if not r["warmup"]}
    assert a.keys() == b.keys()
    paired = []
    for k in a:
        x, y = a[k], b[k]
        assert x["prompt_tokens"] == y["prompt_tokens"]
        assert len(x["output_ids"]) == len(y["output_ids"])
        paired.append({"label": x["label"], "dp_rank": x["dp_rank"],
                       "source_request_id": x["source_request_id"], "family": x["family"],
                       "prompt_tokens": x["prompt_tokens"], "output_tokens": len(x["output_ids"]),
                       "instrumented_e2e_s": x["e2e_s"], "clean_e2e_s": y["e2e_s"],
                       "clean_ttft_s": y["ttft_s"], "clean_tpot_s": y["tpot_s"],
                       "output_token_exact": x["output_ids"] == y["output_ids"],
                       "observer_overhead_pct": 100*(x["e2e_s"]/y["e2e_s"]-1)})
    save_csv(args.out/"request_observer_control.csv", paired)
    request_summary = []
    for family in sorted({r["family"] for r in paired}):
        rows = [r for r in paired if r["family"] == family]
        observed_mean = statistics.mean(r["instrumented_e2e_s"] for r in rows)
        clean_mean = statistics.mean(r["clean_e2e_s"] for r in rows)
        request_summary.append({"family": family, "requests_per_run": len(rows),
                                "instrumented_mean_e2e_s": observed_mean,
                                "clean_mean_e2e_s": clean_mean,
                                "observer_overhead_pct": 100*(observed_mean/clean_mean-1),
                                "output_exact_fraction": statistics.mean(r["output_token_exact"] for r in rows),
                                "clean_mean_ttft_s": statistics.mean(r["clean_ttft_s"] for r in rows),
                                "clean_mean_tpot_s": statistics.mean(r["clean_tpot_s"] for r in rows),
                                "zero_entire_ttft_e2e_bound_pct": 100*sum(r["clean_ttft_s"] for r in rows)/sum(r["clean_e2e_s"] for r in rows),
                                "bound_scope": "fixed_request_timeline_prefill_only_not_online_queue_counterfactual"})
    save_csv(args.out/"request_summary.csv", request_summary)

    invocation = defaultdict(dict)
    layer_profiles = defaultdict(list)
    proof, unknown, kinds = [], set(), Counter()
    per_step_layers = defaultdict(dict)
    for path in sorted((args.instrumented/"operations").glob("*.jsonl")):
        for line in path.open():
            row = json.loads(line)
            kinds[row["kind"]] += 1
            if row["kind"] == "runtime_proof":
                proof.append(row)
                continue
            ids = row.get("request_ids", [])
            unknown.update(rid for rid in ids if rid not in request_map)
            requests = [request_map[rid] for rid in ids if rid in request_map]
            real = requests and all(not r["warmup"] for r in requests)
            families = {r["family"] for r in requests}
            family = next(iter(families)) if len(families) == 1 else "mixed_or_dummy"
            if row["kind"] == "moe":
                identity = row["invocation"], row["layer"]
                assert row["ep_rank"] not in invocation[identity], "Duplicate rank observation"
                invocation[identity][row["ep_rank"]] = {
                    **{k: row[k] for k in ("moe_ms", "dispatch_ms", "expert_ms", "combine_ms", "local_M", "phase")},
                    "request_ids": ids, "real": bool(real), "family": family,
                    "local_histogram": row.get("local_expert_histogram")}
            elif real and row["kind"] == "operation" and row["stage"] == "layer":
                assert 0 <= row["layer"] < 48
                per_step_layers[row["ep_rank"], row["step"]][row["layer"]] = row["layer_ms"]
                layer_profiles[family, row["phase"], row["layer"]].append(row["layer_ms"])
    logical = []
    coverage = Counter()
    for (index, layer), ranks in sorted(invocation.items()):
        coverage[len(ranks)] += 1
        if len(ranks) != 4 or not any(x["real"] for x in ranks.values()):
            continue
        rows = list(ranks.values())
        ids = sorted({rid for x in rows for rid in x["request_ids"]})
        family_set = {request_map[rid]["family"] for rid in ids if rid in request_map}
        phases = {x["phase"] for x in rows if x["phase"] != "dummy"}
        critical = max(ranks, key=lambda rank: ranks[rank]["moe_ms"])
        histograms = [x["local_histogram"] for x in rows]
        logical.append({"invocation": index, "layer": layer,
                        "family": next(iter(family_set)) if len(family_set) == 1 else "mixed",
                        "phase": next(iter(phases)) if len(phases) == 1 else "mixed",
                        "unique_requests": len(ids), "request_ids": json.dumps(ids),
                        "critical_rank": critical,
                        "moe_critical_ms": ranks[critical]["moe_ms"],
                        "dispatch_on_critical_rank_ms": ranks[critical]["dispatch_ms"],
                        "expert_on_critical_rank_ms": ranks[critical]["expert_ms"],
                        "combine_on_critical_rank_ms": ranks[critical]["combine_ms"],
                        "expert_rank_max_mean": max(x["expert_ms"] for x in rows)/statistics.mean(x["expert_ms"] for x in rows),
                        "sum_local_M": sum(x["local_M"] for x in rows),
                        "assignments": sum(sum(h) for h in histograms) if all(h is not None for h in histograms) else None})
    save_csv(args.out/"logical_moe.csv", logical)
    phase_summary = []
    for phase in sorted({r["phase"] for r in logical}):
        rows = [r for r in logical if r["phase"] == phase]
        values = [r["moe_critical_ms"] for r in rows]
        phase_summary.append({"phase": phase, "logical_invocations": len(rows),
                              "moe_p50_ms": statistics.median(values),
                              "moe_p90_ms": percentile(values, .90), "moe_p99_ms": percentile(values, .99),
                              "dispatch_median_ms": statistics.median(r["dispatch_on_critical_rank_ms"] for r in rows),
                              "expert_median_ms": statistics.median(r["expert_on_critical_rank_ms"] for r in rows),
                              "combine_median_ms": statistics.median(r["combine_on_critical_rank_ms"] for r in rows)})
    save_csv(args.out/"phase_timing_summary.csv", phase_summary)
    partition_rows = []
    for family, phase in sorted({(f, p) for f, p, _ in layer_profiles}):
        if not all((family, phase, layer) in layer_profiles for layer in range(48)):
            continue
        costs = [statistics.median(layer_profiles[family, phase, layer]) for layer in range(48)]
        for groups in (4, 12, 16, 24):
            size = 48//groups
            equal_max = max(sum(costs[i:i+size]) for i in range(0, 48, size))
            optimum, cuts = balanced_partition(costs, groups)
            partition_rows.append({"family": family, "phase": phase, "groups": groups,
                                   "summed_layer_medians_ms": sum(costs),
                                   "equal_group_max_ms": equal_max, "cost_balanced_group_max_ms": optimum,
                                   "group_makespan_proxy_gain_pct": 100*(1-optimum/equal_max),
                                   "optimal_boundaries": json.dumps(cuts),
                                   "direct_e2e_gain": "NOT_ESTABLISHED",
                                   "scope": "instrumented_vllm_transfer_not_native_layered_prefill"})
    save_csv(args.out/"contiguous_partition_proxy.csv", partition_rows)
    summary = {"status": "CPU_DIAGNOSTIC_ONLY", "runtime_proofs": proof,
               "request_count_per_full_run": len(a), "request_summary": request_summary,
               "trace_record_counts": dict(kinds), "rank_join_coverage": dict(coverage),
               "unknown_request_ids": sorted(unknown), "valid_measured_logical_moe_invocations": len(logical),
               "complete_48_layer_local_steps": sum(len(x)==48 for x in per_step_layers.values()),
               "phase_summary": phase_summary,
               "limitations": ["single instrumented/clean restart pair, not randomized causal comparison",
                               "observer overhead must not be presented as optimizable scheduler waste",
                               "per-phase medians do not add to median total",
                               "contiguous group makespan is not request E2E gain",
                               "all three native successor milestones remain incomplete"]}
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("request_count_per_full_run", "rank_join_coverage",
                     "unknown_request_ids", "valid_measured_logical_moe_invocations", "phase_summary")}, indent=2))


if __name__ == "__main__":
    main()
