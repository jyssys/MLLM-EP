"""Existing-policy envelope only; not a new scheduler or partition oracle."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    parser = argparse.ArgumentParser()
    parser.add_argument("screen", type=Path)
    parser.add_argument("--policies", nargs="+")
    parser.add_argument("--scope", default="dense_existing_policies_only_correctness_validation_pending")
    parser.add_argument("--output", default="existing_policy_envelope.json")
    args = parser.parse_args()
    rows = list(csv.DictReader((args.screen/"request_run_summary.csv").open()))
    if args.policies:
        rows = [r for r in rows if r["policy"] in args.policies]
    policies = sorted({r["policy"] for r in rows})
    workloads = sorted({r["workload"] for r in rows})
    median_cost = {w: {p: statistics.median(float(r["e2e_s_mean"]) for r in rows
                                           if r["workload"]==w and r["policy"]==p)
                       for p in policies} for w in workloads}
    counts = {w: int(next(r["requests"] for r in rows if r["workload"]==w)) for w in workloads}
    results = []
    for weight_kind, weights in [("observed_request_mix", counts),
                                  ("equal_workload_weight", {w: 1 for w in workloads})]:
        static = {p: sum(weights[w]*median_cost[w][p] for w in workloads)/sum(weights.values())
                  for p in policies}
        best = min(static, key=static.get)
        oracle = sum(weights[w]*min(median_cost[w].values()) for w in workloads)/sum(weights.values())
        results.append({"weighting": weight_kind, "best_static": best,
                        "best_static_mean_e2e_s": static[best], "per_regime_oracle_e2e_s": oracle,
                        "oracle_gain_pct": 100*(1-oracle/static[best]),
                        "per_regime_choices": {w: min(median_cost[w], key=median_cost[w].get) for w in workloads}})
    recoveries = []
    for block in sorted({r["block"] for r in rows}):
        group = {r["policy"]:float(r["e2e_s_mean"]) for r in rows
                 if r["block"]==block and r["workload"]=="heterogeneous_bursty"}
        if {"alp", "pp_only", "greedy"} <= group.keys() and group["alp"] > group["pp_only"]:
            recoveries.append({"block": block,
                               "greedy_recovery_of_alp_excess_pct": 100*(group["alp"]-group["greedy"])/(group["alp"]-group["pp_only"])})
    # Select knobs on other restart blocks, then score the untouched block.
    # This is still the same workload pool, not generalization to new arrivals.
    held_out = []
    blocks = sorted({r["block"] for r in rows})
    if len(blocks) >= 2:
        for block in blocks:
            train = [r for r in rows if r["block"] != block]
            test = {(r["policy"], r["workload"]): r for r in rows if r["block"] == block}
            if len(test) != len(policies) * len(workloads):
                continue
            costs = {w: {p: statistics.mean(float(r["e2e_s_mean"]) for r in train
                                           if r["workload"] == w and r["policy"] == p)
                         for p in policies} for w in workloads}
            static = min(policies, key=lambda p: sum(counts[w] * costs[w][p] for w in workloads))
            choices = {w: min(costs[w], key=costs[w].get) for w in workloads}
            fixed = sum(counts[w] * float(test[static, w]["e2e_s_mean"]) for w in workloads)
            selected = sum(counts[w] * float(test[choices[w], w]["e2e_s_mean"]) for w in workloads)
            held_out.append({"held_out_block": block, "trained_static": static,
                             "trained_per_regime": choices,
                             "held_out_reduction_pct": 100 * (1 - selected / fixed)})
    result = {"scope": args.scope,
              "policies": policies, "median_cost_by_workload": median_cost,
              "envelopes": results, "bursty_trivial_fix": recoveries,
              "leave_one_restart_out": held_out,
              "limitations": ["in-sample oracle and leave-one-restart-out knob selection, not new-workload validation",
                              "does not bound untested chunk/partition or MoE/MLLM successors",
                              "zero switching/restart cost assumed at workload boundary",
                              "long continuation correctness still under investigation",
                              "not SLO capacity goodput or per-invocation oracle"]}
    (args.screen/args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
