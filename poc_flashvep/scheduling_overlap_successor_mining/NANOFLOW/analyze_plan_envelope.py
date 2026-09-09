"""Correctness-gated finite native-plan envelope, not a searched optimum."""
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
    args = parser.parse_args()
    manifest = json.loads((args.screen/"manifest.json").read_text())
    assert not manifest.get("nsight_profiled"), "Profiled timings cannot be an envelope"
    baseline = "plain" if manifest.get("phase") == "prefill" else "graph"
    with (args.screen/"request_run_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with (args.screen/"paired_results.csv").open() as stream:
        pairs = list(csv.DictReader(stream))
    workloads = sorted({r["workload"] for r in rows})
    variants = sorted({r["variant"] for r in rows})
    eligible = {baseline}
    exclusions = []
    for variant in variants:
        if variant == baseline:
            continue
        for workload in workloads:
            matches = [r for r in pairs if r["variant"] == variant and r["workload"] == workload]
            if len(matches) < 3 or any(r["status"] != "TOKEN_EXACT_DESCRIPTIVE_ONLY" for r in matches):
                exclusions.append({"variant": variant, "workload": workload,
                                   "pairs": len(matches), "reason": "needs_three_token_exact_restart_pairs"})
                break
        else:
            eligible.add(variant)
    costs = {w: {v: statistics.median(float(r["mean_e2e_s"]) for r in rows
                                    if r["workload"] == w and r["variant"] == v)
                 for v in sorted(eligible)} for w in workloads}
    weights = {w: int(next(r["requests"] for r in rows if r["workload"] == w)) for w in workloads}
    static = {v: sum(weights[w]*costs[w][v] for w in workloads)/sum(weights.values()) for v in eligible}
    best = min(static, key=static.get)
    oracle = sum(weights[w]*min(costs[w].values()) for w in workloads)/sum(weights.values())
    result = {"scope": "FINITE_MANUAL_PLAN_PORTFOLIO_NOT_PAPER_SEARCHED_OPTIMUM",
              "eligible_variants": sorted(eligible), "excluded": exclusions,
              "status": "FINITE_ENVELOPE_ONLY" if len(eligible) > 1 else "INSUFFICIENT_CORRECTNESS_GATED_PLANS",
              "median_request_e2e_costs": costs, "best_static": best,
              "per_workload_choices": {w: min(costs[w], key=costs[w].get) for w in workloads},
              "best_static_e2e_s": static[best], "per_workload_oracle_e2e_s": oracle,
              "additional_e2e_envelope_pct": 100*(1-oracle/static[best]) if len(eligible) > 1 else None,
              "limitations": ["fixed real-content cohorts, not continuous arrivals",
                               "same searched objective and full MoE profile database unavailable",
                               ("prefill plans initialized in two warmups; eager execution"
                                if manifest.get("phase") == "prefill" else
                                "first decode plan/capture costs remain charged to each fresh target"),
                               "small portfolio versus per-workload oracle identical for this finite set",
                               "zero switching cost; not a bound over all legal schedules"]}
    (args.screen/"finite_plan_envelope.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
