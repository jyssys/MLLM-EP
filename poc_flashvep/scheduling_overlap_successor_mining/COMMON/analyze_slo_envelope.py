"""Finite existing-knob attained-goodput envelope on fixed arrival traces.

This is not maximum sustainable SLO capacity and does not invent unseen plans.
Successful requests must meet TTFT AND every observed token gap. Never replace
maximum ITL by mean TPOT or interpret coalesced streams as exact token timing.
"""
import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
import statistics


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    parser = argparse.ArgumentParser()
    parser.add_argument("screen", type=Path)
    parser.add_argument("--policies", nargs="+")
    args = parser.parse_args()
    rows = list(csv.DictReader((args.screen/"request_run_summary.csv").open()))
    if args.policies:
        rows = [r for r in rows if r["policy"] in args.policies]
    workloads = sorted({r["workload"] for r in rows})
    policies = sorted({r["policy"] for r in rows})
    assert len(policies)**len(workloads) <= 20000
    metrics = [name for name in rows[0] if name.startswith("joint_all_token_slo_")]
    results = []
    for metric in metrics:
        values = {}
        for w in workloads:
            for p in policies:
                observed = [r for r in rows if (r["policy"], r["workload"]) == (p, w)]
                assert observed and all(r[metric] != "" for r in observed), "Missing/unreliable SLO observations"
                values[p, w] = (statistics.median(int(r["requests"])*float(r[metric]) for r in observed),
                                statistics.median(float(r["elapsed_s"]) for r in observed))
        def rate(choices):
            return sum(values[p, w][0] for p, w in zip(choices, workloads))/sum(
                values[p, w][1] for p, w in zip(choices, workloads))
        static = {p: rate([p]*len(workloads)) for p in policies}
        best = max(static, key=static.get)
        selected = max(itertools.product(policies, repeat=len(workloads)), key=rate)
        oracle = rate(selected)
        gain = 100*(oracle/static[best]-1) if static[best] else None
        results.append({"slo": metric, "best_static": best,
                        "best_static_attained_requests_s": static[best],
                        "per_workload_attained_requests_s": oracle,
                        "additional_goodput_pct": gain,
                        "per_workload_choices": dict(zip(workloads, selected))})
    output = {"scope": "FINITE_EXISTING_KNOBS_FIXED_TRACE_ATTAINED_GOODPUT_NOT_CAPACITY",
              "policies": policies, "results": results,
              "limitations": ["in-sample zero-cost workload-boundary knob changes",
                              "median restart aggregate, not confidence interval",
                              "not a new policy, port, or quality-matched successor claim",
                              "thresholds predeclared in the request analyzer, no selected best threshold headline"]}
    (args.screen/"existing_knob_slo_envelope.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
