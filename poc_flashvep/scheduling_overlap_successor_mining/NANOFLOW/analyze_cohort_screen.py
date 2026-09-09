"""Descriptive request metrics with a hard cross-plan token-equality gate.

These simultaneous cohorts do not establish continuous-batching performance.
The searched paper optimum is not represented by the hand-written plan set.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root):
    manifest = json.loads((root / "manifest.json").read_text())
    baseline_variant = "plain" if manifest.get("phase") == "prefill" else "graph"
    collected = {}
    run_rows = []
    for record in manifest["records"]:
        target = root / (record["name"] + ".json")
        if record.get("returncode") != 0 or not target.exists():
            continue
        data = json.loads(target.read_text())
        requests = {row["request_id"]: row for row in data["requests"]}
        assert len(requests) == len(data["requests"])
        collected[record["block"], record["workload"], record["variant"]] = requests
        run_rows.append({
            "block": record["block"], "workload": record["workload"],
            "variant": record["variant"], "requests": len(requests),
            "mean_e2e_s": statistics.mean(x["e2e_s"] for x in requests.values()),
            "mean_ttft_s": statistics.mean(x["ttft_s"] for x in requests.values()),
            "mean_tpot_s": statistics.mean(x["tpot_s"] for x in requests.values())
                if all(x["tpot_s"] is not None for x in requests.values()) else None,
            "steady_itl_mean_s": statistics.mean(value for x in requests.values() for value in x["itl_s"][1:])
                if all(len(x["itl_s"]) > 1 for x in requests.values()) else None,
            "tokens_s": data["tokens_s"],
            "scope": "fixed_cohort_not_continuous_batching",
        })
    pairs = []
    for (block, workload, variant), treatment in collected.items():
        if variant == baseline_variant or (block, workload, baseline_variant) not in collected:
            continue
        baseline = collected[block, workload, baseline_variant]
        assert baseline.keys() == treatment.keys()
        exact = []
        common_prefix = []
        for key in baseline:
            a, b = baseline[key], treatment[key]
            assert a["input_ids"] == b["input_ids"]
            assert len(a["output_ids"]) == len(b["output_ids"])
            exact.append(a["output_ids"] == b["output_ids"])
            prefix = next((i for i, (x, y) in enumerate(zip(a["output_ids"], b["output_ids"]))
                           if x != y), len(a["output_ids"]))
            common_prefix.append(prefix)
        mean = lambda rows, field: statistics.mean(r[field] for r in rows.values())
        pairs.append({
            "block": block, "workload": workload, "variant": variant,
            "requests": len(baseline), "token_exact_requests": sum(exact),
            "token_exact_fraction": statistics.mean(exact),
            "mean_common_prefix_tokens": statistics.mean(common_prefix),
            "e2e_reduction_pct": 100 * (1-mean(treatment, "e2e_s")/mean(baseline, "e2e_s")),
            "ttft_reduction_pct": 100 * (1-mean(treatment, "ttft_s")/mean(baseline, "ttft_s")),
            "tpot_reduction_pct": 100 * (1-mean(treatment, "tpot_s")/mean(baseline, "tpot_s"))
                if all(x["tpot_s"] is not None for x in treatment.values()) else None,
            "status": "TOKEN_EXACT_DESCRIPTIVE_ONLY" if all(exact) else "EXCLUDE_PENDING_CORRECTNESS",
        })
    write_csv(root / "request_run_summary.csv", run_rows)
    write_csv(root / "paired_results.csv", pairs)
    summary = {"collected_runs": len(run_rows), "pairs": pairs,
               "limitations": ["simultaneous cohort, not online arrival trace",
                               "manual plan subset, not native searched optimum",
                               "reported restart count must support any effect; pilot alone cannot",
                               ("prefill plans initialized during two same-shape warmups; no CUDA graph"
                                if manifest.get("phase") == "prefill" else
                                "first decode graph capture and plan creation charged to request")]}
    (root / "analysis.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root), indent=2))
