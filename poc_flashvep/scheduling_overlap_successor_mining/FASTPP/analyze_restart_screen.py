"""Descriptive restart-level analysis, never best-run cherry picking."""
import argparse
from collections import defaultdict
import csv
import json
import itertools
from pathlib import Path
import random
import statistics


def quantile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    low = int(index)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bootstrap_median_interval(values):
    """Resample independent restart pairs, never requests within a shared run."""
    n = len(values)
    if n < 3:
        return None, None
    if n <= 5:
        estimates = [statistics.median(sample) for sample in itertools.product(values, repeat=n)]
    else:
        rng = random.Random(20260909)
        estimates = [statistics.median(rng.choices(values, k=n)) for _ in range(10000)]
    return quantile(estimates, .025), quantile(estimates, .975)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("screen", type=Path)
    parser.add_argument("--manifest", default="plan.json")
    parser.add_argument("--baseline", default="pp_only")
    parser.add_argument("--completed-prefix", default="REQUEST_COLLECTION_COMPLETE")
    args = parser.parse_args()
    manifest = json.loads((args.screen / args.manifest).read_text())
    run_rows, comparisons, correctness = [], [], []
    samples = {}
    for item in manifest["plan"]:
        directory = args.screen / f"b{item['block']}_{item['variant']}"
        path = directory / "run.json"
        if not path.exists():
            continue
        run = json.loads(path.read_text())
        if not run["status"].startswith(args.completed_prefix):
            continue
        smoke = rows(directory / "correctness.jsonl")
        references = {"smoke_arithmetic": "42", "smoke_capital": "Paris"}
        actual = {x['request_id']: x for x in smoke}
        correctness.append({"run": directory.name,
                            "short_answers_pass": all(k in actual and actual[k]["output_text"].strip() == v
                                                      for k, v in references.items()),
                            "outputs": {x["request_id"]: x["output_text"] for x in smoke}})
        for measured in run["request_runs"]:
            if not measured["label"].startswith("measure"):
                continue
            requests = rows(directory / (measured["label"] + ".jsonl"))
            assert len(requests) == len({x["request_id"] for x in requests})
            assert all(x["status"] == "PASS" for x in requests)
            workload = Path(measured["trace"]).stem
            samples[item["block"], item["variant"], workload] = {x["request_id"]: x for x in requests}
            row = {"block": item["block"], "policy": item["variant"], "workload": workload,
                   "requests": len(requests), "tokens_s": measured["tokens_per_s"],
                   "elapsed_s": measured["elapsed_s"],
                   "joint_slo_fraction_ttft2_tpot02": statistics.mean(
                       x["ttft_s"] <= 2 and x["tpot_s"] <= .2 for x in requests),
                   "max_arrival_lateness_ms": 1000 * max(x["arrival_lateness_s"] for x in requests)}
            # Mean TPOT can hide a decode stall. Keep client-observed all-token
            # latency distinct, and do not interpret coalesced SSE as exact ITL.
            coalesced = sum(x.get("stream_coalesced", False) for x in requests)
            row["coalesced_requests"] = coalesced
            max_itls = [max(x["itl_s"], default=0) for x in requests]
            row["request_max_itl_mean_s"] = statistics.mean(max_itls) if not coalesced else ""
            row["request_max_itl_p99_s"] = quantile(max_itls, .99) if not coalesced else ""
            for ttft in (1, 2, 5):
                for tbt_ms in (50, 100, 200):
                    row[f"joint_all_token_slo_ttft{ttft}_tbt{tbt_ms}ms"] = (
                        statistics.mean(x["ttft_s"] <= ttft and gap <= tbt_ms / 1000
                                        for x, gap in zip(requests, max_itls))
                        if not coalesced else "")
            for metric in ("e2e_s", "ttft_s", "tpot_s"):
                values = [x[metric] for x in requests]
                row.update({metric + "_mean": statistics.mean(values),
                            metric + "_p50": statistics.median(values),
                            metric + "_p90": quantile(values, .9),
                            metric + "_p99": quantile(values, .99)})
            run_rows.append(row)
    for (block, policy, workload), target in samples.items():
        if policy == args.baseline or (block, args.baseline, workload) not in samples:
            continue
        baseline = samples[block, args.baseline, workload]
        assert set(baseline) == set(target)
        ids = sorted(baseline)
        assert all(baseline[i]["output_tokens"] == target[i]["output_tokens"] for i in ids)
        effect = [100 * (1 - target[i]["e2e_s"] / baseline[i]["e2e_s"]) for i in ids]
        # Same text hash is a useful diagnostic, not a numerical correctness proof.
        comparisons.append({"block": block, "policy": policy, "workload": workload,
                            "requests": len(ids),
                            "e2e_mean_reduction_pct": 100 * (1 - sum(target[i]["e2e_s"] for i in ids)
                                                               / sum(baseline[i]["e2e_s"] for i in ids)),
                            "paired_request_reduction_median_pct": statistics.median(effect),
                            "paired_request_reduction_p10_pct": quantile(effect, .1),
                            "paired_request_reduction_p90_pct": quantile(effect, .9),
                            "output_text_exact_fraction": statistics.mean(
                                baseline[i]["output_sha256"] == target[i]["output_sha256"] for i in ids),
                            "output_prefix_32char_exact_fraction": statistics.mean(
                                baseline[i]["output_text"][:32] == target[i]["output_text"][:32] for i in ids)})
    groups = defaultdict(list)
    for row in comparisons:
        groups[row["policy"], row["workload"]].append(row)
    aggregate = []
    for (policy, workload), records in groups.items():
        effects = [x["e2e_mean_reduction_pct"] for x in records]
        lower, upper = bootstrap_median_interval(effects)
        aggregate.append({"policy": policy, "workload": workload, "restart_pairs": len(effects),
                          "median_e2e_mean_reduction_pct": statistics.median(effects),
                          "mean_e2e_mean_reduction_pct": statistics.mean(effects),
                          "min_e2e_mean_reduction_pct": min(effects),
                          "max_e2e_mean_reduction_pct": max(effects),
                          "bootstrap_median_ci95_lower_pct": lower,
                          "bootstrap_median_ci95_upper_pct": upper,
                          "ci_unit": "independent_restart_pair_small_n_descriptive",
                          "status": "DESCRIPTIVE_NOT_SUCCESSOR_HEADROOM"})
    within_config = []
    configurations = sorted({(policy, workload) for _, policy, workload in samples})
    for policy, workload in configurations:
        blocks = sorted(block for block, p, w in samples if p == policy and w == workload)
        for first, second in itertools.combinations(blocks, 2):
            a, b = samples[first, policy, workload], samples[second, policy, workload]
            assert set(a) == set(b)
            within_config.append({
                "policy": policy, "workload": workload,
                "first_block": first, "second_block": second,
                "requests": len(a),
                "output_text_exact_fraction": statistics.mean(
                    a[i]["output_sha256"] == b[i]["output_sha256"] for i in a),
                "output_prefix_32char_exact_fraction": statistics.mean(
                    a[i]["output_text"][:32] == b[i]["output_text"][:32] for i in a),
                "e2e_mean_change_pct": 100 * (
                    sum(b[i]["e2e_s"] for i in a) / sum(a[i]["e2e_s"] for i in a) - 1),
                "interpretation": "SAME_CONFIG_RESTART_CONTROL_NOT_CORRECTNESS_PROOF",
            })
    for name, data in (("request_run_summary", run_rows), ("paired_comparisons", comparisons),
                       ("restart_aggregate", aggregate), ("within_config_restart", within_config)):
        if data:
            with (args.screen / (name + ".csv")).open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    (args.screen / "correctness_screen.json").write_text(json.dumps(correctness, indent=2))
    print(json.dumps({"completed_restarts": len(correctness), "restart_aggregate": aggregate,
                      "note": "Original-method gain is NOT additional successor headroom; incomplete blocks stay incomplete."}, indent=2))


if __name__ == "__main__":
    main()
