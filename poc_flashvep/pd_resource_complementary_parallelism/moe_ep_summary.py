"""Summarize critical-rank EP2 event timings and draw sharded replay figures."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import statistics


def percentile(values, fraction):
    ordered = sorted(values)
    pos = fraction * (len(ordered) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] + (pos - lo) * (ordered[hi] - ordered[lo])


def critical(rows):
    return [max(group) for group in zip(*rows, strict=True)]


def reduce(ranks):
    first = ranks[0]
    if {r["rank"] for r in ranks} != {0, 1}:
        raise ValueError("Expected exactly one trace per physical owner")
    fields = ("model_revision", "backend", "workload_sha256", "decode_batch", "decode_context", "prefill_rows")
    if any(r[field] != first[field] for r in ranks for field in fields):
        raise ValueError("Mismatched input or topology receipts")
    if any(not r["qualified"] for r in ranks):
        raise ValueError("Physical correctness gate failed")
    isolated = {}
    for label in first["isolated"]:
        samples = critical([[row["first_ms"] for row in rank["isolated"][label]] for rank in ranks])
        isolated[label] = {"p50_ms": statistics.median(samples), "p95_ms": percentile(samples, .95),
                           "min_ms": min(samples), "max_ms": max(samples), "n": len(samples)}
    comparison = []
    for i, example in enumerate(first["pairs"]):
        label_a, label_b = example["first"], example["second"]
        for rank in ranks:
            if (rank["pairs"][i]["first"], rank["pairs"][i]["second"]) != (label_a, label_b):
                raise ValueError("Mismatched pairing across ranks")
        stats = {}
        for mode in ("serial", "streams", "priority"):
            stats[mode] = {}
            for key in ("first_ms", "second_ms", "makespan_ms", "first_host_enqueue_ms"):
                samples = critical([[r[key] for r in rank["pairs"][i][mode]] for rank in ranks])
                stats[mode][key] = {"p50": statistics.median(samples), "p95": percentile(samples, .95),
                                    "min": min(samples), "max": max(samples)}
        d_isolated = isolated[label_a]["p50_ms"]
        d_p95_isolated = isolated[label_a]["p95_ms"]
        for mode in ("streams", "priority"):
            d_overlap = stats[mode]["first_ms"]["p50"]
            d_p95_overlap = stats[mode]["first_ms"]["p95"]
            saves = 100 * (1 - stats[mode]["makespan_ms"]["p50"] / stats["serial"]["makespan_ms"]["p50"])
            comparison.append({
                "first": label_a, "second": label_b, "mode": mode, "serial": stats["serial"],
                "concurrent": stats[mode], "decode_p50_slowdown_pct": 100 * (d_overlap / d_isolated - 1),
                "decode_p95_slowdown_pct": 100 * (d_p95_overlap / d_p95_isolated - 1),
                "combined_p50_makespan_savings_pct": saves,
                "decode_guard_pass": d_overlap <= 1.05 * d_isolated and d_p95_overlap <= 1.05 * d_p95_isolated,
            })
    return {
        "evidence": "MEASURED_PHYSICAL",
        "scope": "Fixed-route layer-0 BF16 CUDA-event replay using true EP2 on H200",
        **{field: first[field] for field in fields}, "rank_correctness": [r["correctness"] for r in ranks],
        "transport": first["backend_receipt"], "visible_devices": first["visible_devices"],
        "routing": [r["routing"] for r in ranks],
        "isolated_critical_rank": isolated, "paired_critical_rank": comparison,
    }


def figures(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phases = ["decode", *(f"prefill_{m}" for m in data["prefill_rows"])]
    parts = (("dispatch", "Dispatch"), ("experts", "Expert GEMM"), ("combine", "Combine"),
             ("ep_full", "Dependent path"))
    x = range(len(phases))
    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    for i, (key, label) in enumerate(parts):
        points = [(j + (i - 1.5) * .18, data["isolated_critical_rank"][f"{phase}_{key}"]["p50_ms"])
                  for j, phase in enumerate(phases) if f"{phase}_{key}" in data["isolated_critical_rank"]]
        if points:
            points_x, points_y = zip(*points)
            ax.bar(points_x, points_y, .17, label=label)
    ax.set_xticks(list(x), [phase.replace("_", "\n") for phase in phases])
    ax.set_yscale("log")
    ax.set_ylabel("Critical-rank p50 (ms)")
    ax.set_title(f"Qwen3-30B-A3B layer 0 EP2 · {data['backend']}")
    ax.legend(ncols=2)
    fig.savefig(out / "moe_ep_communication_compute.png", dpi=170)
    plt.close(fig)

    pairs = [row for row in data["paired_critical_rank"] if row["mode"] == "priority"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), layout="constrained")
    labels = [f'{r["first"].replace("decode_", "D ")} + {r["second"].replace("prefill_", "P ")}' for r in pairs]
    y = list(range(len(labels)))
    axes[0].barh(y, [r["decode_p95_slowdown_pct"] for r in pairs],
                 color=["#156c68" if r["decode_guard_pass"] else "#ad5b3c" for r in pairs])
    axes[0].axvline(5, color="black", linestyle="--", lw=1, label="5% guard")
    axes[0].axvline(0, color="0.6", lw=.6)
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("Decode operator p95 slowdown (%)")
    axes[0].legend()
    axes[1].barh(y, [r["combined_p50_makespan_savings_pct"] for r in pairs], color="#326aa8")
    axes[1].axvline(0, color="0.6", lw=.6)
    axes[1].set_yticks(y, [])
    axes[1].set_xlabel("Paired p50 savings vs. serial (%)")
    for ax in axes:
        ax.invert_yaxis()
    fig.savefig(out / "moe_ep_operator_pairing.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    ranks = []
    for rank in range(2):
        path = args.raw / f"moe_ep_rank{rank}.json"
        if path.exists():
            ranks.append(json.loads(path.read_text()))
        else:
            with gzip.open(path.with_suffix(".json.gz"), "rt") as source:
                ranks.append(json.load(source))
    result = reduce(ranks)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "moe_ep_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.figures:
        figures(result, args.out)


if __name__ == "__main__":
    main()
