"""Render whole-request deflection and equal-budget native serving controls."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASES = (("8K/O16\n4.5 req/s", 8192, 4.5), ("16K/O16\n1.9 req/s", 16384, 1.9),
         ("16K/O16\n2.6 req/s", 16384, 2.6))
LAYOUTS = (
    ("dense-controls/p-light/", "Native 1P/1D"),
    ("dense-controls/pd-cross/", "Cross-layer NIXL"),
    ("dense-controls/pd-deflect/", "D-local request deflection"),
    ("dense-controls/dp2/", "DP2 at 8,192"),
    ("dense-controls/dp2-chunk2048/", "DP2 at 2,048"),
    ("dense-controls/dp2-no-chunk/", "DP2 unchunked"),
    ("native-static-3gpu/p2_d1/", "2P/1D"),
    ("native-static-3gpu/p_tp2_d1/", "TP2-P/1D"),
    ("native-static-3gpu/p1_d2/", "1P/2D"),
)


def is_control(row):
    return not (row["profiled"] or row.get("shape_observed") or row["path"].endswith("_clean_after"))


def matched(rows, prefix, size, qps):
    return next((row for row in rows if row["path"].startswith(prefix)
                 and row["inputs"] == [size] and row["output_contract"] == [16]
                 and row["qps"] == qps and row["count"] == 64), None)


def matched_goodput(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained", sharey=True)
    for axis, budget in zip(axes, (2, 3)):
        for prefix, name in LAYOUTS:
            arm = [(i, row["goodput"]) for i, (_, size, qps) in enumerate(CASES)
                   if (row := matched(rows, prefix, size, qps)) and row["gpu_budget"] == budget]
            if arm:
                x, goodput = zip(*arm)
                axis.plot(x, goodput, "o-", label=name)
        axis.set(xticks=range(len(CASES)), xticklabels=[label for label, _, _ in CASES],
                 title=f"{budget} H200 GPUs", ylabel="Joint SLO goodput (req/s)")
        if axis.lines:
            axis.legend(fontsize=7)
    fig.savefig(out / "native_same_budget_goodput.png", dpi=150)
    plt.close(fig)


def useful_work(rows, out):
    cohort = [(label, matched(rows, "dense-controls/pd-deflect/", size, qps))
              for label, size, qps in CASES]
    cohort = [(label, row) for label, row in cohort if row is not None]
    if not cohort:
        return
    labels, measurements = zip(*cohort)
    all_rows = [value["prompt_rows"] for value in measurements]
    moved = [value["deflected_completed_input_tokens"] for value in measurements]
    fig, axis = plt.subplots(figsize=(7.6, 4.0), layout="constrained")
    axis.bar(labels, [rows / 1024 for rows in all_rows], color="#d2d8e1", label="All completed input")
    axis.bar(labels, [rows / 1024 for rows in moved], color="#235989", label="Completed D-local input")
    for i, (row, actual) in enumerate(zip(measurements, moved)):
        axis.annotate(f'{row["deflected_count"]}/{row["count"]} requests',
                      (i, actual / 1024), xytext=(0, 7), textcoords="offset points", ha="center")
    axis.set(ylabel="Input tokens (Ki)", title="Actual work moved by generic whole-request deflection")
    axis.legend(fontsize=8)
    fig.savefig(out / "native_deflection_useful_work.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for row in json.loads(args.summary.read_text()) if is_control(row)]
    args.out.mkdir(parents=True, exist_ok=True)
    matched_goodput(rows, args.out)
    useful_work(rows, args.out)


if __name__ == "__main__":
    main()
