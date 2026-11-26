"""Recreate the historical panels that motivate the H200 Dynamic Split experiment."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

REFERENCE_COMMIT = "55870ccafb210d32dc956320d02226d26f2efee6"
REFERENCE_PROFILE = "free_q5_full_lowfreq_burners_on"
FREE_ARMS = {
    "pilot_free_native_common": "NativeCommon Free",
    "pilot_free_static10": "Static10 Free",
    "pilot_free_idle10": "Idle10 Free",
}
COLORS = ["#2e6277", "#b15349", "#549b74"]


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def exactly_one(rows, description):
    if len(rows) != 1:
        raise ValueError(f"Expected one {description}, found {len(rows)}")
    return rows[0]


def persist(fig, output, stem):
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def annotate_bars(ax, bars, suffix, decimals=1):
    ax.bar_label(bars, labels=[f"{bar.get_height():.{decimals}f}{suffix}" for bar in bars],
                 padding=3, fontsize=8)


def plot_motivation(profiles, output):
    row = exactly_one([item for item in profiles if item["run"] == REFERENCE_PROFILE],
                      "predeclared historical Free QPS5 profile")
    kernel_busy = row["kernel_busy_fraction"]
    scheduler_waiting = row["p_scheduler"]["window_fraction_recent_sample_waiting"]
    quiet = row["robust_window_160_ms"]["p_busy_overlap_fraction"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), layout="constrained")
    bars = axes[0].bar(["Vision E", "Prefill P", "Decode D"],
                       [100 * kernel_busy[stage] for stage in ("encoder", "prefill", "decode")],
                       color=COLORS)
    axes[0].set(title="GPU kernel activity", ylabel="% of central trace", ylim=(0, 110))
    annotate_bars(axes[0], bars, "%")
    bars = axes[1].bar(["P sampled\nwaiting", "D physical\nidle", "Quiet ≥160 ms\nwith P backlog"],
                       [100 * scheduler_waiting, 100 * (1 - kernel_busy["decode"]), 100 * quiet],
                       color=COLORS)
    axes[1].set(title="Queue and idle indicators", ylabel="Percent", ylim=(0, 110))
    axes[1].text(0.5, -0.25, "Quiet overlap is normalized by P busy time.\n"
                 "P sampled waiting and D physical idle use the central trace.",
                 transform=axes[1].transAxes, ha="center", fontsize=8)
    annotate_bars(axes[1], bars, "%")
    fig.suptitle("Historical H200 QPS5 Free · DIAGNOSTIC · observer gate failed")
    persist(fig, output, "fig01_historical_prefill_heavy")
    return {
        "figure": 1, "start": REFERENCE_PROFILE, "evidence": row["evidence"],
        "kernel_busy_fraction_by_gpu": kernel_busy,
        "p_scheduler_recent_wait_fraction_central": scheduler_waiting,
        "d_physical_idle_fraction_central": 1 - kernel_busy["decode"],
        "quiet_160ms_fraction_of_p_busy": quiet,
        "quiet_160ms_windows": row["robust_window_160_ms"]["windows"],
        "motivation_status": row["status"],
    }


def plot_dynamic_motivation(profiles, output):
    selected = [row for row in profiles if row["qps"] == 5 and str(row["decode_cap"]) in ("8", "free")]
    if not selected:
        return {"figure": 1, "current": "NOT_MEASURED"}
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.8), layout="constrained")
    width = .8 / len(selected)
    for index, row in enumerate(selected):
        label = f"{row['decode_cap']} · {row['evidence']}"
        busy = row["kernel_busy_fraction"]
        offset = width * (index - (len(selected) - 1) / 2)
        bars = axes[0].bar([offset + j for j in range(3)], [100 * busy[role] for role in (
            "encoder", "prefill", "decode")], width=width, label=label)
        annotate_bars(axes[0], bars, "%")
        bars = axes[1].bar([offset + j for j in range(3)], [
            100 * row["p_scheduler"]["window_fraction_recent_sample_waiting"],
            100 * (1 - busy["decode"]), 100 * row["robust_window_160_ms"]["p_busy_overlap_fraction"],
        ], width=width)
        annotate_bars(axes[1], bars, "%")
    axes[0].set(title="GPU kernel activity", ylabel="% of central trace", ylim=(0, 120),
                xticks=[0, 1, 2], xticklabels=["Vision E", "Prefill P", "Decode D"])
    axes[1].set(title="P waiting and D quiet", ylabel="Percent", ylim=(0, 120), xticks=[0, 1, 2],
                xticklabels=["P sampled waiting", "D physical idle", "Quiet ≥160 ms"])
    axes[1].text(.5, -.23, "Quiet overlap with P backlog uses P busy time as the denominator.",
                 transform=axes[1].transAxes, ha="center", fontsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Current campaign · NativeCommon QPS5")
    persist(fig, output, "fig01_current_motivation")
    return {"figure": 1, "current": [{"run": row["run"], "evidence": row["evidence"],
                                       "motivation_status": row["status"]} for row in selected]}


def select_free_counter(rows, start):
    return exactly_one([row for row in rows if row["start"] == start
                        and row["observation"] == "batch_observed" and float(row["QPS"]) == 5.0],
                       f"Counter cohort for {start}")


def select_clean_wait(rows, start):
    return exactly_one([row for row in rows if row["start"] == start
                        and row["observation"] == "clean" and float(row["QPS"]) == 5.0
                        and row["stage"] == "P ready to D submit"], f"clean D wait for {start}")


def plot_admission(counter, critical, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    bottom = [0.0] * len(FREE_ARMS)
    arm_data = []
    for start, name in FREE_ARMS.items():
        observed = select_free_counter(counter, start)
        histogram = json.loads(observed["D_prepared_decode_call_histogram"])
        clean = select_clean_wait(critical, start)
        arm_data.append({"start": start, "method": name, "counter_evidence": observed["batch_evidence"],
                         "prepared_decode_call_histogram": histogram, "host_evidence": clean["evidence"],
                         "mean_coordinator_wait_ms": float(clean["mean_ms"])})
    for batch, color in ((1, "#b15349"), (2, "#ddb073"), ("≥3", "#2e6277")):
        heights = []
        for row in arm_data:
            histogram = row["prepared_decode_call_histogram"]
            numerator = (sum(value for size, value in histogram.items() if int(size) >= 3)
                         if batch == "≥3" else histogram.get(str(batch), 0))
            heights.append(100 * numerator / sum(histogram.values()))
        axes[0].bar(list(FREE_ARMS.values()), heights, bottom=bottom, color=color, label=str(batch))
        bottom = [base + height for base, height in zip(bottom, heights)]
    axes[0].set(title="Prepared decode distribution · separate Counter", ylabel="% of forwards", ylim=(0, 105))
    axes[0].legend(title="Actual requests", frameon=False, fontsize=8)
    waits = [row["mean_coordinator_wait_ms"] / 1_000 for row in arm_data]
    bars = axes[1].bar(list(FREE_ARMS.values()), waits, color=COLORS)
    axes[1].set(title="Clean mean D coordinator wait", ylabel="Seconds from P-ready to D-submit", ylim=(0, 6.2))
    annotate_bars(axes[1], bars, " s", decimals=3)
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=15, labelsize=8)
    fig.suptitle("Historical H200 Free QPS5 Natural EOS")
    persist(fig, output, "fig02_historical_static_admission")
    return {"figure": 2, "arms": arm_data}


def plot_mixed_correctness(receipt, output):
    probes = [row for run in receipt["runs"] for row in run["mixed"] if row.get("decode_geometry")]
    if not probes:
        return {"evidence": "NOT_MEASURED"}
    labels = [row["case"].replace("mixed_", "") + "%" for row in probes]
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.6), layout="constrained")
    base = [0.] * len(probes)
    for batch, color in ((1, "#b15349"), (2, "#ddb073"), ("≥3", "#2e6277")):
        shares = []
        for row in probes:
            hist = row["decode_geometry"]["decode_request_count_per_forward"]
            calls = (sum(n for size, n in hist.items() if int(size) >= 3)
                     if batch == "≥3" else hist.get(str(batch), 0))
            shares.append(100 * calls / sum(hist.values()))
        axes[0].bar(labels, shares, bottom=base, label=str(batch), color=color)
        base = [a + b for a, b in zip(base, shares)]
    axes[0].set(title="Prepared forwards containing decode", ylabel="% of these forwards", xlabel="Proposed suffix ratio")
    axes[0].legend(title="Actual decode requests", frameon=False, fontsize=8)
    rows = [row["split_count"] for row in probes]
    ok = [row["full_cache_and_coverage_pass_count"] for row in probes]
    axes[1].plot(labels, rows, marker="o", label="Executed D suffix", color=COLORS[0])
    axes[1].plot(labels, ok, marker="o", label="Full-cache and coverage pass", color=COLORS[1])
    axes[1].plot(labels, [row["exact_greedy_count"] for row in probes], marker="o", label="Exact Greedy", color=COLORS[2])
    axes[1].set(title="Route and correctness outcomes", ylabel="Requests out of eight", ylim=(0, 8.5),
                xlabel="Proposed suffix ratio")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Current H200 · mixed N8 correctness probes · DIAGNOSTIC")
    persist(fig, output, "fig02_12_mixed_correctness_batch_and_outcomes")
    return {"figures": [2, 12], "evidence": "DIAGNOSTIC", "probes": [row["case"] for row in probes],
            "scope": "Instrumented probes establish mixed-route execution, without a clean serving attribution."}


def plot_natural_correctness(receipt, output):
    runs = [run for run in receipt["runs"] if run["serial"]]
    no_hold = next((run for run in runs if run["start"] == "natural_nohold_correctness"), None)
    if no_hold is None:
        return {"figure": 12, "evidence": "NOT_MEASURED"}
    split_cases = [case for case in no_hold["serial"] if case.get("route") == "split" and case["actual_prefill_chunks"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 3.8), layout="constrained")
    cut_rows = []
    for index, row in enumerate(split_cases):
        actual = [end for _, end in row["actual_prefill_chunks"]["chunks"]]
        target = row["requested_prefix_tokens"]
        axes[0].scatter(actual, [index] * len(actual), color=COLORS[0], s=23, label="Actual P chunk end" if index == 0 else None)
        axes[0].scatter([target], [index], color=COLORS[1], marker="|", s=85,
                        label="Requested cut" if index == 0 else None)
        cut_rows.append({"case": row["case"], "target": target, "actual": actual,
                         "skipped_in_flight": len(row["natural_skips"] or [])})
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set(title="No hold · stock P chunks", xlabel="Original input token position",
                yticks=range(len(split_cases)),
                yticklabels=[str(case["full_prompt_tokens"]) + " tokens" for case in split_cases])
    outcomes = [{"run": run["start"], "tests": len(run["serial"]),
                 "exact": sum(row["passed"] for row in run["serial"]),
                 "executed_suffix": sum(row["ran_on_D"] is True for row in run["serial"])} for run in runs]
    labels = [row["run"].removeprefix("natural_").removesuffix("_correctness").replace("_", " ") for row in outcomes]
    bars = axes[1].barh(labels, [row["tests"] for row in outcomes], color="#dce7eb", label="Probe cases")
    axes[1].barh(labels, [row["executed_suffix"] for row in outcomes], color=COLORS[0], label="D suffix witnessed")
    axes[1].bar_label(bars, labels=[f"{row['exact']}/{row['tests']} vs E+P teacher" for row in outcomes],
                      padding=4, fontsize=7)
    axes[1].set(title="Natural boundary route outcomes", xlabel="Serial correctness cases",
                xlim=(0, max(row["tests"] for row in outcomes) + 3))
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("Current H200 · natural P boundary correctness probes")
    persist(fig, output, "fig12_natural_boundary_correctness")
    return {"figure": 12, "evidence": "DIAGNOSTIC", "raw_chunk_ends": cut_rows, "outcomes": outcomes,
            "scope": "Exact counts include Greedy sequence against the serial E+P teacher. "
                     "The same-D continuation receipt is separate. No online timing attribution."}


def plot_lead_decode(receipt, output):
    cases = [case for run in receipt["runs"] for case in run["paired_geometry"]
             if case["lead_boundary_control"] and case["lead_shared_decode"]]
    if not cases:
        return {"figure": 2, "evidence": "NOT_MEASURED"}
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.0), layout="constrained")
    names, histograms, shared = [], [], []
    for case in cases:
        label = "Image edge" if "edge" in case["case"] else "Image interior"
        for arm in ("native", "late"):
            names.append(f"{label}\n{arm.capitalize()}")
            histograms.append(case[f"decode_geometry_{arm}"]["decode_request_count_per_forward"])
            shared.append(case["lead_shared_decode"][arm])
    baseline = [0.] * len(names)
    for sizes, label, color in (({1}, "1", "#b15349"), ({2}, "2", "#ddb073"),
                                ({3, 4}, "3–4", "#7bb2b3"), ({5, 6, 7, 8}, "5–8", "#2e6277")):
        values = [100 * sum(count for size, count in hist.items() if int(size) in sizes) / sum(hist.values())
                  for hist in histograms]
        axes[0].bar(names, values, bottom=baseline, label=label, color=color)
        baseline = [a + b for a, b in zip(baseline, values, strict=True)]
    axes[0].set(title="Prepared decode batch distribution", ylabel="% of decode-containing forwards", ylim=(0, 105))
    axes[0].legend(title="Decode requests", frameon=False, fontsize=7, ncols=2)
    bars = axes[1].bar(names, [row["shared_steps"] for row in shared], color=[COLORS[i % 2] for i in range(len(names))])
    axes[1].bar_label(bars, labels=[f"max {row['max_decode_batch']}" for row in shared], fontsize=8, padding=4)
    axes[1].set(title="Lead and Native follower share a forward", ylabel="Prepared decode forwards",
                ylim=(0, max(row["shared_steps"] for row in shared) * 1.2))
    fig.suptitle("Current H200 · paired staged correctness · Greedy 7/8 in each fixture")
    persist(fig, output, "fig02_staged_natural_lead_decode")
    return {"figure": 2, "evidence": "DIAGNOSTIC",
            "prepared_decode_histograms": dict(zip(names, histograms, strict=True)),
            "lead_shared_decode": dict(zip(names, shared, strict=True)),
            "scope": "The lead first passes P. Native followers are released at the lead P response. "
                     "Batch co-execution is observed under this correctness fixture."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wide", type=Path, required=True, help="Prior wide experiment results directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motivation", type=Path, help="Current campaign motivation gate receipt")
    parser.add_argument("--correctness", type=Path, help="Current campaign correctness evidence")
    parser.add_argument("--natural-correctness", type=Path, help="Current natural-boundary correctness evidence")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gate, counter, critical = [args.wide / name for name in
                               ("discovery_gates_wide.json", "decode_cap.csv", "critical_path.csv")]
    receipt = {
        "historical_commit": REFERENCE_COMMIT,
        "sources": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (gate, counter, critical)},
        "panels": [plot_motivation(json.loads(gate.read_text())["profiles"], args.output),
                   plot_admission(csv_rows(counter), csv_rows(critical), args.output)],
        "new_dynamic_method_observations": "NOT_MEASURED_IN_THESE_HISTORICAL_PANELS",
    }
    if args.motivation:
        receipt["current_motivation"] = plot_dynamic_motivation(json.loads(args.motivation.read_text())["profiles"],
                                                               args.output)
        receipt["current_motivation_sha256"] = hashlib.sha256(args.motivation.read_bytes()).hexdigest()
    if args.correctness:
        receipt["current_correctness"] = plot_mixed_correctness(json.loads(args.correctness.read_text()), args.output)
        receipt["current_correctness_sha256"] = hashlib.sha256(args.correctness.read_bytes()).hexdigest()
    if args.natural_correctness:
        natural = json.loads(args.natural_correctness.read_text())
        receipt["natural_correctness"] = plot_natural_correctness(natural, args.output)
        receipt["natural_lead_decode"] = plot_lead_decode(natural, args.output)
        receipt["natural_correctness_sha256"] = hashlib.sha256(args.natural_correctness.read_bytes()).hexdigest()
    (args.output / "historical_figure_data.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
