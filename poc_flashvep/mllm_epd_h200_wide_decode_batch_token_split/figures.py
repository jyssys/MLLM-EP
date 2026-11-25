"""Render wide H200 confirmation figures from the published analysis tables."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

from .analysis import CAPS, CONTEXT, NATIVE_LOADS, group
from ..mllm_epd_h200_prefill_heavy_batch_matched.analysis import CRITICAL_PATH, csv_file
from ..mllm_epd_h200_prefill_heavy_batch_matched.trace import intersect, merged, scheduler_intervals


REFERENCE_START = "free_q5_burners_on"
REFERENCE_COHORT = "native_common_free_5_profile"
ROLE_LABELS = {"encoder": "E", "prefill": "P", "decode": "D"}
CAP_LABELS = {cap: ("Free" if cap == "free" else f"B{cap}") for cap in CAPS}
METHOD_ORDER = {"native-strong": 0, "native-common": 1, "static10": 2, "idle10": 3}
GIB = 2 ** 30


def table(folder, name):
    path = folder / f"{name}.csv"
    if not path.is_file():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def optional(value):
    return float(value) if value not in (None, "") else None


def value_label(row):
    return f"{row['method_family']} · {CAP_LABELS[row['decode_cap']]}"


def candidates(rows):
    # Predeclared conditional pilot rows. Screen, profile, and batch-counter
    # observations are omitted from the latency and SLO figure.
    return [row for row in rows if row["start"].startswith("pilot_") and row["observation"] == "clean"]


def capacity(plots, rows):
    if not rows:
        return
    fields = ("tokens_per_second", "ITL_p99_ms", "TPOT_p95_ms")
    fig, axes = plots.subplots(1, 3, figsize=(11, 3.75))
    for field, label, ax in zip(fields, ("Output tokens/s", "ITL p99 (ms)", "TPOT p95 (ms)"), axes, strict=True):
        for _, samples in group(rows, ("start", "repeat")).items():
            points = [(CAPS.index(r["cap"]), float(r[field])) for r in samples if optional(r.get(field)) is not None]
            if points:
                x, y = zip(*sorted(points))
                ax.plot(x, y, marker="o", linewidth=1.2)
        ax.set(xticks=range(len(CAPS)), xticklabels=list(CAP_LABELS.values()),
               xlabel="Configured D HTTP cap", ylabel=label)
        ax.grid(alpha=.2)
    fig.suptitle("Isolated D · 5,269 replay output tokens · actual prepared NIXL KV")
    plots.save(fig, "D_only_output_ITL_TPOT_by_cap")


def planned_kv(plots, rows, runs):
    points = []
    for row in rows:
        path = runs / row["start"] / "analysis" / f"capacity_decode_wide_{row['cap']}_r{row['repeat']}.json"
        if not path.is_file():
            continue
        record = json.loads(path.read_text())
        plans = [cohort["planned_D_peak_KV_bytes"] for cohort in record["cohorts"]
                 if "planned_D_peak_KV_bytes" in cohort]
        diagnostics = record["diagnostics"]
        allocated = diagnostics.get("before", {}).get("D_KV_storage_bytes")
        peak = diagnostics.get("D_actual_peak_KV_usage_bytes")
        if plans and allocated is not None:
            points.append({"cap": row["cap"], "start": row["start"], "repeat": row["repeat"],
                           "planned_upper_bound_bytes": max(plans), "allocated_pool_bytes": allocated,
                           "observed_KV_peak_bytes": peak,
                           "observed_peak_status": "MEASURED" if peak is not None else "NOT_MEASURED"})
    if not points:
        return
    csv_file(plots.output / "D_only_KV_capacity_plot_data.csv", points)
    fig, ax = plots.subplots(figsize=(7.4, 3.7))
    for index, row in enumerate(points):
        cap = CAPS.index(row["cap"])
        ax.scatter(cap, row["planned_upper_bound_bytes"] / GIB, marker="o", color="#086788",
                   label="Planned maximum cohort bound" if not index else None)
        ax.scatter(cap, row["allocated_pool_bytes"] / GIB, marker="_", s=150, color="#616161",
                   label="Allocated D KV pool" if not index else None)
        if row["observed_KV_peak_bytes"] is not None:
            ax.scatter(cap, row["observed_KV_peak_bytes"] / GIB, marker="x", color="#b54a14")
    if not any(row["observed_KV_peak_bytes"] is not None for row in points):
        ax.text(.02, .06, "Active peak KV occupancy: not measured", transform=ax.transAxes, fontsize=9)
    ax.set(xticks=range(len(CAPS)), xticklabels=list(CAP_LABELS.values()),
           xlabel="Configured D HTTP cap", ylabel="GiB", ylim=(0, 46))
    ax.legend(fontsize=8, loc="lower right")
    plots.save(fig, "D_only_KV_bound_and_allocation")


def pilot_figures(plots, critical, slo):
    def pilot_groups(rows):
        for key, condition in sorted(group(candidates(rows), CONTEXT).items()):
            arms = group(condition, ("start", "cohort"))
            lineup = sorted(arms.values(), key=lambda arm: (
                CAPS.index(arm[0]["decode_cap"]), METHOD_ORDER.get(arm[0]["method_family"], 99),
                arm[0]["start"]
            ))
            mode = "natural" if key[0] == "Natural EOS" else "replay"
            yield f"{mode}_{key[1]}x_{key[2].lower()}_{key[3].lower()}_{key[-1]}", lineup

    for tag, arms in pilot_groups(critical):
        fig, ax = plots.subplots(figsize=(9, max(3.8, len(arms) * .42)))
        positions = range(len(arms))
        left = [0.] * len(arms)
        for stage in CRITICAL_PATH:
            durations = [sum(float(row["mean_ms"]) for row in arm if row["stage"] == stage) for arm in arms]
            ax.barh(positions, durations, left=left, label=stage)
            left = [a + b for a, b in zip(left, durations, strict=True)]
        ax.set(yticks=positions, yticklabels=[value_label(arm[0]) for arm in arms],
               xlabel="Mean TTFT (ms)", title="Conditional pilot · additive host boundaries")
        ax.invert_yaxis()
        ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.18))
        plots.save(fig, f"pilot_host_critical_path_{tag}")

    for tag, arms in pilot_groups(slo):
        fig, ax = plots.subplots(figsize=(9, max(3.8, len(arms) * .42)))
        categories = sorted({row["category"] for arm in arms for row in arm},
                            key=lambda label: (label != "JOINT_PASS", label))
        left = [0.] * len(arms)
        positions = range(len(arms))
        for category in categories:
            counts = [sum(int(row["requests"]) for row in arm if row["category"] == category) for arm in arms]
            ax.barh(positions, counts, left=left, label=category.replace("_", " "))
            left = [a + b for a, b in zip(left, counts, strict=True)]
        ax.set(yticks=positions, yticklabels=[value_label(arm[0]) for arm in arms],
               xlabel="Requests", title="Conditional pilot · joint client SLO outcomes")
        ax.invert_yaxis()
        ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.18))
        plots.save(fig, f"pilot_joint_SLO_failures_{tag}")


def quiet_heatmaps(plots, rows):
    selected = [row for row in rows if row["method_family"] == "native-common" and row["output_mode"] == "Natural EOS"]
    for key, condition in sorted(group(selected, CONTEXT).items()):
        observed = group(condition, ("decode_cap", "QPS", "minimum_window_ms"))
        loads = sorted(set(NATIVE_LOADS) | {float(row["QPS"]) for row in condition})
        thresholds = sorted({int(float(row["minimum_window_ms"])) for row in condition})
        fig, axes = plots.subplots(1, len(thresholds), figsize=(12, 5), squeeze=False)
        for threshold, ax in zip(thresholds, axes.flat, strict=True):
            points = plots.np.full((len(CAPS), len(loads)), plots.np.nan)
            annotations = {}
            for i, cap in enumerate(CAPS):
                for j, qps in enumerate(loads):
                    entries = [r for (c, q, t), records in observed.items()
                               if c == cap and float(q) == qps and float(t) == threshold
                               for r in records if optional(r["fraction_P_busy"]) is not None]
                    if not entries:
                        continue
                    qualified = [r for r in entries if r["evidence"] == "TRACE_ANALYSIS"]
                    valid = qualified or entries
                    points[i, j] = min(float(row["fraction_P_busy"]) for row in valid) * 100
                    annotations[i, j] = ("" if qualified else "*") + (f"\nn={len(valid)}" if len(valid) > 1 else "")
            cmap = plots.plt.get_cmap("viridis").copy()
            cmap.set_bad("#f3f3f3")
            img = ax.imshow(plots.np.ma.masked_invalid(points), cmap=cmap, aspect="auto", vmin=0, vmax=100)
            fig.colorbar(img, ax=ax, fraction=.05, pad=.04)
            for i in range(len(CAPS)):
                for j in range(len(loads)):
                    if not plots.np.isnan(points[i, j]):
                        ax.text(j, i, f"{points[i, j]:.1f}{annotations[i, j]}", ha="center", va="center",
                                fontsize=8, color="white" if points[i, j] < 56 else "black")
            ax.set(xticks=range(len(loads)), xticklabels=[f"{q:g}" for q in loads],
                   yticks=range(len(CAPS)), yticklabels=list(CAP_LABELS.values()),
                   xlabel="Offered QPS", title=f"Quiet period ≥{threshold} ms")
        fig.suptitle(f"Observed P-busy coverage (%) · {key[1]}× Main100 · burners {key[2]}\n"
                     "Blank means unmeasured. * Unqualified trace. Minimum shown when multiple starts exist.")
        tag = f"{key[1]}x_{key[2].lower()}_{key[3].lower()}_{key[-1]}"
        plots.save(fig, f"D_quiet_coverage_{tag}")


def binned_timeline(plots, groups, duration):
    edges = plots.np.linspace(0, duration, max(2, int(duration / .025) + 1))
    values = {}
    for name, intervals in groups.items():
        i = 0
        fractions = []
        for left, right in zip(edges, edges[1:]):
            while i < len(intervals) and intervals[i][1] <= left:
                i += 1
            j = i
            busy = 0.
            while j < len(intervals) and intervals[j][0] < right:
                a, b = intervals[j]
                busy += max(0., min(b, right) - max(a, left))
                j += 1
            fractions.append(100 * busy / (right - left))
        values[name] = fractions
    return edges, values


def representative_timeline(plots, runs, quiet):
    run = runs / REFERENCE_START
    trace_file = run / "analysis/native_common_free_5_trace.json"
    if not trace_file.is_file():
        return
    # These files are the compact finalized interval and sampled scheduler
    # receipts. No per-kernel CUPTI dump or event stream is reparsed.
    trace = json.loads(trace_file.read_text())
    left, right = trace["window_monotonic"]
    duration = right - left
    evidence = trace["evidence"]
    roi = run / "raw/profiles" / REFERENCE_COHORT
    queue = []
    for role, label in ROLE_LABELS.items():
        notes = json.loads((roi / role / "scheduler.json").read_text())
        for sample in notes["rows"]:
            instant = sample["begin_ns"] / 1e9
            if left <= instant < right:
                queue.append({"seconds": instant - left, "stage": label,
                              "waiting": sample["before"]["waiting"],
                              "running": sample["before"]["running"],
                              "interval_ms": notes["sampling_interval_ms"], "evidence": evidence})
    csv_file(plots.output / "free_q5_on_scheduler_waiting_samples.csv", queue, ("seconds", "stage", "waiting"))
    if queue:
        fig, axes = plots.subplots(3, 1, figsize=(9.7, 6), sharex=True)
        for ax, label in zip(axes, ROLE_LABELS.values(), strict=True):
            samples = [row for row in queue if row["stage"] == label]
            ax.scatter([row["seconds"] for row in samples], [row["waiting"] for row in samples], s=15)
            ax.set(ylabel=f"{label} waiting", xlim=(0, duration))
            ax.grid(alpha=.2)
        axes[-1].set_xlabel("Seconds since central p20 arrival")
        fig.suptitle(f"{evidence} · first Native Free QPS5 with burners on\n"
                     "Sampled engine scheduler waiting · gaps remain unobserved")
        plots.save(fig, "E_P_D_sampled_queue_Free_QPS5_first_burners_on")

    windows = [(float(row["begin_seconds"]) + left, float(row["end_seconds"]) + left)
               for row in quiet if row["start"] == REFERENCE_START and row["cohort"] == REFERENCE_COHORT
               and float(row["duration_ms"]) >= 160]
    if not windows:
        return
    p_busy = merged(trace["kernel_intervals"]["prefill"], left, right)
    d_busy = merged(trace["kernel_intervals"]["decode"], left, right)
    p_waiting, _ = scheduler_intervals(roi / "prefill", left, right)
    overlap = intersect(intersect(p_busy, p_waiting), merged(windows, left, right))
    spans = {
        "P kernel": p_busy, "D kernel": d_busy,
        "D quiet ≥160 ms": merged(windows, left, right),
        "P busy, P waiting, D quiet": overlap,
    }
    edges, values = binned_timeline(plots, {k: [(a - left, b - left) for a, b in v] for k, v in spans.items()}, duration)
    cpu_rows = [{"begin_seconds": a, "end_seconds": b, "evidence": evidence,
                 **{key: percentages[i] for key, percentages in values.items()}}
                for i, (a, b) in enumerate(zip(edges, edges[1:]))]
    csv_file(plots.output / "free_q5_on_diagnostic_PD_overlap_bins.csv", cpu_rows)
    fig, axes = plots.subplots(4, 1, figsize=(10, 6.8), sharex=True)
    for ax, (name, percentages) in zip(axes, values.items(), strict=True):
        ax.stairs(percentages, edges, fill=True, alpha=.75)
        ax.set_ylim(0, 100)
        ax.set_ylabel(name + " (%)", rotation=0, labelpad=8, ha="right", va="center")
        ax.grid(alpha=.15)
    axes[-1].set_xlabel("Seconds since central p20 arrival")
    fig.suptitle(f"{evidence} · first Native Free QPS5 with burners on\n"
                 "Unioned CUPTI intervals, observed quiet and recent sampled P waiting")
    plots.save(fig, "PD_overlap_Free_QPS5_first_burners_on_diagnostic")


class Plots:
    def __init__(self, output):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        self.plt, self.np = plt, np
        self.output = output
        self.files = []

    def subplots(self, *args, **kwargs):
        return self.plt.subplots(*args, **kwargs)

    def save(self, fig, stem):
        fig.tight_layout()
        for ext in (".png", ".pdf"):
            fig.savefig(self.output / (stem + ext), dpi=180)
            self.files.append(stem + ext)
        self.plt.close(fig)


def render(results, runs):
    plots = Plots(results)
    d_only = table(results, "wide_decode_capacity")
    quiet = table(results, "quiet_windows")
    capacity(plots, d_only)
    planned_kv(plots, d_only, runs)
    pilot_figures(plots, table(results, "critical_path"), table(results, "slo_breakdown"))
    quiet_heatmaps(plots, table(results, "quiet_overlap"))
    representative_timeline(plots, runs, quiet)
    return {"figures": plots.files, "representative_start": REFERENCE_START,
            "representative_cohort": REFERENCE_COHORT,
            "central_window": "Scheduled arrival p20–p80 of the predetermined first Free QPS5 start.",
            "ratio_sweep": "NOT_RUN · no ratio-sweep figure generated"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args()
    if not (args.results / "receipt.json").is_file():
        parser.error("Complete the main analysis before rendering supplementary figures")
    receipt = render(args.results, args.runs)
    (args.results / "supplementary_figures_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
