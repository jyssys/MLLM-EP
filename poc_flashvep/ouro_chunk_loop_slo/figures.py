"""Plots use measured data only. Labels identify aggregation and sample limits."""

import csv
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import ROOT

OUT = ROOT / "figures"
MENU = [
    "atomic_C1024",
    "atomic_C256",
    "split_C512",
    "grouped_C1024",
    "grouped_C512",
    "mixed_C128",
    "mixed_C512",
    "staged4_C1024",
]


def read(name):
    p = ROOT / name
    if not p.exists():
        return []
    return list(csv.DictReader(p.open()))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / (name + ".png"), dpi=160)
    fig.savefig(OUT / (name + ".svg"))
    plt.close(fig)


def fixed():
    rows = [
        r for r in read("fixed_grid_summary.csv") if r["P"] == "8192" and r["N"] == "8"
    ]
    raw = defaultdict(list)
    for p in (ROOT / "raw").glob("grid*/*P8192_N8_T*.json"):
        d = json.loads(p.read_text())
        key = f"{d['mode']}_C{d['chunk_tokens']}"
        for r in d["requests"]:
            if r["background"]:
                raw[key].extend(r["measured_ITLs_s"])
    fig, ax = plt.subplots(figsize=(9, 5))
    for key in MENU:
        if key in raw:
            vals = np.sort(np.array(raw[key]) * 1000)
            unique, counts = np.unique(vals, return_counts=True)
            ax.step(
                unique,
                (len(vals) - np.cumsum(counts)) / len(vals),
                where="post",
                label=f"{key}, gaps={len(vals)}",
            )
    for x in (100, 200):
        ax.axvline(x, color="gray", ls="--", lw=0.8)
    ax.set(
        xlabel="Observed background token gap (ms)",
        ylabel="Fraction of observed gaps above x",
        yscale="log",
        xlim=(0, 900),
        ylim=(0.0005, 1),
        title="A. LIVE bounded-128 cohorts, P≈8K / N8\nPooled descriptive CCDF. Tokens are not independent statistical samples.",
    )
    ax.legend(fontsize=8, ncol=2)
    save(fig, "A_ITL_CCDF")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, metric, title in zip(
        axes.flat,
        (
            "long_TTFT_ms",
            "ITL_p95_ms",
            "ITL_p99_ms",
            "host_quantum_mean_ms",
            "long_prefill_submissions",
            "staged_bytes",
        ),
        (
            "Mean long TTFT (ms)",
            "Mean cohort p95 ITL (ms)",
            "Mean cohort p99 ITL (ms)",
            "Mean HOST submission span (ms)",
            "Mean core-containing prefill submissions",
            "Staged hidden memory (MiB)",
        ),
    ):
        arr = np.full((4, 4), np.nan)
        for r in rows:
            mode, chunk = r["policy"].split("_C")
            if mode in ("atomic", "grouped", "split", "mixed"):
                arr[
                    ("atomic", "grouped", "split", "mixed").index(mode),
                    (128, 256, 512, 1024).index(int(chunk)),
                ] = float(r[metric]) / (2**20 if metric == "staged_bytes" else 1)
        im = ax.imshow(arr, aspect="auto", cmap="viridis")
        for i in range(4):
            for j in range(4):
                ax.text(
                    j,
                    i,
                    f"{arr[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )
        ax.set(
            xticks=range(4),
            xticklabels=[128, 256, 512, 1024],
            yticks=range(4),
            yticklabels=["Atomic G4", "Grouped G2", "Split G1", "Mixed G1"],
            title=title,
            xlabel="Configured C (tokens)",
        )
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(
        "B. LIVE P≈8K/N8 response surface. 3 cohorts/full-grid cell, 10/shortlist cell.\nHost submission duration is not GPU compute or E2E critical-path time.",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.85)
    save(fig, "B_CG_heatmaps")
    fig, ax = plt.subplots(figsize=(9, 6))
    for r in rows:
        mode = r["policy"].split("_C")[0]
        color = {
            "atomic": "C0",
            "split": "C1",
            "grouped": "C2",
            "mixed": "C3",
            "staged4": "C4",
        }[mode]
        x, y = float(r["long_TTFT_ms"]), float(r["ITL_p99_ms"])
        if "long_TTFT_ms_mean_CI95_low" in r:
            ax.errorbar(
                x,
                y,
                xerr=[
                    [max(0, x - float(r["long_TTFT_ms_mean_CI95_low"]))],
                    [max(0, float(r["long_TTFT_ms_mean_CI95_high"]) - x)],
                ],
                yerr=[
                    [max(0, y - float(r["ITL_p99_ms_mean_CI95_low"]))],
                    [max(0, float(r["ITL_p99_ms_mean_CI95_high"]) - y)],
                ],
                color=color,
                fmt="o",
                capsize=2,
                alpha=0.8,
            )
        else:
            ax.scatter(x, y, c=color)
        ax.annotate(
            r["policy"],
            (float(r["long_TTFT_ms"]), float(r["ITL_p99_ms"])),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.set(
        xlabel="Mean long TTFT (ms)",
        ylabel="Mean background cohort-p99 ITL (ms)",
        title="C. LIVE P≈8K/N8, engine-block/cohort bootstrap 95% CI\nDifferent cohort counts. Point dominance is not statistical dominance.",
    )
    save(fig, "C_TTFT_TBT_pareto")
    expose = [
        r
        for r in read("token_prefill_exposure_summary.csv")
        if r["P"] == "8192" and r["N"] == "8"
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in expose:
        key = f"{r['mode']}_C{r['C']}"
        if key in MENU:
            ax.scatter(
                float(r["exposure_host_ms_mean"]),
                float(r["exposed_token_intervals"]) / float(r["cohorts"]),
            )
            ax.annotate(
                key,
                (
                    float(r["exposure_host_ms_mean"]),
                    float(r["exposed_token_intervals"]) / float(r["cohorts"]),
                ),
                fontsize=8,
            )
    ax.set(
        xlabel="Mean host submission exposure in affected token gap (ms)",
        ylabel="Affected token gaps / cohort",
        title="E. Duration–frequency trade-off, LIVE P≈8K/N8\nHost interval exposure, not attributed GPU waiting.",
    )
    save(fig, "E_duration_frequency")


def serving():
    rows = read("slo_attainment_by_rate.csv")
    if not rows:
        return
    for pattern in ("poisson", "burst"):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        for ax, metric, title in zip(
            axes,
            ("TTFT_pass", "TBT_all_token_pass", "joint_pass"),
            ("TTFT ≤5s", "Every TBT ≤271.893ms", "Joint natural-response completion"),
        ):
            for policy in MENU:
                selected = sorted(
                    [
                        r
                        for r in rows
                        if r["pattern"] == pattern
                        and r["policy"] == policy
                        and r["SLO"] == "moderate"
                        and float(r["TTFT_limit_s"]) == 5.0
                    ],
                    key=lambda r: float(r["rate"]),
                )
                if not selected:
                    continue
                x = np.array([float(r["rate"]) for r in selected])
                y = np.array([float(r[metric + "_fraction"]) * 100 for r in selected])
                ax.plot(x, y, marker="o", label=policy)
                ax.fill_between(
                    x,
                    [100 * float(r[metric + "_CI95_low"]) for r in selected],
                    [100 * float(r[metric + "_CI95_high"]) for r in selected],
                    alpha=0.08,
                )
            ax.axhline(90, color="gray", ls="--")
            ax.set(
                title=title,
                xlabel="Offered requests/s per independent GPU",
                ylim=(0, 102),
            )
        axes[0].set_ylabel("Offered-request attainment (%)")
        axes[-1].legend(fontsize=7, loc="upper right")
        fig.suptitle(
            f"D. LIVE in-process {pattern}, actual natural generation. Engine/family bootstrap 95% CI.\nSmall finite windows. Exploratory rate brackets, not production capacity or HTTP latency.",
            fontsize=11,
        )
        save(fig, f"D_attainment_{pattern}")
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
        for i, a in enumerate((2.0, 5.0)):
            for j, tier in enumerate(("strict", "moderate", "relaxed")):
                ax = axes[i, j]
                for policy in MENU:
                    selected = sorted(
                        [
                            r
                            for r in rows
                            if r["pattern"] == pattern
                            and r["policy"] == policy
                            and r["SLO"] == tier
                            and float(r["TTFT_limit_s"]) == a
                        ],
                        key=lambda r: float(r["rate"]),
                    )
                    if selected:
                        x = [float(r["rate"]) for r in selected]
                        ax.plot(
                            x,
                            [100 * float(r["joint_pass_fraction"]) for r in selected],
                            "o-",
                            label=policy,
                        )
                        ax.fill_between(
                            x,
                            [100 * float(r["joint_pass_CI95_low"]) for r in selected],
                            [100 * float(r["joint_pass_CI95_high"]) for r in selected],
                            alpha=0.07,
                        )
                for target in (80, 90, 95):
                    ax.axhline(
                        target, color="gray", ls="--" if target == 90 else ":", lw=0.8
                    )
                ax.set(
                    title=f"TTFT≤{a:g}s / {tier} TBT",
                    xlabel="Configured arrival rate (req/s)",
                    ylim=(0, 102),
                )
        axes[0, 0].set_ylabel("Joint natural-response attainment (%)")
        axes[1, 0].set_ylabel("Joint natural-response attainment (%)")
        axes[-1, -1].legend(fontsize=7, ncol=2)
        fig.suptitle(
            f"LIVE {pattern} relative SLO sensitivity, 95% engine/source-family CI\n135.947/271.893/679.733ms TBT. 80/90/95% reference targets. Exploratory finite-window rates.",
            fontsize=11,
        )
        save(fig, f"D_relative_SLO_grid_{pattern}")
    for ttft in (2.0, 5.0):
        selected = [
            r
            for r in rows
            if r["pattern"] == "poisson"
            and float(r["TTFT_limit_s"]) == ttft
            and r["SLO"].startswith("absolute")
        ]
        rates = sorted({float(r["rate"]) for r in selected})
        limits = sorted({float(r["TBT_limit_s"]) * 1000 for r in selected})
        arr = np.zeros((len(rates), len(limits)))
        fig, ax = plt.subplots(figsize=(12, 5))
        for i, rate in enumerate(rates):
            for j, limit in enumerate(limits):
                options = [
                    r
                    for r in selected
                    if float(r["rate"]) == rate
                    and abs(float(r["TBT_limit_s"]) * 1000 - limit) < 1e-6
                ]
                best = max(options, key=lambda r: float(r["joint_pass_fraction"]))
                ties = sum(
                    float(r["joint_pass_fraction"])
                    == float(best["joint_pass_fraction"])
                    for r in options
                )
                arr[i, j] = 100 * float(best["joint_pass_fraction"])
                ax.text(
                    j,
                    i,
                    f"{best['policy']}{' +ties' if ties > 1 else ''}\n{arr[i, j]:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        im = ax.imshow(arr, aspect="auto", vmin=0, vmax=100, cmap="YlGnBu", alpha=0.55)
        ax.set(
            xticks=range(len(limits)),
            xticklabels=[f"{v:g}" for v in limits],
            yticks=range(len(rates)),
            yticklabels=rates,
            xlabel="Request maximum TBT limit (ms)",
            ylabel="Offered requests/s per GPU",
            title=f"F. Observed best FIXED policy by SLO, TTFT≤{ttft:g}s\nPost-hoc sensitivity only. No evaluated adaptive controller or qualified capacity claim.",
        )
        fig.colorbar(im, ax=ax, label="Joint attainment (%)")
        save(fig, f"F_observed_best_SLO_TTFT{ttft:g}")
    fig, ax = plt.subplots(figsize=(9, 5))
    for policy in MENU:
        selected = sorted(
            [
                r
                for r in rows
                if r["pattern"] == "poisson"
                and r["policy"] == policy
                and r["SLO"] == "moderate"
                and float(r["TTFT_limit_s"]) == 5.0
            ],
            key=lambda r: float(r["rate"]),
        )
        if selected:
            ax.plot(
                [float(r["rate"]) for r in selected],
                [float(r["mean_output_tokens"]) for r in selected],
                "o-",
                label=policy,
            )
    ax.set(
        xlabel="Offered requests/s",
        ylabel="Mean actual output length (tokens)",
        title="Natural-output workload confounding. Same inputs, not identical output token work.",
    )
    ax.legend(fontsize=8, ncol=2)
    save(fig, "natural_output_lengths")
    capacity = read("slo_capacity_brackets.csv")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for col, ttft in enumerate((2.0, 5.0)):
        for policy in MENU:
            selected = sorted(
                [
                    r
                    for r in capacity
                    if r["pattern"] == "poisson"
                    and r["policy"] == policy
                    and float(r["TTFT_limit_s"]) == ttft
                    and float(r["attainment_target"]) == 0.9
                    and float(r["TBT_limit_s"]) * 1000
                    in (75, 100, 125, 150, 200, 250, 500)
                ],
                key=lambda r: float(r["TBT_limit_s"]),
            )
            if selected:
                axes[0, col].plot(
                    [1000 * float(r["TBT_limit_s"]) for r in selected],
                    [
                        float(r["highest_tested_point_passing_rate"])
                        if r["highest_tested_point_passing_rate"]
                        else np.nan
                        for r in selected
                    ],
                    "o-",
                    label=policy,
                )
            goodput = sorted(
                [
                    r
                    for r in rows
                    if r["pattern"] == "poisson"
                    and r["policy"] == policy
                    and float(r["rate"]) == 0.25
                    and float(r["TTFT_limit_s"]) == ttft
                    and r["SLO"].startswith("absolute")
                ],
                key=lambda r: float(r["TBT_limit_s"]),
            )
            if goodput:
                x = [1000 * float(r["TBT_limit_s"]) for r in goodput]
                axes[1, col].plot(
                    x,
                    [float(r["SLO_goodput_offered_window_req_s"]) for r in goodput],
                    "o-",
                    label=policy,
                )
                axes[1, col].fill_between(
                    x,
                    [
                        float(r["joint_pass_CI95_low"])
                        * float(r["realized_offered_rate_per_s"])
                        for r in goodput
                    ],
                    [
                        float(r["joint_pass_CI95_high"])
                        * float(r["realized_offered_rate_per_s"])
                        for r in goodput
                    ],
                    alpha=0.06,
                )
        axes[0, col].set(
            title=f"TTFT≤{ttft:g}s: exploratory 90% point-attainment screen",
            xlabel="Every-token TBT limit (ms)",
            ylabel="Highest tested passing rate (req/s)",
            xlim=(60, 515),
            ylim=(0, 1.05),
            yticks=(0, 0.125, 0.25, 0.5, 1.0),
        )
        axes[1, col].set(
            title=f"TTFT≤{ttft:g}s: configured Poisson 0.25 req/s",
            xlabel="Every-token TBT limit (ms)",
            ylabel="Declared-drain SLO goodput (req/s)",
            xlim=(60, 515),
        )
    axes[1, 1].legend(fontsize=7, ncol=2)
    fig.suptitle(
        "LIVE natural generation. Upper panels require all finite-window stability screens.\n"
        "No qualified capacity: fewer than 1000 requests/arm/rate. Blank means no tested passing rate, not zero capacity.",
        fontsize=10,
    )
    save(fig, "G_goodput_capacity_sensitivity")


def heavy():
    rows = read("quantum_events.csv")
    groups = defaultdict(list)
    for r in rows:
        if (
            r.get("kind") != "PREFILL_CORE"
            or r.get("P") != "8192"
            or r.get("N") != "8"
            or r.get("contains_long_prefill") != "True"
        ):
            continue
        groups[(r["mode"], int(r["C"]))].append(float(r["GPU_elapsed_ms"]))
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for (mode, c), vals in sorted(groups.items()):
        ax.scatter(np.mean(vals), len(vals), label=f"{mode} C{c}, n={len(vals)}")
        ax.annotate(f"{mode} C{c}", (np.mean(vals), len(vals)), fontsize=7)
    ax.set(
        xlabel="Mean CUDA-event quantum stream elapsed (ms)",
        ylabel="Observed prefill/mixed submission count",
        title="Separate HEAVY P≈8K/N8 trace. Staging and stream waits included.\nThis is not clean E2E timing or pure GPU compute.",
    )
    save(fig, "quantum_GPU_duration_frequency")


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    fixed()
    serving()
    heavy()
    print("FIGURES_COMPLETE", len(list(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
