"""Figures use live timing records. No modeled speedups."""

import csv
from .common import ROOT


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader((ROOT / "results_raw.csv").open()))
    target = ROOT / "figures"
    target.mkdir(exist_ok=True)
    colors = {
        "atomic": "#3760a0",
        "split": "#3b926b",
        "mixed": "#9a6bac",
        "grouped": "#d37b30",
    }
    for run in ("stock_calibration_v1", "static_grid_v1", "fa4_grid_v1"):
        selected = [r for r in rows if r["run"] == run and r["trace"] == "False"]
        if not selected:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
        for ax, (p, n) in zip(axes.flat, [(2048, 4), (2048, 8), (8192, 4), (8192, 8)]):
            for mode in colors:
                points = [
                    r
                    for r in selected
                    if r["mode"] == mode and int(r["P"]) == p and int(r["N"]) == n
                ]
                if not points:
                    continue
                points.sort(key=lambda r: int(r["C"]))
                ax.plot(
                    [float(r["long_TTFT_s"]) for r in points],
                    [1000 * float(r["ITL_p95_s"]) for r in points],
                    "o-",
                    label=mode,
                    color=colors[mode],
                )
                for r in points:
                    ax.annotate(
                        r["C"],
                        (float(r["long_TTFT_s"]), 1000 * float(r["ITL_p95_s"])),
                        fontsize=7,
                    )
            ax.set(
                title=f"Prompt bin {p}, concurrency {n}",
                xlabel="Long-request TTFT (s)",
                ylabel="Background ITL p95 (ms)",
            )
            ax.grid(alpha=0.3)
        if axes[0, 0].get_legend_handles_labels()[0]:
            axes[0, 0].legend()
        fig.suptitle(run + " — LIVE, natural EOS with cap 128; pilot tails")
        fig.savefig(target / (run + "_TTFT_ITL.png"), dpi=150)
        plt.close(fig)
    summary = list(csv.DictReader((ROOT / "warm_confirmation_summary.csv").open()))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for i, (p, n) in enumerate((("2048", "4"), ("8192", "8"))):
        for j, tail in enumerate(("95", "99")):
            ax = axes[i, j]
            for r in summary:
                if (r["P"], r["N"]) != (p, n) or r["C"] != "1024":
                    continue
                x = float(r["mean_cohort_long_TTFT_s"]) * 1000
                y = float(r[f"mean_cohort_ITL_p{tail}_s"]) * 1000
                ax.scatter(x, y, color=colors[r["mode"]], s=65)
                ax.annotate(
                    r["mode"] + " C" + r["C"],
                    (x, y),
                    xytext={
                        "atomic": (5, -12),
                        "grouped": (5, 7),
                        "split": (5, 7),
                        "mixed": (5, -12),
                    }[r["mode"]],
                    textcoords="offset points",
                    fontsize=8,
                )
            ax.set(
                title=f"Prompt bin {p}, concurrency {n}",
                xlabel="Long TTFT (ms)",
                ylabel=f"Mean cohort ITL p{tail} (ms)",
            )
            ax.grid(alpha=0.3)
            ax.margins(x=0.25, y=0.2)
    fig.suptitle(
        "Warm C1024 confirmation — 10 paired cohorts per policy, three process starts\nMeans of cohort statistics, not pooled-token quantiles; Mixed C128 remains in the full table"
    )
    fig.savefig(target / "warm_confirmation_tradeoff.png", dpi=160)
    plt.close(fig)
    points = [r for r in rows if r["run"] == "static_grid_v1"]
    if points:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
        for mode in colors:
            series = [
                r
                for r in points
                if r["P"] == "8192" and r["N"] == "8" and r["mode"] == mode
            ]
            series.sort(key=lambda r: int(r["C"]))
            for ax, field, label in zip(
                axes,
                ["long_TTFT_s", "ITL_p99_s", "peak_HBM_bytes"],
                ["TTFT(s)", "ITL p99(s)", "Peak allocated (GiB)"],
            ):
                scale = 2**30 if field == "peak_HBM_bytes" else 1
                ax.plot(
                    [int(r["C"]) for r in series],
                    [float(r[field]) / scale for r in series],
                    "o-",
                    label=mode,
                    color=colors[mode],
                )
                ax.set(xlabel="Chunk tokens", ylabel=label)
                ax.grid(alpha=0.3)
        axes[0].legend()
        fig.suptitle("Triton live grid, 8K / concurrency8")
        fig.savefig(target / "grid_metrics.png", dpi=150)
        plt.close(fig)
    trace_rows = []
    import json

    for path in sorted((ROOT / "raw").glob("*/*.json")):
        d = json.loads(path.read_text())
        if not d.get("heavy"):
            continue
        gpu = d.get("gpu_quantums", [])
        if not gpu:
            continue
        fig, ax = plt.subplots(figsize=(13, 3), layout="constrained")
        start = min(g["gpu_start_ms"] for g in gpu)
        for g in gpu:
            if "quantum" in g:
                continue
            long = any(r.startswith("long_") for r in g["ids"])
            y = 1 if long else 0
            ax.broken_barh(
                [(g["gpu_start_ms"] - start, g["gpu_end_ms"] - g["gpu_start_ms"])],
                (y - 0.35, 0.7),
                facecolors="#d37b30" if long else "#3760a0",
                alpha=0.8,
            )
            trace_rows.append(dict(file=str(path.relative_to(ROOT)), **g))
        ax.set(
            yticks=[0, 1],
            yticklabels=["background", "long/mixed"],
            xlabel="Same-GPU CUDA-event timeline (ms)",
            title=path.stem + " — executed model calls, staging excluded from bars",
        )
        fig.savefig(
            target / (path.parent.name + "_" + path.stem + "_timeline.png"), dpi=140
        )
        plt.close(fig)
    if trace_rows:
        with (ROOT / "gpu_timeline.jsonl").open("w") as f:
            for r in trace_rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
