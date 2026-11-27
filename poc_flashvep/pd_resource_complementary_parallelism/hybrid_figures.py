"""Render the measured Direction D critical-path and conversion comparisons."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "tp2": "Uniform TP2",
    "cp2_contiguous": "Contiguous CP2",
    "cp2_zigzag": "Balanced CP2",
    "tp_attention_row_mlp": "TP attention + row MLP",
    "cp_attention_tp_mlp": "CP attention + TP MLP",
    "local_control": "Single-GPU control",
}
COLORS = ("#334155", "#a16207", "#2563eb", "#15803d", "#7c3aed", "#94a3b8")


def records(paths):
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else Path.open
        with opener(path, "rt") as stream:
            yield from (json.loads(line) for line in stream if line.strip())


def critical_rank(row):
    return max(row["ranks"], key=lambda item: median(rep["cuda_ms"] for rep in item["stage_events_eager"]))


def stage_medians(row):
    trials = critical_rank(row)["stage_events_eager"]
    stage_names = {name for trial in trials for name in trial["stages_cuda_ms"]}
    return {name: median(trial["stages_cuda_ms"].get(name, 0) for trial in trials) for name in stage_names}


def render(rows, out):
    by_shape = defaultdict(list)
    for row in rows:
        geom = row["geometry"]
        by_shape[(geom["layers"], geom["batch"], row["mode"])].append(row)
    for (layers, batch, mode), selected in by_shape.items():
        fig, ax = plt.subplots(figsize=(8.2, 4.7), layout="constrained")
        for (arm, label), color in zip(LABELS.items(), COLORS, strict=True):
            series = sorted((row for row in selected if row["arm"] == arm),
                            key=lambda row: row["geometry"]["input_length"])
            if not series:
                continue
            ax.plot([row["geometry"]["input_length"] for row in series],
                    [row["critical_cuda_ms_p50"] for row in series], marker="o", label=label, color=color)
        ax.set(xlabel="Input tokens per request", ylabel="Critical rank CUDA time (ms)",
               title=f"Qwen3-8B {layers} layer(s), batch {batch}, {mode.replace('_', ' ')}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.legend(fontsize="small", ncol=2)
        ax.grid(axis="y", alpha=.2)
        fig.savefig(out / f"d_latency_l{layers}_b{batch}_{mode}.png", dpi=150)
        plt.close(fig)

        if mode != "eager":
            continue
        largest = max(row["geometry"]["input_length"] for row in selected)
        snapshot = [row for row in selected if row["geometry"]["input_length"] == largest
                    and abs(row["event_instrumentation_overhead_pct"]) <= 5
                    and row["arm"] != "local_control"]
        if not snapshot:
            continue
        table = [(LABELS[row["arm"]], stage_medians(row), row) for row in snapshot]
        groups = {
            "Attention and projection": ("attention_qkv_rope", "attention_fa3", "attention_out_projection"),
            "MLP": ("mlp",),
            "NCCL communication": ("hidden_communication", "kv_communication", "final_output_communication"),
            "Tensor layout conversion": ("hidden_layout", "kv_layout"),
        }
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 5), layout="constrained")
        baseline = [0.0] * len(table)
        for (name, stages), color in zip(groups.items(), ("#2563eb", "#5b21b6", "#d97706", "#475569"), strict=True):
            amount = [sum(stage.get(key, 0) for key in stages) for _, stage, _ in table]
            axes[0].barh([label for label, _, _ in table], amount, left=baseline, color=color, label=name)
            baseline = [a + b for a, b in zip(baseline, amount, strict=True)]
        axes[0].legend(fontsize="small")
        axes[0].set(xlabel="Observed CUDA stage time (ms)", title="Critical rank stage medians")
        axes[1].barh([label for label, _, _ in table],
                     [max(row["physical_transmit_payload_bytes_per_rank"]) / 2**20 for _, _, row in table],
                     color="#2563eb")
        axes[1].set(xlabel="Max rank modeled TX payload (MiB)", title="NCCL payload by layout")
        fig.suptitle(f"{largest:,} tokens, {layers} layer(s), instrumentation overhead ≤ 5%")
        fig.savefig(out / f"d_layout_l{layers}_b{batch}_{mode}.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    render(list(records(args.raw)), args.out)


if __name__ == "__main__":
    main()
