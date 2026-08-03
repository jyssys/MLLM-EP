"""Summarize CUDA-event FusedMoE timings from vLLM workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NUM_LAYERS = 48
EP_DEGREE = 8


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _reduction_pct(asis: float, tobe: float) -> float:
    return float((1.0 - tobe / asis) * 100.0) if asis else 0.0


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "sum": 0.0,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
        "sum": float(values.sum()),
    }


def _load_jsonl(path: Path, *, drop_first_per_layer_rank: int) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
            if row.get("elapsed_ms") is None:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"no valid MoE timing rows found in {path}")

    df = pd.DataFrame(rows)
    df = df[(df["layer"] >= 0) & (df["ep_rank"] >= 0)].copy()
    df["elapsed_ms"] = df["elapsed_ms"].astype(float)
    df["call_index"] = df["call_index"].astype(int)
    if drop_first_per_layer_rank:
        df = df[df["call_index"] >= drop_first_per_layer_rank].copy()
    df["profile_call_index"] = df["call_index"] - int(drop_first_per_layer_rank)
    return df


def _summarize_variant(df: pd.DataFrame) -> dict[str, Any]:
    per_rank_call = (
        df.groupby(["layer", "profile_call_index", "ep_rank"], as_index=False)[
            "elapsed_ms"
        ]
        .sum()
        .sort_values(["layer", "profile_call_index", "ep_rank"])
    )

    critical = per_rank_call.groupby(["layer", "profile_call_index"])[
        "elapsed_ms"
    ].max()
    layer_critical_total = critical.groupby(level=0).sum()
    layer_critical_mean = critical.groupby(level=0).mean()

    imbalance_values = []
    for (_layer, _call), group in per_rank_call.groupby(["layer", "profile_call_index"]):
        values = group["elapsed_ms"].to_numpy(dtype=float)
        mean = values.mean()
        if mean > 0:
            imbalance_values.append(float(values.max() / mean))

    rank_total = (
        df.groupby("ep_rank")["elapsed_ms"].sum().reindex(range(EP_DEGREE), fill_value=0.0)
    )
    layer_rank_total = (
        df.groupby(["layer", "ep_rank"])["elapsed_ms"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(index=range(NUM_LAYERS), columns=range(EP_DEGREE), fill_value=0.0)
    )
    profile_calls = (
        df.groupby(["layer", "ep_rank"])["profile_call_index"].nunique().to_numpy()
    )

    rank_mean = float(rank_total.mean()) if len(rank_total) else 0.0
    rank_max_over_mean = float(rank_total.max() / rank_mean) if rank_mean else 0.0

    return {
        "num_records": int(len(df)),
        "layers": sorted(int(x) for x in df["layer"].unique()),
        "ranks": sorted(int(x) for x in df["ep_rank"].unique()),
        "profile_calls_per_layer_rank": _stats(profile_calls),
        "gpu_work_ms": _stats(df["elapsed_ms"].to_numpy(dtype=float)),
        "gpu_work_total_ms": float(df["elapsed_ms"].sum()),
        "critical_path_moe_ms": {
            "total": float(critical.sum()),
            "per_layer_call": _stats(critical.to_numpy(dtype=float)),
            "per_layer_total": [float(x) for x in layer_critical_total.tolist()],
            "per_layer_mean": [float(x) for x in layer_critical_mean.tolist()],
        },
        "timing_imbalance_max_over_mean": _stats(np.asarray(imbalance_values)),
        "rank_total_ms": [float(x) for x in rank_total.tolist()],
        "rank_total_max_over_mean": rank_max_over_mean,
        "layer_rank_total_ms": layer_rank_total.to_numpy(dtype=float).tolist(),
    }


def _plot_summary(asis: dict[str, Any], tobe: dict[str, Any], path: Path) -> None:
    labels = [
        "Critical total\\nMoE ms",
        "Mean layer-call\\ncritical ms",
        "P95 layer-call\\ncritical ms",
        "P95 timing\\nimbalance",
    ]
    asis_values = [
        asis["critical_path_moe_ms"]["total"],
        asis["critical_path_moe_ms"]["per_layer_call"]["mean"],
        asis["critical_path_moe_ms"]["per_layer_call"]["p95"],
        asis["timing_imbalance_max_over_mean"]["p95"],
    ]
    tobe_values = [
        tobe["critical_path_moe_ms"]["total"],
        tobe["critical_path_moe_ms"]["per_layer_call"]["mean"],
        tobe["critical_path_moe_ms"]["per_layer_call"]["p95"],
        tobe["timing_imbalance_max_over_mean"]["p95"],
    ]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, asis_values, width, label="As-Is linear", color="#4c78a8")
    ax.bar(x + width / 2, tobe_values, width, label="To-Be layer-wise", color="#f58518")
    for i, (a, t) in enumerate(zip(asis_values, tobe_values, strict=True)):
        ax.text(i, max(a, t) * 1.02, f"{_reduction_pct(a, t):.1f}% down", ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("CUDA event metric")
    ax.set_title("MoE-only CUDA Timing Summary")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_by_layer(asis: dict[str, Any], tobe: dict[str, Any], path: Path) -> None:
    layers = np.arange(NUM_LAYERS)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        layers,
        asis["critical_path_moe_ms"]["per_layer_mean"],
        label="As-Is linear",
        color="#4c78a8",
        linewidth=2,
    )
    ax.plot(
        layers,
        tobe["critical_path_moe_ms"]["per_layer_mean"],
        label="To-Be layer-wise",
        color="#f58518",
        linewidth=2,
    )
    ax.set_xlabel("MoE layer")
    ax.set_ylabel("Mean critical MoE time per call (ms)")
    ax.set_title("MoE-only Critical Path by Layer")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_rank_total(asis: dict[str, Any], tobe: dict[str, Any], path: Path) -> None:
    ranks = np.arange(EP_DEGREE)
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        ranks - width / 2,
        asis["rank_total_ms"],
        width,
        label="As-Is linear",
        color="#4c78a8",
    )
    ax.bar(
        ranks + width / 2,
        tobe["rank_total_ms"],
        width,
        label="To-Be layer-wise",
        color="#f58518",
    )
    ax.set_xlabel("EP rank")
    ax.set_ylabel("Total FusedMoE CUDA time (ms)")
    ax.set_title("MoE-only Time by Rank")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_layer_rank_heatmap(summary: dict[str, Any], title: str, path: Path) -> None:
    data = np.asarray(summary["layer_rank_total_ms"], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xlabel("EP rank")
    ax.set_ylabel("MoE layer")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Total FusedMoE CUDA time (ms)")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asis-jsonl", required=True)
    parser.add_argument("--tobe-jsonl", required=True)
    parser.add_argument("--output-dir", default="outputs/moe_cuda_timing")
    parser.add_argument("--drop-first-per-layer-rank", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    asis_df = _load_jsonl(
        Path(args.asis_jsonl),
        drop_first_per_layer_rank=args.drop_first_per_layer_rank,
    )
    tobe_df = _load_jsonl(
        Path(args.tobe_jsonl),
        drop_first_per_layer_rank=args.drop_first_per_layer_rank,
    )
    asis = _summarize_variant(asis_df)
    tobe = _summarize_variant(tobe_df)

    comparison = {
        "drop_first_per_layer_rank": args.drop_first_per_layer_rank,
        "asis": asis,
        "tobe": tobe,
        "delta": {
            "critical_path_total_reduction_pct": _reduction_pct(
                asis["critical_path_moe_ms"]["total"],
                tobe["critical_path_moe_ms"]["total"],
            ),
            "critical_path_mean_reduction_pct": _reduction_pct(
                asis["critical_path_moe_ms"]["per_layer_call"]["mean"],
                tobe["critical_path_moe_ms"]["per_layer_call"]["mean"],
            ),
            "critical_path_p95_reduction_pct": _reduction_pct(
                asis["critical_path_moe_ms"]["per_layer_call"]["p95"],
                tobe["critical_path_moe_ms"]["per_layer_call"]["p95"],
            ),
            "timing_imbalance_p95_reduction_pct": _reduction_pct(
                asis["timing_imbalance_max_over_mean"]["p95"],
                tobe["timing_imbalance_max_over_mean"]["p95"],
            ),
            "rank_total_max_over_mean_reduction_pct": _reduction_pct(
                asis["rank_total_max_over_mean"],
                tobe["rank_total_max_over_mean"],
            ),
        },
        "figures": {
            "summary": str(output_dir / "moe_cuda_summary.png"),
            "by_layer": str(output_dir / "moe_cuda_by_layer.png"),
            "rank_total": str(output_dir / "moe_cuda_rank_total.png"),
            "asis_heatmap": str(output_dir / "moe_cuda_asis_layer_rank.png"),
            "tobe_heatmap": str(output_dir / "moe_cuda_tobe_layer_rank.png"),
        },
    }

    _write_json(output_dir / "moe_cuda_timing_summary.json", comparison)
    _plot_summary(asis, tobe, output_dir / "moe_cuda_summary.png")
    _plot_by_layer(asis, tobe, output_dir / "moe_cuda_by_layer.png")
    _plot_rank_total(asis, tobe, output_dir / "moe_cuda_rank_total.png")
    _plot_layer_rank_heatmap(
        asis, "As-Is MoE-only CUDA Time by Layer/Rank", output_dir / "moe_cuda_asis_layer_rank.png"
    )
    _plot_layer_rank_heatmap(
        tobe, "To-Be MoE-only CUDA Time by Layer/Rank", output_dir / "moe_cuda_tobe_layer_rank.png"
    )

    print(json.dumps(comparison["delta"], indent=2))


if __name__ == "__main__":
    main()
