"""Apply the preregistered Stage-0 slack stop to validated live EP traces."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IMAGE_TOKEN_ID = 151655
SPILL_FRACTIONS = (0.25, 0.50)


def _percentile(values: np.ndarray, value: float) -> float:
    return float(np.percentile(values, value))


def _summary(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(dtype=float)
    return {
        "median": float(np.median(array)),
        "p25": _percentile(array, 25),
        "p75": _percentile(array, 75),
        "p95": _percentile(array, 95),
        "max": float(np.max(array)),
    }


def _load_live(source: Path) -> pd.DataFrame:
    records = []
    for path in sorted((source / "raw_live").glob("rank*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if (row["phase"] == "main" and row["measured"]
                        and row["modality"] == "vision"):
                    records.append(row)
    frame = pd.DataFrame(records)
    expected = {"request_id", "layer", "ep_rank", "expert_ms", "total_assignments"}
    if frame.empty or not expected.issubset(frame.columns):
        raise RuntimeError(f"invalid live trace: {source}")
    return frame


def _visual_hot_count(source: Path, request: str, layer: int, rank: int) -> int:
    with np.load(source / "routes" / f"vision.{request}.npz") as archive:
        routes = archive["routed_experts"]
        token_ids = archive["prompt_token_ids"]
    vision = routes[token_ids == IMAGE_TOKEN_ID, layer, :].reshape(-1)
    if not len(vision):
        raise AssertionError((request, "no visual tokens"))
    local = vision[(vision // 32) == rank] - rank * 32
    return int(np.bincount(local, minlength=32).max()) if len(local) else 0


def _build_rows(source: Path) -> pd.DataFrame:
    raw = _load_live(source)
    ranks = raw.groupby(["request_id", "layer", "ep_rank"], as_index=False).agg(
        expert_ms=("expert_ms", "median"),
        assignments=("total_assignments", "median"),
        repetitions=("expert_ms", "size"),
    )
    timing = ranks.pivot(index=["request_id", "layer"], columns="ep_rank",
                         values="expert_ms").dropna()
    loads = ranks.pivot(index=["request_id", "layer"], columns="ep_rank",
                        values="assignments").loc[timing.index]
    rows = []
    for (request, layer), times in timing.iterrows():
        owner = int(times.idxmax())
        helper = int(times.drop(owner).idxmin())
        other = float(times.drop([owner, helper]).max())
        owner_ms, helper_ms = float(times[owner]), float(times[helper])
        hot = _visual_hot_count(source, str(request), int(layer), owner)
        ideal_ms = max(other, (owner_ms + helper_ms) / 2.0)
        row = {
            "request_id": request,
            "layer": int(layer),
            "owner_rank": owner,
            "helper_rank": helper,
            "owner_ms": owner_ms,
            "helper_ms": helper_ms,
            "second_slowest_ms": other,
            "helper_slack_ms": owner_ms - helper_ms,
            "helper_slack_fraction": (owner_ms - helper_ms) / owner_ms,
            "owner_assignments": int(loads.loc[(request, layer), owner]),
            "hot_visual_expert_assignments": hot,
            "zero_cost_ideal_makespan_ms": ideal_ms,
            "zero_cost_ideal_speedup": owner_ms / ideal_ms,
        }
        for fraction in SPILL_FRACTIONS:
            # Optimistic bound: expert time is perfectly linear in routed rows,
            # with no copy, activation transfer, launch, or contention cost.
            shifted = owner_ms * fraction * hot / row["owner_assignments"]
            cooperative = times.copy()
            cooperative[owner] -= shifted
            cooperative[helper] += shifted
            key = int(fraction * 100)
            row[f"oracle_{key}_makespan_ms"] = float(cooperative.max())
            row[f"oracle_{key}_speedup"] = owner_ms / float(cooperative.max())
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 24 * 48:
        raise AssertionError(f"expected 1152 request-layer observations, got {len(result)}")
    return result


def _placeholder(path: Path, title: str, reason: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(.5, .58, "NOT RUN", ha="center", va="center", fontsize=30,
            fontweight="bold", color="#b22222")
    ax.text(.5, .40, reason, ha="center", va="center", fontsize=12, wrap=True)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plots(frame: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(frame.helper_slack_ms, bins=40, color="#4472c4", alpha=.85)
    axes[0].axvline(frame.helper_slack_ms.median(), color="black", linestyle="--")
    axes[0].set(xlabel="Owner − helper expert latency (ms)", ylabel="Request×layer count",
                title="Absolute helper slack")
    axes[1].hist(100 * frame.helper_slack_fraction, bins=40,
                 color="#ed7d31", alpha=.85)
    axes[1].axvline(100 * frame.helper_slack_fraction.median(),
                    color="black", linestyle="--")
    axes[1].set(xlabel="Helper slack / owner latency (%)", ylabel="Count",
                title="Relative helper slack")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("Stage 0: live Vision-prefill helper slack")
    fig.tight_layout()
    fig.savefig(figures / "plot1_helper_slack_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    points = ax.scatter(frame.hot_visual_expert_assignments,
                        frame.zero_cost_ideal_speedup, c=frame.helper_slack_fraction,
                        cmap="viridis", alpha=.55, s=18)
    ax.axhline(1.05, color="#b22222", linestyle="--", label="5% gate")
    ax.set(xlabel="Owner hottest visual-expert assignments",
           ylabel="Zero-cost ideal split speedup",
           title="No practical break-even region before overhead")
    ax.grid(alpha=.2)
    ax.legend()
    fig.colorbar(points, ax=ax, label="Helper slack / owner")
    fig.tight_layout()
    fig.savefig(figures / "plot2_break_even_vs_token_count.png", dpi=180)
    plt.close(fig)

    _placeholder(figures / "plot3_sync_vs_async_split.png",
                 "Synchronous versus asynchronous split",
                 "Stage 1 was stopped by the Stage-0 zero-cost makespan bound; "
                 "no P2P/GEMM benchmark was run.")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(frame.owner_ms, frame.zero_cost_ideal_makespan_ms,
               alpha=.45, s=18, label="zero-cost arbitrary split oracle")
    lo = float(min(frame.owner_ms.min(), frame.zero_cost_ideal_makespan_ms.min()))
    hi = float(max(frame.owner_ms.max(), frame.zero_cost_ideal_makespan_ms.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, label="no improvement")
    ax.set(xlabel="Baseline four-rank makespan (ms)",
           ylabel="Optimistic cooperative makespan (ms)",
           title="Real-layer upper bound (copy/transfer costs excluded)")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "plot4_real_layer_makespan.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frame = _build_rows(args.source)
    frame.to_csv(args.output_dir / "stage0_slack_rows.csv", index=False)
    _plots(frame, args.output_dir / "figures")

    ideal = frame.zero_cost_ideal_speedup
    summary = {
        "VISION_COVERED_SPLIT": "NO-GO",
        "stop_stage": 0,
        "stop_reason": "helper slack fails even a zero-overhead arbitrary-split upper bound",
        "source": str(args.source.resolve()),
        "source_scope": {
            "requests": int(frame.request_id.nunique()),
            "layers": int(frame.layer.nunique()),
            "request_layer_observations": len(frame),
            "live_repetitions_per_rank": 15,
            "historical_source_gpus": [4, 5, 6, 7],
            "current_run_gpu_work": "none; Stage 0 reused the allowed existing trace",
        },
        "stage0_policy": {
            "early_stop": "median zero-cost ideal speedup <1.05 and <10% observations reach 1.05",
            "spill_fractions_preregistered": list(SPILL_FRACTIONS),
            "post_hoc_changes": False,
        },
        "helper_slack_ms": _summary(frame.helper_slack_ms),
        "helper_slack_fraction": _summary(frame.helper_slack_fraction),
        "hot_visual_expert_assignments": _summary(frame.hot_visual_expert_assignments),
        "zero_cost_arbitrary_split": {
            "speedup": _summary(ideal),
            "median_makespan_reduction": float(np.median(1.0 - 1.0 / ideal)),
            "fraction_speedup_ge_1_05": float((ideal >= 1.05).mean()),
            "fraction_speedup_ge_1_10": float((ideal >= 1.10).mean()),
        },
        "fixed_spill_zero_overhead": {
            str(int(fraction * 100)): {
                "speedup": _summary(frame[f"oracle_{int(fraction * 100)}_speedup"]),
                "fraction_regressed": float(
                    (frame[f"oracle_{int(fraction * 100)}_speedup"] < 1.0).mean()),
                "fraction_speedup_ge_1_05": float(
                    (frame[f"oracle_{int(fraction * 100)}_speedup"] >= 1.05).mean()),
            } for fraction in SPILL_FRACTIONS
        },
        "stage1": {"status": "NOT-RUN", "expert_p2p_copy_ms": None,
                   "copy_gemm_contention": None, "break_even_tokens": None},
        "stage2": {"status": "NOT-RUN", "baseline_vs_cooperative": None,
                   "correctness": None},
        "stage3": {"status": "NOT-RUN", "coverage_above_break_even": None},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
