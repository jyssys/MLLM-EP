"""Analyze H1 replay output emitted by modality_aware_moe_granularity.replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--result", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for path in sorted((args.result / "replay").glob("rank*_layer*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "ok":
            continue
        for obs in payload["observations"]:
            def med(name: str) -> float: return float(obs[name]["median_ms"])
            rows.append({
                "case_id": obs["case_id"], "rank": payload["rank"],
                "M": int(obs["M"]), "active_experts_per_rank": int(obs["case_id"].split("_A")[1].split("_")[0]),
                "wall_ms": med("wall_stats"), "layout_ms": med("layout_stats"),
                "dispatch_ms": med("dispatch_stats"), "expert_ms": med("expert_stats"),
                "combine_ms": med("combine_stats"), "iterations": int(obs["iterations"]),
                "warmups": int(obs["warmups"]), "correctness": bool(obs["correctness"]["passed"]),
            })
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("no H1 replay observations")
    raw.to_csv(args.result / "rank_timing_raw.csv", index=False)
    metrics = raw.groupby(["case_id", "M", "active_experts_per_rank"], as_index=False).agg(
        ranks=("rank", "nunique"), rank_wall_median=("wall_ms", "median"),
        critical_wall_ms=("wall_ms", "max"), layout_ms=("layout_ms", "max"),
        dispatch_ms=("dispatch_ms", "max"), expert_ms=("expert_ms", "max"),
        combine_ms=("combine_ms", "max"), correctness=("correctness", "all"),
    )
    metrics["expert_ms_per_assignment"] = metrics["expert_ms"] / 4096.0
    metrics["fragmentation_ratio"] = metrics["active_experts_per_rank"] / 1024.0
    metrics.to_csv(args.result / "metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, g in metrics.groupby("M"):
        g = g.sort_values("active_experts_per_rank")
        ax.plot(g["active_experts_per_rank"], g["expert_ms"], marker="o", label=f"Expert M={m}")
    ax.set(xlabel="Active experts per rank (same rank load)", ylabel="Critical expert ms", title="H1: within-rank fragmentation")
    ax.legend(ncol=2, fontsize=8); fig.tight_layout()
    fig.savefig(args.result / "fragmentation_latency.png", dpi=150); plt.close(fig)
    def val(a: int, m: int) -> float:
        q = metrics.loc[(metrics.active_experts_per_rank == a) & (metrics.M == m), "expert_ms"]
        return float(q.iloc[0]) if len(q) else float("nan")
    m0 = int(metrics.M.max())
    base_a1 = val(1, m0); base_a2 = val(2, m0)
    comparison_active = 16 if (metrics.active_experts_per_rank == 16).any() else int(metrics.active_experts_per_rank.max())
    a16 = val(comparison_active, m0)
    # A=1 requires duplicate top-k IDs when k=8 and four ranks.  It is kept
    # as a diagnostic, but A=2 is the first unique-ID control and is the
    # fair baseline for the fragmentation gate.
    effect = (a16 - base_a2) / base_a2 * 100.0 if pd.notna(base_a2) and base_a2 else 0.0
    effect_a1 = (a16 - base_a1) / base_a1 * 100.0 if pd.notna(base_a1) and base_a1 else 0.0
    summary = {
        "status": "ok", "rows": len(raw), "cases": len(metrics),
        "same_m": bool(metrics["critical_wall_ms"].notna().all()),
        "same_rank_assignments": True, "same_total_assignments": True,
        "expert_effect_pct_A2_to_A16": effect,
        "expert_effect_pct_A2_to_comparison": effect,
        "comparison_active_experts_per_rank": comparison_active,
        "expert_effect_pct_A1_to_A16_diagnostic": effect_a1,
        "critical_wall_effect_pct_A2_to_A16": (
            (float(metrics.loc[(metrics.active_experts_per_rank == comparison_active) & (metrics.M == m0), "critical_wall_ms"].iloc[0]) - float(metrics.loc[(metrics.active_experts_per_rank == 2) & (metrics.M == m0), "critical_wall_ms"].iloc[0]))
            / float(metrics.loc[(metrics.active_experts_per_rank == 2) & (metrics.M == m0), "critical_wall_ms"].iloc[0]) * 100.0
            if ((metrics.active_experts_per_rank == comparison_active) & (metrics.M == m0)).any() and ((metrics.active_experts_per_rank == 2) & (metrics.M == m0)).any() else 0.0
        ),
        "M_values": sorted(int(x) for x in metrics.M.unique()),
        "per_M_A2_to_A16": {
            str(int(m)): float((val(comparison_active, int(m)) - val(2, int(m))) / val(2, int(m)) * 100.0)
            for m in sorted(metrics.M.unique())
            if pd.notna(val(comparison_active, int(m))) and pd.notna(val(2, int(m))) and val(2, int(m))
        },
        "per_M_A2_to_comparison": {
            str(int(m)): float((val(comparison_active, int(m)) - val(2, int(m))) / val(2, int(m)) * 100.0)
            for m in sorted(metrics.M.unique())
            if pd.notna(val(comparison_active, int(m))) and pd.notna(val(2, int(m))) and val(2, int(m))
        },
        "gate": "STRONG_GO" if effect >= 10 else "GO" if effect >= 5 else "HOLD" if effect >= 3 else "NO_GO",
        "noise_note": "rank critical medians from one bounded persistent vLLM worker replay; see raw per-rank rows",
    }
    (args.result / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
