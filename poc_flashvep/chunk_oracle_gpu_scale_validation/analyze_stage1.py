"""Aggregate bounded fixed-vs-route-oracle GPU replay and make figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "poc_flashvep/deepep_revalidation/results/chunk_oracle_gpu_scale_validation_20260831_221000"
OFFLINE = ROOT / "poc_flashvep/deepep_revalidation/results/spatial_chunked_prefill_20260831_200000/analysis"
FIG = RESULT / "figures"
ANALYSIS = RESULT / "analysis"
FIG.mkdir(parents=True, exist_ok=True)
ANALYSIS.mkdir(parents=True, exist_ok=True)


def load_gpu() -> pd.DataFrame:
    rows = []
    for path in sorted((RESULT / "replay").glob("rank*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "ok":
            raise RuntimeError(f"failed replay: {path}: {payload}")
        for item in payload["observations"]:
            row = {k: item[k] for k in ("request_id", "category", "pair_id", "budget", "layer", "rank", "strategy", "total_tokens", "vision_tokens", "chunks", "boundaries", "chunk_sizes")}
            for stage in ("wall", "expert", "dispatch", "combine"):
                row[f"{stage}_median_ms"] = item[f"{stage}_stats"]["median_ms"]
                row[f"{stage}_p25_ms"] = item[f"{stage}_stats"]["p25_ms"]
                row[f"{stage}_p75_ms"] = item[f"{stage}_stats"]["p75_ms"]
                row[f"{stage}_cv"] = item[f"{stage}_stats"]["cv"]
            row["correctness_passed"] = bool(item["correctness"].get("passed", False))
            row["route_identity"] = bool(item.get("route_identity"))
            row["token_partition_identity"] = bool(item.get("token_partition_identity"))
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate(gpu: pd.DataFrame) -> pd.DataFrame:
    keys = ["request_id", "category", "pair_id", "budget", "layer", "strategy", "total_tokens", "vision_tokens", "chunks"]
    out = []
    for key, g in gpu.groupby(keys, sort=False):
        row = dict(zip(keys, key))
        for stage in ("wall", "expert", "dispatch", "combine"):
            row[f"rank_max_{stage}_median_ms"] = float(g[f"{stage}_median_ms"].max())
            row[f"rank_mean_{stage}_median_ms"] = float(g[f"{stage}_median_ms"].mean())
            row[f"rank_max_{stage}_p25_ms"] = float(g[f"{stage}_p25_ms"].max())
            row[f"rank_max_{stage}_p75_ms"] = float(g[f"{stage}_p75_ms"].max())
            row[f"rank_max_{stage}_cv"] = float(g[f"{stage}_cv"].max())
        row["all_correctness"] = bool(g.correctness_passed.all())
        row["all_route_identity"] = bool(g.route_identity.all())
        row["all_partition_identity"] = bool(g.token_partition_identity.all())
        # Boundaries are identical across ranks; retain one for inspection.
        row["boundaries"] = json.dumps(g.iloc[0]["boundaries"])
        row["chunk_sizes"] = json.dumps(g.iloc[0]["chunk_sizes"])
        out.append(row)
    return pd.DataFrame(out)


def add_reductions(obs: pd.DataFrame) -> pd.DataFrame:
    keys = ["request_id", "budget", "layer"]
    fixed = obs[obs.strategy == "fixed"].set_index(keys)
    for stage in ("wall", "expert", "dispatch", "combine"):
        oracle = obs[obs.strategy == "oracle"].set_index(keys)
        obs[f"{stage}_reduction"] = np.nan
        common = fixed.index.intersection(oracle.index)
        vals = 1.0 - oracle.loc[common, f"rank_max_{stage}_median_ms"].to_numpy() / fixed.loc[common, f"rank_max_{stage}_median_ms"].to_numpy()
        lookup = pd.Series(vals, index=common)
        obs.loc[pd.MultiIndex.from_frame(obs[keys]).isin(common), f"{stage}_reduction"] = pd.MultiIndex.from_frame(obs.loc[pd.MultiIndex.from_frame(obs[keys]).isin(common), keys]).map(lookup).to_numpy()
    return obs


def offline_proxy() -> pd.DataFrame:
    p = OFFLINE / "chunk_layer_metrics.csv"
    # Keep only selected routes and the two measured layers/budgets.  This is
    # the same route-aware metric used to derive the previous offline claim.
    selected = {"coins", "cat", "logo", "coffee", "coffee_rocket", "model_card", "retina", "method"}
    df = pd.read_csv(p, usecols=["request_id", "budget", "strategy", "chunk", "layer", "scope", "effective_tiles", "padding_rows"])
    df = df[df.request_id.isin(selected) & df.budget.isin([128, 256]) & df.layer.isin([0, 12, 24, 36, 47])]
    x = df.groupby(["request_id", "budget", "layer", "strategy", "scope"], as_index=False).agg(tile_sum=("effective_tiles", "sum"), padding_sum=("padding_rows", "sum"))
    return x


def main() -> None:
    gpu = load_gpu()
    gpu.to_csv(ANALYSIS / "stage1_per_rank.csv", index=False)
    obs = add_reductions(aggregate(gpu))
    obs.to_csv(ANALYSIS / "stage1_per_observation.csv", index=False)
    # Pair-level summary, primary expert kernel metric.
    pair_rows = []
    for (budget, strategy), g in obs.groupby(["budget", "strategy"]):
        pair_rows.append({
            "budget": int(budget), "strategy": strategy, "observations": len(g),
            "expert_median_ms": float(g.rank_max_expert_median_ms.median()),
            "wall_median_ms": float(g.rank_max_wall_median_ms.median()),
            "dispatch_median_ms": float(g.rank_max_dispatch_median_ms.median()),
            "combine_median_ms": float(g.rank_max_combine_median_ms.median()),
            "expert_p25_ms": float(g.rank_max_expert_median_ms.quantile(.25)),
            "expert_p75_ms": float(g.rank_max_expert_median_ms.quantile(.75)),
        })
    pair = pd.DataFrame(pair_rows)
    pair.to_csv(ANALYSIS / "stage1_summary_by_strategy.csv", index=False)
    offline = offline_proxy()
    offline.to_csv(ANALYSIS / "stage1_offline_proxy_selected.csv", index=False)
    # Join vision/all proxy against GPU reduction for the same request/layer.
    proxy = offline.pivot_table(index=["request_id", "budget", "layer", "scope"], columns="strategy", values="tile_sum").reset_index()
    proxy["offline_reduction"] = 1 - proxy["oracle"] / proxy["fixed"]
    gpu_pair = obs[obs.strategy == "oracle"][["request_id", "budget", "layer", "expert_reduction", "wall_reduction"]].copy()
    merged = proxy.merge(gpu_pair, on=["request_id", "budget", "layer"], how="left")
    merged.to_csv(ANALYSIS / "stage1_offline_vs_gpu.csv", index=False)

    # Figure 1: paired max-rank expert latency.
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for budget, color in [(128, "#2563eb"), (256, "#dc2626")]:
        f = obs[(obs.budget == budget) & (obs.strategy == "fixed")].set_index(["request_id", "layer"])
        o = obs[(obs.budget == budget) & (obs.strategy == "oracle")].set_index(["request_id", "layer"])
        idx = f.index.intersection(o.index)
        ax.scatter(f.loc[idx, "rank_max_expert_median_ms"], o.loc[idx, "rank_max_expert_median_ms"], s=22, alpha=.7, label=f"budget {budget}", color=color)
    lim = ax.get_xlim(); ax.plot(lim, lim, "k--", lw=.8); ax.set_xlabel("Fixed max-rank expert median (ms)"); ax.set_ylabel("Route-oracle max-rank expert median (ms)"); ax.set_title("GPU expert latency: fixed vs exact route-oracle cuts"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "plot1_fixed_vs_oracle_gpu_latency.png", dpi=180); plt.close(fig)

    # Figure 2: distribution of oracle reduction.
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    vals = [obs[obs.budget == b].expert_reduction.dropna().to_numpy() for b in (128, 256)]
    ax.boxplot(vals, tick_labels=["128", "256"], showmeans=True); ax.axhline(0, color="k", lw=.8); ax.axhline(.10, color="#16a34a", ls="--", lw=.8, label="10% gate"); ax.set_ylabel("Expert latency reduction (1 − oracle/fixed)"); ax.set_xlabel("Chunk budget"); ax.set_title("Exact route-oracle GPU reduction"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "plot2_gpu_speedup_distribution.png", dpi=180); plt.close(fig)

    # Figure 3: previous offline proxy against GPU reduction (scope all and vision).
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for scope, marker, color in [("all", "o", "#7c3aed"), ("vision", "s", "#ea580c")]:
        z = merged[merged.scope == scope].dropna(subset=["offline_reduction", "expert_reduction"])
        ax.scatter(z.offline_reduction * 100, z.expert_reduction * 100, marker=marker, alpha=.65, label=scope, color=color)
    ax.axhline(0, color="k", lw=.8); ax.axvline(0, color="k", lw=.8); ax.set_xlabel("Offline tile proxy reduction (%)"); ax.set_ylabel("GPU expert latency reduction (%)"); ax.set_title("Offline proxy vs GPU replay (selected request/layers)"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "plot3_offline_proxy_vs_gpu.png", dpi=180); plt.close(fig)

    # Additional stage breakdown for report.
    breakdown = []
    for b in (128, 256):
        f = obs[(obs.budget == b) & (obs.strategy == "fixed")].set_index(["request_id", "layer"])
        o = obs[(obs.budget == b) & (obs.strategy == "oracle")].set_index(["request_id", "layer"])
        idx = f.index.intersection(o.index)
        row = {"budget": b}
        for stage in ("wall", "expert", "dispatch", "combine"):
            row[f"{stage}_reduction_median"] = float((1 - o.loc[idx, f"rank_max_{stage}_median_ms"] / f.loc[idx, f"rank_max_{stage}_median_ms"]).median())
        breakdown.append(row)
    pd.DataFrame(breakdown).to_csv(ANALYSIS / "stage1_breakdown.csv", index=False)

    summary = {
        "status": "ok",
        "run_id": RESULT.name.rsplit("_", 2)[-2] + "_" + RESULT.name.rsplit("_", 1)[-1],
        "requests": sorted(obs.request_id.unique().tolist()),
        "layers": [0, 12, 24, 36, 47], "budgets": [128, 256],
        "observations_per_strategy": int(len(obs) // 2),
        "correctness_all": bool(obs.all_correctness.all()),
        "route_identity_all": bool(obs.all_route_identity.all() and obs.all_partition_identity.all()),
        "by_budget": {},
    }
    for b in (128, 256):
        f = obs[(obs.budget == b) & (obs.strategy == "fixed")].set_index(["request_id", "layer"])
        o = obs[(obs.budget == b) & (obs.strategy == "oracle")].set_index(["request_id", "layer"])
        idx = f.index.intersection(o.index)
        summary["by_budget"][str(b)] = {
            "fixed_expert_median_ms": float(f.loc[idx, "rank_max_expert_median_ms"].median()),
            "oracle_expert_median_ms": float(o.loc[idx, "rank_max_expert_median_ms"].median()),
            "expert_reduction_median": float((1 - o.loc[idx, "rank_max_expert_median_ms"] / f.loc[idx, "rank_max_expert_median_ms"]).median()),
            "wall_reduction_median": float((1 - o.loc[idx, "rank_max_wall_median_ms"] / f.loc[idx, "rank_max_wall_median_ms"]).median()),
            "dispatch_reduction_median": float((1 - o.loc[idx, "rank_max_dispatch_median_ms"] / f.loc[idx, "rank_max_dispatch_median_ms"]).median()),
            "combine_reduction_median": float((1 - o.loc[idx, "rank_max_combine_median_ms"] / f.loc[idx, "rank_max_combine_median_ms"]).median()),
            "expert_reduction_positive_fraction": float((1 - o.loc[idx, "rank_max_expert_median_ms"] / f.loc[idx, "rank_max_expert_median_ms"] > 0).mean()),
            "expert_reduction_ge_5_fraction": float((1 - o.loc[idx, "rank_max_expert_median_ms"] / f.loc[idx, "rank_max_expert_median_ms"] >= .05).mean()),
        }
    reductions = [summary["by_budget"][str(b)]["expert_reduction_median"] for b in (128, 256)]
    summary["stage1_gate"] = "GO" if min(reductions) >= .10 else "HOLD" if min(reductions) >= .05 else "NO-GO"
    summary["stage2_status"] = "NOT_RUN" if summary["stage1_gate"] == "NO-GO" else "PENDING_BOUNDED_SCALE_RUN"
    (RESULT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
