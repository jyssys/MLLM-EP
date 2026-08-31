"""Aggregate fair chunk decomposition GPU replay and make figures."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULT = Path(os.environ.get("FAIR_ANALYSIS_RESULT", str(ROOT / "poc_flashvep/deepep_revalidation/results/fair_chunk_oracle_decomposition_20260831_230000")))
ANALYSIS = RESULT / "gpu_analysis"
FIG = RESULT / "figures"
ANALYSIS.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
STRATEGIES = ("fixed", "balanced", "same_count", "strict", "relaxed")
BUDGETS = (128, 256, 512, 1024)


def load() -> pd.DataFrame:
    rows = []
    for path in sorted((RESULT / "replay").glob("rank*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "ok":
            raise RuntimeError(path)
        for item in payload["observations"]:
            row = {k: item[k] for k in ("request_id", "source", "category", "budget", "layer", "rank", "strategy", "total_tokens", "vision_tokens", "chunks", "boundaries", "chunk_sizes")}
            for stage in ("wall", "expert", "dispatch", "combine"):
                stats = item[f"{stage}_stats"]
                row[f"{stage}_median_ms"] = stats["median_ms"]
                row[f"{stage}_p25_ms"] = stats["p25_ms"]
                row[f"{stage}_p75_ms"] = stats["p75_ms"]
                row[f"{stage}_cv"] = stats["cv"]
            row["correctness_passed"] = bool(item["correctness"].get("passed", False))
            row["route_identity"] = bool(item.get("route_identity"))
            row["token_partition_identity"] = bool(item.get("token_partition_identity"))
            sizes = np.asarray(item["chunk_sizes"], dtype=float)
            row["min_chunk"] = float(sizes.min())
            row["max_chunk"] = float(sizes.max())
            row["chunk_size_cv"] = float(sizes.std() / max(sizes.mean(), 1e-12))
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["request_id", "source", "category", "budget", "layer", "strategy", "total_tokens", "vision_tokens", "chunks", "min_chunk", "max_chunk", "chunk_size_cv"]
    rows = []
    for key, group in raw.groupby(keys, sort=False):
        row = dict(zip(keys, key))
        for stage in ("wall", "expert", "dispatch", "combine"):
            row[f"rank_max_{stage}_median_ms"] = float(group[f"{stage}_median_ms"].max())
            row[f"rank_mean_{stage}_median_ms"] = float(group[f"{stage}_median_ms"].mean())
            row[f"rank_max_{stage}_p25_ms"] = float(group[f"{stage}_p25_ms"].max())
            row[f"rank_max_{stage}_p75_ms"] = float(group[f"{stage}_p75_ms"].max())
            row[f"rank_max_{stage}_cv"] = float(group[f"{stage}_cv"].max())
        row["correctness_all"] = bool(group.correctness_passed.all())
        row["route_identity_all"] = bool(group.route_identity.all() and group.token_partition_identity.all())
        row["boundaries"] = json.dumps(group.iloc[0]["boundaries"])
        row["chunk_sizes"] = json.dumps(group.iloc[0]["chunk_sizes"])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_reductions(obs: pd.DataFrame) -> pd.DataFrame:
    out = obs.copy()
    key = ["request_id", "source", "budget", "layer"]
    for stage in ("expert", "wall", "dispatch", "combine"):
        piv = out.pivot_table(index=key, columns="strategy", values=f"rank_max_{stage}_median_ms")
        for strategy in STRATEGIES:
            col = f"{stage}_{strategy}_reduction_vs_fixed"
            out[col] = np.nan
            if strategy in piv:
                values = 1.0 - piv[strategy] / piv["fixed"]
                idx = pd.MultiIndex.from_frame(out[key])
                out[col] = idx.map(values).to_numpy()
    return out


def summary(obs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, budget, strategy), g in obs.groupby(["source", "budget", "strategy"], sort=False):
        row = {"source": source, "budget": int(budget), "strategy": strategy, "observations": len(g),
               "expert_median_ms": float(g.rank_max_expert_median_ms.median()),
               "wall_median_ms": float(g.rank_max_wall_median_ms.median()),
               "dispatch_median_ms": float(g.rank_max_dispatch_median_ms.median()),
               "combine_median_ms": float(g.rank_max_combine_median_ms.median()),
               "median_chunks": float(g.chunks.median()), "median_min_chunk": float(g.min_chunk.median()),
               "median_max_chunk": float(g.max_chunk.median()), "median_chunk_cv": float(g.chunk_size_cv.median()),
               "correctness_all": bool(g.correctness_all.all()), "route_identity_all": bool(g.route_identity_all.all())}
        for stage in ("expert", "wall", "dispatch", "combine"):
            values = g[f"{stage}_{strategy}_reduction_vs_fixed"].dropna()
            row[f"{stage}_reduction_median"] = float(values.median()) if len(values) else 0.0
            row[f"{stage}_reduction_p25"] = float(values.quantile(.25)) if len(values) else 0.0
            row[f"{stage}_reduction_p75"] = float(values.quantile(.75)) if len(values) else 0.0
            row[f"{stage}_positive_fraction"] = float((values > 0).mean()) if len(values) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def decomposition(obs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, budget), g in obs.groupby(["source", "budget"], sort=False):
        piv = g.pivot_table(index=["request_id", "layer"], columns="strategy", values="rank_max_expert_median_ms")
        for a in STRATEGIES:
            if a not in piv: raise RuntimeError(a)
        def red(a: str, b: str) -> np.ndarray:
            return (1.0 - piv[b].to_numpy() / np.maximum(piv[a].to_numpy(), 1e-12))
        rows.append({"source": source, "budget": int(budget),
                     "tail_balancing": float(np.median(red("fixed", "balanced"))),
                     "routing_only": float(np.median(red("balanced", "same_count"))),
                     "chunk_count_flexibility": float(np.median(red("same_count", "strict"))),
                     "relaxed_gt_budget": float(np.median(red("strict", "relaxed"))),
                     "total_fixed_to_relaxed": float(np.median(red("fixed", "relaxed"))),
                     "routing_only_ge_10_fraction": float(np.mean(red("balanced", "same_count") >= .10)),
                     "routing_only_positive_fraction": float(np.mean(red("balanced", "same_count") > 0))})
    return pd.DataFrame(rows)


def figures(obs: pd.DataFrame) -> None:
    # Strategy reductions relative to fixed, split by source and budget.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, source, title in zip(axes, ("short", "long"), ("Short representative routes", "Long multi-image routes")):
        x = np.arange(len(BUDGETS)); width = .16
        for j, strategy in enumerate(STRATEGIES[1:]):
            vals = []
            for budget in BUDGETS:
                g = obs[(obs.source == source) & (obs.budget == budget) & (obs.strategy == strategy)]
                vals.append(100 * (g["expert_" + strategy + "_reduction_vs_fixed"].median() if len(g) else np.nan))
            ax.bar(x + (j - 1.5) * width, vals, width, label=strategy)
        ax.axhline(0, color="k", lw=.7); ax.set_xticks(x, [str(b) for b in BUDGETS]); ax.set_title(title); ax.set_xlabel("Budget"); ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Max-rank expert latency reduction vs fixed (%)"); axes[0].legend(fontsize=8); fig.suptitle("Fair GPU decomposition: strategy effects"); fig.tight_layout(); fig.savefig(FIG / "plot1_gpu_decomposition.png", dpi=180); plt.close(fig)

    # Chunk size fairness/relaxation visibility.
    fig, ax = plt.subplots(figsize=(10, 4.8))
    data = []
    labels = []
    for budget in BUDGETS:
        for strategy in STRATEGIES:
            g = obs[(obs.budget == budget) & (obs.strategy == strategy)]
            data.append(g.chunk_size_cv.to_numpy()); labels.append(f"{budget}\n{strategy}")
    ax.boxplot(data, tick_labels=labels, showfliers=False); ax.set_ylabel("Chunk-size coefficient of variation"); ax.set_title("Chunk-size balance and relaxed-size effect"); ax.tick_params(axis="x", labelsize=7); fig.tight_layout(); fig.savefig(FIG / "plot2_chunk_size_fairness.png", dpi=180); plt.close(fig)

    dec = decomposition(obs)
    fig, ax = plt.subplots(figsize=(9, 4.8)); x = np.arange(len(BUDGETS)); width=.19
    for j, col in enumerate(("tail_balancing", "routing_only", "chunk_count_flexibility", "relaxed_gt_budget")):
        vals = [100 * float(dec[dec.budget == b][col].iloc[0]) for b in BUDGETS]
        ax.bar(x + (j - 1.5) * width, vals, width, label=col)
    ax.axhline(0, color="k", lw=.7); ax.set_xticks(x, [str(b) for b in BUDGETS]); ax.set_xlabel("Budget"); ax.set_ylabel("Paired max-rank expert reduction (%)"); ax.set_title("GPU component decomposition"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "plot3_component_breakdown.png", dpi=180); plt.close(fig)


def main() -> None:
    raw = load(); raw.to_csv(ANALYSIS / "per_rank.csv", index=False)
    obs = paired_reductions(aggregate(raw)); obs.to_csv(ANALYSIS / "per_observation.csv", index=False)
    sm = summary(obs); sm.to_csv(ANALYSIS / "strategy_summary.csv", index=False)
    dec = decomposition(obs); dec.to_csv(ANALYSIS / "gpu_decomposition.csv", index=False)
    figures(obs)
    piv = sm[sm.source == "short"].pivot_table(index="budget", columns="strategy", values="expert_median_ms")
    red = dec[dec.source == "short"].set_index("budget")
    gate_vals = red.loc[list(BUDGETS[:2]), "routing_only"].to_numpy()
    gate = "STRONG_GO" if np.all(gate_vals >= .10) else "GO" if np.sum(gate_vals >= .10) >= 1 and np.all(gate_vals > 0) else "HOLD" if np.all(gate_vals >= .05) else "NO-GO"
    result = {"status": "ok", "stage1_gpu_gate": gate, "routing_only_expert_reduction_short_128_256": gate_vals.tolist(), "correctness_all": bool(obs.correctness_all.all()), "route_identity_all": bool(obs.route_identity_all.all()), "strategies": list(STRATEGIES), "budgets": list(BUDGETS)}
    (RESULT / "gpu_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(sm.to_string(index=False)); print("\nDECOMPOSITION\n", dec.to_string(index=False)); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
