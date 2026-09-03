"""Aggregate rank timings, paired modality curves, figures, and preregistered gate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _q(x: pd.Series, p: float) -> float:
    return float(x.quantile(p)) if len(x) else float("nan")


def _load(result: Path) -> pd.DataFrame:
    route = pd.read_csv(result / "route_statistics.csv")
    rows: list[dict] = []
    for path in sorted((result / "replay").glob("rank*_layer*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "ok":
            continue
        for obs in payload.get("observations", []):
            def val(name: str, stat: str = "median_ms") -> float:
                return float(obs[name][stat])
            rows.append({
                "case_id": obs["case_id"], "request_id": obs["request_id"],
                "category": obs["category"], "modality": obs["modality"],
                "layer": int(obs["layer"]), "M": int(obs["M"]),
                "rank": int(obs["rank"]), "token_count": int(obs["token_count"]),
                "wall_ms": val("wall_stats"), "wall_p25_ms": val("wall_stats", "p25_ms"),
                "wall_p95_ms": val("wall_stats", "p95_ms"),
                "layout_ms": val("layout_stats"),
                "dispatch_ms": val("dispatch_stats"),
                "expert_ms": val("expert_stats"),
                "combine_ms": val("combine_stats"),
                "warmups": int(obs["warmups"]), "iterations": int(obs["iterations"]),
                "output_shape": str(obs["correctness"].get("output_shape")),
                "route_identity": bool(obs["route_identity"]),
                "token_partition_identity": bool(obs["token_partition_identity"]),
            })
    if not rows:
        raise RuntimeError("no replay observations")
    raw = pd.DataFrame(rows)
    route = route.drop(columns=["source_route", "source_sha256"], errors="ignore")
    return raw.merge(route, on=["case_id", "request_id", "category", "modality", "layer", "M"], how="left")


def _aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["case_id", "request_id", "category", "modality", "layer", "M"]
    out_rows = []
    for key, group in raw.groupby(keys, sort=True):
        row = dict(zip(keys, key, strict=True))
        row["ranks"] = int(group["rank"].nunique())
        row["rank_median_wall_ms"] = float(group["wall_ms"].median())
        row["critical_wall_ms"] = float(group["wall_ms"].max())
        row["rank_min_wall_ms"] = float(group["wall_ms"].min())
        row["rank_spread_wall_ms"] = row["critical_wall_ms"] - row["rank_min_wall_ms"]
        row["critical_layout_ms"] = float(group["layout_ms"].max())
        for metric in ("dispatch_ms", "expert_ms", "combine_ms"):
            row[f"{metric}_mean_rank_ms"] = float(group[metric].mean())
            row[f"{metric}_critical_ms"] = float(group[metric].max())
        row["total_ms_per_token"] = row["critical_wall_ms"] / int(key[-1])
        row["dispatch_ms_per_token"] = row["dispatch_ms_critical_ms"] / int(key[-1])
        row["expert_ms_per_token"] = row["expert_ms_critical_ms"] / int(key[-1])
        row["combine_ms_per_token"] = row["combine_ms_critical_ms"] / int(key[-1])
        for col in raw.columns:
            if col not in keys and col not in {"rank", "wall_ms", "wall_p25_ms", "wall_p95_ms", "layout_ms", "dispatch_ms", "expert_ms", "combine_ms", "warmups", "iterations", "output_shape", "route_identity", "token_partition_identity"} and col not in row:
                row[col] = group[col].iloc[0]
        row["correctness"] = bool(group["route_identity"].all() and group["token_partition_identity"].all())
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _make_figures(result: Path, agg: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"vision": "#d95f02", "text": "#1b9e77"}
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for modality, g in agg.groupby("modality"):
        c = g.groupby("M")["total_ms_per_token"].median()
        ax.plot(c.index, c.values, marker="o", label=modality, color=colors[modality])
    ax.set(xlabel="M tokens", ylabel="Critical MoE ms/token", title="Latency per token vs execution granularity")
    ax.set_xticks(sorted(agg.M.unique())); ax.legend(); fig.tight_layout()
    fig.savefig(result / "latency_per_token_vs_granularity.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=True)
    for modality, g in agg.groupby("modality"):
        c = colors[modality]
        for ax, metric, title in zip(axes, ("dispatch_ms_per_token", "expert_ms_per_token", "combine_ms_per_token"), ("Dispatch", "Expert", "Combine"), strict=True):
            s = g.groupby("M")[metric].median()
            ax.plot(s.index, s.values, marker="o", label=modality, color=c)
            ax.set_title(title); ax.set_xlabel("M"); ax.set_ylabel("ms/token")
    axes[0].legend(); fig.suptitle("Phase breakdown vs granularity (critical rank)"); fig.tight_layout()
    fig.savefig(result / "phase_breakdown_vs_granularity.png", dpi=150); plt.close(fig)

    # Each dot is a real request/layer case; local expert M is route-derived.
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for modality, g in agg.groupby("modality"):
        ax.scatter(g["M"], g["p50_active_expert_m"], s=24, alpha=.65, label=modality, color=colors[modality])
    ax.set(xlabel="Global M tokens", ylabel="p50 assignments / active expert", title="Local expert-M distribution")
    ax.legend(); fig.tight_layout(); fig.savefig(result / "local_expert_m_distribution.png", dpi=150); plt.close(fig)


def _write_csvs(result: Path, raw: pd.DataFrame, agg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw.to_csv(result / "rank_timing_raw.csv", index=False)
    agg.to_csv(result / "granularity_results.csv", index=False)
    curves = agg.groupby(["modality", "M"], as_index=False).agg(
        cases=("case_id", "count"),
        total_ms_per_token_median=("total_ms_per_token", "median"),
        total_ms_per_token_p25=("total_ms_per_token", lambda x: _q(x, .25)),
        total_ms_per_token_p75=("total_ms_per_token", lambda x: _q(x, .75)),
        dispatch_ms_per_token_median=("dispatch_ms_per_token", "median"),
        expert_ms_per_token_median=("expert_ms_per_token", "median"),
        combine_ms_per_token_median=("combine_ms_per_token", "median"),
        active_experts_median=("active_experts", "median"),
        effective_experts_median=("effective_experts", "median"),
        p50_local_expert_m_median=("p50_active_expert_m", "median"),
        rank_cv_median=("rank_cv", "median"),
    )
    curves.to_csv(result / "modality_curves.csv", index=False)
    piv = agg.pivot_table(index=["request_id", "category", "layer", "M"], columns="modality", values=["critical_wall_ms", "total_ms_per_token", "dispatch_ms_per_token", "expert_ms_per_token", "combine_ms_per_token", "active_experts", "effective_experts", "p50_active_expert_m", "rank_cv"], aggfunc="median").reset_index()
    piv.columns = ["_".join(str(x) for x in c if str(x) != "") if isinstance(c, tuple) else str(c) for c in piv.columns]
    for metric in ("critical_wall_ms", "total_ms_per_token", "dispatch_ms_per_token", "expert_ms_per_token", "combine_ms_per_token"):
        v, t = f"{metric}_vision", f"{metric}_text"
        if v in piv and t in piv:
            piv[f"{metric}_vision_minus_text_pct"] = (piv[v] - piv[t]) / piv[t].replace(0, np.nan) * 100.0
    piv.to_csv(result / "matched_pair_results.csv", index=False)
    return curves, piv


def _gate(result: Path, agg: pd.DataFrame, curves: pd.DataFrame) -> dict:
    best: dict[str, int] = {}
    curve_med: dict[str, dict[str, float]] = {}
    for modality, g in curves.groupby("modality"):
        values = {str(int(r.M)): float(r.total_ms_per_token_median) for r in g.itertuples()}
        curve_med[modality] = values
        best[modality] = int(g.loc[g.total_ms_per_token_median.idxmin(), "M"])
    common_m = 128
    gains = {}
    for modality, m in best.items():
        base = curve_med[modality].get(str(common_m), float("nan"))
        opt = curve_med[modality].get(str(m), float("nan"))
        gains[modality] = float((base - opt) / base * 100.0) if base else float("nan")
    best_rows = []
    for key, g in agg.groupby(["request_id", "layer", "modality"], sort=False):
        best_rows.append({"request_id": key[0], "layer": key[1], "modality": key[2],
                          "best_M": int(g.loc[g.total_ms_per_token.idxmin(), "M"])})
    req_best = pd.DataFrame(best_rows)
    repeated = {}
    for modality, g in req_best.groupby("modality"):
        repeated[modality] = {int(m): int((g.best_M == m).sum()) for m in sorted(g.best_M.unique())}
    separation = (best.get("vision", 0) >= 2 * best.get("text", 0) or best.get("text", 0) >= 2 * best.get("vision", 0)) if len(best) == 2 else False
    repeated_separation = False
    if len(repeated) == 2:
        v = max(repeated["vision"], key=repeated["vision"].get)
        t = max(repeated["text"], key=repeated["text"].get)
        repeated_separation = v >= 2 * t or t >= 2 * v
    max_gain = max(gains.values()) if gains else 0.0
    if separation and repeated_separation and max_gain >= 8.0:
        status = "STRONG_GO"
    elif separation and repeated_separation and max_gain >= 5.0:
        status = "GO"
    elif best.get("vision") != best.get("text") and max_gain >= 2.0:
        status = "HOLD"
    else:
        status = "NO_GO"
    gate = {
        "status": status, "common_fixed_M": common_m, "best_M": best,
        "common_fixed_to_own_opt_gain_pct": gains,
        "curve_median_ms_per_token": curve_med,
        "per_request_layer_best_M_counts": repeated,
        "separation_at_least_2x": separation,
        "repeated_separation_at_least_2x": repeated_separation,
        "correctness_all": bool(agg["correctness"].all()),
        "route_identity_all": bool(agg["correctness"].all()),
        "measurement": {"rank_count": int(agg["ranks"].min()), "warmups": 3, "iterations": 20},
    }
    (result / "gate_summary.json").write_text(json.dumps(gate, indent=2) + "\n")
    return gate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, required=True)
    args = ap.parse_args()
    raw = _load(args.result)
    agg = _aggregate(raw)
    curves, pairs = _write_csvs(args.result, raw, agg)
    _make_figures(args.result, agg)
    gate = _gate(args.result, agg, curves)
    print(json.dumps({"rows": len(agg), "matched_rows": len(pairs), "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
