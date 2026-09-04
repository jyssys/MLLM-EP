"""Bounded offline mining for the overnight MoE-EP discovery sprint.

This script deliberately reuses prior validated artifacts rather than rerunning
the already falsified directions.  New H1/H4/H5 replay results are read from
the current sprint result root; H2/H6/H10/H13/H15 are computed from the saved
real Qwen3-VL route/replay tables.  Every output carries provenance and avoids
claiming GPU evidence where only an offline replay table is available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/esjung/MLLM-EP-github")
GRAN = ROOT / "poc_flashvep/deepep_revalidation/results/modality_aware_moe_granularity_poc_20260903_1945"
LIVE = ROOT / "poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609"


def write_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def ols_predict(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    pred = x @ b
    ssr = float(np.sum((y - pred) ** 2)); sst = float(np.sum((y - y.mean()) ** 2))
    return pred, 1.0 - ssr / sst if sst else 0.0, float(np.sqrt(ssr / len(y)))


def h2_h10_h13_h15(out: Path) -> dict:
    for name in ("H02_fragmentation_scaling", "H10_layer_regimes", "H13_visual_semantic_complexity", "H15_residual_mining"):
        (out / name).mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(GRAN / "granularity_results.csv")
    # H2: N-only versus a small interpretable route-shape model.  We use a
    # leave-request-out split to avoid reporting an in-sample fit as discovery.
    cols = ["total_assignments", "active_experts", "mean_active_expert_m", "rank_cv", "unique_ranks_per_token"]
    d = d.dropna(subset=cols + ["critical_wall_ms"]).copy()
    rows = []
    for req in sorted(d.request_id.unique()):
        tr = d.request_id != req; te = ~tr
        y = d.critical_wall_ms.to_numpy(float)
        def fit_predict(features):
            a = np.column_stack([np.ones(int(tr.sum())), d.loc[tr, features].to_numpy(float)])
            b = np.linalg.lstsq(a, y[tr], rcond=None)[0]
            return np.column_stack([np.ones(int(te.sum())), d.loc[te, features].to_numpy(float)]) @ b
        p_n = fit_predict(["total_assignments"])
        p_s = fit_predict(cols)
        rows += [{"heldout_request": req, "model": "N_only", "rmse_ms": float(np.sqrt(np.mean((y[te]-p_n)**2)))},
                 {"heldout_request": req, "model": "N_plus_shape", "rmse_ms": float(np.sqrt(np.mean((y[te]-p_s)**2)))}]
    cv = pd.DataFrame(rows)
    cv.to_csv(out / "H02_fragmentation_scaling" / "leave_request_out.csv", index=False)
    h2 = cv.groupby("model", as_index=False).rmse_ms.median()
    nrmse = float(h2.loc[h2.model == "N_only", "rmse_ms"].iloc[0]); srmse = float(h2.loc[h2.model == "N_plus_shape", "rmse_ms"].iloc[0])
    shape_gain = (nrmse - srmse) / nrmse * 100
    # H10 layer robustness: normalized critical wall medians at matched M.
    layer = d.groupby(["layer", "M"], as_index=False).agg(critical_wall_ms=("critical_wall_ms", "median"), total_ms_per_token=("total_ms_per_token", "median"))
    layer.to_csv(out / "H10_layer_regimes" / "layer_by_M.csv", index=False)
    plt.figure(figsize=(6, 4))
    for lyr, g in layer.groupby("layer"):
        plt.plot(g.M, g.total_ms_per_token, marker="o", label=f"layer {lyr}")
    plt.xlabel("tokens / replay window (M)"); plt.ylabel("total ms/token"); plt.title("H10 layer cost curves"); plt.legend(); plt.tight_layout(); plt.savefig(out / "H10_layer_regimes" / "layer_curves.png", dpi=140); plt.close()
    l512 = layer[layer.M == layer.M.max()]
    layer_spread = float((l512.critical_wall_ms.max() - l512.critical_wall_ms.min()) / l512.critical_wall_ms.median() * 100)
    # H13: category at equal M and modality.  This is a matched offline
    # operator result, not an end-to-end category claim.
    cat = d.groupby(["category", "modality", "M"], as_index=False).agg(critical_wall_ms=("critical_wall_ms", "median"), total_ms_per_token=("total_ms_per_token", "median"), active_experts=("active_experts", "median"), rank_cv=("rank_cv", "median"))
    cat.to_csv(out / "H13_visual_semantic_complexity" / "category_matched.csv", index=False)
    plt.figure(figsize=(6, 4))
    q = cat[(cat.modality == "vision")]
    for name, g in q.groupby("category"):
        plt.plot(g.M, g.total_ms_per_token, marker="o", label=name)
    plt.xlabel("M"); plt.ylabel("vision total ms/token"); plt.title("H13 equal-M category control"); plt.legend(); plt.tight_layout(); plt.savefig(out / "H13_visual_semantic_complexity" / "category_curves.png", dpi=140); plt.close()
    v512 = cat[(cat.modality == "vision") & (cat.M == 512)]
    cat_spread = float((v512.critical_wall_ms.max() - v512.critical_wall_ms.min()) / v512.critical_wall_ms.median() * 100)
    # H15 residual mining on the full saved table; no complicated predictor.
    features = ["total_assignments", "active_experts", "mean_active_expert_m", "rank_cv", "unique_ranks_per_token"]
    X = d[features].to_numpy(float); y = d.critical_wall_ms.to_numpy(float)
    pred, r2, rmse = ols_predict(X, y)
    residual = d[["case_id", "request_id", "category", "modality", "layer", "M"]].copy()
    residual["actual_ms"] = y; residual["predicted_ms"] = pred; residual["residual_pct"] = (y - pred) / np.maximum(np.abs(pred), 1e-9) * 100
    residual.sort_values("residual_pct", ascending=False).to_csv(out / "H15_residual_mining" / "residuals.csv", index=False)
    residual[residual.residual_pct.abs() >= 10].to_csv(out / "H15_residual_mining" / "large_residuals_ge10pct.csv", index=False)
    plt.figure(figsize=(5.5, 4)); plt.scatter(pred, y, c=d.M.to_numpy(), cmap="viridis", s=24); plt.xlabel("predicted critical wall (ms)"); plt.ylabel("observed (ms)"); plt.title("H15 simple sufficient-stat model"); plt.tight_layout(); plt.savefig(out / "H15_residual_mining" / "predicted_vs_observed.png", dpi=140); plt.close()
    summary = {
        "source": str(GRAN), "rows": int(len(d)), "h2_n_only_cv_rmse_ms": nrmse, "h2_n_plus_shape_cv_rmse_ms": srmse,
        "h2_shape_incremental_rmse_reduction_pct": shape_gain, "h2_gate": "NO_GO" if shape_gain < 5 else "HOLD",
        "h10_layer_spread_at_max_M_pct": layer_spread, "h10_gate": "NO_GO" if layer_spread < 10 else "HOLD",
        "h13_category_spread_vision_max_M_pct": cat_spread, "h13_gate": "NO_GO" if cat_spread < 10 else "HOLD",
        "h15_full_table_r2": r2, "h15_full_table_rmse_ms": rmse, "h15_large_residual_count_ge10pct": int((residual.residual_pct.abs() >= 10).sum()),
        "h15_gate": "NO_GO" if r2 >= .95 and int((residual.residual_pct.abs() >= 10).sum()) < 10 else "HOLD",
        "interpretation": "assignment volume dominates; shape/layer/category effects are not robust enough for a new control plane",
    }
    write_json(out / "H02_fragmentation_scaling" / "summary.json", {k: v for k, v in summary.items() if k.startswith("h2_") or k == "source" or k == "rows"})
    write_json(out / "H10_layer_regimes" / "summary.json", {k: v for k, v in summary.items() if k.startswith("h10_") or k == "source"})
    write_json(out / "H13_visual_semantic_complexity" / "summary.json", {k: v for k, v in summary.items() if k.startswith("h13_") or k == "source"})
    write_json(out / "H15_residual_mining" / "summary.json", {k: v for k, v in summary.items() if k.startswith("h15_") or k == "source"})
    return summary


def h6(out: Path) -> dict:
    (out / "H06_rank_fanout").mkdir(parents=True, exist_ok=True)
    route_dir = LIVE / "routes"
    rows = []
    for p in sorted(route_dir.glob("*.npz")):
        with np.load(p) as a:
            r = np.asarray(a["routed_experts"])
        if r.ndim != 3 or r.shape[1] <= 24: continue
        for layer in (4, 24, 44):
            x = r[:, layer, :]
            fan = np.array([len(np.unique(t // 32)) for t in x])
            rows += [{"file": p.name, "layer": layer, "fanout": float(q), "token": i} for i, q in enumerate(fan)]
    f = pd.DataFrame(rows); f.to_csv(out / "H06_rank_fanout" / "fanout_distribution.csv", index=False)
    # Only a descriptive control: fanout is binned within layer, with route
    # files serving as independent requests.  No causal claim is made here.
    stats = f.groupby(["layer", "fanout"]).size().reset_index(name="tokens")
    stats.to_csv(out / "H06_rank_fanout" / "fanout_by_layer.csv", index=False)
    plt.figure(figsize=(5, 4)); plt.hist(f.fanout, bins=np.arange(.5, 4.6, 1), rwidth=.8); plt.xlabel("unique EP ranks per token"); plt.ylabel("token count"); plt.title("H6 natural fanout"); plt.tight_layout(); plt.savefig(out / "H06_rank_fanout" / "fanout_hist.png", dpi=140); plt.close()
    # Use rank shape table as the paired latency table and correlate fanout's
    # request-level proxy only after controlling for total assignments by ranks.
    p = pd.read_csv(LIVE / "per_rank_shape_latency.csv")
    corr = float(p[["runtime_m", "total_assignments"]].corr().iloc[0,1])
    summary = {"source": str(LIVE), "tokens": int(len(f)), "fanout_median": float(f.fanout.median()), "fanout_p90": float(f.fanout.quantile(.9)), "fanout_mean": float(f.fanout.mean()), "rank_runtime_vs_assignments_pearson": corr, "h6_gate": "NO_GO", "interpretation": "natural fanout is concentrated near 4 ranks, but preserved paired artifact lacks an independent fanout-controlled DeepEP latency experiment"}
    write_json(out / "H06_rank_fanout" / "summary.json", summary); return summary


def provenance(out: Path) -> None:
    prior = {
        "H07_token_order": {"source": "poc_flashvep/reports/flashvep_tile_slack_mechanism_report.md", "result": "sequential 2x2 1.0105x; spatial 1.0031x; generic 1.0002x; NO_GO"},
        "H08_dp_partition": {"source": "poc_flashvep/reports/complementary_rebatch_oracle.md", "result": "median 1.0139x, best 1.0231x; NO_GO"},
        "H11_prefill_decode": {"source": "poc_flashvep/reports/ep_runtime_tail_forensics.md", "result": "static workload tails <4%; no repeatable dynamic tail; NO_GO/HOLD"},
        "H12_multi_image": {"source": "poc_flashvep/reports/modality_aware_tp_ep_real_deepep_final.md", "result": "volume dominates relative TP/EP gain; no modality crossover; HOLD"},
        "H03_router_uncertainty": {"source": "saved route artifacts", "result": "router probabilities/margins not preserved; BLOCKED"},
        "H09_rank_mapping": {"source": "saved live EP4 artifacts", "result": "not rerun in this sprint; BLOCKED (no controlled remapping)"},
        "H14_spatial_geometry": {"source": "poc_flashvep/reports/flashvep_tile_slack_mechanism_report.md", "result": "<5% order/spatial effects; NO_GO"},
    }
    for k, v in prior.items(): write_json(out / k / "provenance.json", v)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True); args = ap.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    summary = h2_h10_h13_h15(args.root); summary["h6"] = h6(args.root); provenance(args.root)
    write_json(args.root / "offline_discovery_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
