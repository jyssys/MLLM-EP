#!/usr/bin/env python3
"""Analyze causal Sonic/QuACK measurements and render preregistered figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict


def fit_cv(rows: list[dict], names: list[str]) -> dict:
    x = np.asarray([[float(r[n]) for n in names] for r in rows])
    y = np.asarray([float(r["median_ms"]) for r in rows])
    pred = cross_val_predict(LinearRegression(), x, y, cv=KFold(5, shuffle=True, random_state=1100))
    model = LinearRegression().fit(x, y)
    return {
        "features": names,
        "cv_r2": float(r2_score(y, pred)),
        "cv_rmse_ms": float(mean_squared_error(y, pred) ** 0.5),
        "fit_r2": float(model.score(x, y)),
        "coefficients": [float(x) for x in model.coef_],
    }


def gate(effect_pct: float, staircase: bool) -> str:
    if effect_pct >= 10 and staircase:
        return "GO"
    if effect_pct >= 5 and staircase:
        return "HOLD"
    return "NO-GO"


def analyze_shape(shape: dict) -> dict:
    hist = [r for r in shape["records"] if r["kind"] == "histogram"]
    boundary = [r for r in shape["records"] if r["kind"] == "boundary"]
    aligned = next(r for r in hist if r["name"] == "aligned")
    heavy = next(r for r in hist if r["name"] == "boundary_heavy")
    gap = (heavy["median_ms"] / aligned["median_ms"] - 1) * 100
    largest = (max(r["median_ms"] for r in hist) / min(r["median_ms"] for r in hist) - 1) * 100
    steps = []
    for multiple in (1, 2, 3):
        zero = next(r for r in boundary if r["multiple"] == multiple and r["delta"] == 0)
        neighbors = [r for r in boundary if r["multiple"] == multiple and abs(r["delta"]) == 1]
        steps.append(float(np.median([r["median_ms"] for r in neighbors]) / zero["median_ms"] - 1))
    # A true tile staircase should increase on both sides of every boundary.
    staircase = all(x > 0.01 for x in steps)
    models = {
        "M0": fit_cv(shape["records"], ["N"]),
        "M1": fit_cv(shape["records"], ["N", "G"]),
        "M2": fit_cv(shape["records"], ["N", "G", "Q"]),
        "M3": fit_cv(shape["records"], ["full_tiles", "tail_count", "tail_rows"]),
    }
    lut = {int(r["rows"]): float(r["median_ms"]) for r in shape["tail_lut"]}
    b = max(lut)
    b_cost = lut[b]
    oracle = []
    for r in hist:
        tails = [x % b for x in r["counts"] if x % b]
        full = sum(x // b for x in r["counts"])
        # Interpolate measured single-group relative tail cost. This intentionally
        # does not assume FLOPs scale linearly with rows.
        xs = np.asarray(sorted(lut))
        ys = np.asarray([lut[int(x)] for x in xs])
        tail_equiv = sum(float(np.interp(t, xs, ys)) / b_cost for t in tails)
        current_equiv = full + len(tails)
        ideal_equiv = full + tail_equiv
        oracle.append({
            "name": r["name"], "current_full_tile_equiv": current_equiv,
            "measured_tail_equiv": ideal_equiv,
            "oracle_speedup": current_equiv / ideal_equiv if ideal_equiv else 1.0,
        })
    return {
        "name": shape["name"],
        "aligned_ms": aligned["median_ms"], "boundary_heavy_ms": heavy["median_ms"],
        "aligned_boundary_gap_pct": gap, "largest_iso_ng_gap_pct": largest,
        "boundary_step_ratios": steps, "staircase": staircase,
        "poc1_status": gate(max(abs(gap), largest), staircase),
        "models": models, "oracle": oracle,
        "oracle_median_speedup": float(np.median([x["oracle_speedup"] for x in oracle])),
        "oracle_p95_speedup": float(np.percentile([x["oracle_speedup"] for x in oracle], 95)),
    }


def figures(data: dict, summary: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(data["shapes"]), figsize=(12, 4), squeeze=False)
    for ax, shape in zip(axes[0], data["shapes"]):
        rows = [r for r in shape["records"] if r["kind"] == "histogram"]
        ax.bar([r["name"] for r in rows], [r["median_ms"] for r in rows])
        ax.tick_params(axis="x", rotation=55)
        ax.set_title(shape["name"]); ax.set_ylabel("expert up+down latency (ms)")
    fig.tight_layout(); fig.savefig(out / "plot1_iso_ng_tile_causality.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(len(data["shapes"]), 1, figsize=(10, 7), squeeze=False)
    for ax, shape in zip(axes[:, 0], data["shapes"]):
        for multiple in (1, 2, 3):
            rows = [r for r in shape["records"] if r["kind"] == "boundary" and r["multiple"] == multiple]
            ax.plot([r["delta"] for r in rows], [r["median_ms"] for r in rows], marker="o", label=f"{multiple}×B")
        ax.axvline(0, color="black", lw=.7); ax.legend(); ax.set_title(shape["name"])
        ax.set_xlabel("paired expert offset from boundary"); ax.set_ylabel("latency (ms)")
    fig.tight_layout(); fig.savefig(out / "plot2_boundary_staircase.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, len(data["shapes"]), figsize=(11, 4), squeeze=False)
    for ax, shape in zip(axes[0], data["shapes"]):
        rows = shape["records"]
        sc=ax.scatter([r["Q"] for r in rows], [r["median_ms"] for r in rows], c=[r["tail_count"] for r in rows])
        ax.set_title(shape["name"]); ax.set_xlabel("effective tiles Q"); ax.set_ylabel("latency (ms)")
        fig.colorbar(sc, ax=ax, label="tail count")
    fig.tight_layout(); fig.savefig(out / "plot3_tile_features_vs_latency.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, len(summary["shapes"]), figsize=(11, 4), squeeze=False)
    for ax, item in zip(axes[0], summary["shapes"]):
        ax.bar([x["name"] for x in item["oracle"]], [x["oracle_speedup"] for x in item["oracle"]])
        ax.axhline(1, color="black", lw=.7); ax.tick_params(axis="x", rotation=55)
        ax.set_title(item["name"]); ax.set_ylabel("measured-LUT oracle speedup")
    fig.tight_layout(); fig.savefig(out / "plot6_exact_routing_oracle_headroom.png", dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.result_dir / "synthetic.json").read_text())
    summary = {"schema": 1, "shapes": [analyze_shape(s) for s in data["shapes"]]}
    summary["poc1_status"] = "NO-GO" if any(s["poc1_status"] == "NO-GO" for s in summary["shapes"]) else "HOLD"
    m2_gain = [s["models"]["M2"]["cv_r2"] - s["models"]["M1"]["cv_r2"] for s in summary["shapes"]]
    summary["poc2_status"] = "GO" if min(m2_gain) >= .05 else ("HOLD" if max(m2_gain) >= .01 else "NO-GO")
    summary["poc4_status"] = "GO" if min(s["oracle_median_speedup"] for s in summary["shapes"]) >= 1.10 else "NO-GO"
    tr_path = args.result_dir / "sonic_tr.json"
    if tr_path.exists():
        tr = json.loads(tr_path.read_text())["rows"]
        summary["sonic_token_rounding"] = tr
        fig, ax = plt.subplots(figsize=(7, 4))
        for row in tr:
            ax.scatter(100 * row["routing_edit_fraction_of_original"], row["speedup_vs_topk"], label=f"T{row['tokens']} {row['method']}")
        for item in summary["shapes"]:
            ax.scatter(0, item["oracle_median_speedup"], marker="*", s=120, label=f"{item['name']} exact-route oracle")
        ax.axhline(1, color="black", lw=.7); ax.set_xlabel("token-expert assignment edit (% of original)"); ax.set_ylabel("kernel speedup")
        ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(args.result_dir/"figures"/"plot7_sonic_routing_edit_tradeoff.png",dpi=180); plt.close(fig)
    (args.result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figures(data, summary, args.result_dir / "figures")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
