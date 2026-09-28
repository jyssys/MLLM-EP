"""Exploratory nonlinear diagnostics requested after the frozen gate was observed.

This script cannot change the primary verdict. All thresholds, bins, and risk
maps are fit on calibration requests only, then evaluated once on held-out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from ..contract import atomic_json, read_jsonl_gz


FEATURES = ("separated_visual_score", "text_energy", "full_energy", "router_removed_mass_control")


def auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    pos, neg = labels == 1, labels == 0
    if not pos.any() or not neg.any(): return None
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if not (labels == 1).any(): return None
    order = np.argsort(-scores, kind="stable"); ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered == 1].mean())


def jsd(left: np.ndarray, right: np.ndarray) -> float:
    left, right = left / max(left.sum(), 1), right / max(right.sum(), 1)
    middle = 0.5 * (left + right)
    def kl(values: np.ndarray) -> float:
        keep = values > 0
        return float(np.sum(values[keep] * np.log(values[keep] / middle[keep])))
    return 0.5 * (kl(left) + kl(right))


def fit_eval(calibration: list[dict[str, Any]], heldout: list[dict[str, Any]], feature: str,
             label_key: str, absolute_feature: bool = False) -> dict[str, Any]:
    transform = abs if absolute_feature else float
    xcal = np.asarray([transform(row[feature]) for row in calibration], dtype=float)
    ycal = np.asarray([float(row[label_key]) for row in calibration])
    xheld = np.asarray([transform(row[feature]) for row in heldout], dtype=float)
    yheld = np.asarray([float(row[label_key]) for row in heldout])
    unsafe_threshold = float(np.percentile(ycal, 75))
    labels_cal, labels_held = (ycal >= unsafe_threshold).astype(int), (yheld >= unsafe_threshold).astype(int)
    edges = np.asarray(np.percentile(xcal, [33.333333, 66.666667]), dtype=float)
    bins_cal, bins_held = np.digitize(xcal, edges), np.digitize(xheld, edges)
    risk = np.asarray([(labels_cal[bins_cal == index].sum() + 1) / ((bins_cal == index).sum() + 2)
                       for index in range(3)])
    predicted = risk[bins_held]
    damage_edges = np.asarray(np.percentile(ycal, [33.333333, 66.666667]), dtype=float)
    damage_bins = np.digitize(yheld, damage_edges)
    low, high = bins_held == 0, bins_held == 2
    low_hist = np.bincount(damage_bins[low], minlength=3).astype(float) + 0.5
    high_hist = np.bincount(damage_bins[high], minlength=3).astype(float) + 0.5
    return {
        "calibration_unsafe_threshold_p75": unsafe_threshold,
        "calibration_feature_tertile_edges": edges.tolist(),
        "calibration_bin_risk_laplace": risk.tolist(),
        "heldout_n": len(heldout), "heldout_unsafe_prevalence": float(labels_held.mean()),
        "heldout_auroc": auc(labels_held, predicted),
        "heldout_average_precision": average_precision(labels_held, predicted),
        "heldout_brier": float(np.square(predicted - labels_held).mean()),
        "heldout_low_vs_high_damage_histogram_jsd": jsd(low_hist, high_hist),
        "heldout_low_n": int(low.sum()), "heldout_high_n": int(high.sum()),
        "interpretation": "EXPLORATORY_POST_GATE_NONLINEAR_DIAGNOSTIC_NOT_PRIMARY_EVIDENCE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(read_jsonl_gz(args.labels))
    orig_cal = [row for row in rows if row["split"] == "calibration" and row["condition"] == "orig"]
    orig_held = [row for row in rows if row["split"] == "heldout" and row["condition"] == "orig"]
    indexed = {(row["case_id"], row["condition"]): row for row in rows}
    causal_cal, causal_held = [], []
    for case_id in sorted({row["case_id"] for row in rows}):
        orig, cf = indexed[(case_id, "orig")], indexed[(case_id, "cf")]
        target = causal_cal if orig["split"] == "calibration" else causal_held
        target.append({
            "request_id": orig["request_id"],
            "delta_damage": orig["structured_damage"] - cf["structured_damage"],
            **{f"delta_{key}": orig[key] - cf[key] for key in FEATURES},
        })
    result = {
        "status": "EXPLORATORY_POST_GATE_ONLY",
        "reason": "Requested after partial held-out correlation was observed; cannot alter frozen verdict.",
        "original_condition": {key: fit_eval(orig_cal, orig_held, key, "structured_damage") for key in FEATURES},
        "absolute_causal_delta": {
            key: fit_eval(causal_cal, causal_held, f"delta_{key}", "delta_damage", absolute_feature=True)
            for key in FEATURES
        },
    }
    atomic_json(args.output, result); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
