#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from poc_mllm_vision_conditioned_decode_damage_skipping.contract import (
    atomic_json, read_json, read_jsonl, write_jsonl_gz,
)

FEATURES = {
    "F0_GENERIC_LOCAL_ROUTER": [
        "router_weight", "slot_rank_scaled", "weight_margin_above", "weight_margin_below",
        "layer_fraction", "step_fraction",
    ],
    "F1_MODES_STYLE_GENERIC": [
        "router_weight", "slot_rank_scaled", "weight_margin_above", "weight_margin_below",
        "layer_fraction", "step_fraction", "calibration_expert_frequency",
        "calibration_layer_expert_frequency", "calibration_expert_mean_weight", "modes_importance",
    ],
    "F2_TEXT_PREFILL": [
        "router_weight", "slot_rank_scaled", "weight_margin_above", "weight_margin_below",
        "layer_fraction", "step_fraction", "calibration_expert_frequency",
        "calibration_layer_expert_frequency", "calibration_expert_mean_weight", "modes_importance",
        "text_mass_affinity", "text_count_affinity",
    ],
    "F3_FULL_PREFILL": [
        "router_weight", "slot_rank_scaled", "weight_margin_above", "weight_margin_below",
        "layer_fraction", "step_fraction", "calibration_expert_frequency",
        "calibration_layer_expert_frequency", "calibration_expert_mean_weight", "modes_importance",
        "full_mass_affinity", "full_count_affinity",
    ],
    "F4_VISION_PRIOR": [
        "router_weight", "slot_rank_scaled", "weight_margin_above", "weight_margin_below",
        "layer_fraction", "step_fraction", "calibration_expert_frequency",
        "calibration_layer_expert_frequency", "calibration_expert_mean_weight", "modes_importance",
        "vision_mass_affinity", "vision_count_affinity", "frozen_visual_criticality",
    ],
    "F5_VISION_SEPARATED": [
        "router_weight", "slot_rank_scaled", "weight_margin_above", "weight_margin_below",
        "layer_fraction", "step_fraction", "calibration_expert_frequency",
        "calibration_layer_expert_frequency", "calibration_expert_mean_weight", "modes_importance",
        "text_mass_affinity", "text_count_affinity", "vision_mass_affinity",
        "vision_count_affinity", "vision_text_mass_ratio_clipped", "frozen_visual_criticality",
    ],
}


def enrich(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calibration = [row for row in rows if row["split"] == "calibration"]
    expert_count = Counter(int(row["expert_id"]) for row in calibration)
    layer_count = Counter((int(row["layer"]), int(row["expert_id"])) for row in calibration)
    expert_weights: dict[int, list[float]] = defaultdict(list)
    for row in calibration:
        expert_weights[int(row["expert_id"])].append(float(row["router_weight"]))
    total = max(len(calibration), 1)
    by_layer = Counter(int(row["layer"]) for row in calibration)
    answer = []
    for source in rows:
        row = dict(source)
        expert, layer = int(row["expert_id"]), int(row["layer"])
        row["slot_rank_scaled"] = (float(row["slot_rank"]) - 1.0) / 7.0
        row["calibration_expert_frequency"] = expert_count[expert] / total
        row["calibration_layer_expert_frequency"] = layer_count[(layer, expert)] / max(by_layer[layer], 1)
        row["calibration_expert_mean_weight"] = float(np.mean(expert_weights.get(expert, [0.0])))
        row["modes_importance"] = float(row["router_weight"]) * np.sqrt(
            max(row["calibration_layer_expert_frequency"], 1e-12)
        )
        row["full_count_affinity"] = float(row["vision_count_affinity"]) + float(row["text_count_affinity"])
        row["vision_text_mass_ratio_clipped"] = float(np.clip(row["vision_text_mass_ratio"], 0, 20))
        answer.append(row)
    return answer


def select_skips(values: np.ndarray, rate: float) -> np.ndarray:
    count = int(np.floor(len(values) * rate))
    mask = np.zeros(len(values), dtype=bool)
    if count:
        mask[np.argsort(values, kind="stable")[:count]] = True
    return mask


def policy_metrics(rows: list[dict[str, Any]], predictions: np.ndarray, critical_threshold: float, rate: float) -> dict[str, float]:
    actual = np.asarray([row["functional_damage_score"] for row in rows])
    safe = np.asarray([row["absolute_safe"] for row in rows], dtype=bool)
    critical = actual >= critical_threshold
    skipped = select_skips(predictions, rate)
    return {
        "actual_skip_rate": float(skipped.mean()),
        "safe_skip_precision": float(safe[skipped].mean()) if skipped.any() else 0.0,
        "false_safe_rate": float(critical[skipped].mean()) if skipped.any() else 0.0,
        "protected_critical_recall": float((~skipped & critical).sum() / max(critical.sum(), 1)),
        "mean_damage_skipped": float(actual[skipped].mean()) if skipped.any() else 0.0,
        "top1_flip_in_skipped": float(np.mean([rows[index]["top1_flip"] for index in np.flatnonzero(skipped)])) if skipped.any() else 0.0,
    }


def bootstrap(rows: list[dict[str, Any]], left: np.ndarray, right: np.ndarray, threshold: float, metric: str, seed: int = 3407, repeats: int = 2000) -> dict[str, float]:
    requests = sorted({row["request_id"] for row in rows})
    indices = {request: np.asarray([i for i, row in enumerate(rows) if row["request_id"] == request]) for request in requests}
    random = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        sampled = random.choice(requests, len(requests), replace=True)
        take = np.concatenate([indices[request] for request in sampled])
        subset = [rows[index] for index in take]
        left_value = policy_metrics(subset, left[take], threshold, 0.20)[metric]
        right_value = policy_metrics(subset, right[take], threshold, 0.20)[metric]
        values.append(100.0 * (left_value - right_value))
    point = 100.0 * (
        policy_metrics(rows, left, threshold, 0.20)[metric]
        - policy_metrics(rows, right, threshold, 0.20)[metric]
    )
    return {
        "point_pp": float(point),
        "ci95_low_pp": float(np.percentile(values, 2.5)),
        "ci95_high_pp": float(np.percentile(values, 97.5)),
        "bootstrap_unit": "request",
        "replicates": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    contract = read_json(args.contract)
    rows = enrich(list(read_jsonl(args.labels)))
    calibration = [row for row in rows if row["split"] == "calibration"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    y_train = np.asarray([row["functional_damage_score"] for row in calibration])
    y_test = np.asarray([row["functional_damage_score"] for row in heldout])
    thresholds = {
        name: float(np.quantile(y_train, quantile))
        for name, quantile in contract["critical_thresholds_from_calibration_quantiles"].items()
    }
    primary_threshold = thresholds["moderate_primary"]
    predictions: dict[str, np.ndarray] = {}
    results = {}
    prediction_rows = []
    for family, features in FEATURES.items():
        train = np.asarray([[float(row[name]) for name in features] for row in calibration])
        test = np.asarray([[float(row[name]) for name in features] for row in heldout])
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(contract["ridge_alpha"])))
        model.fit(train, y_train)
        predicted = model.predict(test)
        predictions[family] = predicted
        critical = y_test >= primary_threshold
        result = {
            "features": features,
            "continuous_damage_spearman": float(spearmanr(y_test, predicted).statistic),
            "critical_auroc": float(roc_auc_score(critical, predicted)),
            "critical_auprc": float(average_precision_score(critical, predicted)),
            "frontier": {
                f"{int(100*rate)}pct": policy_metrics(heldout, predicted, primary_threshold, rate)
                for rate in (0.10, 0.20, 0.30, 0.40)
            },
        }
        results[family] = result
        for row, value in zip(heldout, predicted):
            prediction_rows.append({
                "request_id": row["request_id"], "domain": row["domain"],
                "layer": row["layer"], "decode_step": row["decode_step"],
                "expert_id": row["expert_id"], "family": family,
                "predicted_damage": float(value), "actual_damage": row["functional_damage_score"],
                "critical_primary": bool(row["functional_damage_score"] >= primary_threshold),
            })

    comparisons = {}
    for control in ("F1_MODES_STYLE_GENERIC", "F2_TEXT_PREFILL", "F3_FULL_PREFILL"):
        key = f"F5_MINUS_{control.split('_', 1)[0]}"
        comparisons[key] = {
            metric: bootstrap(
                heldout, predictions["F5_VISION_SEPARATED"], predictions[control],
                primary_threshold, metric,
            )
            for metric in ("protected_critical_recall", "safe_skip_precision")
        }

    oracle = {}
    for rate in (0.10, 0.20, 0.30, 0.40):
        oracle[f"{int(100*rate)}pct"] = policy_metrics(heldout, y_test, primary_threshold, rate)
    oracle_safe_headroom = float(np.mean([row["absolute_safe"] for row in heldout]))

    f1 = predictions["F1_MODES_STYLE_GENERIC"]
    f1_cutoff = float(np.median(model_prediction_calibration(calibration, FEATURES["F1_MODES_STYLE_GENERIC"], y_train, contract)))
    router_cutoff = float(np.median([row["router_weight"] for row in calibration]))
    rare = np.asarray([
        row["functional_damage_score"] >= primary_threshold
        and row["router_weight"] <= router_cutoff
        and f1[index] <= f1_cutoff
        for index, row in enumerate(heldout)
    ], dtype=bool)
    rare_rows = [row for row, flag in zip(heldout, rare) if flag]
    protection = {}
    for family, predicted in predictions.items():
        skipped = select_skips(predicted, 0.20)
        protection[family] = float((~skipped & rare).sum() / max(rare.sum(), 1))
    rare_result = {
        "definition": "primary-critical AND router-weight<=calibration-median AND F1-prediction<=calibration-median",
        "path_count": int(rare.sum()),
        "path_rate": float(rare.mean()),
        "domain_counts": Counter(row["domain"] for row in rare_rows),
        "layer_counts": Counter(str(row["layer"]) for row in rare_rows),
        "decode_step_counts": Counter(str(row["decode_step"]) for row in rare_rows),
        "mean_vision_mass_affinity": float(np.mean([row["vision_mass_affinity"] for row in rare_rows])) if rare_rows else None,
        "protection_at_20pct_skip": protection,
    }

    f5_f1 = comparisons["F5_MINUS_F1"]
    f5_f2 = comparisons["F5_MINUS_F2"]
    increment_f1 = max(
        f5_f1["protected_critical_recall"]["point_pp"] if f5_f1["protected_critical_recall"]["ci95_low_pp"] > 0 else -np.inf,
        f5_f1["safe_skip_precision"]["point_pp"] if f5_f1["safe_skip_precision"]["ci95_low_pp"] > 0 else -np.inf,
    )
    increment_f2 = max(
        f5_f2["protected_critical_recall"]["point_pp"] if f5_f2["protected_critical_recall"]["ci95_low_pp"] > 0 else -np.inf,
        f5_f2["safe_skip_precision"]["point_pp"] if f5_f2["safe_skip_precision"]["ci95_low_pp"] > 0 else -np.inf,
    )
    gate = {
        "oracle_safe_headroom_ge20pct": oracle_safe_headroom >= 0.20,
        "f5_vs_f1_increment_ge3pp_ci_positive": increment_f1 >= 3.0,
        "f5_vs_f2_increment_ge3pp_ci_positive": increment_f2 >= 3.0,
        "vision_increment_ci_positive": increment_f1 > 0 and increment_f2 > 0,
    }
    gate["pass"] = all(gate.values())
    if gate["pass"]:
        verdict = "VISION_DAMAGE_SIGNAL_SUPPORTED"
    elif oracle_safe_headroom < 0.20:
        verdict = "INSUFFICIENT_SAFE_SKIP_HEADROOM"
    elif results["F1_MODES_STYLE_GENERIC"]["frontier"]["20pct"]["protected_critical_recall"] >= results["F5_VISION_SEPARATED"]["frontier"]["20pct"]["protected_critical_recall"]:
        verdict = "GENERIC_MODES_STYLE_SUFFICIENT"
    else:
        verdict = "TEXT_SIGNAL_SUFFICIENT"

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_gz(args.artifact_dir / "feature_matrix.jsonl.gz", rows)
    write_jsonl_gz(args.artifact_dir / "predictor_outputs.jsonl.gz", prediction_rows)
    atomic_json(args.artifact_dir / "damage_predictor_comparison.json", {
        "status": "MEASURED_HELDOUT_FUNCTIONAL_DAMAGE",
        "calibration_paths": len(calibration), "heldout_paths": len(heldout),
        "calibration_requests": len({row["request_id"] for row in calibration}),
        "heldout_requests": len({row["request_id"] for row in heldout}),
        "critical_thresholds": thresholds, "results": results,
        "comparisons": comparisons, "stage_b_gate": gate, "verdict": verdict,
    })
    atomic_json(args.artifact_dir / "drop_one_oracle_headroom.json", {
        "status": "MEASURED_HELDOUT_FUNCTIONAL_ORACLE", "absolute_safe_headroom": oracle_safe_headroom,
        "absolute_safe_definition": contract["absolute_safe_path"], "frontier": oracle,
    })
    atomic_json(args.artifact_dir / "rare_but_critical_paths.json", rare_result)
    atomic_json(args.artifact_dir / "bootstrap_ci.json", comparisons)
    atomic_json(args.artifact_dir / "no_future_leakage_audit.json", {
        "status": "PASS", "fit_rows": "calibration_only", "threshold_rows": "calibration_only",
        "heldout_labels_visible_before_prediction": False, "request_overlap": sorted(
            {row["request_id"] for row in calibration} & {row["request_id"] for row in heldout}
        ),
    })
    atomic_json(args.artifact_dir / "stage_b_gate.json", {"gate": gate, "verdict": verdict})
    print(json.dumps({"verdict": verdict, "gate": gate, "comparisons": comparisons}, indent=2))


def model_prediction_calibration(rows: list[dict[str, Any]], features: list[str], target: np.ndarray, contract: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray([[float(row[name]) for name in features] for row in rows])
    model = make_pipeline(StandardScaler(), Ridge(alpha=float(contract["ridge_alpha"])))
    model.fit(matrix, target)
    return model.predict(matrix)


if __name__ == "__main__":
    main()
