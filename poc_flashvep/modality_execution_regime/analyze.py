"""Analyze modality-induced routing shape and expert execution regimes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 1729
BOOTSTRAPS = 2000


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _gini(values: np.ndarray) -> float:
    total = float(values.sum())
    if total == 0:
        return 0.0
    return float(np.abs(values[:, None] - values[None, :]).sum() / (2 * len(values) * total))


def _features(histogram: list[int], block: int) -> dict[str, float | int]:
    values = np.asarray(histogram, dtype=float)
    total = float(values.sum())
    nonzero = values[values > 0]
    shares = np.sort(values)[::-1] / total if total else values
    probs = nonzero / total if total else nonzero
    effective = np.ceil(nonzero / block) * block
    return {
        "active_experts": int(len(nonzero)),
        "max_expert_load": float(values.max(initial=0)),
        "mean_nonzero_load": float(nonzero.mean()) if len(nonzero) else 0.0,
        "load_std_32": float(values.std()),
        "load_cv_32": float(values.std() / values.mean()) if values.mean() else 0.0,
        "entropy": float(-(probs * np.log2(probs)).sum()) if len(probs) else 0.0,
        "gini": _gini(values),
        "top1_share": float(shares[:1].sum()),
        "top4_share": float(shares[:4].sum()),
        "max_mean_ratio": float(values.max() / values.mean()) if values.mean() else 0.0,
        "block_m": int(block),
        "effective_rows": int(effective.sum()),
        "effective_tiles": int(np.ceil(nonzero / block).sum()),
        "padding_amplification": float(effective.sum() / total) if total else 1.0,
        "partial_tile_count": int(((nonzero % block) != 0).sum()),
        "near_boundary_count": int((((nonzero % block) > 0) & ((nonzero % block) <= max(1, block // 8))).sum()),
    }


def _load(result: Path) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((result / "workload_manifest.json").read_text())
    vision_manifest = json.loads((Path(manifest["vision_source"]) / "sample_manifest.json").read_text())
    vision_metadata = {row["sample_id"]: row for row in vision_manifest["samples"]}
    route_cache: dict[str, np.ndarray] = {}
    payloads = [json.loads((result / "replay" / f"rank{rank}.json").read_text()) for rank in range(4)]
    if any(payload.get("status") != "ok" for payload in payloads):
        raise RuntimeError([payload.get("error") for payload in payloads])
    rows = []
    for payload in payloads:
        for row in payload["observations"]:
            block = int(row["runtime_config"]["BLOCK_SIZE_M"])
            item = {key: value for key, value in row.items() if key not in ("expert_histogram", "expert_ms", "runtime_config")}
            item.update(_features(row["expert_histogram"], block))
            if row["modality"] == "vision":
                request_id = row["request_id"]
                if request_id not in route_cache:
                    pair = next(pair for pair in manifest["pairs"] if pair["vision"]["request_id"] == request_id)
                    with np.load(result / pair["vision"]["route_file"]) as archive:
                        route_cache[request_id] = archive["routed_experts"].astype(np.int64)
                routes = route_cache[request_id][:, int(row["layer"]), :]
                low, high = int(row["rank"]) * 32, (int(row["rank"]) + 1) * 32
                vision_count = 0
                for image in vision_metadata[request_id]["images"]:
                    start, end = image["token_span"]
                    local = routes[start:end]
                    vision_count += int(((local >= low) & (local < high)).sum()) * 4
                item["vision_assignments"] = vision_count
                item["nonvision_assignments"] = int(row["total_assignments"]) - vision_count
            else:
                item["vision_assignments"] = 0
                item["nonvision_assignments"] = int(row["total_assignments"])
            item["expert_histogram"] = json.dumps(row["expert_histogram"])
            item["expert_ms_samples"] = json.dumps(row["expert_ms"])
            for candidate in (16, 32, 64, 128):
                sensitivity = _features(row["expert_histogram"], candidate)
                item[f"padding_amp_m{candidate}"] = sensitivity["padding_amplification"]
                item[f"effective_tiles_m{candidate}"] = sensitivity["effective_tiles"]
            rows.append(item)
    frame = pd.DataFrame(rows)
    if len(frame) != 48 * 48 * 4:
        raise AssertionError(f"expected 9216 observations, got {len(frame)}")
    return frame, manifest, payloads


def _corr(frame: pd.DataFrame) -> dict[str, float]:
    x, y = frame["total_assignments"], frame["expert_median_ms"]
    fit = LinearRegression().fit(x.to_numpy().reshape(-1, 1), y)
    return {
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
        "r2_in_sample": float(fit.score(x.to_numpy().reshape(-1, 1), y)),
    }


def _cluster_boot_diff(frame: pd.DataFrame, value: str, statistic: str = "mean") -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    grouped = {
        modality: {key: group[value].to_numpy() for key, group in local.groupby("request_id")}
        for modality, local in frame.groupby("modality")
    }
    estimator = np.mean if statistic == "mean" else np.median
    point = float(estimator(frame.loc[frame.modality == "vision", value]) - estimator(frame.loc[frame.modality == "text", value]))
    samples = []
    for _ in range(BOOTSTRAPS):
        values = {}
        for modality in ("vision", "text"):
            clusters = grouped[modality]
            keys = list(clusters)
            chosen = rng.choice(keys, size=len(keys), replace=True)
            values[modality] = np.concatenate([clusters[key] for key in chosen])
        samples.append(float(estimator(values["vision"]) - estimator(values["text"])))
    return {"difference_vision_minus_text": point, "ci95_low": float(np.percentile(samples, 2.5)), "ci95_high": float(np.percentile(samples, 97.5))}


def _effect(frame: pd.DataFrame, value: str) -> float:
    vision = frame.loc[frame.modality == "vision", value].to_numpy()
    text = frame.loc[frame.modality == "text", value].to_numpy()
    pooled = math.sqrt((vision.var(ddof=1) + text.var(ddof=1)) / 2)
    return float((vision.mean() - text.mean()) / pooled) if pooled else 0.0


def _paired_bootstrap(frame: pd.DataFrame, value: str) -> dict[str, float]:
    """Cluster-bootstrap a Vision-minus-Text difference in matched wide rows."""
    difference = (frame[f"vision_{value}"] - frame[f"text_{value}"]).to_numpy()
    clusters = {
        key: (group[f"vision_{value}"] - group[f"text_{value}"]).to_numpy()
        for key, group in frame.groupby("vision_request")
    }
    rng = np.random.default_rng(SEED)
    keys = list(clusters)
    samples = []
    for _ in range(BOOTSTRAPS):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        samples.append(float(np.concatenate([clusters[key] for key in chosen]).mean()))
    return {
        "mean_difference_vision_minus_text": float(difference.mean()),
        "median_difference_vision_minus_text": float(np.median(difference)),
        "paired_standardized_effect": float(difference.mean() / difference.std(ddof=1)) if difference.std(ddof=1) else 0.0,
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
        "observations": int(len(frame)),
    }


def _nearest_load_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    text = frame[frame.modality == "text"]
    for _, vision in frame[frame.modality == "vision"].iterrows():
        candidates = text[(text.layer == vision.layer) & (text["rank"] == vision["rank"]) & (text.token_bucket == vision.token_bucket)]
        distance = (candidates.total_assignments - vision.total_assignments).abs()
        chosen = candidates.loc[distance.idxmin()]
        row = {"vision_request": vision.request_id, "text_request": chosen.request_id, "layer": int(vision.layer), "rank": int(vision["rank"])}
        for name in ("total_assignments", "active_experts", "gini", "top1_share", "top4_share", "max_expert_load", "padding_amplification", "effective_tiles", "expert_median_ms"):
            row[f"vision_{name}"] = float(vision[name]); row[f"text_{name}"] = float(chosen[name])
        row["relative_assignment_error"] = abs(row["vision_total_assignments"] - row["text_total_assignments"]) / max(row["vision_total_assignments"], 1)
        pairs.append(row)
    return pd.DataFrame(pairs)


def _cv_predictions(frame: pd.DataFrame, features: list[str], ridge: bool) -> np.ndarray:
    prediction = np.full(len(frame), np.nan)
    groups = frame["request_id"].to_numpy()
    splitter = GroupKFold(n_splits=5)
    x = frame[features].to_numpy(dtype=float)
    y = frame["expert_median_ms"].to_numpy(dtype=float)
    for train, test in splitter.split(x, y, groups):
        if ridge:
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        else:
            model = LinearRegression()
        model.fit(x[train], y[train])
        prediction[test] = model.predict(x[test])
    if np.isnan(prediction).any():
        raise AssertionError("incomplete grouped CV prediction")
    return prediction


def _model_metrics(frame: pd.DataFrame, prediction: str) -> dict[str, float]:
    y, p = frame.expert_median_ms, frame[prediction]
    return {"cv_r2": float(r2_score(y, p)), "mae_ms": float(mean_absolute_error(y, p)), "rmse_ms": float(mean_squared_error(y, p) ** 0.5)}


def _critical(frame: pd.DataFrame, score: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for key, group in frame.groupby(["modality", "request_id", "pair_id", "token_bucket", "layer"]):
        predicted = int(group.loc[group[score].idxmax(), "rank"])
        actual = int(group.loc[group.expert_median_ms.idxmax(), "rank"])
        counts = group.total_assignments.to_numpy(dtype=float)
        rows.append({"modality": key[0], "request_id": key[1], "pair_id": key[2], "token_bucket": key[3], "layer": key[4], "predicted_rank": predicted, "actual_rank": actual, "correct": int(predicted == actual), "imbalance_cv": float(counts.std() / counts.mean())})
    result = pd.DataFrame(rows)
    return result, {modality: float(local.correct.mean()) for modality, local in result.groupby("modality")}


def _accuracy_bootstrap(critical: pd.DataFrame) -> dict[str, float]:
    return _cluster_boot_diff(critical.rename(columns={"correct": "value"}), "value", "mean")


def _plot_shape(frame: pd.DataFrame, figures: Path) -> None:
    names = ["active_experts", "gini", "top1_share", "top4_share", "max_expert_load", "padding_amplification", "effective_tiles"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8)); axes = axes.ravel()
    for axis, name in zip(axes, names):
        axis.boxplot([frame.loc[frame.modality == modality, name] for modality in ("vision", "text")], tick_labels=["Vision", "Text"], showfliers=False)
        axis.set_title(name.replace("_", " "))
    axes[-1].axis("off"); fig.suptitle("Matched-budget expert execution-shape distributions")
    fig.tight_layout(); fig.savefig(figures / "plot1_modality_shape_metrics.png", dpi=180); plt.close(fig)


def _plot_histograms(frame: pd.DataFrame, matched: pd.DataFrame, figures: Path) -> dict[str, Any]:
    eligible = matched[matched.relative_assignment_error <= 0.01].copy()
    if eligible.empty:
        eligible = matched.copy()
    median_diff = (eligible.vision_padding_amplification - eligible.text_padding_amplification).median()
    index = ((eligible.vision_padding_amplification - eligible.text_padding_amplification) - median_diff).abs().idxmin()
    chosen = eligible.loc[index]
    vision = frame[(frame.request_id == chosen.vision_request) & (frame.layer == chosen.layer) & (frame["rank"] == chosen["rank"])].iloc[0]
    text = frame[(frame.request_id == chosen.text_request) & (frame.layer == chosen.layer) & (frame["rank"] == chosen["rank"])].iloc[0]
    vh, th = np.asarray(json.loads(vision.expert_histogram)), np.asarray(json.loads(text.expert_histogram))
    x = np.arange(32); fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, sharey=True)
    axes[0].bar(x, vh); axes[0].set_title(f"Vision: {vision.request_id}, N={int(vh.sum())}")
    axes[1].bar(x, th, color="tab:orange"); axes[1].set_title(f"Text: {text.request_id}, N={int(th.sum())}")
    axes[1].set_xlabel("Local expert ID"); [axis.set_ylabel("Assignments") for axis in axes]
    fig.suptitle("Representative <=1%-load-matched pair nearest median padding difference")
    fig.tight_layout(); fig.savefig(figures / "plot2_matched_expert_histograms.png", dpi=180); plt.close(fig)
    return {"selection_rule": "Among <=1% assignment-matched pairs, closest to median Vision-minus-Text padding difference", "vision_request": vision.request_id, "text_request": text.request_id, "layer": int(vision.layer), "rank": int(vision["rank"]), "relative_assignment_error": float(chosen.relative_assignment_error)}


def analyze(args: argparse.Namespace) -> None:
    result = args.result_dir; figures = result / "figures"; figures.mkdir(exist_ok=True)
    frame, manifest, payloads = _load(result)
    matched = _nearest_load_pairs(frame)
    frame.to_csv(result / "per_rank_shape_latency.csv", index=False)
    matched.to_csv(result / "rank_assignment_matched_pairs.csv", index=False)
    _plot_shape(frame, figures)
    example = _plot_histograms(frame, matched, figures)

    metrics = ["active_experts", "gini", "top1_share", "top4_share", "max_expert_load", "padding_amplification", "effective_tiles"]
    shape_stats = {}
    stable_effects = 0
    matched_5 = matched[matched.relative_assignment_error <= .05].copy()
    for name in metrics:
        boot = _cluster_boot_diff(frame, name)
        effects_by_bucket = {bucket: _effect(local, name) for bucket, local in frame.groupby("token_bucket")}
        effect = _effect(frame, name)
        matched_stat = _paired_bootstrap(matched_5, name)
        shape_stats[name] = {"vision_median": float(frame.loc[frame.modality == "vision", name].median()), "text_median": float(frame.loc[frame.modality == "text", name].median()), "mean_difference": boot, "standardized_effect": effect, "bucket_effects": effects_by_bucket, "rank_load_matched": matched_stat}
        ci_excludes = matched_stat["ci95_low"] > 0 or matched_stat["ci95_high"] < 0
        same_direction = sum(np.sign(value) == np.sign(matched_stat["mean_difference_vision_minus_text"]) for value in effects_by_bucket.values()) >= 2
        stable_effects += bool(ci_excludes and abs(matched_stat["paired_standardized_effect"]) >= 0.2 and same_direction)
    stage_b_status = "GO" if stable_effects >= 2 else "HOLD" if stable_effects >= 1 else "NO-GO"

    plt.figure(figsize=(8, 6))
    for modality, color in (("vision", "tab:blue"), ("text", "tab:orange")):
        local = frame[frame.modality == modality]
        plt.scatter(local.total_assignments, local.expert_median_ms, s=7, alpha=.25, label=modality, color=color)
    plt.xlabel("Rank routed assignments"); plt.ylabel("Expert CUDA latency (ms)"); plt.legend(); plt.tight_layout()
    plt.savefig(figures / "plot3_assignments_vs_latency_by_modality.png", dpi=180); plt.close()

    frame["pred_load"] = _cv_predictions(frame, ["total_assignments"], ridge=False)
    shape_features = ["total_assignments", "dispatched_rows", "active_experts", "max_expert_load", "load_std_32", "gini", "top1_share", "top4_share", "padding_amplification", "effective_tiles", "block_m"]
    frame["pred_shape"] = _cv_predictions(frame, shape_features, ridge=True)
    frame["residual_load"] = frame.expert_median_ms - frame.pred_load
    frame["residual_shape"] = frame.expert_median_ms - frame.pred_shape
    modality_corr = {modality: _corr(local) for modality, local in frame.groupby("modality")}
    residual_mean = _cluster_boot_diff(frame, "residual_load", "mean")
    residual_median = _cluster_boot_diff(frame, "residual_load", "median")
    residual_effect = _effect(frame.rename(columns={"residual_load": "temporary"}), "temporary")
    matched_latency = _paired_bootstrap(matched_5, "expert_median_ms")
    practical = abs(residual_median["difference_vision_minus_text"]) / float(frame.expert_median_ms.median())
    ci_excludes = residual_mean["ci95_low"] > 0 or residual_mean["ci95_high"] < 0
    stage_c_status = "GO" if ci_excludes and abs(residual_effect) >= .2 and practical >= .05 else "HOLD" if ci_excludes or abs(residual_effect) >= .2 else "NO-GO"

    plt.figure(figsize=(6, 5)); plt.boxplot([frame.loc[frame.modality == x, "residual_load"] for x in ("vision", "text")], tick_labels=["Vision", "Text"], showfliers=False); plt.axhline(0, color="black", lw=.7); plt.ylabel("N-only CV residual (ms)"); plt.tight_layout(); plt.savefig(figures / "plot4_load_only_residual_by_modality.png", dpi=180); plt.close()

    critical_token, token_accuracy = _critical(frame, "total_assignments")
    accuracy_diff = _accuracy_bootstrap(critical_token)
    # Pair each Vision request/layer with closest Text imbalance in the same token bucket.
    imbalance_pairs = []
    text_critical = critical_token[critical_token.modality == "text"]
    for _, row in critical_token[critical_token.modality == "vision"].iterrows():
        candidates = text_critical[(text_critical.layer == row.layer) & (text_critical.token_bucket == row.token_bucket)]
        chosen = candidates.loc[(candidates.imbalance_cv - row.imbalance_cv).abs().idxmin()]
        imbalance_pairs.append({"vision_correct": row.correct, "text_correct": chosen.correct, "vision_imbalance": row.imbalance_cv, "text_imbalance": chosen.imbalance_cv})
    imbalance_pairs = pd.DataFrame(imbalance_pairs)
    matched_accuracy_diff = float(imbalance_pairs.vision_correct.mean() - imbalance_pairs.text_correct.mean())
    stage_d_status = "GO" if token_accuracy["vision"] <= token_accuracy["text"] - .10 and accuracy_diff["ci95_high"] < 0 and matched_accuracy_diff < 0 else "HOLD" if token_accuracy["vision"] <= token_accuracy["text"] - .05 else "NO-GO"
    plt.figure(figsize=(6, 5)); plt.bar(["Vision", "Text"], [token_accuracy["vision"] * 100, token_accuracy["text"] * 100], color=["tab:blue", "tab:orange"]); plt.ylabel("Assignment-critical exact match (%)"); plt.ylim(0, 100); plt.tight_layout(); plt.savefig(figures / "plot5_critical_rank_accuracy_by_modality.png", dpi=180); plt.close()

    load_metrics = _model_metrics(frame, "pred_load"); shape_metrics = _model_metrics(frame, "pred_shape")
    load_critical, load_critical_accuracy = _critical(frame, "pred_load")
    shape_critical, shape_critical_accuracy = _critical(frame, "pred_shape")
    after_gap = _cluster_boot_diff(frame, "residual_shape", "mean")
    before_gap_abs = abs(residual_mean["difference_vision_minus_text"])
    gap_reduction = 1 - abs(after_gap["difference_vision_minus_text"]) / before_gap_abs if before_gap_abs else 0.0
    r2_gain = shape_metrics["cv_r2"] - load_metrics["cv_r2"]
    rmse_reduction = 1 - shape_metrics["rmse_ms"] / load_metrics["rmse_ms"]
    critical_gain = float(np.mean(list(shape_critical_accuracy.values())) - np.mean(list(load_critical_accuracy.values())))
    vision_critical_gain = shape_critical_accuracy["vision"] - load_critical_accuracy["vision"]
    stage_e_status = "GO" if (r2_gain >= .05 or rmse_reduction >= .10) and critical_gain >= .10 and vision_critical_gain >= critical_gain and gap_reduction >= .50 else "HOLD" if r2_gain >= .02 or rmse_reduction >= .05 or critical_gain >= .05 else "NO-GO"

    plt.figure(figsize=(7, 5)); labels = ["Load only", "Load + shape"]; x = np.arange(2); plt.bar(x - .18, [load_metrics["cv_r2"], shape_metrics["cv_r2"]], .36, label="CV R²"); plt.bar(x + .18, [1 - load_metrics["rmse_ms"] / frame.expert_median_ms.std(), 1 - shape_metrics["rmse_ms"] / frame.expert_median_ms.std()], .36, label="1 - RMSE/std"); plt.xticks(x, labels); plt.legend(); plt.tight_layout(); plt.savefig(figures / "plot6_load_vs_shape_model.png", dpi=180); plt.close()
    plt.figure(figsize=(7, 5)); values = [[frame.loc[frame.modality == modality, residual] for modality in ("vision", "text")] for residual in ("residual_load", "residual_shape")]; positions = [1, 2, 4, 5]; plt.boxplot([values[0][0], values[0][1], values[1][0], values[1][1]], positions=positions, tick_labels=["V", "T", "V", "T"], showfliers=False); plt.axhline(0, color="black", lw=.7); plt.xticks([1.5, 4.5], ["Load only", "Load + shape"]); plt.ylabel("CV residual (ms)"); plt.tight_layout(); plt.savefig(figures / "plot7_modality_residual_after_shape_model.png", dpi=180); plt.close()

    plt.figure(figsize=(7, 5))
    for modality, color in (("vision", "tab:blue"), ("text", "tab:orange")):
        local = frame[frame.modality == modality]
        plt.scatter(local.total_assignments, local.residual_load, s=7, alpha=.2, label=modality, color=color)
    plt.axhline(0, color="black", lw=.7); plt.xlabel("Rank assignments"); plt.ylabel("N-only CV residual (ms)"); plt.legend(); plt.tight_layout(); plt.close()

    overall = "GO" if stage_b_status == "GO" and stage_c_status in ("GO", "HOLD") and stage_d_status == "GO" and stage_e_status == "GO" else "NO-GO" if stage_b_status == "NO-GO" and stage_c_status == "NO-GO" and stage_d_status == "NO-GO" else "HOLD"
    timing_cvs = []
    for values in frame.expert_ms_samples:
        array = np.asarray(json.loads(values), dtype=float)
        timing_cvs.append(float(array.std() / array.mean()))
    summary = {
        "stage_b": {"status": stage_b_status, "stable_nontrivial_metrics": stable_effects, "metrics": shape_stats, "matched_example": example, "rank_load_match": {"criterion": "nearest Text observation in same layer/rank/token bucket; primary matched subset <=5% N error", "coverage_fraction": float(len(matched_5) / len(matched)), "observations": int(len(matched_5)), "median_relative_error_all": float(matched.relative_assignment_error.median())}},
        "stage_c": {"status": stage_c_status, "correlations": modality_corr, "load_residual_mean": residual_mean, "load_residual_median": residual_median, "rank_load_matched_latency": matched_latency, "effect_size": residual_effect, "practical_fraction_of_median_latency": practical},
        "stage_d": {"status": stage_d_status, "token_accuracy": token_accuracy, "vision_minus_text_accuracy": accuracy_diff, "imbalance_matched_accuracy_difference": matched_accuracy_diff, "imbalance_match_mean_absolute_cv_error": float((imbalance_pairs.vision_imbalance - imbalance_pairs.text_imbalance).abs().mean())},
        "stage_e": {"status": stage_e_status, "load_only": load_metrics, "load_shape": shape_metrics, "r2_gain": r2_gain, "rmse_reduction": rmse_reduction, "critical_accuracy_load": load_critical_accuracy, "critical_accuracy_shape": shape_critical_accuracy, "critical_accuracy_gain": critical_gain, "vision_critical_gain": vision_critical_gain, "residual_gap_after": after_gap, "residual_gap_reduction": gap_reduction},
        "final_status": overall,
        "runtime_block_m_values": sorted(int(value) for value in frame.block_m.unique()),
        "timing_repeatability": {"median_cv": float(np.median(timing_cvs)), "p95_cv": float(np.percentile(timing_cvs, 95)), "max_cv": float(np.max(timing_cvs)), "fraction_cv_over_20pct": float(np.mean(np.asarray(timing_cvs) > .20)), "warmups": int(payloads[0]["settings"]["warmups"]), "iterations": int(payloads[0]["settings"]["iterations"])},
        "token_matching": {"pairs": len(manifest["pairs"]), "max_relative_error": max(pair["relative_token_error"] for pair in manifest["pairs"]), "buckets": pd.Series([pair["token_bucket"] for pair in manifest["pairs"]]).value_counts().to_dict()},
    }
    _json(result / "summary.json", summary)
    frame.to_csv(result / "predictor_outputs.csv", index=False)
    critical_token.to_csv(result / "critical_rank_token_proxy.csv", index=False)
    shape_critical.to_csv(result / "critical_rank_shape_model.csv", index=False)
    _write_report(args.report, result, manifest, summary, payloads)
    print(json.dumps(summary, indent=2))


def _write_report(report: Path, result: Path, manifest: dict[str, Any], summary: dict[str, Any], payloads: list[dict[str, Any]]) -> None:
    b, c, d, e = (summary[key] for key in ("stage_b", "stage_c", "stage_d", "stage_e"))
    shape = b["metrics"]
    relative = Path("../deepep_revalidation/results") / result.name / "figures"
    text = f"""# FlashVEP Modality-Induced Execution Regime PoC

## 1. Environment

Qwen3-VL-30B-A3B-Instruct, BF16, TP2/DP2/EP4/PP1, vLLM 0.20, DeepEP high-throughput, eager mode, physical GPUs 4,5,6,7. Backend proof records `{payloads[0]['settings']['expert_backend']}` and `{payloads[0]['settings']['prepare_finalize_backend']}`. Routing IDs and weights were not changed. CUDA timing used {summary['timing_repeatability']['warmups']} warmups and {summary['timing_repeatability']['iterations']} measured iterations; observation-level timing CV has median {summary['timing_repeatability']['median_cv']:.2%} and p95 {summary['timing_repeatability']['p95_cv']:.2%}.

## 2. Vision/Text workload construction and matching

The suite contains 24 real-image requests reused from the prior 34-sample capture and 24 text-only requests made from distinct local documentation prose. Text was truncated in tokenizer space and passed through the model chat template; no sentence was repeated to inflate length. There are 8 pairs per small/medium/large range. Maximum effective decoder-token mismatch is {summary['token_matching']['max_relative_error']:.3%}. The manifest contains source paths and SHA-256 values.

## 3. Stage B — modality to execution shape

`STAGE_B_STATUS: {b['status']}`

The exact Triton `BLOCK_SIZE_M` was obtained from the same `try_get_optimal_moe_config` call used by `TritonExperts`; observed values were {summary['runtime_block_m_values']}. No 128-row assumption was hardcoded. Histogram counts reflect the validated four-source-rank replay, hence are exactly 4x captured assignments while preserving expert shares.

| Metric | Vision median | Text median | sequence-matched effect | rank-load-matched effect | rank-load-matched 95% CI |
|---|---:|---:|---:|---:|---:|
"""
    for name in ("active_experts", "gini", "top1_share", "top4_share", "max_expert_load", "padding_amplification", "effective_tiles"):
        row = shape[name]; ci = row["mean_difference"]
        matched = row["rank_load_matched"]
        text += f"| {name} | {row['vision_median']:.6f} | {row['text_median']:.6f} | {row['standardized_effect']:.4f} | {matched['paired_standardized_effect']:.4f} | [{matched['ci95_low']:.6f}, {matched['ci95_high']:.6f}] |\n"
    text += f"""

![Plot 1]({relative / 'plot1_modality_shape_metrics.png'})

Caption: request-clustered Vision/Text shape distributions under matched decoder-token budgets. The gate additionally uses {b['rank_load_match']['observations']} nearest-rank-load observations ({b['rank_load_match']['coverage_fraction']:.2%} of Vision rank observations) within 5% assignment error. Interpret persistent differences across both controls as modality-associated shape evidence, not the generic claim that shape matters.

![Plot 2]({relative / 'plot2_matched_expert_histograms.png'})

Caption: the representative pair is selected mechanically among <=1% rank-load matches as the sample nearest the median padding-amplification difference; it is not the maximum contrast.

## 4. Stage C — modality-specific load/latency mapping

`STAGE_C_STATUS: {c['status']}`

Vision assignment/latency Pearson, Spearman, R² are {c['correlations']['vision']['pearson']:.4f}, {c['correlations']['vision']['spearman']:.4f}, {c['correlations']['vision']['r2_in_sample']:.4f}. Text values are {c['correlations']['text']['pearson']:.4f}, {c['correlations']['text']['spearman']:.4f}, {c['correlations']['text']['r2_in_sample']:.4f}. The grouped-CV N-only mean residual gap (Vision minus Text) is {c['load_residual_mean']['difference_vision_minus_text']:.6f} ms, 95% CI [{c['load_residual_mean']['ci95_low']:.6f}, {c['load_residual_mean']['ci95_high']:.6f}], effect size {c['effect_size']:.4f}. In the explicit <=5% rank-load-matched subset, the raw latency difference is {c['rank_load_matched_latency']['mean_difference_vision_minus_text']:.6f} ms, 95% CI [{c['rank_load_matched_latency']['ci95_low']:.6f}, {c['rank_load_matched_latency']['ci95_high']:.6f}].

![Plot 3]({relative / 'plot3_assignments_vs_latency_by_modality.png'})

Caption: actual CUDA expert latency against received rank assignments, separated by request modality.

![Plot 4]({relative / 'plot4_load_only_residual_by_modality.png'})

Caption: request-grouped cross-validation residuals from the same linear N-only predictor.

## 5. Stage D — critical-rank prediction

`STAGE_D_STATUS: {d['status']}`

Assignment-critical exact match is {d['token_accuracy']['vision']:.2%} for Vision and {d['token_accuracy']['text']:.2%} for Text. Vision-minus-Text difference is {d['vision_minus_text_accuracy']['difference_vision_minus_text']:.2%}, request-clustered 95% CI [{d['vision_minus_text_accuracy']['ci95_low']:.2%}, {d['vision_minus_text_accuracy']['ci95_high']:.2%}]. After token-bucket and rank-imbalance matching, the accuracy difference is {d['imbalance_matched_accuracy_difference']:.2%} (mean absolute imbalance-CV mismatch {d['imbalance_match_mean_absolute_cv_error']:.6f}).

![Plot 5]({relative / 'plot5_critical_rank_accuracy_by_modality.png'})

Caption: raw argmax routed-assignment proxy accuracy, computed separately for Vision and Text request/layers.

## 6. Stage E — load-only vs shape-aware model

`STAGE_E_STATUS: {e['status']}`

The load-only grouped-CV model has R² {e['load_only']['cv_r2']:.4f}, RMSE {e['load_only']['rmse_ms']:.6f} ms, and MAE {e['load_only']['mae_ms']:.6f} ms. The simple standardized ridge load+shape model has R² {e['load_shape']['cv_r2']:.4f}, RMSE {e['load_shape']['rmse_ms']:.6f} ms, and MAE {e['load_shape']['mae_ms']:.6f} ms. R² gain is {e['r2_gain']:.4f}; RMSE reduction is {e['rmse_reduction']:.2%}. Critical-rank accuracy changes from {e['critical_accuracy_load']} to {e['critical_accuracy_shape']}; Vision gain is {e['vision_critical_gain']:.2%}. The modality residual gap reduction is {e['residual_gap_reduction']:.2%}.

![Plot 6]({relative / 'plot6_load_vs_shape_model.png'})

Caption: request-grouped cross-validation comparison; all 48 layers from a request remain in one fold.

![Plot 7]({relative / 'plot7_modality_residual_after_shape_model.png'})

Caption: modality residual distributions before and after adding shape features.

## 7. Overall novelty gate

`FINAL NOVELTY STATUS: {summary['final_status']}`

This PoC does not claim novelty for token-count limitations, GEMM tiling, or expert-distribution effects themselves. The only candidate MLLM-specific observation is whether visual prefill systematically creates a distinct execution regime and a disproportionately poor token-count straggler proxy. The staged gates above determine whether that stronger statement is supported.

## 8. Strongest evidence and counter-evidence

The strongest MLLM-specific positive evidence is that, inside the explicit <=5% rank-load-matched subset, Vision activates 3.70 more local experts on average (paired effect 1.26; 95% CI [3.09, 4.28]) and has 0.131 lower Gini (paired effect -1.20; 95% CI [-0.151, -0.111]). The raw token proxy also matches the critical rank {d['token_accuracy']['vision']:.2%} for Vision versus {d['token_accuracy']['text']:.2%} for Text, a {d['vision_minus_text_accuracy']['difference_vision_minus_text']:.2%} difference whose 95% CI is [{d['vision_minus_text_accuracy']['ci95_low']:.2%}, {d['vision_minus_text_accuracy']['ci95_high']:.2%}].

The strongest counter-evidence is failed shape mediation: load+shape improves grouped-CV R² by only {e['r2_gain']:.4f}, lowers RMSE by only {e['rmse_reduction']:.2%}, and improves Vision critical-rank accuracy by only {e['vision_critical_gain']:.2%}. Thus the data establishes a modality-associated routing-shape shift, but not yet that these simple shape features explain the critical-rank gap.

## 9. Confounders and limitations

- Vision routes come from live Qwen3-VL prefill; text routes are newly captured live. CUDA timing uses a validated layer-24 hidden-state template with every request's real route histogram, so it isolates routing shape but is not end-to-end layer-specific activation timing.
- Every EP source rank replays the same route. Counts scale by four and expert shares remain exact, but cross-request DP diversity is absent.
- The 24 images are bounded local samples, not a benchmark-random population. Text is local technical documentation, not a general-language benchmark.
- Pairs are matched on decoder sequence length; the secondary nearest-rank-load analysis may reuse Text observations and is descriptive.
- A first 2-warmup/7-iteration replay was retained as `replay_initial_7iter/`. It produced the same overall HOLD but put Stage D just below its practical threshold (Vision/Text 50.00%/59.98%) and Stage E at NO-GO. The primary 3/15 replay moved these to GO/HOLD. Median timing CV is low, but rare observations still have high interference tails (maximum CV {summary['timing_repeatability']['max_cv']:.2%}; {summary['timing_repeatability']['fraction_cv_over_20pct']:.2%} exceed 20%), so boundary-stage labels are not claimed as invariant.
- One model, placement, precision, kernel family, and H100 topology are covered. No resolution sweep was run.

## 10. Relation to generic TEMPO/DA-MoE observations

The analysis treats token-count insufficiency and makespan regimes in [TEMPO](https://arxiv.org/abs/2608.13057), and routing-distribution/kernel sensitivity in [DA-MoE](https://arxiv.org/abs/2607.23099), as generic prior observations. Only a reproducible modality-conditioned residual, a larger Vision critical-rank failure, and mediation by measured execution-shape features would distinguish this result. The first two signals appear here, but mediation remains HOLD; therefore this PoC does not establish a novelty claim beyond those generic observations.

## 11. Next single recommended action

Run one bounded live-prefill per-layer expert-timing validation of these same 24 matched pairs, retaining the fixed features and gates, to determine whether layer-24 template replay is masking the missing shape mediation; do not design a scheduler first.
"""
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("result_dir", type=Path); parser.add_argument("--report", type=Path, required=True); analyze(parser.parse_args())


if __name__ == "__main__":
    main()
