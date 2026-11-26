"""Reproduce the isolated absolute/incremental costs and causal P-ready predictor."""
import argparse
from collections import Counter
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from ..mllm_epd_h200_modality_aware_dynamic_split.policy import attention_pairs, consume_budget
from .cut_slack import analyze_cohort, read_jsonl
from .slack_policy import (
    BUDGET_CAPS_MS, CALLBACK_TO_D_SUBMIT_MS, SlackPolicy, cost_features,
    predict, ready_features, suffix_class, upper_cost, within_bounds,
)


def fit(rows, feature, target, lower=10, upper=95):
    values = np.array([feature(row) for row in rows], dtype=float)
    x = np.column_stack((np.ones(len(rows)), values))
    actual = np.array([row[target] for row in rows], dtype=float)
    penalty = np.eye(x.shape[1]) * .1
    penalty[0, 0] = 0.
    coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ actual)
    residuals = actual - x @ coefficients
    return {
        "intercept": float(coefficients[0]), "weights": coefficients[1:].tolist(),
        "lower_residual_ms": min(0., float(np.percentile(residuals, lower))),
        "upper_residual_ms": max(0., float(np.percentile(residuals, upper))),
    }


def errors(predicted, actual, upper=None, lower=None):
    if not predicted:
        return {"n": 0}
    signed = np.array(predicted) - np.array(actual)
    result = {
        "n": len(predicted), "MAE_ms": float(np.mean(np.abs(signed))),
        "median_signed_error_ms": float(np.median(signed)),
        "p95_underprediction_ms": float(np.percentile(-signed, 95)),
        "p95_overprediction_ms": float(np.percentile(signed, 95)),
    }
    if upper is not None:
        result["upper_cost_undercoverage"] = sum(p < t for p, t in zip(upper, actual, strict=True))
    if lower is not None:
        result["lower_slack_overcoverage"] = sum(p > t for p, t in zip(lower, actual, strict=True))
    return result


def summarize_fit(model, rows, feature, target, positive=False):
    point = [predict(model, feature(row)) for row in rows]
    if positive:
        point = [max(0., value) for value in point]
    actual = [row[target] for row in rows]
    upper = [value + model["upper_residual_ms"] for value in point]
    lower = [max(0., value + model["lower_residual_ms"]) if positive else value + model["lower_residual_ms"]
             for value in point]
    return errors(point, actual, upper, lower)


def validate_suffix_margins(model, train, validation, feature, target):
    classes = {}
    for row in train:
        key = suffix_class(row["length"] - row["prefix"])
        classes.setdefault(key, []).append(row[target] - predict(model, feature(row)))
    margins = {key: max(0., float(np.percentile(residuals, 95)))
               for key, residuals in classes.items() if len(residuals) >= 20}
    candidate = model | {"upper_margin_by_suffix": margins}
    actual = [row[target] for row in validation]
    point = [predict(model, feature(row)) for row in validation]
    upper = [upper_cost(candidate, feature(row), row["length"] - row["prefix"]) for row in validation]
    result = errors(point, actual, upper=upper)
    if not validation:
        return
    model["segmented_upper_validation"] = result | {"training_margins_ms": margins}
    limit = model["heldout_within_bounds"]["upper_cost_undercoverage"]
    if result["upper_cost_undercoverage"] <= limit:
        model["upper_margin_by_suffix"] = margins


def heldout_upper_misses(model):
    scope = "segmented_upper_validation" if "upper_margin_by_suffix" in model else "heldout_within_bounds"
    return model[scope]["upper_cost_undercoverage"]


def cost_data(measures):
    if {row["transport"] for row in measures} != {"inline_redirect"}:
        raise ValueError("Require measured common inline cost calibration")
    transport = {row.get("transport_variant", "whole") for row in measures}
    regime = {row.get("numeric_regime", "bi0") for row in measures}
    preissue_kind = {row.get("preissue_crossing") == "True" for row in measures}
    if len(transport) != 1 or len(regime) != 1 or len(preissue_kind) != 1:
        raise ValueError("Separate transport and numerical regimes before fitting")
    preissue, = preissue_kind
    baseline = {(row["request_id"], row["repeat"]): row for row in measures if int(row["suffix_tokens"]) == 0}
    for row in measures:
        if int(row["suffix_tokens"]) == 0:
            continue
        if preissue and row["D_transport_route"] != "NO_RPC_VERIFIED":
            continue
        native = baseline[row["request_id"], row["repeat"]]
        full, prefix = int(row["input_tokens"]), int(row["prefix_tokens"])
        yield {
            "trial": row["trial"], "request_id": row["request_id"], "fit_group": row["fit_group"],
            "transport": next(iter(transport)), "numeric_regime": next(iter(regime)),
            "preissue_crossing": preissue,
            "crosses_image_span": bool(int(row["cross_boundary_image_count"])),
            "length": full, "prefix": prefix, "absolute": float(row["D_first_token_http_ms"]),
            "incremental": float(row["D_first_token_http_ms"]) - float(native["D_first_token_http_ms"]),
            "P_service_saved": float(native["P_source_http_ms"]) - float(row["P_source_http_ms"]),
        }


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cost_calibration(path):
    with path.open() as source:
        measures = [row for row in csv.DictReader(source) if row["series"] == "cost"]
    pairs = list(cost_data(measures))
    train = [row for row in pairs if row["fit_group"] == "fit"]
    heldout = [row for row in pairs if row["fit_group"] == "validation"]
    train_ids, heldout_ids = ({row["request_id"] for row in subset} for subset in (train, heldout))
    if train_ids & heldout_ids or not train_ids or not heldout_ids:
        raise ValueError("Cost calibration must have request-disjoint validation")

    def feature(row):
        return cost_features(row["length"], row["prefix"])

    receipt = {
        "schema": "split_cost_v1", "evidence": "ISOLATED_GPU_MEASUREMENT", "calibration_SHA256": digest(path),
        "transport": "inline_" + train[0]["transport"], "numeric_regime": train[0]["numeric_regime"],
        "preissue_crossing": train[0]["preissue_crossing"],
        "observed_D_transport_routes": dict(Counter(row["D_transport_route"] for row in measures
                                                    if row.get("D_transport_route"))),
        "boundary_to_D_submit_allowance_ms": CALLBACK_TO_D_SUBMIT_MS,
        "scope": "Common-inline isolated B1. Absolute is full D HTTP to one greedy token. "
                 "Incremental is paired split minus native. P-service saved is native P HTTP minus "
                 "prefix P HTTP at fixed request/repeat. These host envelopes include other work.",
        "training_points": len(train), "heldout_points": len(heldout),
        "training_requests": sorted(train_ids), "heldout_requests": sorted(heldout_ids),
        "bounds": [[min(series), max(series)] for series in (
            [r["length"] for r in train], [r["length"] - r["prefix"] for r in train],
            [attention_pairs(r["prefix"], r["length"] - r["prefix"]) for r in train])],
    }
    for target in ("absolute", "incremental", "P_service_saved"):
        model = fit(train, feature, target, lower=10, upper=95)
        selected = [r for r in heldout if within_bounds(receipt, r["length"], r["prefix"])]
        model["heldout_all"] = summarize_fit(model, heldout, feature, target, positive=target == "absolute")
        model["heldout_within_bounds"] = summarize_fit(model, selected, feature, target, positive=target == "absolute")
        if target in ("absolute", "incremental"):
            validate_suffix_margins(model, train, selected, feature, target)
        receipt[target] = model
    # Fit on identical geometry and repeat splits. The binary feature captures
    # an additional descriptor/clone even when slice payload is one extra row.
    def modality_feature(row):
        return (*feature(row), float(row["crosses_image_span"]))

    supports_boundary_ablation = all(len({row["crosses_image_span"] for row in subset}) == 2
                                     for subset in (train, heldout))
    modality = {"enabled": False, "feature": "crosses_image_span", "same_feature_budget": True,
                "partition_scope": "No additional candidate cuts. Only replacement predicted cost."}
    if supports_boundary_ablation:
        enable = True
        for target in ("absolute", "incremental"):
            model = fit(train, modality_feature, target)
            heldout_rows = [r for r in heldout if within_bounds(receipt, r["length"], r["prefix"])]
            model["heldout_within_bounds"] = summarize_fit(
                model, heldout_rows, modality_feature, target, positive=target == "absolute")
            validate_suffix_margins(model, train, heldout_rows, modality_feature, target)
            before, after = receipt[target]["heldout_within_bounds"], model["heldout_within_bounds"]
            enable &= (after["MAE_ms"] < before["MAE_ms"]
                       and after["p95_underprediction_ms"] <= before["p95_underprediction_ms"]
                       and heldout_upper_misses(model) <= heldout_upper_misses(receipt[target]))
            modality[target] = model
        modality["enabled"] = enable
    receipt["modality_cost"] = modality
    return receipt, pairs


def slice_check(receipt, trials_path):
    rows = [row for row in read_jsonl(trials_path)
            if row.get("transport_variant", row.get("mode")) == "slice"]
    if not rows:
        raise ValueError("No explicitly identified isolated slice cells")
    model = receipt["absolute"]
    supported = [r for r in rows if within_bounds(receipt, r["input_tokens"], r["prefix_tokens"])]
    predicted = [max(0., predict(model, cost_features(r["input_tokens"], r["prefix_tokens"])))
                 for r in supported]
    actual = [r["D_first_token_http_ms"] for r in supported]
    upper = [x + model["upper_residual_ms"] for x in predicted]
    return {"receipt_SHA256": digest(trials_path), "rows": len(rows),
            "supported_rows": len(supported), "whole_model_on_slice": errors(predicted, actual, upper=upper)}


def timing_features(row):
    progress = row["progress"]
    return ready_features(row["input_tokens"], row["prefix"], progress["competitors"],
                          progress["running"], progress["waiting"], progress["ready_competitors"],
                          progress["batch_size"], progress["batch_tokens"])


def feature_rows(rows):
    return [row for row in rows if row.get("progress") and timing_features(row) is not None
            and row["P_next_other_ready_after_cut_ms"] is not None and not row["drain"]]


def ready_calibration(train, evaluate, manifest, same_start_repeat=(), named_evaluation=None):
    workload = json.loads(manifest.read_text())
    discovery, heldout = set(workload["discovery_ids"]), set(workload["heldout_ids"])
    if discovery & heldout:
        raise ValueError("Frozen workload folds overlap")
    training = [r for r in feature_rows(train) if r["dataset_request_id"] in discovery]
    validation = [r for r in feature_rows(train) if r["dataset_request_id"] in heldout]
    independent = [r for r in feature_rows(evaluate) if r["dataset_request_id"] in heldout]
    repeated = [r for r in feature_rows(same_start_repeat) if r["dataset_request_id"] in heldout]
    groups = {"discovery": training, "same_start_heldout": validation,
              "same_start_repeat_heldout": repeated, "independent_heldout": independent}
    for name, rows in (named_evaluation or {}).items():
        available = feature_rows(rows)
        groups[f"{name}_heldout"] = [r for r in available if r["dataset_request_id"] in heldout]
        groups[f"{name}_discovery"] = [r for r in available if r["dataset_request_id"] in discovery]
    if not training:
        raise ValueError("No trainable stock P-ready data")
    receipt = {
        "schema": "causal_p_ready_v1", "evidence": "DIAGNOSTIC",
        "manifest_SHA256": digest(manifest),
        "training_requests": sorted({r["dataset_request_id"] for r in training}),
        "scope": "Train only on Discovery request IDs from the declared Native stock diagnostic epoch. "
                 "Future native P-ready labels are offline supervision. Online features are P scheduler "
                 "queue and progress, with a hard rejection if another P result is already ready.",
        "quantiles": {"budget_lower_residual": .1, "own_ready_lower_residual": .1},
    }
    for model_name, target in (("other_ready", "P_next_other_ready_after_cut_ms"),
                               ("own_ready", "P_own_native_ready_after_cut_ms")):
        model = fit(training, timing_features, target)
        for name, subset in groups.items():
            model[name] = summarize_fit(model, subset, timing_features, target, positive=True)
            if model_name == "other_ready":
                old = [consume_budget(row["progress"]["competitors"], 80_000., 160.) for row in subset]
                actual = [row[target] for row in subset]
                model[name]["old_80k_160ms"] = errors(old, actual, lower=old)
                model[name]["old_zero_budget"] = sum(value == 0. for value in old)
                retired = [
                    consume_budget(
                        [{"remaining_prompt_tokens": item["prompt_tokens"] - item["retired_prompt_tokens"]}
                         for item in row["progress"]["competitors"]], 80_000., 160.)
                    for row in subset]
                model[name]["retired_work_80k_160ms"] = errors(retired, actual, lower=retired)
                model[name]["retired_work_zero_budget"] = sum(value == 0. for value in retired)
        receipt[model_name] = model
    return receipt, groups


def sensitivity(cost, ready, cohorts, ratios=None, gain_estimate="lower"):
    evaluate = SlackPolicy("b-new", cost, ready, None, 10., True, False, True, gain_estimate=gain_estimate)
    if ratios:
        evaluate = replace(evaluate, ratios=tuple(ratios))
    report = {}
    for name, rows in cohorts.items():
        posthoc_active = name.startswith("posthoc_active_")
        quiet = [row for row in rows if row["candidate_geometry"]
                 and (row["central_ROI"] or posthoc_active and row.get("actual_offload"))
                 and row["D_active_http_at_cut"] == row["D_pending_from_P_ready_at_cut"] == 0
                 and (posthoc_active or not row["oracle_other_D_HTTP_gap_right_censored"])]
        eligible = [row for row in quiet if row.get("D_engine_scheduler_at_cut") is None
                    or ((row.get("D_engine_queue_quiet_gap_ms") or 0.) > 0
                        and (row.get("D_engine_KV_free_blocks_at_cut") or 0) >= 256)]
        report[name] = {}
        for cap in BUDGET_CAPS_MS:
            policy = replace(evaluate, max_idle_budget_ms=cap)
            events = []
            for row in eligible:
                accepted, reason, details = policy.choose(
                    length=row["input_tokens"], prefix=row["prefix"], spans=[], cuts=[],
                    competitors=[], state_observed=True, progress=row["progress"])
                estimated = details["cost_prediction_ms"]
                if estimated is None:
                    continue
                paired_saved = predict(cost["P_service_saved"], cost_features(row["input_tokens"], row["prefix"]))
                paired_gain = (paired_saved - details["incremental_cost_point_ms"]
                               - details["boundary_to_D_submit_allowance_ms"])
                observed = row["oracle_other_D_HTTP_gap_ms"]
                cost_fit = estimated + policy.safety_ms <= observed
                if not cost_fit and row["oracle_other_D_HTTP_gap_right_censored"]:
                    cost_fit = None
                events.append({"dataset_request_id": row["dataset_request_id"], "accepted": accepted,
                    "predicted_fit": details["budget_ms"] >= estimated + policy.safety_ms,
                    "cost_fits_observed_proxy": cost_fit, "stock_boundary_without_hold":
                    row["usable_boundary_without_hold"], "estimated_absolute_ms": estimated,
                    "estimated_net_gain_ms": details["expected_gain_ms"], "reason": reason,
                    "paired_P_service_saved_ms": paired_saved, "paired_P_service_gain_ms": paired_gain,
                    "additional_queue_relief_for_break_even_ms": max(0., -paired_gain),
                    "P_waiting_at_cut": row.get("prefill_waiting"),
                    "observed_other_D_gap_ms": observed, "suffix_tokens": row["suffix"],
                    "actual_offload_under_recorded_policy": row.get("actual_offload"),
                    "actual_boundary_to_D_first_token_ms": row.get("actual_boundary_to_D_first_token_ms"),
                    "actual_D_EC_export_RPC_ms": row.get("D_EC_export_RPC_ms")})
            report[name][str(cap or "adaptive")] = {
                "observed_HTTP_idle_cut_candidates": len(quiet),
                "after_known_D_engine_checks": len(eligible),
                "with_unknown_D_engine": sum(row.get("D_engine_scheduler_at_cut") is None for row in eligible),
                "cost_supported": len(events), "predicted_fit": sum(event["predicted_fit"] for event in events),
                "predicted_fit_with_positive_gain": sum(event["accepted"] for event in events),
                "estimated_fit_with_positive_paired_P_service_gain": sum(
                    event["predicted_fit"] and event["paired_P_service_gain_ms"] > 0. for event in events),
                "predicted_fit_with_positive_gain_without_hold": sum(
                    event["accepted"] and event["stock_boundary_without_hold"] for event in events),
                "oracle_fit_under_cost_upper": sum(event["cost_fits_observed_proxy"] is True for event in events),
                "accepted_but_cost_upper_exceeds_observed": sum(
                    event["accepted"] and event["cost_fits_observed_proxy"] is False for event in events),
                "measured_active_outside_cost_upper": sum(1 for event in events
                    if event["actual_boundary_to_D_first_token_ms"] is not None and
                    event["actual_boundary_to_D_first_token_ms"] > event["estimated_absolute_ms"]),
                "events": events,
            }
    return report


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def epoch(run, cohort):
    path = Path(run)
    environment = json.loads((path / "state" / "environment.json").read_text())
    config = path / "configs" / f"{cohort}.json"
    return {"run": path.name, "cohort": cohort, "numeric_regime": environment.get("numeric_regime"),
            "config_SHA256": digest(config) if config.exists() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated-csv", type=Path)
    parser.add_argument("--stock-train", nargs=2, metavar=("EPOCH", "COHORT"))
    parser.add_argument("--stock-eval", nargs=2, action="append", default=[], metavar=("EPOCH", "COHORT"))
    parser.add_argument("--score-active", nargs=2, action="append", default=[], metavar=("EPOCH", "COHORT"))
    parser.add_argument("--main-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slice-trials", type=Path)
    parser.add_argument("--qualified-ratios", nargs="+", type=float, choices=(.01, .025, .05, .10, .20, .30))
    args = parser.parse_args()
    if bool(args.stock_train) != bool(args.main_manifest):
        parser.error("Stock calibration requires the original Main100 request folds")
    if not args.stock_train and (args.stock_eval or args.score_active):
        parser.error("P-ready evaluation requires a stock training epoch")
    if not (args.stock_train or args.isolated_csv) or not args.isolated_csv and (args.slice_trials or args.score_active):
        parser.error("Choose stock and/or isolated calibration. Cost checks require isolated measurements.")
    args.output.mkdir(parents=True, exist_ok=True)
    cost = None
    if args.isolated_csv:
        cost, pairs = cost_calibration(args.isolated_csv)
        write(args.output / "matched_isolated_cost_pairs.json", pairs)
        if args.slice_trials:
            cost["slice_cross_check"] = slice_check(cost, args.slice_trials)
        write(args.output / "split_cost_model.json", cost)
    if args.stock_train:
        train, _ = analyze_cohort(Path(args.stock_train[0]), args.stock_train[1], "stock")
        training_run = Path(args.stock_train[0]).resolve()
        matched, independent, named = [], [], {}
        for run, cohort in args.stock_eval:
            rows = analyze_cohort(Path(run), cohort, "stock")[0]
            same_start = Path(run).resolve() == training_run
            group = matched if same_start else independent
            group.extend(rows)
            scope = "same_start" if same_start else "independent_start"
            named[f"{scope}_{Path(run).name}_{cohort}"] = rows
        predictors, qualified_rows = ready_calibration(train, independent, args.main_manifest, matched, named)
        for run, cohort in args.score_active:
            active, _ = analyze_cohort(Path(run), cohort, "active")
            qualified_rows[f"posthoc_active_{Path(run).name}_{cohort}"] = [
                row for row in active if row.get("progress") and timing_features(row) is not None]
        predictors["epochs"] = {
            "train": epoch(args.stock_train[0], args.stock_train[1]),
            "same_start_repeated": [epoch(run, cohort) for run, cohort in args.stock_eval
                                    if Path(run).resolve() == training_run],
            "independent": [epoch(run, cohort) for run, cohort in args.stock_eval
                            if Path(run).resolve() != training_run],
        }
        write(args.output / "causal_p_ready_model.json", predictors)
        if cost is not None:
            write(args.output / "budget_cap_sensitivity_oracle.json", {
                "all_geometry": sensitivity(cost, predictors, qualified_rows),
                "requested_qualified_ratios": args.qualified_ratios,
                "qualified_geometry_only": (sensitivity(cost, predictors, qualified_rows, args.qualified_ratios)
                                            if args.qualified_ratios else None),
                "point_gain_geometry_only": (sensitivity(cost, predictors, qualified_rows, args.qualified_ratios,
                                                         gain_estimate="point") if args.qualified_ratios else None),
            })


if __name__ == "__main__":
    main()
