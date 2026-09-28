from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from ..contract import atomic_json, read_json, read_jsonl, write_jsonl_gz


def log_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - float(np.max(values))
    return shifted - np.log(np.exp(shifted).sum())


def damage_metrics(base: np.ndarray, changed: np.ndarray, target: int) -> dict[str, Any]:
    base_lp, changed_lp = log_softmax(base), log_softmax(changed)
    base_p, changed_p = np.exp(base_lp), np.exp(changed_lp)
    middle = np.maximum(0.5 * (base_p + changed_p), 1e-300)
    jsd = 0.5 * float(np.sum(base_p * (base_lp - np.log(middle))))
    jsd += 0.5 * float(np.sum(changed_p * (changed_lp - np.log(middle))))
    drop = float(base_lp[target] - changed_lp[target])
    flip = int(np.argmax(base)) != int(np.argmax(changed))
    return {
        "target_logprob_drop_nats": drop,
        "next_token_jsd": jsd,
        "top1_flip": flip,
        "functional_damage_score": max(0.0, drop) + 2.0 * jsd + float(flip),
    }


def locate(runs: list[Path], relative: Path) -> Path:
    matches = [run / relative for run in runs if (run / relative).is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {relative}, got {matches}")
    return matches[0]


def functional_labels(plan: dict[str, Any], runs: list[Path]) -> tuple[dict[tuple[str, int], dict], dict]:
    labels, marker_rows = {}, []
    for case in plan["cases"]:
        baseline_path = locate(runs, Path("logits") / f"{case['case_id']}.baseline.npz")
        with np.load(baseline_path) as payload:
            baseline = payload["logits"]
            target = int(payload["target_token_id"])
        for path in case["paths"]:
            probe = f"{case['case_id']}.drop_slot{path['slot_rank']}"
            with np.load(locate(runs, Path("logits") / f"{probe}.npz")) as payload:
                changed = payload["logits"]
                if int(payload["target_token_id"]) != target:
                    raise RuntimeError(f"target mismatch {probe}")
            markers = []
            for run in runs:
                markers.extend(
                    json.loads(Path(value).read_text(encoding="utf-8"))
                    for value in glob.glob(str(run / "markers" / f"{probe}.pid*.json"))
                )
            marker = {
                "probe_id": probe,
                "marker_count": len(markers),
                "owned_target_count": sum(bool(value["owned_target"]) for value in markers),
                "all_fingerprint_match": all(value["fingerprint"] == case["target_hidden_hash"] for value in markers),
            }
            if marker["marker_count"] < 1 or marker["owned_target_count"] != 1 or not marker["all_fingerprint_match"]:
                raise RuntimeError(f"marker audit failed: {marker}")
            marker_rows.append(marker)
            labels[(case["base_case_id"], int(path["expert_id"]))] = {
                **damage_metrics(baseline, changed, target),
                "probe_id": probe,
            }
    return labels, {
        "probes": len(marker_rows),
        "all_pass": True,
        "marker_count_distribution": dict(Counter(row["marker_count"] for row in marker_rows)),
    }


def rho(rows: list[dict], score: str, label: str) -> float | None:
    left, right = [], []
    for row in rows:
        if row.get(score) is not None and row.get(label) is not None:
            left.append(float(row[score])); right.append(float(row[label]))
    if len(left) < 3 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    value = spearmanr(left, right).statistic
    return None if np.isnan(value) else float(value)


def separation(rows: list[dict], score: str, label: str, low: float, high: float) -> dict[str, Any]:
    lows = [float(row[label]) for row in rows if row.get(score) is not None and float(row[score]) <= low]
    highs = [float(row[label]) for row in rows if row.get(score) is not None and float(row[score]) >= high]
    return {
        "low_n": len(lows), "high_n": len(highs),
        "low_mean": None if not lows else float(np.mean(lows)),
        "high_mean": None if not highs else float(np.mean(highs)),
        "high_minus_low_mean": None if not lows or not highs else float(np.mean(highs) - np.mean(lows)),
        "low_median": None if not lows else float(np.median(lows)),
        "high_median": None if not highs else float(np.median(highs)),
        "high_minus_low_median": None if not lows or not highs else float(np.median(highs) - np.median(lows)),
    }


def bootstrap_request_difference(rows: list[dict], score: str, label: str, low: float, high: float, repeats: int = 10000) -> dict[str, Any]:
    by_request: dict[str, list[dict]] = defaultdict(list)
    for row in rows: by_request[row["request_id"]].append(row)
    requests = sorted(by_request)
    rng, values = np.random.default_rng(3407), []
    for _ in range(repeats):
        sample = rng.choice(requests, size=len(requests), replace=True)
        picked = [row for request in sample for row in by_request[str(request)]]
        result = separation(picked, score, label, low, high)["high_minus_low_mean"]
        if result is not None: values.append(result)
    return {
        "unit": "request", "repeats": repeats, "valid_repeats": len(values),
        "ci95": None if not values else [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
    }


def thresholds(rows: list[dict], key: str) -> tuple[float, float]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.percentile(values, 33.333333)), float(np.percentile(values, 66.666667))


def bootstrap_proxy_increment(
    rows: list[dict], candidate: str, baseline: str,
    candidate_thresholds: tuple[float, float], baseline_thresholds: tuple[float, float],
    repeats: int = 10000,
) -> dict[str, Any]:
    by_request: dict[str, list[dict]] = defaultdict(list)
    for row in rows: by_request[row["request_id"]].append(row)
    requests, values = sorted(by_request), []
    rng = np.random.default_rng(3407)
    for _ in range(repeats):
        sample = rng.choice(requests, size=len(requests), replace=True)
        picked = [row for request in sample for row in by_request[str(request)]]
        c = separation(picked, candidate, "d_visual", *candidate_thresholds)["high_minus_low_mean"]
        b = separation(picked, baseline, "d_visual", *baseline_thresholds)["high_minus_low_mean"]
        if c is not None and b is not None: values.append(c - b)
    return {"metric": "candidate_minus_router_high_low_d_visual", "unit": "request",
            "repeats": repeats, "valid_repeats": len(values),
            "point": None if not values else float(np.mean(values)),
            "ci95": None if not values else [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]}


def affinity(compact: Path, calibration_ids: set[str]) -> dict[tuple[int, int], float]:
    vision, text = defaultdict(float), defaultdict(float)
    for row in read_jsonl(compact):
        if row["request_id"] not in calibration_ids: continue
        for layer, payload in enumerate(row["prefill_layers"]):
            vm, tm = payload["vision_mass"], payload["text_mass"]
            vden, tden = max(sum(map(float, vm.values())), 1e-12), max(sum(map(float, tm.values())), 1e-12)
            for expert in range(128):
                vision[(layer, expert)] += float(vm.get(str(expert), 0.0)) / vden
                text[(layer, expert)] += float(tm.get(str(expert), 0.0)) / tden
    return {key: value / max(value + text[key], 1e-12) for key, value in vision.items()}


def proxy_scores(rows: list[dict], compact: Path) -> dict[str, Any]:
    calibration = [row for row in rows if row["split"] == "calibration"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    by_le, by_e, by_l = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in calibration:
        value = float(row["path_contribution_shift"])
        by_le[(row["layer"], row["expert_id"])].append(value)
        by_e[row["expert_id"]].append(value); by_l[row["layer"]].append(value)
    global_value = float(np.median([row["path_contribution_shift"] for row in calibration]))
    calibration_ids = {row["request_id"] for row in calibration}
    modality = affinity(compact, calibration_ids)
    for row in rows:
        key = (row["layer"], row["expert_id"])
        if len(by_le[key]) >= 2: causal = float(np.median(by_le[key])); support = "layer_expert"
        elif len(by_e[row["expert_id"]]) >= 2: causal = float(np.median(by_e[row["expert_id"]])); support = "expert"
        elif by_l[row["layer"]]: causal = float(np.median(by_l[row["layer"]])); support = "layer"
        else: causal, support = global_value, "global"
        row["proxy_p0_router"] = float(row["router_weight_orig"])
        row["proxy_p1_static_causal"] = float(row["router_weight_orig"]) * causal
        row["proxy_p1_fallback"] = support
        row["proxy_p2_modality_affinity"] = float(row["router_weight_orig"]) * modality.get(key, 0.5)
    output = {}
    for key in ("proxy_p0_router", "proxy_p1_static_causal", "proxy_p2_modality_affinity"):
        low, high = thresholds(calibration, key)
        sep = separation(heldout, key, "d_visual", low, high)
        boot = bootstrap_request_difference(heldout, key, "d_visual", low, high)
        output[key] = {
            "calibration_thresholds": {"low_upper": low, "high_lower": high},
            "heldout_d_visual_spearman": rho(heldout, key, "d_visual"),
            "heldout_d_orig_spearman": rho(heldout, key, "d_orig"),
            "heldout_separation": sep,
            "heldout_bootstrap": boot,
        }
    baseline_thresholds = thresholds(calibration, "proxy_p0_router")
    for key in ("proxy_p1_static_causal", "proxy_p2_modality_affinity"):
        output[key]["increment_vs_router_bootstrap"] = bootstrap_proxy_increment(
            heldout, key, "proxy_p0_router", thresholds(calibration, key), baseline_thresholds
        )
    generic = output["proxy_p0_router"]
    candidates = ["proxy_p1_static_causal", "proxy_p2_modality_affinity"]
    best = max(candidates, key=lambda key: output[key]["heldout_d_visual_spearman"] or -999)
    best_result = output[best]
    generic_rho = generic["heldout_d_visual_spearman"] or 0.0
    best_rho = best_result["heldout_d_visual_spearman"] or 0.0
    increment_ci = best_result["increment_vs_router_bootstrap"]["ci95"]
    proxy_pass = best_rho > generic_rho and increment_ci is not None and increment_ci[0] > 0 and (
        best_result["heldout_separation"]["high_minus_low_mean"] or 0.0
    ) > (
        generic["heldout_separation"]["high_minus_low_mean"] or 0.0
    )
    return {"status": "RUN", "results": output, "best_modality_proxy": best,
            "adds_signal_beyond_router": proxy_pass}


def subgroup(rows: list[dict], key: str, low: float, high: float) -> dict[str, Any]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows: grouped[str(row[key])].append(row)
    return {
        value: {
            "requests": len({row["request_id"] for row in values}),
            "paths": len(values),
            "pcs_d_visual_spearman": rho(values, "path_contribution_shift", "d_visual"),
            "high_low": separation(values, "path_contribution_shift", "d_visual", low, high),
        }
        for value, values in sorted(grouped.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--signal-summary", type=Path, required=True)
    parser.add_argument("--orig-plan", type=Path, required=True)
    parser.add_argument("--cf-plan", type=Path, required=True)
    parser.add_argument("--orig-run", type=Path, action="append", required=True)
    parser.add_argument("--cf-run", type=Path, action="append", required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--verdict", type=Path, required=True)
    args = parser.parse_args()

    signals = list(read_jsonl(args.signals))
    orig, orig_audit = functional_labels(read_json(args.orig_plan), args.orig_run)
    cf, cf_audit = functional_labels(read_json(args.cf_plan), args.cf_run)
    labels = []
    for row in signals:
        key = (row["case_id"], int(row["expert_id"]))
        if key not in orig: raise RuntimeError(f"missing original damage {key}")
        do = orig[key]["functional_damage_score"]
        cf_value = cf.get(key, {
            "functional_damage_score": 0.0, "target_logprob_drop_nats": 0.0,
            "next_token_jsd": 0.0, "top1_flip": False,
        })
        dc = cf_value["functional_damage_score"]
        labels.append({**row, "d_orig": float(do), "d_cf": float(dc),
                       "d_visual": float(do - dc), "d_visual_pos": max(float(do - dc), 0.0),
                       "d_orig_target_logprob_drop_nats": orig[key]["target_logprob_drop_nats"],
                       "d_orig_next_token_jsd": orig[key]["next_token_jsd"],
                       "d_orig_top1_flip": orig[key]["top1_flip"],
                       "d_cf_target_logprob_drop_nats": cf_value["target_logprob_drop_nats"],
                       "d_cf_next_token_jsd": cf_value["next_token_jsd"],
                       "d_cf_top1_flip": cf_value["top1_flip"],
                       "damage_metric": "max(0,target_logprob_drop_nats)+2*JSD+1[top1_flip]",
                       "functional_evidence": "MEASURED_TRUE_EP4_EXACT_DROP_ONE_REMAINING_FORWARD"})
    calibration = [row for row in labels if row["split"] == "calibration"]
    heldout = [row for row in labels if row["split"] == "heldout"]
    frozen = read_json(args.signal_summary)["frozen_calibration_thresholds"]
    sep = separation(heldout, "path_contribution_shift", "d_visual", frozen["low_upper"], frozen["high_lower"])
    boot = bootstrap_request_difference(heldout, "path_contribution_shift", "d_visual", frozen["low_upper"], frozen["high_lower"])
    primary_rho = rho(heldout, "path_contribution_shift", "d_visual")
    ci = boot["ci95"]
    gate_pass = bool(primary_rho is not None and primary_rho > 0 and
                     sep["high_minus_low_mean"] is not None and sep["high_minus_low_mean"] > 0 and
                     ci is not None and ci[0] > 0)
    gate = {
        "status": "PASS" if gate_pass else "FAIL",
        "fresh_heldout_requests": len({row["request_id"] for row in heldout}),
        "fresh_heldout_paths": len(heldout),
        "primary_pcs_d_visual_spearman": primary_rho,
        "primary_pcs_d_orig_spearman": rho(heldout, "path_contribution_shift", "d_orig"),
        "high_low": sep,
        "request_bootstrap": boot,
        "diagnostics": {
            key: {"d_visual_spearman": rho(heldout, key, "d_visual"),
                  "d_orig_spearman": rho(heldout, key, "d_orig")}
            for key in ("router_shift", "hidden_delta", "expert_output_shift")
        },
        "topk_membership_flip_rate": float(np.mean([row["topk_membership_flip"] for row in heldout])),
        "retention_ablation": {
            label: {
                "paths": len(values),
                "pcs_d_visual_spearman": rho(values, "path_contribution_shift", "d_visual"),
                "high_low": separation(values, "path_contribution_shift", "d_visual", frozen["low_upper"], frozen["high_lower"]),
            }
            for label, values in {
                "retained": [row for row in heldout if row["counterfactual_topk_retained"]],
                "left_counterfactual_topk": [row for row in heldout if not row["counterfactual_topk_retained"]],
            }.items()
        },
        "per_layer": subgroup(heldout, "layer", frozen["low_upper"], frozen["high_lower"]),
        "per_domain": subgroup(heldout, "domain", frozen["low_upper"], frozen["high_lower"]),
        "marker_audit": {"orig": orig_audit, "cf": cf_audit},
        "gate_rule": "rho>0 AND high-low>0 AND request-bootstrap-CI-lower>0",
    }
    if gate_pass:
        proxy = proxy_scores(labels, args.compact)
    else:
        proxy = {"status": "NOT_RUN_Q2_2A_GATE_FAILED", "best_modality_proxy": None,
                 "adds_signal_beyond_router": False}
        for row in labels: row["proxy_status"] = "NOT_RUN_Q2_2A_GATE_FAILED"
    if gate_pass and proxy["adds_signal_beyond_router"]:
        verdict = "MLLM_CAUSAL_LINK_AND_PROXY_SUPPORTED"
        next_action = "Proceed to a separately gated vision-protected rank-aware candidate; no runtime claim is made here."
    elif gate_pass:
        verdict = "MLLM_CAUSAL_PATH_LINK_SUPPORTED"
        next_action = "Keep the causal premise, but solve cheap-signal discovery before any modified-route EP timing."
    elif float(np.mean([row["router_shift"] for row in heldout])) > 0:
        verdict = "VISUAL_ROUTING_CHANGES_BUT_NOT_PATH_IMPORTANCE"
        next_action = "Stop the MLLM-specific protection branch and retain the generic rank-aware EP direction."
    else:
        verdict = "NO_MLLM_PATH_CAUSAL_LINK"
        next_action = "Stop visual path protection and retain only the generic rank-aware EP observation."
    final = {
        "schema_version": 1, "q2_2_a_gate": gate["status"], "q2_2_b_proxy": proxy["status"],
        "verdict": verdict, "next_action": next_action,
        "strict_non_claims": ["TPOT", "E2E", "throughput", "production deployability", "EP8 speedup"],
    }
    write_jsonl_gz(args.labels, labels)
    atomic_json(args.gate, gate); atomic_json(args.proxy, proxy); atomic_json(args.verdict, final)
    print(json.dumps({**final, "primary": gate}, indent=2))


if __name__ == "__main__":
    main()
