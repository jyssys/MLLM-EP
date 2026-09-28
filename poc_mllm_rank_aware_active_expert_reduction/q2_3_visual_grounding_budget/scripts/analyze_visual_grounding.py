from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import spearmanr

from ..contract import atomic_json, read_json, read_jsonl_gz, write_jsonl_gz


def log_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - float(np.max(values))
    return shifted - np.log(np.exp(shifted).sum())


def damage(base: np.ndarray, changed: np.ndarray, target: int) -> dict[str, Any]:
    blp, clp = log_softmax(base), log_softmax(changed)
    bp, cp = np.exp(blp), np.exp(clp)
    middle = np.maximum(0.5 * (bp + cp), 1e-300)
    jsd = 0.5 * float(np.sum(bp * (blp - np.log(middle))))
    jsd += 0.5 * float(np.sum(cp * (clp - np.log(middle))))
    drop = float(blp[target] - clp[target])
    flip = int(np.argmax(base) != np.argmax(changed))
    return {"target_logprob_drop_nats": drop, "jsd": jsd, "top1_flip": flip,
            "damage": max(0.0, drop) + 2.0 * jsd + float(flip)}


def _rho_values(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    value = spearmanr(left, right).statistic
    return None if np.isnan(value) else float(value)


def rho(rows: list[dict[str, Any]], x: str, y: str) -> float | None:
    return _rho_values([float(row[x]) for row in rows], [float(row[y]) for row in rows])


def request_bootstrap(rows: list[dict[str, Any]], statistic: Callable[[list[dict[str, Any]]], float | None],
                      repeats: int = 10000) -> dict[str, Any]:
    by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: by_request[str(row["request_id"])].append(row)
    requests, values = sorted(by_request), []
    rng = np.random.default_rng(3407)
    for _ in range(repeats):
        sample = rng.choice(requests, len(requests), replace=True)
        picked = [row for request in sample for row in by_request[str(request)]]
        value = statistic(picked)
        if value is not None and np.isfinite(value): values.append(float(value))
    return {"unit": "request", "repeats": repeats, "valid_repeats": len(values),
            "ci95": None if not values else [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]}


def load_logits(run: Path, probe: str, step: int) -> tuple[np.ndarray, int]:
    path = run / "logits" / f"{probe}.step{step}.npz"
    with np.load(path) as payload:
        return payload["logits"], int(payload["target_token_id"])


def marker_audit(run: Path, probe_cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for case in probe_cases:
        for keep in (6, 4):
            probe = f"{case['probe_case_id']}.k{keep}"
            markers = [json.loads(Path(value).read_text()) for value in glob.glob(str(run / "markers" / f"{probe}.pid*.json"))]
            row = {
                "probe_id": probe, "markers": len(markers),
                "fingerprint_match": bool(markers) and all(value["fingerprint"] == case["target_hidden_hash"] for value in markers),
                "owned_kept": sum(int(value["owned_kept_count"]) for value in markers),
                "owned_removed": sum(int(value["owned_removed_count"]) for value in markers),
                "expected_kept": keep, "expected_removed": 8 - keep,
            }
            row["pass"] = row["fingerprint_match"] and row["owned_kept"] == keep and row["owned_removed"] == 8 - keep
            rows.append(row)
    failed = [row for row in rows if not row["pass"]]
    if failed: raise RuntimeError(f"structured intervention marker audit failed: {failed[:3]}")
    return {"all_pass": True, "probes": len(rows),
            "marker_count_distribution": dict(Counter(row["markers"] for row in rows))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--probe-run", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--verdict", type=Path, required=True)
    args = parser.parse_args()
    manifest, plan = read_json(args.manifest), read_json(args.plan)
    signals = {row["capture_id"]: row for row in read_jsonl_gz(args.signals)}
    labels = []
    for case in plan["cases"]:
        step, horizon = int(case["probe_step"]), int(case["horizon"])
        per_keep = {}
        for keep in (6, 4):
            immediate, horizon_values = None, []
            for offset in range(horizon):
                base, target = load_logits(args.probe_run, f"{case['probe_case_id']}.k8", step + offset)
                changed, changed_target = load_logits(args.probe_run, f"{case['probe_case_id']}.k{keep}", step + offset)
                if target != changed_target: raise RuntimeError((case["probe_case_id"], step + offset))
                metric = damage(base, changed, target)
                horizon_values.append(metric["damage"])
                if offset == 0: immediate = metric
            per_keep[keep] = {**immediate, "horizon_mean_damage": float(np.mean(horizon_values)),
                              "horizon_damage": horizon_values}
        signal = signals[case["probe_case_id"]]
        labels.append({
            **signal,
            "request_id": case["request_id"], "condition": case["condition"],
            "split": case["split"], "domain": case["domain"],
            "k6_damage": per_keep[6]["damage"], "k4_damage": per_keep[4]["damage"],
            "structured_damage": 0.5 * (per_keep[6]["damage"] + per_keep[4]["damage"]),
            "k6_horizon_damage": per_keep[6]["horizon_mean_damage"],
            "k4_horizon_damage": per_keep[4]["horizon_mean_damage"],
            "structured_horizon_damage": 0.5 * (per_keep[6]["horizon_mean_damage"] + per_keep[4]["horizon_mean_damage"]),
            "k6_target_logprob_drop_nats": per_keep[6]["target_logprob_drop_nats"],
            "k4_target_logprob_drop_nats": per_keep[4]["target_logprob_drop_nats"],
            "k6_jsd": per_keep[6]["jsd"], "k4_jsd": per_keep[4]["jsd"],
            "k6_top1_flip": per_keep[6]["top1_flip"], "k4_top1_flip": per_keep[4]["top1_flip"],
            "evidence": "MEASURED_EXACT_SEMANTIC_INTERVENTION_NOT_PHYSICAL_SAVING",
        })
    write_jsonl_gz(args.labels, labels)
    calibration = [row for row in labels if row["split"] == "calibration" and row["condition"] == "orig"]
    heldout_orig = [row for row in labels if row["split"] == "heldout" and row["condition"] == "orig"]
    controls = ["text_energy", "full_energy", "router_removed_mass_control"]
    control_calibration = {key: rho(calibration, key, "structured_damage") for key in controls}
    strongest = max(controls, key=lambda key: control_calibration[key] if control_calibration[key] is not None else -999)
    by_case_condition = {(row["case_id"], row["condition"]): row for row in labels}
    causal_rows = []
    for case in manifest["cases"]:
        if case["split"] != "heldout": continue
        orig, cf = by_case_condition[(case["case_id"], "orig")], by_case_condition[(case["case_id"], "cf")]
        causal_rows.append({
            "request_id": case["request_id"], "case_id": case["case_id"], "domain": case["domain"],
            "delta_separated_visual_score": orig["separated_visual_score"] - cf["separated_visual_score"],
            "delta_structured_damage": orig["structured_damage"] - cf["structured_damage"],
            "delta_text_energy": orig["text_energy"] - cf["text_energy"],
            "delta_full_energy": orig["full_energy"] - cf["full_energy"],
            "delta_router_removed_mass_control": orig["router_removed_mass_control"] - cf["router_removed_mass_control"],
        })
    candidate = "separated_visual_score"
    assoc_point = rho(heldout_orig, candidate, "structured_damage")
    assoc_boot = request_bootstrap(heldout_orig, lambda rows: rho(rows, candidate, "structured_damage"))
    causal_point = rho(causal_rows, "delta_separated_visual_score", "delta_structured_damage")
    causal_boot = request_bootstrap(causal_rows, lambda rows: rho(rows, "delta_separated_visual_score", "delta_structured_damage"))
    candidate_rho = assoc_point
    control_rho = rho(heldout_orig, strongest, "structured_damage")
    control_boot = request_bootstrap(heldout_orig, lambda rows: rho(rows, strongest, "structured_damage"))
    increment = None if candidate_rho is None or control_rho is None else candidate_rho - control_rho
    increment_boot = request_bootstrap(
        heldout_orig,
        lambda rows: None if rho(rows, candidate, "structured_damage") is None or rho(rows, strongest, "structured_damage") is None
        else rho(rows, candidate, "structured_damage") - rho(rows, strongest, "structured_damage"),
    )
    def positive(point: float | None, boot: dict[str, Any]) -> bool:
        return point is not None and point > 0 and boot["ci95"] is not None and boot["ci95"][0] > 0
    pass_assoc, pass_causal = positive(assoc_point, assoc_boot), positive(causal_point, causal_boot)
    pass_increment = increment is not None and increment > 0 and increment_boot["ci95"] is not None and increment_boot["ci95"][0] > 0
    passed = pass_assoc and pass_causal and pass_increment
    marker = marker_audit(args.probe_run, plan["cases"])
    gate = {
        "schema_version": 1, "primary_score": candidate,
        "primary_damage": "mean(K6_damage,K4_damage)", "heldout_requests": len(heldout_orig),
        "association": {"spearman": assoc_point, "bootstrap": assoc_boot, "pass": pass_assoc},
        "causal_delta": {"spearman": causal_point, "bootstrap": causal_boot, "pass": pass_causal},
        "control_selected_on_calibration_only": strongest,
        "calibration_control_spearman": control_calibration,
        "heldout_control": {"spearman": control_rho, "bootstrap": control_boot,
                            "pass": positive(control_rho, control_boot)},
        "increment_vs_frozen_control": {"spearman_difference": increment, "bootstrap": increment_boot, "pass": pass_increment},
        "heldout_all_feature_spearman": {key: rho(heldout_orig, key, "structured_damage") for key in [candidate] + controls},
        "marker_audit": marker,
        "gate_pass": passed,
        "phase2_promoted": passed,
    }
    atomic_json(args.gate, gate)
    if passed:
        verdict = "VISUAL_GROUNDING_BUDGET_SIGNAL_SUPPORTED"
        next_action = "DEBATE_AND_SEPARATELY_SPECIFY_PHASE2_EP_AWARE_EXPERT_SELECTION"
    elif pass_assoc and not (pass_causal and pass_increment):
        verdict = "VISUAL_GROUNDING_ASSOCIATION_ONLY"
        next_action = "STOP_PHASE2; DO_NOT_USE_ASSOCIATION_AS_VISUAL_PROTECTION"
    elif positive(control_rho, control_boot):
        verdict = "GENERIC_UNIT_RISK_ONLY"
        next_action = "STOP_MLLM_SPECIFIC_PHASE2"
    else:
        verdict = "NO_STRUCTURED_DAMAGE_SIGNAL"
        next_action = "STOP_PHASE2"
    result = {"verdict": verdict, "next_action": next_action, "phase1_gate": "PASS" if passed else "FAIL",
              "phase2_promoted": passed, "latency_claim": "NOT_RUN", "modes_comparison": "NOT_RUN_BY_USER_SCOPE"}
    atomic_json(args.verdict, result)
    print(json.dumps({"gate": gate, "verdict": result}, indent=2))


if __name__ == "__main__": main()
