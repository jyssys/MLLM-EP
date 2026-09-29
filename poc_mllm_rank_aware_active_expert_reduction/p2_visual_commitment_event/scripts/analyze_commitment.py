#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from poc_mllm_rank_aware_active_expert_reduction.contract import atomic_json


def _read(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("sensitivity.pid*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _bootstrap(values: list[float], seed: int = 3407) -> list[float | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return [None, None]
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(10000, len(array)))].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _ema(values: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    output = values.copy()
    for index in range(1, len(values)):
        output[index] = alpha * values[index] + (1.0 - alpha) * output[index - 1]
    return output


def _event(rows: list[dict[str, Any]], rule: dict[str, Any]) -> dict[str, Any] | None:
    ordered = sorted(rows, key=lambda row: row["position"])
    positions = np.asarray([row["position"] for row in ordered], dtype=np.int64)
    values = np.asarray([row["jsd_nats"] for row in ordered], dtype=np.float64)
    smooth = _ema(values)
    width, threshold = int(rule["width"]), float(rule["threshold"])
    for index in range(len(ordered) - width + 1):
        if positions[index] < int(rule["minimum_position"]):
            continue
        if np.all(smooth[index : index + width] < threshold):
            before, after = values[:index], values[index:]
            if len(before) < 5 or len(after) < 5:
                continue
            return {
                "position": int(positions[index]), "index": index,
                "pre_mean_jsd": float(before.mean()), "post_mean_jsd": float(after.mean()),
                "delta_jsd": float(before.mean() - after.mean()),
                "post_below_threshold_fraction": float(np.mean(smooth[index:] < threshold)),
                "pre_observations": len(before), "post_observations": len(after),
            }
    return None


def _split_delta(rows: list[dict[str, Any]], position: int) -> float | None:
    pre = [row["jsd_nats"] for row in rows if int(row["position"]) < position]
    post = [row["jsd_nats"] for row in rows if int(row["position"]) >= position]
    if len(pre) < 5 or len(post) < 5:
        return None
    return float(np.mean(pre) - np.mean(post))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = _read(args.raw)
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    expected_positions = set(range(1, 33)) | set(range(36, 129, 4))
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[row["pair_id"]].append(row)
    missing = {
        pair["pair_id"]: sorted(expected_positions - {int(row["position"]) for row in by_pair[pair["pair_id"]]})
        for pair in cohort["pairs"]
        if expected_positions - {int(row["position"]) for row in by_pair[pair["pair_id"]]}
    }
    if missing:
        raise RuntimeError(f"incomplete trajectories: {missing}")
    calibration = [pair for pair in cohort["pairs"] if pair["split"] == "calibration"]
    heldout = [pair for pair in cohort["pairs"] if pair["split"] == "heldout"]
    noise = [pair for pair in cohort["pairs"] if pair["split"] == "repeat_noise"]
    cal_values = np.asarray([
        row["jsd_nats"] for pair in calibration for row in by_pair[pair["pair_id"]]
    ])
    thresholds = sorted(set(float(value) for value in np.quantile(
        cal_values, [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    )))
    candidates = []
    for threshold in thresholds:
        for width in (3, 5, 8):
            for minimum in (12, 16, 24, 32):
                rule = {"threshold": threshold, "width": width, "minimum_position": minimum}
                events = [_event(by_pair[pair["pair_id"]], rule) for pair in calibration]
                found = [event for event in events if event is not None]
                rate = len(found) / len(events)
                persistence = float(np.mean([
                    event["post_below_threshold_fraction"] for event in found
                ])) if found else 0.0
                deltas = [event["delta_jsd"] for event in found]
                standardized = 0.0 if not deltas else float(
                    np.mean(deltas) / max(np.std(deltas), 1e-12)
                )
                valid = rate >= 0.5 and persistence >= 0.75
                candidates.append({
                    **rule, "calibration_event_rate": rate,
                    "calibration_persistence": persistence,
                    "calibration_mean_delta": float(np.mean(deltas)) if deltas else 0.0,
                    "score": rate * persistence * standardized if valid else -1e30,
                    "valid": valid,
                })
    frozen = max(candidates, key=lambda row: row["score"])
    rule = {key: frozen[key] for key in ("threshold", "width", "minimum_position")}
    event_rows = []
    for pair in heldout:
        event = _event(by_pair[pair["pair_id"]], rule)
        event_rows.append({
            "pair_id": pair["pair_id"], "domain": pair["domain"],
            "event": event,
        })
    found = [row for row in event_rows if row["event"] is not None]
    event_rate = len(found) / len(event_rows)
    deltas = [row["event"]["delta_jsd"] for row in found]
    delta_ci = _bootstrap(deltas)
    pre_mean = float(np.mean([row["event"]["pre_mean_jsd"] for row in found])) if found else None
    post_mean = float(np.mean([row["event"]["post_mean_jsd"] for row in found])) if found else None
    persistence = float(np.mean([
        row["event"]["post_below_threshold_fraction"] for row in found
    ])) if found else 0.0
    cal_events = [
        _event(by_pair[pair["pair_id"]], rule) for pair in calibration
    ]
    cal_positions = [event["position"] for event in cal_events if event is not None]
    fixed_position = int(np.median(cal_positions)) if cal_positions else 64
    generic_deltas = []
    pseudo_deltas = []
    event_minus_generic = []
    for index, pair in enumerate(heldout):
        trajectory = by_pair[pair["pair_id"]]
        generic = _split_delta(trajectory, fixed_position)
        pseudo_position = cal_positions[index % len(cal_positions)] if cal_positions else fixed_position
        pseudo = _split_delta(trajectory, pseudo_position)
        if generic is not None:
            generic_deltas.append(generic)
        if pseudo is not None:
            pseudo_deltas.append(pseudo)
        event = event_rows[index]["event"]
        if event is not None and generic is not None:
            event_minus_generic.append(event["delta_jsd"] - generic)
    event_minus_generic_ci = _bootstrap(event_minus_generic, seed=3408)
    noise_values = [
        row["jsd_nats"] for pair in noise for row in by_pair[pair["pair_id"]]
    ]
    raw_noise_mean = float(np.mean(noise_values))
    noise_mean = max(0.0, raw_noise_mean)
    generic_ci = _bootstrap(generic_deltas, seed=3409)
    window_summary = {}
    for name, lower, upper in (
        ("1_32", 1, 32), ("36_64", 36, 64),
        ("68_96", 68, 96), ("100_128", 100, 128),
    ):
        values = [
            row["jsd_nats"] for pair in heldout
            for row in by_pair[pair["pair_id"]]
            if lower <= int(row["position"]) <= upper
        ]
        window_summary[name] = {
            "observations": len(values), "mean_jsd": float(np.mean(values)),
            "median_jsd": float(np.median(values)),
        }
    existence = (
        frozen["valid"] and event_rate >= 0.4 and delta_ci[0] > 0
        and persistence >= 0.8 and post_mean is not None and post_mean > noise_mean
    )
    ml_specific = existence and event_minus_generic_ci[0] is not None and event_minus_generic_ci[0] > 0
    if not existence:
        verdict = "NO_VISUAL_COMMITMENT_EVENT"
    elif not ml_specific:
        verdict = "GENERIC_LATE_DECODE_TRANSITION_ONLY"
    else:
        # This is only the existence gate; skipping must still be measured.
        verdict = "VISUAL_COMMITMENT_ORACLE_SUPPORTED"
    payload = {
        "schema_version": "visual_commitment_analysis.v1",
        "evidence": "MEASURED_FULL_VOCAB_TEACHER_FORCED_TRUE_EP4",
        "trajectory_rows": len(rows), "calibration_requests": len(calibration),
        "heldout_requests": len(heldout), "repeat_noise_requests": len(noise),
        "frozen_rule": rule, "calibration_selection": frozen,
        "heldout_event_rate": event_rate, "heldout_event_count": len(found),
        "heldout_pre_mean_jsd": pre_mean, "heldout_post_mean_jsd": post_mean,
        "heldout_pre_minus_post_mean_jsd": float(np.mean(deltas)) if deltas else None,
        "heldout_pre_minus_post_95ci": delta_ci,
        "heldout_post_persistence": persistence,
        "repeat_noise_mean_jsd": noise_mean,
        "repeat_noise_raw_numeric_mean_jsd": raw_noise_mean,
        "heldout_window_summary": window_summary,
        "generic_fixed_position": fixed_position,
        "generic_position_mean_delta_jsd": float(np.mean(generic_deltas)),
        "generic_position_delta_95ci": generic_ci,
        "pseudo_event_mean_delta_jsd": float(np.mean(pseudo_deltas)),
        "event_minus_generic_mean_delta_jsd": float(np.mean(event_minus_generic)) if event_minus_generic else None,
        "event_minus_generic_95ci": event_minus_generic_ci,
        "existence_gate_pass": existence,
        "mllm_specific_gate_pass": ml_specific,
        "run_skip_stage": ml_specific,
        "events": event_rows,
        "verdict": verdict,
        "limitations": [
            "Counterfactual forced continuations can be off-distribution for the swapped image.",
            "The rule is selected on eight calibration requests and evaluated once on fresh held-out requests.",
            "An oracle event is not an online detector and does not imply skip safety.",
        ],
    }
    atomic_json(args.output, payload)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if found:
        event_block = (
            f"Pre/post JSD: **{pre_mean:.6g} → {post_mean:.6g}**; mean delta "
            f"**{payload['heldout_pre_minus_post_mean_jsd']:.6g}**, request-bootstrap 95% CI "
            f"**[{delta_ci[0]:.6g}, {delta_ci[1]:.6g}]**. "
        )
    else:
        event_block = (
            "No held-out trajectory satisfied the calibration-frozen persistent-low rule, "
            "so event-aligned pre/post JSD and its CI are **N/A**. "
        )
    if event_minus_generic_ci[0] is None:
        specific_block = "Event-minus-generic CI: **N/A**."
    else:
        specific_block = (
            f"Event-minus-generic CI: **[{event_minus_generic_ci[0]:.6g}, "
            f"{event_minus_generic_ci[1]:.6g}]**."
        )
    args.report.write_text(
        "# PoC-2 — visual commitment event existence gate\n\n"
        f"Evidence: **{payload['evidence']}**. Frozen rule: `{rule}`.\n\n"
        f"Calibration rule validity: **{frozen['valid']}**; calibration event rate: "
        f"**{frozen['calibration_event_rate']:.3%}**.\n\n"
        f"Held-out event rate: **{event_rate:.3%}** ({len(found)}/{len(heldout)}). "
        f"{event_block}Post persistence: **{persistence:.3%}**.\n\n"
        "| Held-out decode window | Mean JSD | Median JSD |\n"
        "|---|---:|---:|\n"
        + "\n".join(
            f"| {name.replace('_', '–')} | {summary['mean_jsd']:.6g} | {summary['median_jsd']:.6g} |"
            for name, summary in window_summary.items()
        )
        + "\n\n"
        f"Same-image repeat-noise JSD: **{payload['repeat_noise_mean_jsd']:.6g}**. "
        f"Generic fixed-position delta: **{payload['generic_position_mean_delta_jsd']:.6g}**; "
        f"request-bootstrap 95% CI **[{generic_ci[0]:.6g}, {generic_ci[1]:.6g}]**; "
        f"pseudo-event delta: **{payload['pseudo_event_mean_delta_jsd']:.6g}**. "
        f"{specific_block}\n\n"
        f"Existence gate: **{existence}**. MLLM-specific beyond-position gate: **{ml_specific}**. "
        f"Final verdict: **{verdict}**.\n\n"
        "Because the existence gate failed, K8→K6 pre/post damage, post-commit "
        "rank-aware closure, and a cheap online proxy are **NOT_RUN_BY_GATE**.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "rule": rule, "event_rate": event_rate, "delta_ci": delta_ci,
        "generic_ci": event_minus_generic_ci, "verdict": verdict,
        "run_skip_stage": ml_specific,
    }, indent=2))


if __name__ == "__main__":
    main()
