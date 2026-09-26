from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from poc_mllm_ep_foundation.evaluation.metrics import (
    chartqa_relaxed_correct,
    extract_mcq_letter,
)
from poc_mllm_ep_foundation.io import atomic_json


LAYERS = 48


def _read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    ids = [str(row["request_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate request IDs in factorial results")
    return rows


def _score(row: dict[str, Any]) -> tuple[str | None, bool]:
    output = str(row.get("output_text", "")).strip()
    answer = str(row.get("answer", "")).strip()
    if row["task"] == "mmstar":
        parsed = extract_mcq_letter(output)
        return parsed, parsed == answer.upper()
    return output, chartqa_relaxed_correct(output, answer)


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _median(values: list[float]) -> float | None:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.median(finite)) if finite else None


def _bootstrap_delta(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    seed: int = 3407,
    repeats: int = 10_000,
) -> list[float] | None:
    ids = sorted(set(baseline) & set(candidate))
    if not ids:
        return None
    delta = np.asarray([
        float(candidate[key]["correct"]) - float(baseline[key]["correct"])
        for key in ids
    ])
    rng = np.random.default_rng(seed)
    sampled = delta[rng.integers(0, len(delta), size=(repeats, len(delta)))].mean(1)
    return [float(np.quantile(sampled, 0.025) * 100),
            float(np.quantile(sampled, 0.975) * 100)]


def _config(row: dict[str, Any]) -> str:
    return f"{row['resolution_label']}_K{int(row['k_visual'])}"


def _base_id(row: dict[str, Any]) -> str:
    return str(row.get("base_request_id") or str(row["request_id"]).split("__", 1)[0])


def analyze(
    rows: list[dict[str, Any]],
    physical_costs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    enriched: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        parsed, correct = _score(row)
        row["parsed_answer"] = parsed
        row["correct"] = correct
        row["config"] = _config(row)
        row["base_request_id"] = _base_id(row)
        row["logical_visual_expert_pairs"] = (
            int(row["vision_token_count"]) * int(row["k_visual"]) * LAYERS
        )
        row["fixed_width_visual_slots"] = int(row["vision_token_count"]) * 8 * LAYERS
        row["invalid_visual_slots"] = (
            int(row["vision_token_count"]) * (8 - int(row["k_visual"])) * LAYERS
        )
        enriched.append(row)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        grouped[row["config"]].append(row)
    expected_configs = {
        f"{resolution}_K{k}"
        for resolution in ("R0", "R05", "R025")
        for k in (8, 6, 4)
    }
    if set(grouped) != expected_configs:
        raise RuntimeError(f"incomplete factorial configs: {sorted(set(grouped))}")

    baseline_by_id = {
        row["base_request_id"]: row for row in grouped["R0_K8"]
    }
    configs: dict[str, dict[str, Any]] = {}
    for name in sorted(grouped):
        values = grouped[name]
        by_id = {row["base_request_id"]: row for row in values}
        strata = sorted({str(row["part2_stratum"]) for row in values})
        tasks = sorted({str(row["task"]) for row in values})
        stratum_accuracy = {
            key: _mean([float(row["correct"]) for row in values
                        if row["part2_stratum"] == key])
            for key in strata
        }
        task_accuracy = {
            key: _mean([float(row["correct"]) for row in values if row["task"] == key])
            for key in tasks
        }
        paired_delta = _mean([
            (float(row["correct"]) - float(baseline_by_id[row["base_request_id"]]["correct"]))
            * 100 for row in values
        ])
        configs[name] = {
            "n": len(values),
            "accuracy": _mean([float(row["correct"]) for row in values]),
            "stratum_macro_accuracy": _mean(list(stratum_accuracy.values())),
            "stratum_accuracy": stratum_accuracy,
            "task_accuracy": task_accuracy,
            "baseline_delta_pp": paired_delta,
            "baseline_delta_cluster_bootstrap_95ci_pp": _bootstrap_delta(
                baseline_by_id, by_id
            ),
            "baseline_correct_to_wrong": sum(
                baseline_by_id[row["base_request_id"]]["correct"] and not row["correct"]
                for row in values
            ),
            "baseline_wrong_to_correct": sum(
                not baseline_by_id[row["base_request_id"]]["correct"] and row["correct"]
                for row in values
            ),
            "mean_visual_tokens": _mean([float(row["vision_token_count"]) for row in values]),
            "median_visual_tokens": _median([float(row["vision_token_count"]) for row in values]),
            "mean_actual_visual_ratio": _mean([
                float(row["actual_visual_token_ratio"]) for row in values
            ]),
            "duplicate_resolution_count": sum(
                bool(row["duplicate_resolution_grid"]) for row in values
            ),
            "mean_logical_visual_pairs": _mean([
                float(row["logical_visual_expert_pairs"]) for row in values
            ]),
            "mean_fixed_width_visual_slots": _mean([
                float(row["fixed_width_visual_slots"]) for row in values
            ]),
            "mean_invalid_visual_slots": _mean([
                float(row["invalid_visual_slots"]) for row in values
            ]),
            "mean_ttft_ms": _mean([float(row["ttft_ms"]) for row in values
                                    if row.get("ttft_ms") is not None]),
            "median_ttft_ms": _median([float(row["ttft_ms"]) for row in values
                                        if row.get("ttft_ms") is not None]),
            "mean_fresh_image_ttft_ms": _mean([
                float(row["processor_ms"]) + float(row["ttft_ms"])
                for row in values if row.get("ttft_ms") is not None
            ]),
            "mean_natural_engine_ms": _mean([
                float(row["request_engine_ms"]) for row in values
                if row.get("request_engine_ms") is not None
            ]),
            "mean_fresh_image_natural_ms": _mean([
                float(row["processor_ms"]) + float(row["request_engine_ms"])
                for row in values if row.get("request_engine_ms") is not None
            ]),
            "mean_output_tokens": _mean([
                float(row["output_token_count"]) for row in values
                if row.get("output_token_count") is not None
            ]),
            "generation_timeout_count": sum(
                row.get("finish_reason") == "generation_timeout" for row in values
            ),
            "normal_termination_fraction": _mean([
                float(row.get("finish_reason") == "stop") for row in values
            ]),
        }

    baseline = configs["R0_K8"]
    for value in configs.values():
        value["ttft_gain_vs_baseline_pct"] = 100.0 * (
            1.0 - float(value["mean_ttft_ms"]) / float(baseline["mean_ttft_ms"])
        )
        value["natural_engine_gain_vs_baseline_pct"] = 100.0 * (
            1.0
            - float(value["mean_natural_engine_ms"])
            / float(baseline["mean_natural_engine_ms"])
        )
        value["fresh_image_ttft_gain_vs_baseline_pct"] = 100.0 * (
            1.0
            - float(value["mean_fresh_image_ttft_ms"])
            / float(baseline["mean_fresh_image_ttft_ms"])
        )
        value["fresh_image_natural_gain_vs_baseline_pct"] = 100.0 * (
            1.0
            - float(value["mean_fresh_image_natural_ms"])
            / float(baseline["mean_fresh_image_natural_ms"])
        )
        value["visual_pair_reduction_vs_baseline_pct"] = 100.0 * (
            1.0
            - float(value["mean_logical_visual_pairs"])
            / float(baseline["mean_logical_visual_pairs"])
        )
    if physical_costs is not None:
        for name, value in configs.items():
            measured = physical_costs.get(name)
            if measured is None:
                raise RuntimeError(f"physical cost summary missing {name}")
            for key in (
                "ep_critical_ms", "ttft_ms", "natural_e2e_ms",
                "fixed32_ms", "fixed128_ms", "physical_padded_rows",
                "max_mean", "dispatch_ms", "expert_ms", "combine_ms",
            ):
                if key in measured:
                    value[f"measured_{key}"] = measured[key]
        physical_baseline = configs["R0_K8"]
        for value in configs.values():
            if value.get("measured_ttft_ms") is not None:
                value["measured_ttft_gain_vs_baseline_pct"] = 100.0 * (
                    1.0
                    - float(value["measured_ttft_ms"])
                    / float(physical_baseline["measured_ttft_ms"])
                )
            if value.get("measured_natural_e2e_ms") is not None:
                value["measured_natural_e2e_gain_vs_baseline_pct"] = 100.0 * (
                    1.0
                    - float(value["measured_natural_e2e_ms"])
                    / float(physical_baseline["measured_natural_e2e_ms"])
                )

    # Same quality constraint is used by all control families. Discovery is
    # only a freeze screen; non-inferiority is re-evaluated once on held-out.
    eligible = {
        name for name, value in configs.items()
        if float(value["baseline_delta_pp"]) >= -1.0
    }

    def best(names: set[str], objective: str) -> str | None:
        candidates = sorted(
            name for name in names & eligible
            if configs[name].get(objective) is not None
        )
        return min(candidates, key=lambda name: float(configs[name][objective])) \
            if candidates else None

    resolution_only = best({f"{r}_K8" for r in ("R0", "R05", "R025")},
                           "mean_logical_visual_pairs")
    expert_only = best({f"R0_K{k}" for k in (8, 6, 4)},
                       "mean_logical_visual_pairs")
    joint_pair = best(set(configs), "mean_logical_visual_pairs")
    joint_ep = best(set(configs), "measured_ep_critical_ms")

    # Coordinate choices in both orders implement the independent-tuning
    # control without giving it fewer configurations than the joint controls.
    r_from_k8 = None if resolution_only is None else resolution_only.split("_", 1)[0]
    independent_r_then_k = None if r_from_k8 is None else best(
        {f"{r_from_k8}_K{k}" for k in (8, 6, 4)}, "mean_logical_visual_pairs"
    )
    k_from_r0 = None if expert_only is None else expert_only.split("_K", 1)[1]
    independent_k_then_r = None if k_from_r0 is None else best(
        {f"{r}_K{k_from_r0}" for r in ("R0", "R05", "R025")},
        "mean_logical_visual_pairs",
    )

    # The baseline is always evaluated from the frozen Part-1 artifact and is
    # not itself a Part-2 policy candidate.  If the EP objective selects C0,
    # freeze the strongest non-baseline EP screen explicitly; this preserves
    # the negative discovery result while still allowing one honest held-out
    # confirmation rather than silently calling the baseline a new policy.
    nonbaseline_ep = best(set(configs) - {"R0_K8"}, "measured_ep_critical_ms")
    ep_candidate = joint_ep if joint_ep != "R0_K8" else nonbaseline_ep
    frozen = []
    for name, reason in ((ep_candidate, (
                            "DISCOVERY_JOINT_EP_CRITICAL_TTFT"
                            if joint_ep != "R0_K8"
                            else "DISCOVERY_BEST_NONBASELINE_EP_SCREEN_C0_WON"
                         )),
                         (joint_pair, "DISCOVERY_JOINT_TOTAL_PAIR")):
        if name is not None and name not in [row["config"] for row in frozen]:
            frozen.append({"config": name, "reason": reason})
        if len(frozen) == 2:
            break

    matched_pairs = []
    names = sorted(configs)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            left_pairs = float(configs[left]["mean_logical_visual_pairs"])
            right_pairs = float(configs[right]["mean_logical_visual_pairs"])
            relative = abs(left_pairs - right_pairs) / max(left_pairs, right_pairs)
            if relative > 0.05:
                continue
            left_rows = {row["base_request_id"]: row for row in grouped[left]}
            right_rows = {row["base_request_id"]: row for row in grouped[right]}
            ids = sorted(set(left_rows) & set(right_rows))
            quality_delta = np.asarray([
                float(right_rows[key]["correct"]) - float(left_rows[key]["correct"])
                for key in ids
            ])
            rng = np.random.default_rng(3407)
            boot = quality_delta[
                rng.integers(0, len(ids), size=(10_000, len(ids)))
            ].mean(1)
            matched_pairs.append({
                "left": left,
                "right": right,
                "mean_pair_difference_fraction": relative,
                "right_minus_left_quality_pp": float(quality_delta.mean() * 100),
                "quality_bootstrap_95ci_pp": [
                    float(np.quantile(boot, 0.025) * 100),
                    float(np.quantile(boot, 0.975) * 100),
                ],
                "right_minus_left_ttft_ms": (
                    None
                    if configs[right].get("measured_ttft_ms") is None
                    or configs[left].get("measured_ttft_ms") is None
                    else float(configs[right]["measured_ttft_ms"])
                    - float(configs[left]["measured_ttft_ms"])
                ),
                "ttft_evidence": (
                    "PAIRED_CLEAN_TRUE_EP4_REPRESENTATIVE_COHORT"
                    if configs[right].get("measured_ttft_ms") is not None
                    and configs[left].get("measured_ttft_ms") is not None
                    else "N/A_UNMATCHED_SEMANTIC_RUN_SCHEDULER_SETTINGS"
                ),
            })

    interactions = {}
    for resolution in ("R05", "R025"):
        for k in (6, 4):
            name = f"{resolution}_K{k}"
            joint_rows = {row["base_request_id"]: row for row in grouped[name]}
            r_rows = {row["base_request_id"]: row for row in grouped[f"{resolution}_K8"]}
            k_rows = {row["base_request_id"]: row for row in grouped[f"R0_K{k}"]}
            ids = sorted(set(baseline_by_id) & set(joint_rows) & set(r_rows) & set(k_rows))
            values = np.asarray([
                float(joint_rows[key]["correct"])
                - float(r_rows[key]["correct"])
                - float(k_rows[key]["correct"])
                + float(baseline_by_id[key]["correct"])
                for key in ids
            ])
            rng = np.random.default_rng(3407)
            boot = values[rng.integers(0, len(ids), size=(10_000, len(ids)))].mean(1)
            interactions[name] = {
                "interaction_pp": float(values.mean() * 100),
                "bootstrap_95ci_pp": [float(np.quantile(boot, 0.025) * 100),
                                      float(np.quantile(boot, 0.975) * 100)],
            }

    return {
        "schema_version": "mllm_resolution_expert_tradeoff.analysis.v1",
        "evidence": "MEASURED_TRUE_EP4_DISCOVERY_OFFICIAL_REPRODUCTION_PUBLIC_PROTOCOL_PARTIAL",
        "rows": len(enriched),
        "base_requests": len({row["base_request_id"] for row in enriched}),
        "configs": configs,
        "quality_constraint": "discovery paired accuracy delta >= -1pp; heldout CI decides",
        "controls": {
            "C0_reference": "R0_K8",
            "C1_resolution_only": resolution_only,
            "C2_expert_only": expert_only,
            "C3_independent_resolution_then_k": independent_r_then_k,
            "C3_independent_k_then_resolution": independent_k_then_r,
            "C4_joint_pair_only": joint_pair,
            "C5_joint_ep_critical_path": (
                joint_ep if joint_ep is not None
                else "PENDING_CLEAN_PHYSICAL_EP4_COST_SUMMARY"
            ),
        },
        "matched_pair_comparisons": matched_pairs,
        "quality_interactions": interactions,
        "frozen_candidates": frozen,
        "scored_rows": enriched,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-policy", type=Path)
    parser.add_argument(
        "--physical-cost-summary",
        type=Path,
        help="clean/profiled representative TRUE EP4 cost for all nine configs",
    )
    args = parser.parse_args()
    physical_costs = None
    if args.physical_cost_summary:
        cost_payload = json.loads(
            args.physical_cost_summary.read_text(encoding="utf-8")
        )
        physical_costs = cost_payload["configs"]
    analysis = analyze(_read_jsonl(args.results), physical_costs)
    atomic_json(args.output, analysis)
    if args.frozen_policy:
        atomic_json(args.frozen_policy, {
            "schema_version": "mllm_resolution_expert_tradeoff.frozen_policy.v1",
            "frozen_from": str(args.output.resolve()),
            "selection_split": "discovery96",
            "heldout_accessed": False,
            "candidates": analysis["frozen_candidates"],
            "quality_constraint": analysis["quality_constraint"],
        })
    print(json.dumps({
        "rows": analysis["rows"],
        "controls": analysis["controls"],
        "frozen_candidates": analysis["frozen_candidates"],
    }, indent=2))


if __name__ == "__main__":
    main()
