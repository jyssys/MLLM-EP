#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from poc_mllm_rank_aware_active_expert_reduction.contract import atomic_json, read_jsonl_gz


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def spearman(values_a: list[float], values_b: list[float]) -> float | None:
    ranked_a = _average_ranks(np.asarray(values_a, dtype=float))
    ranked_b = _average_ranks(np.asarray(values_b, dtype=float))
    if len(ranked_a) < 3 or np.std(ranked_a) == 0 or np.std(ranked_b) == 0:
        return None
    return float(np.corrcoef(ranked_a, ranked_b)[0, 1])


def load_vds(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in glob.glob(str(root / "raw_vds" / "worker_pid*.jsonl")):
        rows.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["request_id"], row["layer"], row["decode_step"])].append(row)
    result = []
    for key, values in grouped.items():
        result.append({
            "request_id": key[0], "layer": key[1], "decode_step": key[2],
            "domain": values[0]["domain"], "tp_observations": len(values),
            "vds_attn_mean": float(np.mean([row["vds_attn_mean"] for row in values])),
            "vds_attn_max": float(np.mean([row["vds_attn_max"] for row in values])),
            "vds_attn_top_quartile_mean": float(np.mean([
                row["vds_attn_top_quartile_mean"] for row in values])),
            "vds_contribution_mean": float(np.mean([
                row["vds_contribution_mean"] for row in values])),
            "vds_contribution_max": float(np.mean([
                row["vds_contribution_max"] for row in values])),
        })
    return result


def corr(rows: list[dict[str, Any]], key: str) -> float | None:
    if len(rows) < 3 or np.std([row[key] for row in rows]) == 0:
        return None
    return spearman([row[key] for row in rows],
                    [row["functional_damage_score"] for row in rows])


def bootstrap_difference(rows: list[dict[str, Any]], low: float, high: float,
                         key: str, seed: int = 3407) -> dict[str, Any]:
    requests = sorted({row["request_id"] for row in rows})
    rng = np.random.default_rng(seed)
    def statistic(local: list[dict[str, Any]]) -> float | None:
        a = [row["functional_damage_score"] for row in local if row[key] <= low]
        b = [row["functional_damage_score"] for row in local if row[key] >= high]
        return None if not a or not b else float(np.mean(b) - np.mean(a))
    point = statistic(rows)
    values = []
    for _ in range(2000):
        chosen = rng.choice(requests, len(requests), replace=True)
        local = [row for request in chosen for row in rows if row["request_id"] == request]
        value = statistic(local)
        if value is not None: values.append(value)
    return {"high_minus_low_mean_damage": point,
            "request_bootstrap_95ci": None if not values else [
                float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
            "bootstrap_samples": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vds-run", type=Path, required=True)
    parser.add_argument("--damage-labels", type=Path, required=True)
    parser.add_argument("--caption-causal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--joined", type=Path, required=True)
    args = parser.parse_args()
    vds = load_vds(args.vds_run)
    lookup = {(row["request_id"], row["layer"], row["decode_step"]): row for row in vds}
    labels = read_jsonl_gz(args.damage_labels)
    joined = [{**row, **{key: value for key, value in lookup[
        (row["request_id"], row["layer"], row["decode_step"])].items()
        if key.startswith("vds_") or key == "tp_observations"}}
              for row in labels
              if (row["request_id"], row["layer"], row["decode_step"]) in lookup]
    calibration = [row for row in joined if row["split"] == "calibration"]
    heldout = [row for row in joined if row["split"] == "heldout"]
    primary = "vds_attn_mean"
    if calibration:
        low, high = map(float, np.quantile([row[primary] for row in calibration], [1/3, 2/3]))
    else:
        low = high = float("nan")
    bins = {}
    for split, rows in (("calibration", calibration), ("heldout", heldout)):
        bins[split] = {
            "count": len(rows), "request_count": len({row["request_id"] for row in rows}),
            "vds_attn_damage_spearman": corr(rows, "vds_attn_mean"),
            "vds_contribution_damage_spearman": corr(rows, "vds_contribution_mean"),
            "damage_by_frozen_vds_bin": {
                "low": float(np.mean([row["functional_damage_score"] for row in rows
                                      if row[primary] <= low])) if any(row[primary] <= low for row in rows) else None,
                "mid": float(np.mean([row["functional_damage_score"] for row in rows
                                      if low < row[primary] < high])) if any(low < row[primary] < high for row in rows) else None,
                "high": float(np.mean([row["functional_damage_score"] for row in rows
                                       if row[primary] >= high])) if any(row[primary] >= high for row in rows) else None,
            },
        }
    difference = bootstrap_difference(heldout, low, high, primary) if heldout else {}
    causal = json.loads(args.caption_causal.read_text())
    caption_cross_domain = (causal.get("aggregate", {}).get("cross_domain", {})
                            .get("1-256", {}).get("4", {}))
    direct_alignment = False
    monotonic = False
    values = bins.get("heldout", {}).get("damage_by_frozen_vds_bin", {})
    if values and all(values.get(key) is not None for key in ("low", "mid", "high")):
        monotonic = values["low"] <= values["mid"] <= values["high"]
    ci = difference.get("request_bootstrap_95ci")
    positive = ci is not None and ci[0] > 0
    payload = {
        "schema_version": "rank_aware_q2_vds.v1",
        "status": "MEASURED_DEPLOYABLE_VDS_A_B_AND_FUNCTIONAL_DROP_ONE",
        "vds_run": str(args.vds_run.resolve()), "vds_points": len(vds),
        "functional_label_count": len(labels), "joined_path_count": len(joined),
        "primary_vds_frozen_before_damage_analysis": primary,
        "calibration_tertiles": {"low_max": low, "high_min": high},
        "results": bins, "heldout_high_vs_low": difference,
        "causal_vds_c": {
            "status": "PARTIAL_EXTERNAL_CAUSAL_ALIGNMENT_ONLY",
            "direct_same_observation_vds_to_image_swap_alignment": direct_alignment,
            "caption_image_swap_expert_change_pct": (
                100.0 * caption_cross_domain["expert_changed_fraction"]
                if "expert_changed_fraction" in caption_cross_domain else None),
            "caption_image_swap_ep4_changed_rank_pct": (
                100.0 * caption_cross_domain["changed_rank_fraction"]
                if "changed_rank_fraction" in caption_cross_domain else None),
            "limitation": "Existing causal caption lane establishes image dependence but is not the same 32-request functional-damage cohort.",
        },
        "k8_to_k6_k4_functional_diagnostic": "NOT_RUN_Q2_GATE_REQUIRES_DROP_ONE_LINK_FIRST",
        "gate": {
            "monotonic_heldout_damage": monotonic,
            "heldout_high_vs_low_ci_positive": positive,
            "direct_causal_alignment": direct_alignment,
            "q2_pass": monotonic and positive and direct_alignment,
        },
        "verdict": "VISUAL_PROTECTION_SUPPORTED" if monotonic and positive and direct_alignment
                   else "VISUAL_DEPENDENCY_NOT_LINKED_TO_SKIP_DAMAGE",
    }
    atomic_json(args.output, payload)
    from poc_mllm_rank_aware_active_expert_reduction.contract import write_jsonl_gz
    write_jsonl_gz(args.joined, joined)
    print(json.dumps({k: payload[k] for k in (
        "status", "vds_points", "functional_label_count", "joined_path_count",
        "heldout_high_vs_low", "gate", "verdict")}, indent=2))


if __name__ == "__main__": main()
