#!/usr/bin/env python3
"""CPU-only recovery checks for the semantic-first EP-refinement PoC.

This script intentionally makes no GPU latency claim.  It re-scores already
captured full-model answers, solves a same-assignment/risk-budget rank-load
MILP under virtual EP placements, and audits the prior layer replay results.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from poc_flashvep.mllm_moe_transient_branch_compression.analyze_branches import (
    modality_and_coords,
)
from poc_semantic_first_topk.policy_core import (
    FULL_K,
    keep_from_k,
    semantic_allocation,
    spatial_schedule_k,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalized_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9.%+-]+", " ", text.lower()).split())


def numeric_value(text: str) -> float | None:
    value = normalized_text(text)
    try:
        return float(value[:-1]) / 100.0 if value.endswith("%") else float(value)
    except ValueError:
        return None


def semantically_equivalent(left: str, right: str) -> bool:
    """Conservative equivalence: normalized identity or equal numeric value."""
    if normalized_text(left) == normalized_text(right):
        return True
    a, b = numeric_value(left), numeric_value(right)
    return bool(a is not None and b is not None and
                np.isclose(a, b, rtol=1e-6, atol=1e-9))


def chartqa_relaxed(prediction: str, target: str, tolerance: float = 0.05) -> bool:
    """Official ChartQA relaxed numeric accuracy semantics."""
    pred, gold = numeric_value(prediction), numeric_value(target)
    if pred is not None and gold not in (None, 0.0):
        return abs(pred - gold) / abs(gold) <= tolerance
    return normalized_text(prediction) == normalized_text(target)


def official_score(dataset: str, prediction: str, answers: list[str]) -> bool:
    if dataset == "chartqa":
        return any(chartqa_relaxed(prediction.strip(), str(answer).strip())
                   for answer in answers)
    # This project's GQA screening protocol uses answer-list exact match.
    return any(normalized_text(prediction) == normalized_text(str(answer)) for answer in answers)


def quality_recheck(quality_root: Path, requests_path: Path, output: Path) -> dict:
    manifest = {row["id"]: row for row in read_jsonl(requests_path)}
    records = [row for path in quality_root.glob("quality_*.jsonl")
               for row in read_jsonl(path)]
    grouped: dict[str, dict[str, dict]] = {}
    for row in records:
        grouped.setdefault(row["request_id"], {})[row["policy"]] = row
    rows = []
    for request_id in sorted(grouped):
        policies = grouped[request_id]
        if "stock" not in policies or "ep_refined_0.3_s0.05" not in policies:
            continue
        base, candidate = policies["stock"], policies["ep_refined_0.3_s0.05"]
        request = manifest[request_id]
        base_score = official_score(base["dataset"], base["prediction"], request["answers"])
        candidate_score = official_score(candidate["dataset"], candidate["prediction"],
                                         request["answers"])
        rows.append({
            "request_id": request_id,
            "dataset": base["dataset"],
            "question": request["question"],
            "answers": json.dumps(request["answers"], ensure_ascii=False),
            "stock_prediction": base["prediction"],
            "policy_prediction": candidate["prediction"],
            "token_exact": bool(candidate["short_greedy_exact"]),
            "semantic_equivalent_to_stock": semantically_equivalent(
                base["prediction"], candidate["prediction"]),
            "stock_official_correct": bool(base_score),
            "policy_official_correct": bool(candidate_score),
            "paired_score_delta": int(candidate_score) - int(base_score),
            "kl_per_answer_token": candidate["kl_per_answer_token"],
            "teacher_forced_greedy_equal": candidate["teacher_forced_greedy_equal"],
        })
    write_csv(output / "quality_recheck_rows.csv", rows)
    deltas = np.asarray([row["paired_score_delta"] for row in rows], dtype=float)
    rng = np.random.default_rng(20260911)
    bootstrap = np.asarray([
        rng.choice(deltas, size=len(deltas), replace=True).mean() * 100
        for _ in range(10000)
    ])
    result = {
        "requests": len(rows),
        "stock_official_correct": sum(row["stock_official_correct"] for row in rows),
        "policy_official_correct": sum(row["policy_official_correct"] for row in rows),
        "stock_accuracy_pct": 100 * np.mean([row["stock_official_correct"] for row in rows]),
        "policy_accuracy_pct": 100 * np.mean([row["policy_official_correct"] for row in rows]),
        "paired_delta_pp": 100 * deltas.mean(),
        "paired_bootstrap_95ci_pp": np.quantile(bootstrap, [.025, .975]).tolist(),
        "stock_correct_to_policy_wrong": int(np.count_nonzero(deltas < 0)),
        "stock_wrong_to_policy_correct": int(np.count_nonzero(deltas > 0)),
        "token_nonexact": int(sum(not row["token_exact"] for row in rows)),
        "semantically_equivalent_nonexact": int(sum(
            not row["token_exact"] and row["semantic_equivalent_to_stock"] for row in rows)),
        "changed_rows": [row for row in rows if not row["token_exact"]],
        "evidence_boundary": (
            "measured saved full-model greedy outputs; official-style ChartQA relaxed and "
            "project GQA exact scoring; n=32 is a bounded safety check, not a population proof"
        ),
    }
    (output / "quality_recheck.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def virtual_rank_load(ids: np.ndarray, k: np.ndarray, ep_size: int) -> np.ndarray:
    if 128 % ep_size:
        raise ValueError(f"EP{ep_size} does not divide 128 experts")
    ranks = ids // (128 // ep_size)
    keep = keep_from_k(k, ids.shape[1])
    return np.asarray([np.count_nonzero(keep & (ranks == rank))
                       for rank in range(ep_size)], dtype=np.int64)


def optimal_refinement(ids: np.ndarray, risk: np.ndarray, modality: np.ndarray,
                       semantic_k: np.ndarray, ep_size: int,
                       risk_slack: float = 0.05) -> tuple[np.ndarray, dict]:
    """Exact min-max rank-load oracle at fixed assignments and risk budget.

    Each vision token chooses a prefix k in [1,8]. Text remains k=8. The MILP
    preserves the semantic allocation's exact assignment count and permits at
    most ``risk_slack`` additional summed omitted router risk.
    """
    vision = np.flatnonzero(modality == "vision")
    text = np.flatnonzero(modality != "vision")
    options = 8
    variables = len(vision) * options + 1
    z_index = variables - 1
    rows = len(vision) + 2 + ep_size
    matrix = lil_matrix((rows, variables), dtype=np.float64)
    lower = np.full(rows, -np.inf)
    upper = np.full(rows, np.inf)
    rank_of = ids // (128 // ep_size)
    semantic_risk = float(sum(risk[token, int(semantic_k[token]):].sum()
                              for token in vision))
    target_k = int(semantic_k[vision].sum())
    text_load = virtual_rank_load(ids[text], np.full(len(text), 8, dtype=np.int16),
                                  ep_size) if len(text) else np.zeros(ep_size, dtype=np.int64)
    for local, token in enumerate(vision):
        for choice, k in enumerate(range(1, 9)):
            column = local * options + choice
            matrix[local, column] = 1
            matrix[len(vision), column] = k
            matrix[len(vision) + 1, column] = float(risk[token, k:].sum())
            for rank in range(ep_size):
                matrix[len(vision) + 2 + rank, column] = int(
                    np.count_nonzero(rank_of[token, :k] == rank))
        lower[local] = upper[local] = 1
    lower[len(vision)] = upper[len(vision)] = target_k
    upper[len(vision) + 1] = semantic_risk * (1 + risk_slack) + 1e-10
    for rank in range(ep_size):
        matrix[len(vision) + 2 + rank, z_index] = -1
        upper[len(vision) + 2 + rank] = -int(text_load[rank])
    objective = np.zeros(variables); objective[z_index] = 1
    bounds = Bounds(np.zeros(variables), np.r_[np.ones(variables - 1), np.inf])
    integrality = np.r_[np.ones(variables - 1), 1]
    solved = milp(objective, integrality=integrality, bounds=bounds,
                  constraints=LinearConstraint(matrix.tocsr(), lower, upper),
                  options={"time_limit": 20.0, "mip_rel_gap": 0.0})
    if not solved.success or solved.x is None:
        raise RuntimeError(f"MILP failed: {solved.message}")
    k = np.full(len(modality), 8, dtype=np.int16)
    choices = solved.x[:-1].reshape(len(vision), options).argmax(axis=1) + 1
    k[vision] = choices.astype(np.int16)
    final_risk = float(sum(risk[token, int(k[token]):].sum() for token in vision))
    if int(k[vision].sum()) != target_k or final_risk > semantic_risk * (1 + risk_slack) + 1e-6:
        raise AssertionError((target_k, int(k[vision].sum()), semantic_risk, final_risk))
    return k, {
        "solver": "scipy.optimize.milp/HiGHS exact min-max",
        "semantic_risk": semantic_risk,
        "final_risk": final_risk,
        "risk_ratio": final_risk / max(semantic_risk, 1e-12),
        "target_vision_assignments": target_k,
        "objective_max_load": int(round(solved.fun)),
    }


def load_metrics(stock_load: np.ndarray, semantic_load: np.ndarray,
                 refined_load: np.ndarray) -> dict:
    stock_max = int(stock_load.max())
    sem_max, ref_max = int(semantic_load.max()), int(refined_load.max())
    return {
        "stock_max_rank_load": stock_max,
        "semantic_max_rank_load": sem_max,
        "semantic_mean_rank_load": float(semantic_load.mean()),
        "semantic_max_mean": sem_max / max(float(semantic_load.mean()), 1e-12),
        "refined_max_rank_load": ref_max,
        "refined_mean_rank_load": float(refined_load.mean()),
        "refined_max_mean": ref_max / max(float(refined_load.mean()), 1e-12),
        "additional_max_rank_work_removed": sem_max - ref_max,
        "incremental_max_rank_oracle_pct": 100 * (sem_max - ref_max) / max(sem_max, 1),
        "semantic_max_rank_reduction_vs_stock_pct": 100 * (stock_max - sem_max) /
                                                       max(stock_max, 1),
        "refined_max_rank_reduction_vs_stock_pct": 100 * (stock_max - ref_max) /
                                                      max(stock_max, 1),
        "incremental_max_rank_reduction_pp": 100 * (sem_max - ref_max) / max(stock_max, 1),
    }


def summarize_scale(rows: list[dict], source: str) -> list[dict]:
    output = []
    for ep_size in sorted({int(row["ep_size"]) for row in rows if row["source"] == source}):
        part = [row for row in rows if row["source"] == source and row["ep_size"] == ep_size]
        record = {"source": source, "ep_size": ep_size, "n": len(part)}
        for metric in ("semantic_max_rank_load", "semantic_max_mean",
                       "refined_max_rank_load", "refined_max_mean",
                       "additional_max_rank_work_removed", "incremental_max_rank_oracle_pct",
                       "semantic_max_rank_reduction_vs_stock_pct",
                       "refined_max_rank_reduction_vs_stock_pct",
                       "incremental_max_rank_reduction_pp"):
            values = np.asarray([row[metric] for row in part], dtype=float)
            record[f"{metric}_median"] = float(np.median(values))
            record[f"{metric}_p10"] = float(np.quantile(values, .1))
            record[f"{metric}_p90"] = float(np.quantile(values, .9))
        output.append(record)
    return output


def captured_virtual_scale(capture: Path, schedules: Path, output: Path) -> list[dict]:
    manifest = json.loads((capture / "manifest.json").read_text())
    group_k = json.loads(schedules.read_text())["global_policies"]["semantic_full_0.3"]["group_k"]
    rows = []
    for sample in manifest["samples"]:
        source_rank = next(int(row["source_dp_rank"]) for row in manifest["schedule"]
                           if row["sample_id"] == sample["sample_id"])
        for layer in manifest["policy"]["layers"]:
            # Only router IDs/weights are needed here. Avoid decompressing the
            # multi-GB expert-output capture used by the earlier quality oracle.
            parts = [np.load(capture / "raw" /
                             f"router.{sample['sample_id']}.dp{source_rank}.tp{tp}.layer{layer}.npz")
                     for tp in (0, 1)]
            selected = np.concatenate([part["selected"] for part in parts])[
                :int(sample["prompt_tokens"])]
            positions = np.flatnonzero(selected)
            ids = np.concatenate([part["topk_ids"] for part in parts], axis=0)[
                :int(sample["prompt_tokens"])][positions].astype(np.int64)
            weights = np.concatenate([part["topk_weights"] for part in parts], axis=0)[
                :int(sample["prompt_tokens"])][positions].astype(np.float64)
            modality, coords = modality_and_coords(sample, positions)
            semantic_k = spatial_schedule_k(modality, coords, group_k)
            risk = weights
            for ep_size in (2, 4, 8, 16):
                refined_k, solver = optimal_refinement(
                    ids, risk, modality, semantic_k, ep_size)
                semantic_load = virtual_rank_load(ids, semantic_k, ep_size)
                refined_load = virtual_rank_load(ids, refined_k, ep_size)
                stock_load = virtual_rank_load(ids, np.full(len(ids), 8, dtype=np.int16),
                                               ep_size)
                rows.append({
                    "source": "EP4_output_capture_virtual_linear_remap",
                    "sample_id": sample["sample_id"], "category": sample["category"],
                    "layer": layer, "ep_size": ep_size,
                    "semantic_vision_assignments": int(semantic_k[modality == "vision"].sum()),
                    "refined_vision_assignments": int(refined_k[modality == "vision"].sum()),
                    "risk_ratio": solver["risk_ratio"],
                    "semantic_load": json.dumps(semantic_load.tolist()),
                    "refined_load": json.dumps(refined_load.tolist()),
                    **load_metrics(stock_load, semantic_load, refined_load),
                })
    return rows


def historical_ep8_routes(route_root: Path, output: Path) -> list[dict]:
    """Use unique real EP8 Qwen routes for an EP8-only sensitivity control."""
    candidates = sorted(route_root.glob("route_wave*_layer*_dp0_ep*.npz"))
    grouped: dict[tuple[int, int], list[Path]] = {}
    for path in candidates:
        match = re.search(r"wave(\d+)_layer(\d+)_dp0_ep(\d+)", path.name)
        if match and int(match.group(3)) in (0, 1):
            grouped.setdefault((int(match.group(1)), int(match.group(2))), []).append(path)
    rows, seen = [], set()
    representative_layers = {4, 12, 24, 36, 44, 47}
    for (wave, layer), paths in sorted(grouped.items()):
        if layer not in representative_layers or len(paths) != 2:
            continue
        arrays = [np.load(path) for path in sorted(paths)]
        order = np.argsort(np.concatenate([array["token_positions"] for array in arrays]))
        ids = np.concatenate([array["topk_ids"] for array in arrays])[order].astype(np.int64)
        weights = np.concatenate([array["topk_weights"] for array in arrays])[order].astype(np.float64)
        modality = np.where(np.concatenate([array["modality"] for array in arrays])[order] == 1,
                            "vision", "text")
        digest = hashlib.sha256(ids.tobytes() + weights.tobytes() + modality.tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        risk = weights
        target = int(round(np.count_nonzero(modality == "vision") * 8 * .30))
        semantic = semantic_allocation(risk, modality, target, FULL_K)
        refined_k, solver = optimal_refinement(ids, risk, modality, semantic.k, 8)
        semantic_load = virtual_rank_load(ids, semantic.k, 8)
        refined_load = virtual_rank_load(ids, refined_k, 8)
        stock_load = virtual_rank_load(ids, np.full(len(ids), 8, dtype=np.int16), 8)
        rows.append({
            "source": "historical_real_EP8_routes_router_risk_schedule",
            "sample_id": f"wave{wave}", "category": "historical_ep8",
            "layer": layer, "ep_size": 8,
            "semantic_vision_assignments": int(semantic.k[modality == "vision"].sum()),
            "refined_vision_assignments": int(refined_k[modality == "vision"].sum()),
            "risk_ratio": solver["risk_ratio"],
            "semantic_load": json.dumps(semantic_load.tolist()),
            "refined_load": json.dumps(refined_load.tolist()),
            **load_metrics(stock_load, semantic_load, refined_load),
        })
    return rows


def middle_layer_oracle(replay_csv: Path, scale_rows: list[dict], output: Path) -> dict:
    replay = pd.read_csv(replay_csv)
    selected = []
    # Exact-budget l24 has three requests. Early/late are the camera controls.
    for layer in (4, 24, 44):
        if layer == 24:
            part = replay[(replay.layer == layer) & replay.experiment.str.contains("exactbudget")]
        else:
            part = replay[(replay.layer == layer) &
                          replay.experiment.str.contains(f"global_camera_e448_l{layer}")]
        pivot = part.pivot_table(index=["experiment", "sample", "layer"], columns="policy",
                                 values=["dispatch_ms", "expert_ms", "combine_ms", "moe_ms",
                                         "projected_ttft_reduction_pct"], aggfunc="first")
        semantic = "global_semantic_full_0.3"
        refined = "global_semantic_full_0.3_ep_refined_s0.05"
        for index, row in pivot.iterrows():
            record = {"experiment": index[0], "sample": index[1], "layer": index[2]}
            for metric in ("dispatch_ms", "expert_ms", "combine_ms", "moe_ms",
                           "projected_ttft_reduction_pct"):
                record[f"semantic_{metric}"] = float(row[(metric, semantic)])
                record[f"refined_{metric}"] = float(row[(metric, refined)])
                record[f"refine_delta_{metric}"] = float(row[(metric, refined)] - row[(metric, semantic)])
            selected.append(record)
    write_csv(output / "middle_layer_replay_pairs.csv", selected)
    per_layer = []
    for layer in (4, 24, 44):
        part = [row for row in selected if row["layer"] == layer]
        per_layer.append({
            "layer": layer, "requests": len(part),
            "paired_incremental_ttft_pp_median": float(np.median([
                row["refine_delta_projected_ttft_reduction_pct"] for row in part])),
            "dispatch_delta_ms_median": float(np.median([
                row["refine_delta_dispatch_ms"] for row in part])),
            "expert_delta_ms_median": float(np.median([
                row["refine_delta_expert_ms"] for row in part])),
            "combine_delta_ms_median": float(np.median([
                row["refine_delta_combine_ms"] for row in part])),
            "moe_delta_ms_median": float(np.median([
                row["refine_delta_moe_ms"] for row in part])),
        })
    write_csv(output / "middle_layer_summary.csv", per_layer)
    increments = {row["layer"]: max(row["paired_incremental_ttft_pp_median"], 0.0)
                  for row in per_layer}
    total = sum(increments.values())
    retention = increments[24] / total if total else 0.0

    # A broader logical control over six captured layers. It is assignment-only.
    virtual = [row for row in scale_rows
               if row["source"] == "EP4_output_capture_virtual_linear_remap" and
               row["ep_size"] == 4]
    layer_logical = []
    for layer in sorted({row["layer"] for row in virtual}):
        values = [row["incremental_max_rank_oracle_pct"] for row in virtual
                  if row["layer"] == layer]
        layer_logical.append({"layer": layer, "n": len(values),
                              "incremental_max_rank_oracle_pct_median": float(np.median(values))})
    write_csv(output / "captured_layer_load_oracle.csv", layer_logical)
    ordered = sorted(layer_logical, key=lambda row: row["incremental_max_rank_oracle_pct_median"],
                     reverse=True)
    all_mass = sum(row["incremental_max_rank_oracle_pct_median"] for row in ordered)
    top_rows = []
    running = 0.0
    for count, row in enumerate(ordered, 1):
        running += row["incremental_max_rank_oracle_pct_median"]
        top_rows.append({"selected_layers": count,
                         "layer_ids": json.dumps([item["layer"] for item in ordered[:count]]),
                         "logical_incremental_mass_retained_pct": 100 * running / max(all_mass, 1e-12),
                         "relative_refinement_layer_exposure_pct": 100 * count / len(ordered)})
    write_csv(output / "layer_subset_load_oracle.csv", top_rows)
    result = {
        "measured_paired_replay": per_layer,
        "separate_median_difference_reported_previously_pp": {
            "early_l4": 0.526013,
            "middle_l24": 2.592716,
            "late_l44": 0.573424,
        },
        "paired_middle_incremental_ttft_pp_median": increments[24],
        "three_stratum_incremental_ttft_mass_retained_by_middle_only_pct": 100 * retention,
        "refined_layer_fraction_middle_only_pct": 100 / 3,
        "quality_risk_statement": (
            "No middle-only full-model generation was measured. Restricting refinement from all "
            "layers to a middle stratum reduces refinement exposure to one third, but the quality "
            "benefit is an exposure proxy, not an accuracy measurement."
        ),
        "aggregation_warning": (
            "The prior ~2.59pp middle number subtracts two independently selected cross-request "
            "medians. The statistically aligned paired-request median is reported separately."
        ),
        "captured_layer_assignment_oracle": layer_logical,
        "captured_layer_subset_oracle": top_rows,
        "evidence_boundary": "DeepEP stage measurements for layers 4/24/44; subset extrapolation is an oracle",
    }
    (output / "middle_layer_oracle.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quality-root", type=Path, default=Path(
        "poc_semantic_first_topk/results/proxy_quality_heldout32_eprefine_20260911"))
    parser.add_argument("--requests", type=Path, default=Path(
        "/home/esjung/MLLM-EP-github/poc_flashvep/deepep_revalidation/results/"
        "top_tier_successor_mining_20260907_135950/data/requests.jsonl"))
    parser.add_argument("--capture", type=Path, default=Path(
        "/home/esjung/MLLM-EP-vision-hetero-topk/poc_vision_heterogeneous_topk/results/"
        "vision_capture_fresh_20260911_0050"))
    parser.add_argument("--schedules", type=Path, default=Path(
        "poc_semantic_first_topk/results/combined_calibration_gqa8_chartqa8_20260911/"
        "oracle_schedules_dp.json"))
    parser.add_argument("--historical-ep8", type=Path, default=Path(
        "poc_flashvep/deepep_revalidation/results/"
        "mllm_ep8_critical_rank_trace_20260904_run17/raw_routes"))
    parser.add_argument("--replay-summary", type=Path, default=Path(
        "poc_semantic_first_topk/results/final_analysis/replay_summary.csv"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    quality = quality_recheck(args.quality_root, args.requests, args.output)
    scale_rows = captured_virtual_scale(args.capture, args.schedules, args.output)
    scale_rows.extend(historical_ep8_routes(args.historical_ep8, args.output))
    write_csv(args.output / "ep_scale_oracle_rows.csv", scale_rows)
    summaries = []
    for source in sorted({row["source"] for row in scale_rows}):
        summaries.extend(summarize_scale(scale_rows, source))
    write_csv(args.output / "ep_scale_oracle_summary.csv", summaries)
    middle = middle_layer_oracle(args.replay_summary, scale_rows, args.output)
    result = {
        "quality": quality,
        "ep_scale_summary": summaries,
        "middle_layer": middle,
        "gpu_measurement_added": False,
        "evidence_boundary": (
            "Quality and prior EP4 DeepEP replay rows are measured facts. EP2/8/16 load results "
            "and layer selection are offline assignment oracles, not latency/TTFT measurements."
        ),
    }
    (args.output / "recovery_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "ep_summaries": summaries,
                      "quality": {key: quality[key] for key in (
                          "stock_official_correct", "policy_official_correct", "paired_delta_pp")}},
                     indent=2))


if __name__ == "__main__":
    main()
