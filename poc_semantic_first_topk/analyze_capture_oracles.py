#!/usr/bin/env python3
"""Full integer-K and EP-refinement oracles over real Qwen3-VL captures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from poc_flashvep.mllm_moe_transient_branch_compression.analyze_branches import (
    load_layer,
    modality_and_coords,
)
from poc_semantic_first_topk.policy_core import (
    COARSE_K,
    FULL_K,
    branch_risk,
    ep_same_budget_refinement,
    output_metrics,
    rank_load,
    semantic_allocation,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def row_for(sample: dict, layer: int, policy: str, risk_kind: str,
            target_fraction: float, data: dict[str, np.ndarray], modality: np.ndarray,
            k: np.ndarray, risk_value: float, swaps: int = 0,
            semantic_max_reduction: float | None = None) -> dict:
    baseline_k = np.full(len(k), 8, dtype=np.int16)
    before = rank_load(data["ids"], baseline_k)
    after = rank_load(data["ids"], k)
    vision = modality == "vision"
    possible = int(vision.sum()) * 8
    dropped = int(np.sum(8 - k[vision]))
    metrics = output_metrics(data["weights"], data["outputs"], k, modality, False)
    renorm = output_metrics(data["weights"], data["outputs"], k, modality, True)
    result = {
        "sample_id": sample["sample_id"], "category": sample["category"],
        "edge": sample["fixed_edge"], "layer": layer, "policy": policy,
        "risk_kind": risk_kind, "target_fraction": target_fraction,
        "vision_tokens": int(vision.sum()), "dropped": dropped,
        "vision_assignment_reduction": dropped / max(possible, 1),
        "risk_value": risk_value, "swaps": swaps,
        "max_rank_before": int(before.max()), "max_rank_after": int(after.max()),
        "max_rank_reduction": 1 - float(after.max()) / max(float(before.max()), 1),
        "rank_cv_after": float(after.std() / max(after.mean(), 1e-12)),
        "k1": int(np.count_nonzero(k[vision] == 1)),
        "k2": int(np.count_nonzero(k[vision] == 2)),
        "k3": int(np.count_nonzero(k[vision] == 3)),
        "k4": int(np.count_nonzero(k[vision] == 4)),
        "k5": int(np.count_nonzero(k[vision] == 5)),
        "k6": int(np.count_nonzero(k[vision] == 6)),
        "k7": int(np.count_nonzero(k[vision] == 7)),
        "k8": int(np.count_nonzero(k[vision] == 8)),
        **metrics,
        "renorm_combined_rel_l2": renorm["combined_rel_l2"],
        "renorm_vision_rel_l2": renorm["vision_rel_l2"],
    }
    if semantic_max_reduction is not None:
        result["ep_advantage_over_semantic"] = (
            result["max_rank_reduction"] / max(semantic_max_reduction, 1e-12))
    return result


def summarize(rows: list[dict]) -> list[dict]:
    keys = sorted({(r["policy"], r["risk_kind"], r["target_fraction"]) for r in rows})
    metrics = ["vision_assignment_reduction", "max_rank_reduction", "rank_cv_after",
               "combined_rel_l2", "vision_rel_l2", "combined_cosine",
               "renorm_combined_rel_l2", "renorm_vision_rel_l2", "swaps"]
    output = []
    for policy, risk_kind, fraction in keys:
        selected = [r for r in rows if (r["policy"], r["risk_kind"], r["target_fraction"])
                    == (policy, risk_kind, fraction)]
        record = {"policy": policy, "risk_kind": risk_kind,
                  "target_fraction": fraction, "n": len(selected)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected])
            record[f"{metric}_median"] = float(np.median(values))
            record[f"{metric}_p10"] = float(np.quantile(values, .1))
            record[f"{metric}_p90"] = float(np.quantile(values, .9))
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 12, 24, 36, 44, 47])
    parser.add_argument("--samples", nargs="*", default=[])
    parser.add_argument("--risk-kinds", nargs="+",
                        default=["router", "contribution", "contribution_squared"])
    parser.add_argument("--fractions", type=float, nargs="+",
                        default=[.01, .02, .05, .1, .15, .2, .25, .3, .4, .5, .6])
    parser.add_argument("--risk-slacks", type=float, nargs="+", default=[.01, .02, .05, .1])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.capture / "manifest.json").read_text())
    rows: list[dict] = []
    samples = [row for row in manifest["samples"]
               if not args.samples or row["sample_id"] in set(args.samples)]
    for sample in samples:
        for layer in args.layers:
            data = load_layer(args.capture, manifest, sample, layer)
            modality, _ = modality_and_coords(sample, data["position"])
            total = int(np.count_nonzero(modality == "vision")) * 8
            for risk_kind in args.risk_kinds:
                risk = branch_risk(data["weights"], data["outputs"], risk_kind)
                for fraction in args.fractions:
                    target = int(round(total * fraction))
                    semantic_rows = {}
                    for grid_name, grid in (("full", FULL_K), ("coarse", COARSE_K)):
                        allocation = semantic_allocation(risk, modality, target, grid)
                        base_row = row_for(sample, layer, f"semantic_{grid_name}", risk_kind,
                                           fraction, data, modality, allocation.k, allocation.risk)
                        rows.append(base_row)
                        semantic_rows[grid_name] = (allocation, base_row)
                    allocation, base_row = semantic_rows["full"]
                    for slack in args.risk_slacks:
                        refined = ep_same_budget_refinement(
                            data["ids"], risk, modality, allocation.k, risk_slack=slack)
                        rows.append(row_for(
                            sample, layer, f"ep_refined_s{slack:g}", risk_kind, fraction,
                            data, modality, refined.k, refined.final_risk, refined.swaps,
                            base_row["max_rank_reduction"]))
    write_csv(args.output / "allocation_rows.csv", rows)
    write_csv(args.output / "allocation_summary.csv", summarize(rows))
    (args.output / "manifest.json").write_text(json.dumps({
        "source_capture": str(args.capture.resolve()), "layers": args.layers,
        "samples": [row["sample_id"] for row in samples], "risk_kinds": args.risk_kinds,
        "fractions": args.fractions, "risk_slacks": args.risk_slacks,
        "evidence_boundary": "captured real expert-output logical oracle; no task benchmark or runtime claim",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
