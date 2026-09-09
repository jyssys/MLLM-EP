"""Create compact, reproducible tables from the completed branch analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def describe(group: pd.DataFrame, columns: list[str]) -> dict:
    result = {"n": len(group)}
    for column in columns:
        values = group[column].dropna().to_numpy(float)
        result[column] = {"mean": float(values.mean()), "p10": float(np.quantile(values, .1)),
                          "median": float(np.median(values)), "p90": float(np.quantile(values, .9))}
    return result


def matched_modality(pair: pd.DataFrame) -> list[dict]:
    rows = []
    keys = ["layer", "expert", "pair_kind"]
    for key, local in pair.groupby(keys):
        vision = local[local.modality == "vision"]
        text = local[local.modality == "text"]
        if vision.empty or text.empty:
            continue
        target = np.sort(text.input_relative_l2.to_numpy())
        target_output = text.sort_values("input_relative_l2").output_relative_l2.to_numpy()
        source = vision.input_relative_l2.to_numpy()
        pos = np.searchsorted(target, source).clip(0, len(target) - 1)
        before = np.maximum(pos - 1, 0)
        choose_before = np.abs(target[before] - source) < np.abs(target[pos] - source)
        pos[choose_before] = before[choose_before]
        gap = np.abs(source - target[pos])
        common = gap <= .02
        if not common.any():
            continue
        delta = vision.output_relative_l2.to_numpy()[common] - target_output[pos[common]]
        rows.append({"layer": key[0], "expert": key[1], "pair_kind": key[2], "n": len(delta),
                     "input_match_abs_median": float(np.median(gap[common])),
                     "vision_minus_text_output_rel_mean": float(delta.mean()),
                     "vision_minus_text_output_rel_median": float(np.median(delta))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    analysis = args.root / "analysis_full"
    pair = pd.read_csv(analysis / "pair_atlas.csv")
    policy = pd.read_csv(analysis / "compression_policies.csv")
    rle = pd.read_csv(analysis / "route_runs.csv")
    contribution = pd.read_csv(args.root / "contribution_full" / "contribution_oracle.csv")
    centroid = pd.read_csv(args.root / "centroid_full" / "centroid_oracle.csv")
    reconstruction = pd.read_csv(analysis / "capture_reconstruction.csv")
    manifest = json.loads((args.root / "capture_full" / "manifest.json").read_text())
    summary: dict = {
        "capture": {"samples": len(manifest["samples"]), "layers": manifest["policy"]["layers"],
                    "vision_tokens": sum(row["vision_tokens"] for row in manifest["samples"]),
                    "vision_assignments": sum(row["vision_tokens"] for row in manifest["samples"]) * 8 * len(manifest["policy"]["layers"]),
                    "prompt_tokens": sum(row["prompt_tokens"] for row in manifest["samples"]),
                    "reconstruction_min_cosine": float(reconstruction.min_cosine.min()),
                    "reconstruction_max_rel_l2": float(reconstruction.max_relative_l2.max())},
        "pairs": {}, "policies": {}, "rle": {}, "contribution": {}, "centroid": {}, "quality": {},
    }
    for (modality, kind), local in pair.groupby(["modality", "pair_kind"]):
        item = describe(local, ["input_relative_l2", "output_relative_l2", "contraction_ratio"])
        item["strict_safe_fraction"] = float(local.strict_branch_safe.mean())
        item["contracts_fraction"] = float((local.contraction_ratio < 1).mean())
        item["strong_contracts_fraction"] = float((local.contraction_ratio < .8).mean())
        item["input_output_spearman"] = float(local[["input_relative_l2", "output_relative_l2"]].corr(method="spearman").iloc[0, 1])
        summary["pairs"][f"{modality}:{kind}"] = item
    matches = pd.DataFrame(matched_modality(pair))
    matches.to_csv(args.root / "matched_modality_pairs.csv", index=False)
    if not matches.empty:
        summary["matched_modality"] = []
        for kind, local in matches.groupby("pair_kind"):
            summary["matched_modality"].append({"pair_kind": kind, "n": int(local.n.sum()),
                "input_match_abs_median": float(local.input_match_abs_median.median()),
                "vision_minus_text_output_rel_mean": float(np.average(
                    local.vision_minus_text_output_rel_mean, weights=local.n)),
                "vision_minus_text_output_rel_median": float(local.vision_minus_text_output_rel_median.median())})
    for keys, local in policy.groupby(["modality", "policy", "group_cap"], dropna=False):
        summary["policies"][":".join(map(str, keys))] = {
            "rows": len(local), "branch_reduction_pre_gate_median": float(local.branch_reduction_pre_gate.median()),
            "safe_branch_reduction_median": float(local.safe_branch_reduction.median()),
            "safe_branch_reduction_max": float(local.safe_branch_reduction.max()),
            "combined_rel_l2_median": float(local.combined_rel_l2_median.median()),
            "combined_pass_fraction_median": float(local.combined_pass_fraction_pre_revert.median())}
    summary["rle"] = {column: {"median": float(rle[column].median()), "p90": float(rle[column].quantile(.9))}
                      for column in ["rle_fraction_ge2", "rle_fraction_ge4", "cc_fraction_ge2", "cc_fraction_ge4"]}
    for keys, local in contribution.groupby(["kind", "target_fraction"]):
        summary["contribution"][f"{keys[0]}:{keys[1]:.2f}"] = {
            "achieved_fraction_median": float(local.achieved_fraction.median()),
            "affected_rel_l2_median": float(local.affected_rel_l2_median.median()),
            "affected_rel_l2_p90_median": float(local.affected_rel_l2_p90.median()),
            "affected_pass_1pct_median": float(local.affected_pass_1pct.median()),
            "affected_pass_5pct_median": float(local.affected_pass_5pct.median())}
    for keys, local in centroid.groupby(["grouping", "group_cap", "representative"]):
        summary["centroid"][":".join(map(str, keys))] = {
            "branch_reduction_median": float(local.branch_reduction.median()),
            "affected_rel_l2_median": float(local.affected_rel_l2_median.median()),
            "affected_strict_pass_median": float(local.affected_strict_pass_fraction.median()),
            "affected_strict_pass_max": float(local.affected_strict_pass_fraction.max())}
    for path in sorted(args.root.glob("quality_*_*/quality.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        summary["quality"][path.parent.name] = {
            "requests": len(records), "logit_rel_l2_median": float(np.median([row["logit_rel_l2"] for row in records])),
            "logit_cosine_median": float(np.median([row["logit_cosine"] for row in records])),
            "route_agreement_median": float(np.median([row["next_route_agreement_mean"] for row in records])),
            "greedy_exact_fraction": float(np.mean([row["greedy_exact"] for row in records])),
            "baseline_answer_matches": int(sum(row["baseline_answer_match"] for row in records)),
            "modified_answer_matches": int(sum(row["modified_answer_match"] for row in records))}
    (args.root / "FINDING_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["capture"], indent=2))


if __name__ == "__main__":
    main()
