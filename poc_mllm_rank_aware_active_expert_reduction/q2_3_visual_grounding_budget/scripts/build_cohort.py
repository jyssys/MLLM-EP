from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..contract import HORIZON, LAYERS, SEED, STEPS, atomic_json, read_json, read_jsonl_gz


def balanced(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row["domain"])].append(row)
    for values in by_domain.values():
        values.sort(key=lambda row: (row["selection_key"], row["request_id"]))
    domains = sorted(by_domain)
    result: list[dict[str, Any]] = []
    level = 0
    while len(result) < count:
        progress = False
        for domain in domains:
            if level < len(by_domain[domain]):
                result.append(by_domain[domain][level])
                progress = True
                if len(result) == count:
                    return result
        if not progress:
            break
        level += 1
    raise RuntimeError(f"only {len(result)} balanced rows for requested {count}")


def relative_visual(left: dict[str, Any], right: dict[str, Any]) -> float:
    return abs(left["observed_visual_token_count"] - right["observed_visual_token_count"]) / max(
        left["observed_visual_token_count"], 1
    )


def matchable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows if any(
            other["domain"] == row["domain"]
            and other["image_sha256"] != row["image_sha256"]
            and relative_visual(row, other) <= 0.10
            for other in rows
        )
    ]


def select_pairs(rows: list[dict[str, Any]], split: str, count: int,
                 used: set[str]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Round-robin domains while reserving a unique matched image immediately."""
    originals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["analysis_split"] == split:
            originals[str(row["domain"])].append(row)
    for values in originals.values(): values.sort(key=lambda row: (row["selection_key"], row["request_id"]))
    domains, offsets, result = sorted(originals), defaultdict(int), []
    while len(result) < count:
        progress = False
        for domain in domains:
            values = originals[domain]
            while offsets[domain] < len(values):
                row = values[offsets[domain]]; offsets[domain] += 1
                if row["request_id"] in used: continue
                choices = [other for other in rows if other["domain"] == row["domain"]
                           and other["request_id"] not in used and other["request_id"] != row["request_id"]
                           and relative_visual(row, other) <= 0.10]
                choices.sort(key=lambda other: (relative_visual(row, other), other["selection_key"], other["request_id"]))
                if not choices: continue
                replacement = choices[0]
                used.update((row["request_id"], replacement["request_id"]))
                result.append((row, replacement)); progress = True
                break
            if len(result) == count: return result
        if not progress: break
    raise RuntimeError(f"only {len(result)} disjoint matched pairs for {split}, need {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-cohort", type=Path, required=True)
    parser.add_argument("--natural", type=Path, required=True)
    parser.add_argument("--prior-q2", type=Path, required=True)
    parser.add_argument("--prior-q22", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=int, default=8)
    parser.add_argument("--heldout", type=int, default=24)
    args = parser.parse_args()

    cohort = read_json(args.image_cohort)["requests"]
    natural = {row["request_id"]: row for row in read_jsonl_gz(args.natural)}
    prior_q2_ids = {row["request_id"] for row in read_jsonl_gz(args.prior_q2)}
    q22 = read_json(args.prior_q22)
    q22_ids = {
        value
        for case in q22["cases"]
        for value in (case["request_id"], case["counterfactual_request_id"])
    }
    q22_hashes = {
        value
        for case in q22["cases"]
        for value in (case["original_image_sha256"], case["counterfactual_image_sha256"])
    }
    excluded_ids = prior_q2_ids | q22_ids
    eligible = []
    for row in cohort:
        result = natural.get(row["request_id"])
        if result is None or row["request_id"] in excluded_ids or row["image_sha256"] in q22_hashes:
            continue
        tokens = [int(value) for value in result["output_token_ids"]]
        if len(tokens) <= max(STEPS) + HORIZON:
            continue
        eligible.append({
            **row,
            "observed_visual_token_count": int(result["vision_token_count"]),
            "natural_output_token_count": len(tokens),
            "natural_output_token_ids": tokens,
        })

    pairable = matchable(eligible)
    used: set[str] = set()
    heldout_pairs = select_pairs(pairable, "heldout", args.heldout, used)
    calibration_pairs = select_pairs(pairable, "discovery", args.calibration, used)
    calibration = [row for row, _ in calibration_pairs]
    heldout = [row for row, _ in heldout_pairs]
    replacements = {row["request_id"]: replacement for row, replacement in calibration_pairs + heldout_pairs}

    cases = []
    split_offsets = {"calibration": 0, "heldout": 0}
    for split, rows in (("calibration", calibration), ("heldout", heldout)):
        for row in rows:
            replacement = replacements[row["request_id"]]
            rel = relative_visual(row, replacement)
            index = split_offsets[split]
            split_offsets[split] += 1
            layer = LAYERS[index % len(LAYERS)]
            step = STEPS[(index * 5 + index // len(LAYERS)) % len(STEPS)]
            forced = row["natural_output_token_ids"][: step + HORIZON]
            prefix_hash = hashlib.sha256(",".join(map(str, forced)).encode()).hexdigest()
            cases.append({
                "case_id": f"vg{len(cases):02d}",
                "split": split,
                "request_id": row["request_id"],
                "domain": row["domain"],
                "question": row["question"],
                "original_image_path": row["image_path"],
                "original_image_sha256": row["image_sha256"],
                "counterfactual_request_id": replacement["request_id"],
                "counterfactual_image_path": replacement["image_path"],
                "counterfactual_image_sha256": replacement["image_sha256"],
                "original_visual_token_count": row["observed_visual_token_count"],
                "counterfactual_visual_token_count": replacement["observed_visual_token_count"],
                "visual_token_relative_difference": rel,
                "target_layer": layer,
                "probe_step": step,
                "horizon": HORIZON,
                "forced_token_ids": forced,
                "forced_prefix_sha256": prefix_hash,
                "site_selection": "BALANCED_PREDECLARED_LAYER_STEP_GRID_WITHOUT_ROUTING_OR_DAMAGE",
            })

    payload = {
        "schema_version": 1,
        "seed": SEED,
        "frozen_before_capture": True,
        "selection_uses_routing_grounding_or_damage": False,
        "universal_prompt_sha256": cohort[0]["universal_prompt_sha256"],
        "prior_q2_usage": "EXCLUSION_AND_CALIBRATION_HISTORY_ONLY",
        "calibration_requests": args.calibration,
        "fresh_heldout_requests": args.heldout,
        "domain_counts": {
            split: dict(Counter(case["domain"] for case in cases if case["split"] == split))
            for split in ("calibration", "heldout")
        },
        "target_layers": list(LAYERS),
        "target_steps": list(STEPS),
        "cases": cases,
    }
    atomic_json(args.output, payload)
    print({"cases": len(cases), "domain_counts": payload["domain_counts"]})


if __name__ == "__main__":
    main()
