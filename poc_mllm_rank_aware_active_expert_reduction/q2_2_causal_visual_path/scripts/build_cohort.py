from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from ..contract import LAYERS, SEED, STEPS, atomic_json, read_json, read_jsonl


def choose_balanced(rows: list[dict], count: int) -> list[dict]:
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)
    for values in by_domain.values():
        values.sort(key=lambda row: (row.get("selection_key", ""), row["request_id"]))
    domains = sorted(by_domain)
    selected: list[dict] = []
    cursor = 0
    while len(selected) < count:
        domain = domains[cursor % len(domains)]
        index = cursor // len(domains)
        if index < len(by_domain[domain]):
            selected.append(by_domain[domain][index])
        cursor += 1
        if cursor > count * len(domains) * 4:
            raise RuntimeError("insufficient balanced cohort")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-cohort", type=Path, required=True)
    parser.add_argument("--existing-pairs", type=Path, required=True)
    parser.add_argument("--natural", type=Path, required=True)
    parser.add_argument("--old-q2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=int, default=8)
    parser.add_argument("--heldout", type=int, default=24)
    args = parser.parse_args()

    cohort = read_json(args.image_cohort)["requests"]
    pair_rows = read_json(args.existing_pairs)["anchors"]
    natural = {row["request_id"]: row for row in read_jsonl(args.natural)}
    old_hashes = {
        row["original_image_sha256"]
        for row in read_json(args.old_q2)["anchors"]
    }

    calibration_pool = []
    by_request = {row["request_id"]: row for row in cohort}
    for pair in pair_rows:
        source = natural[pair["original_request_id"]]
        calibration_pool.append({
            **by_request[pair["original_request_id"]],
            "same_domain_replacement_request_id": pair["same_domain_replacement_request_id"],
            "same_domain_replacement_image_path": pair["same_domain_replacement_image_path"],
            "same_domain_replacement_image_sha256": pair["same_domain_replacement_image_sha256"],
            "same_domain_replacement_visual_token_count": pair["same_domain_replacement_visual_token_count"],
            "observed_visual_token_count": int(source["vision_token_count"]),
        })
    calibration = choose_balanced(calibration_pool, args.calibration)

    raw_heldout_pool = [
        {**row, "observed_visual_token_count": int(natural[row["request_id"]]["vision_token_count"])}
        for row in cohort
        if row["analysis_split"] == "heldout" and row["image_sha256"] not in old_hashes
    ]
    raw_by_domain: dict[str, list[dict]] = defaultdict(list)
    for row in raw_heldout_pool:
        raw_by_domain[row["domain"]].append(row)
    heldout_pool = []
    for row in raw_heldout_pool:
        closest = min(
            abs(other["observed_visual_token_count"] - row["observed_visual_token_count"])
            / max(row["observed_visual_token_count"], 1)
            for other in raw_by_domain[row["domain"]]
            if other["request_id"] != row["request_id"]
        )
        if closest <= 0.10:
            heldout_pool.append(row)
    heldout = choose_balanced(heldout_pool, args.heldout)
    heldout_ids = {row["request_id"] for row in heldout}
    all_heldout_by_domain: dict[str, list[dict]] = defaultdict(list)
    for row in raw_heldout_pool:
        all_heldout_by_domain[row["domain"]].append(row)
    for row in heldout:
        candidates = [
            other for other in all_heldout_by_domain[row["domain"]]
            if other["request_id"] != row["request_id"]
            and other["image_sha256"] != row["image_sha256"]
        ]
        candidates.sort(key=lambda other: (
            abs(other["observed_visual_token_count"] - row["observed_visual_token_count"]),
            other["request_id"] in heldout_ids,
            other["selection_key"],
        ))
        replacement = candidates[0]
        row.update({
            "same_domain_replacement_request_id": replacement["request_id"],
            "same_domain_replacement_image_path": replacement["image_path"],
            "same_domain_replacement_image_sha256": replacement["image_sha256"],
            "same_domain_replacement_visual_token_count": replacement["observed_visual_token_count"],
        })

    cases = []
    for split, selected in (("calibration", calibration), ("heldout", heldout)):
        for row in selected:
            output = natural[row["request_id"]]
            digest = hashlib.sha256(f"q2-2::{SEED}::{row['request_id']}".encode()).digest()
            layer = LAYERS[digest[0] % len(LAYERS)]
            valid_steps = [step for step in STEPS if step < len(output["output_token_ids"])]
            if not valid_steps:
                raise RuntimeError(f"no valid frozen step for {row['request_id']}")
            step = valid_steps[digest[1] % len(valid_steps)]
            forced = [int(value) for value in output["output_token_ids"][: step + 1]]
            prefix_hash = hashlib.sha256(
                ",".join(map(str, forced)).encode()
            ).hexdigest()
            original_tokens = int(row["observed_visual_token_count"])
            cf_tokens = int(row["same_domain_replacement_visual_token_count"])
            cases.append({
                "case_id": f"case{len(cases):02d}",
                "request_id": row["request_id"],
                "domain": row["domain"],
                "split": split,
                "question": row["question"],
                "original_image_path": row["image_path"],
                "original_image_sha256": row["image_sha256"],
                "counterfactual_request_id": row["same_domain_replacement_request_id"],
                "counterfactual_image_path": row["same_domain_replacement_image_path"],
                "counterfactual_image_sha256": row["same_domain_replacement_image_sha256"],
                "original_visual_token_count": original_tokens,
                "counterfactual_visual_token_count": cf_tokens,
                "visual_token_relative_difference": abs(cf_tokens - original_tokens) / max(original_tokens, 1),
                "forced_token_ids": forced,
                "forced_prefix_sha256": prefix_hash,
                "probe_step": step,
                "target_layer": layer,
                "selection_rule": "SHA256_REQUEST_ID_OVER_PREDECLARED_Q2_CALIBRATION_LAYER_STEP_GRID",
            })

    heldout_hashes = {row["original_image_sha256"] for row in cases if row["split"] == "heldout"}
    if heldout_hashes & old_hashes:
        raise AssertionError("fresh held-out overlaps previous Q2")
    payload = {
        "schema_version": 1,
        "seed": SEED,
        "frozen_before_q2_2_capture": True,
        "routing_or_damage_used_for_selection": False,
        "old_q2_usage": "CALIBRATION_ONLY",
        "calibration_requests": args.calibration,
        "fresh_heldout_requests": args.heldout,
        "domain_counts": {
            split: dict(Counter(row["domain"] for row in cases if row["split"] == split))
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
