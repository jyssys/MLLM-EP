from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..contract import (
    CONFIRMATION_REQUESTS,
    DISCOVERY_REQUESTS,
    HORIZON,
    LAYERS,
    SEED,
    STEPS,
    atomic_json,
    read_json,
    read_jsonl_gz,
)


def keyed(row: dict[str, Any], salt: str) -> str:
    return hashlib.sha256(f"{SEED}:{salt}:{row['request_id']}".encode()).hexdigest()


def select_matched_pairs(
    rows: list[dict[str, Any]], pair_counts: dict[str, int], salt: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["domain"])].append(row)
    selected: list[dict[str, Any]] = []
    replacements: dict[str, dict[str, Any]] = {}
    for domain, needed in pair_counts.items():
        values = grouped[domain]
        pairs = []
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                lrow, rrow = values[left], values[right]
                relative = abs(lrow["observed_visual_token_count"] - rrow["observed_visual_token_count"]) / max(
                    min(lrow["observed_visual_token_count"], rrow["observed_visual_token_count"]), 1
                )
                tie = hashlib.sha256(
                    f"{SEED}:{salt}:{lrow['request_id']}:{rrow['request_id']}".encode()
                ).hexdigest()
                pairs.append((relative, tie, lrow, rrow))
        used: set[str] = set()
        chosen = []
        for _, _, lrow, rrow in sorted(pairs, key=lambda value: (value[0], value[1])):
            if lrow["request_id"] in used or rrow["request_id"] in used:
                continue
            chosen.append((lrow, rrow))
            used.update((lrow["request_id"], rrow["request_id"]))
            if len(chosen) == needed:
                break
        if len(chosen) != needed:
            raise RuntimeError(f"{domain}: only {len(chosen)} disjoint pairs, need {needed}")
        for lrow, rrow in chosen:
            selected.extend((lrow, rrow))
            replacements[lrow["request_id"]] = rrow
            replacements[rrow["request_id"]] = lrow
    selected.sort(key=lambda row: (row["domain"], keyed(row, salt)))
    return selected, replacements


def prior_hashes(
    cohort: list[dict[str, Any]], q2_labels: Path, q22: Path, q23: Path
) -> tuple[set[str], dict[str, int]]:
    by_id = {row["request_id"]: row["image_sha256"] for row in cohort}
    hashes: set[str] = set()
    counts: dict[str, int] = {}
    q2_ids = {str(row["request_id"]) for row in read_jsonl_gz(q2_labels)}
    q2_hashes = {by_id[value] for value in q2_ids if value in by_id}
    hashes |= q2_hashes
    counts["q2"] = len(q2_hashes)
    for name, path in (("q2_2", q22), ("q2_3", q23)):
        values = {
            str(value)
            for case in read_json(path)["cases"]
            for value in (case["original_image_sha256"], case["counterfactual_image_sha256"])
        }
        hashes |= values
        counts[name] = len(values)
    counts["union"] = len(hashes)
    return hashes, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-cohort", type=Path, required=True)
    parser.add_argument("--natural", type=Path, required=True)
    parser.add_argument("--prior-q2-labels", type=Path, required=True)
    parser.add_argument("--prior-q22", type=Path, required=True)
    parser.add_argument("--prior-q23", type=Path, required=True)
    parser.add_argument("--neutral-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = read_json(args.image_cohort)
    cohort = source["requests"]
    natural = {row["request_id"]: row for row in read_jsonl_gz(args.natural)}
    excluded, exclusion_counts = prior_hashes(
        cohort, args.prior_q2_labels, args.prior_q22, args.prior_q23
    )
    required_tokens = max(STEPS) + HORIZON
    eligible = []
    for row in cohort:
        observed = natural.get(row["request_id"])
        if observed is None or row["image_sha256"] in excluded:
            continue
        tokens = [int(value) for value in observed["output_token_ids"]]
        if len(tokens) < required_tokens:
            continue
        eligible.append({
            **row,
            "observed_visual_token_count": int(observed["vision_token_count"]),
            "natural_output_token_count": len(tokens),
            "natural_output_token_ids": tokens,
            "natural_finish_reason": observed.get("finish_reason"),
        })

    domains = sorted({str(row["domain"]) for row in eligible})
    if len(domains) != 5:
        raise RuntimeError(f"expected five domains, got {domains}")
    confirmation_pair_counts = {domain: 2 + int(index < 2) for index, domain in enumerate(domains)}
    confirmation, confirmation_replacements = select_matched_pairs(
        eligible, confirmation_pair_counts, "confirmation"
    )
    confirmation_hashes = {row["image_sha256"] for row in confirmation}
    discovery_pair_counts = {domain: 3 + int(index == 0) for index, domain in enumerate(domains)}
    discovery, discovery_replacements = select_matched_pairs(
        [row for row in eligible if row["image_sha256"] not in confirmation_hashes],
        discovery_pair_counts,
        "discovery",
    )
    if len(discovery) != DISCOVERY_REQUESTS or len(confirmation) != CONFIRMATION_REQUESTS:
        raise RuntimeError((len(discovery), len(confirmation)))
    cases: list[dict[str, Any]] = []
    for split, rows in (("discovery", discovery), ("confirmation", confirmation)):
        replacements = discovery_replacements if split == "discovery" else confirmation_replacements
        for index, row in enumerate(rows):
            replacement = replacements[row["request_id"]]
            forced = row["natural_output_token_ids"][:required_tokens]
            cases.append({
                "case_id": f"dmc-{split[:4]}-{index:02d}",
                "split": split,
                "request_id": row["request_id"],
                "domain": row["domain"],
                "question": row["question"],
                "original_image_path": row["image_path"],
                "original_image_sha256": row["image_sha256"],
                "counterfactual_request_id": replacement["request_id"],
                "counterfactual_image_path": replacement["image_path"],
                "counterfactual_image_sha256": replacement["image_sha256"],
                "neutral_image_path": str((args.neutral_dir / f"{row['image_sha256']}.png").resolve()),
                "original_visual_token_count": row["observed_visual_token_count"],
                "counterfactual_visual_token_count": replacement["observed_visual_token_count"],
                "visual_token_relative_difference": abs(
                    row["observed_visual_token_count"] - replacement["observed_visual_token_count"]
                ) / max(row["observed_visual_token_count"], 1),
                "natural_output_token_count": row["natural_output_token_count"],
                "natural_finish_reason": row["natural_finish_reason"],
                "forced_token_ids": forced,
                "forced_prefix_sha256": hashlib.sha256(
                    ",".join(map(str, forced)).encode()
                ).hexdigest(),
                "assigned_probe_layer": LAYERS[index % len(LAYERS)],
                "assigned_probe_step": STEPS[(index * 5 + index // len(LAYERS)) % len(STEPS)],
                "selection_key": keyed(row, split),
            })

    payload = {
        "schema_version": 1,
        "seed": SEED,
        "source_prompt_sha256": source["prompt_sha256"],
        "frozen_before_routing_or_damage": True,
        "confirmation_frozen_before_primary_selection": True,
        "selection_uses_routing_or_damage": False,
        "selection_uses_output_content_or_quality": False,
        "minimum_natural_token_contract": required_tokens,
        "prior_exclusion_hash_counts": exclusion_counts,
        "eligible_requests_after_exclusion": len(eligible),
        "split_counts": dict(Counter(case["split"] for case in cases)),
        "domain_counts": {
            split: dict(Counter(case["domain"] for case in cases if case["split"] == split))
            for split in ("discovery", "confirmation")
        },
        "layers": list(LAYERS),
        "steps": list(STEPS),
        "conditions": ["original", "same_domain_swap", "semantic_neutral"],
        "cases": cases,
    }
    atomic_json(args.output, payload)
    print({"cases": len(cases), "domain_counts": payload["domain_counts"], "excluded": exclusion_counts})


if __name__ == "__main__":
    main()
