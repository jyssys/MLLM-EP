from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..contract import HORIZON, TOP_K, atomic_json, read_json, read_jsonl_gz


def redundancy_rank_slots(site: dict[str, Any], paths: list[dict[str, Any]], keep: int) -> list[int]:
    count = TOP_K - keep
    bottleneck = int(site["wave_bottleneck_rank"])
    scored = []
    for path in paths:
        closure = float(path["wave_expert_frequency_b8"] == 1)
        bottleneck_bonus = float(path["owner_rank_ep4"] == bottleneck)
        utility = float(path["max_output_cosine"]) + 0.05 * closure + 0.02 * bottleneck_bonus
        scored.append((utility, -float(path["router_weight"]), int(path["slot_rank"])))
    return sorted(value[2] for value in sorted(scored, reverse=True)[:count])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sites", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--policies", required=True)
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    sites = {
        (row["case_id"], int(row["layer"]), int(row["decode_step"])): row
        for row in read_jsonl_gz(args.sites) if row["condition"] == "original"
    }
    paths: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl_gz(args.paths):
        if row["condition"] == "original": paths[row["site_id"]].append(row)
    policies = [value for value in args.policies.split(",") if value]
    cases = []
    selected_cases = [case for case in manifest["cases"] if case["split"] == args.split]
    if args.limit_cases is not None:
        selected_cases = selected_cases[:args.limit_cases]
    for case in selected_cases:
        key = (case["case_id"], int(case["assigned_probe_layer"]), int(case["assigned_probe_step"]))
        site = sites[key]
        expert_ids = [int(value) for value in site["expert_ids"]]
        site_paths = {int(row["slot_rank"]): row for row in paths[site["site_id"]]}
        variants = [{
            "probe_id": f"{case['case_id']}.baseline", "policy": "baseline", "keep_k": 8,
            "intervene": False, "removed_slots": [], "removed_expert_ids": [],
        }]
        for policy in policies:
            for keep in (6, 4):
                if policy == "redundancy_rank":
                    slots = redundancy_rank_slots(site, list(site_paths.values()), keep)
                else:
                    field = f"{policy}_k{keep}_removed_slots"
                    if field not in site:
                        raise KeyError(field)
                    slots = [int(value) for value in site[field]]
                variants.append({
                    "probe_id": f"{case['case_id']}.{policy}.k{keep}",
                    "policy": policy, "keep_k": keep, "intervene": True,
                    "removed_slots": slots,
                    "removed_expert_ids": [expert_ids[value - 1] for value in slots],
                    "predicted_wave_closure_events": sum(
                        int(site_paths[value]["wave_expert_frequency_b8"] == 1) for value in slots
                    ),
                    "predicted_bottleneck_rank_removals": sum(
                        int(site_paths[value]["owner_rank_ep4"] == int(site["wave_bottleneck_rank"]))
                        for value in slots
                    ),
                })
        cases.append({
            "case_id": case["case_id"], "request_id": case["request_id"],
            "domain": case["domain"], "split": args.split, "condition": "original",
            "question": case["question"], "image_path": case["original_image_path"],
            "forced_token_ids": case["forced_token_ids"], "forced_prefix_sha256": case["forced_prefix_sha256"],
            "target_layer": int(case["assigned_probe_layer"]), "probe_step": int(case["assigned_probe_step"]),
            "horizon": HORIZON, "target_hidden_hash": site["hidden_fingerprint"],
            "native_expert_ids": expert_ids, "native_weights": site["weights"],
            "variants": variants,
        })
    atomic_json(args.output, {
        "schema_version": 1, "split": args.split, "frozen_before_functional_probe": True,
        "policies": policies, "case_count": len(cases),
        "probe_count": sum(len(case["variants"]) for case in cases),
        "cases": cases,
    })
    print({"cases": len(cases), "probes": sum(len(case["variants"]) for case in cases), "policies": policies})


if __name__ == "__main__": main()
