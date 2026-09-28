#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from poc_mllm_rank_aware_active_expert_reduction.contract import (
    EXPERTS, atomic_json, expert_owner, read_jsonl_gz,
)

PER_EXPERT_BYTES = (1536 * 2048 + 2048 * 768) * 2


def paths_for_wave(rows: list[dict[str, Any]], step: int, layer: int) -> list[dict[str, Any]]:
    result = []
    for request_index, row in enumerate(rows):
        matches = [item for item in row["sampled_routes"]
                   if int(item["step"]) == step and int(item["layer"]) == layer]
        if len(matches) != 1:
            raise RuntimeError((row["request_id"], step, layer, len(matches)))
        route = matches[0]
        for slot, (expert, weight) in enumerate(zip(route["route_ids"], route["route_weights"])):
            result.append({"request_index": request_index, "request_id": row["request_id"],
                           "expert": int(expert), "weight": float(weight), "slot": slot,
                           "source_rank": int(route["source_ep_rank"])})
    return result


def removal_native_tail(paths: list[dict[str, Any]], budget: int) -> set[int]:
    return set(sorted(range(len(paths)), key=lambda i: (paths[i]["weight"], -paths[i]["slot"]))[:budget])


def removal_random(paths: list[dict[str, Any]], budget: int, seed: str) -> set[int]:
    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))
    return set(rng.sample(range(len(paths)), budget))


def removal_rank_aware(paths: list[dict[str, Any]], budget: int, ep: int) -> set[int]:
    """Greedy structural upper bound that removes whole expert groups first."""
    groups: dict[int, list[int]] = defaultdict(list)
    for index, path in enumerate(paths):
        groups[path["expert"]].append(index)
    active_by_rank = Counter(expert_owner(expert, ep) for expert in groups)
    removed: set[int] = set()
    remaining = budget
    candidates = []
    for expert, indexes in groups.items():
        rank = expert_owner(expert, ep)
        mean_weight = sum(paths[index]["weight"] for index in indexes) / len(indexes)
        candidates.append((active_by_rank[rank], -len(indexes), -mean_weight,
                           expert, indexes))
    candidates.sort(reverse=True)
    for _, _, _, expert, indexes in candidates:
        if not remaining:
            break
        if len(indexes) > remaining:
            continue
        removed.update(indexes)
        remaining -= len(indexes)
        active_by_rank[expert_owner(expert, ep)] -= 1
    if remaining:
        extras = [i for i in sorted(range(len(paths)), key=lambda i: paths[i]["weight"])
                  if i not in removed]
        removed.update(extras[:remaining])
    if len(removed) != budget:
        raise RuntimeError((len(removed), budget))
    return removed


def metrics(paths: list[dict[str, Any]], removed: set[int], ep: int) -> dict[str, Any]:
    residual = [path for index, path in enumerate(paths) if index not in removed]
    active = [set() for _ in range(ep)]
    assignments = [0] * ep
    remote = 0
    for path in residual:
        rank = expert_owner(path["expert"], ep)
        active[rank].add(path["expert"]); assignments[rank] += 1
        remote += rank != path["source_rank"] % ep
    counts = [len(values) for values in active]
    weight_bytes = [value * PER_EXPERT_BYTES for value in counts]
    return {
        "residual_paths": len(residual), "removed_paths": len(removed),
        "active_experts_by_rank": counts, "max_active_experts": max(counts),
        "mean_active_experts": float(np.mean(counts)), "total_active_experts": sum(counts),
        "active_weight_bytes_by_rank": weight_bytes,
        "max_active_weight_bytes": max(weight_bytes),
        "total_active_weight_bytes": sum(weight_bytes),
        "assignments_by_rank": assignments,
        "rank_max_mean": max(assignments) / max(float(np.mean(assignments)), 1e-12),
        "remote_fraction": remote / max(len(residual), 1),
    }


def reduction(base: dict[str, Any], changed: dict[str, Any], key: str) -> float:
    return 100.0 * (base[key] - changed[key]) / base[key] if base[key] else 0.0


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["split"], row["batch"], row["ep"], row["policy"])].append(row)
    output = []
    for (split, batch, ep, policy), rows in sorted(grouped.items()):
        output.append({
            "split": split, "batch": batch, "ep": ep, "policy": policy,
            "observations": len(rows),
            "path_reduction_pct": float(np.mean([r["path_reduction_pct"] for r in rows])),
            "max_active_expert_reduction_pct": float(np.mean([r["max_active_expert_reduction_pct"] for r in rows])),
            "max_active_weight_reduction_pct": float(np.mean([r["max_active_weight_reduction_pct"] for r in rows])),
            "total_active_expert_reduction_pct": float(np.mean([r["total_active_expert_reduction_pct"] for r in rows])),
            "remote_path_reduction_pct": float(np.mean([r["remote_path_reduction_pct"] for r in rows])),
        })
    return {"rows": output}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=0.20)
    args = parser.parse_args()
    rows = read_jsonl_gz(args.trace)
    discovery = [row for row in rows if row["analysis_split"] == "discovery"]
    heldout = [row for row in rows if row["analysis_split"] == "heldout"]
    # Freeze gate threshold before held-out analysis.
    gate = {"budget": args.budget,
            "pass_if_o4_increment_over_native_tail_max_rank_active_expert_reduction_pp": 10.0,
            "primary_batch": 8, "primary_topology": "EP4",
            "frozen_before_heldout": True}
    records = []
    sampled_steps = {1, 16, 32, 64, 96, 128}
    for split, pool in (("discovery", discovery), ("heldout", heldout)):
        for batch in (1, 2, 4, 8, 16):
            # Frozen first ten waves per stratum bound the oracle runtime while
            # retaining all 48 layers and six decode phases.
            usable = min(len(pool) // batch, 10) * batch
            for offset in range(0, usable, batch):
                wave = pool[offset:offset + batch]
                common_steps = sorted(set.intersection(*[
                    {int(item["step"]) for item in row["sampled_routes"]} for row in wave
                ]))
                for step in [value for value in common_steps if value in sampled_steps]:
                    for layer in range(48):
                        paths = paths_for_wave(wave, step, layer)
                        budget = max(1, int(round(len(paths) * args.budget)))
                        for ep in (2, 4, 8):
                            base = metrics(paths, set(), ep)
                            policies = {
                                "O1_RANDOM": removal_random(paths, budget, f"{split}:{batch}:{offset}:{step}:{layer}:{ep}"),
                                "O2_NATIVE_TAIL": removal_native_tail(paths, budget),
                                "O4_RANK_AWARE_COLLAPSE_STRUCTURAL_UPPER": removal_rank_aware(paths, budget, ep),
                            }
                            for policy, removed in policies.items():
                                changed = metrics(paths, removed, ep)
                                records.append({
                                    "split": split, "batch": batch, "wave": offset // batch,
                                    "step": step, "layer": layer, "ep": ep, "policy": policy,
                                    "path_reduction_pct": 100.0 * len(removed) / len(paths),
                                    "max_active_expert_reduction_pct": reduction(base, changed, "max_active_experts"),
                                    "max_active_weight_reduction_pct": reduction(base, changed, "max_active_weight_bytes"),
                                    "total_active_expert_reduction_pct": reduction(base, changed, "total_active_experts"),
                                    "remote_path_reduction_pct": 100.0 * (base["remote_fraction"] * base["residual_paths"] - changed["remote_fraction"] * changed["residual_paths"]) / max(base["remote_fraction"] * base["residual_paths"], 1e-12),
                                })
    # Compact records are still bounded: only aggregated analysis is committed.
    summary = aggregate(records)
    primary = {(row["policy"], row["ep"]): row for row in summary["rows"]
               if row["split"] == "heldout" and row["batch"] == 8}
    ep4_o2 = primary[("O2_NATIVE_TAIL", 4)]
    ep4_o4 = primary[("O4_RANK_AWARE_COLLAPSE_STRUCTURAL_UPPER", 4)]
    ep8_o4 = primary[("O4_RANK_AWARE_COLLAPSE_STRUCTURAL_UPPER", 8)]
    increment = ep4_o4["max_active_expert_reduction_pct"] - ep4_o2["max_active_expert_reduction_pct"]
    passed = increment >= gate["pass_if_o4_increment_over_native_tail_max_rank_active_expert_reduction_pp"] and ep8_o4["max_active_expert_reduction_pct"] >= ep4_o4["max_active_expert_reduction_pct"]
    payload = {
        "schema_version": "rank_aware_structural_oracle.v1",
        "source_trace": str(args.trace.resolve()),
        "source_request_count": len(rows), "per_expert_weight_bytes": PER_EXPERT_BYTES,
        "evidence": "STRUCTURAL_ROUTE_REPLAY; O4_IS_NONDEPLOYABLE_UPPER_BOUND",
        "ep8_label": "STRUCTURAL_EP8_PROJECTION_ONLY",
        "gate": gate, "gate_pass": passed,
        "primary_ep4_native_tail": ep4_o2,
        "primary_ep4_rank_aware": ep4_o4,
        "primary_ep8_rank_aware": ep8_o4,
        "o4_increment_over_native_tail_pp": increment,
        "aggregate": summary,
        "limitations": [
            "Caption trace lane is structural-only and does not provide future-truth safe labels.",
            "O4 removes complete expert groups with future route knowledge and is not deployable.",
            "No latency is inferred for EP8.",
        ],
    }
    atomic_json(args.output, payload)
    atomic_json(args.records, {"schema_version": "rank_aware_structural_aggregate.v1", **summary})
    print(json.dumps({k: v for k, v in payload.items() if k != "aggregate"}, indent=2))


if __name__ == "__main__":
    main()
