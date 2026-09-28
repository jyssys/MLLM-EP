#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from poc_mllm_vision_conditioned_decode_damage_skipping.contract import (
    atomic_json, read_json, read_jsonl, write_jsonl_gz,
)
from poc_mllm_vision_conditioned_decode_reduction.track_a import load_capture_groups


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def normalized(mapping: dict[str, float], expert: int) -> float:
    values = [max(0.0, float(value)) for value in mapping.values()]
    denominator = sum(values)
    return max(0.0, float(mapping.get(str(expert), 0.0))) / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--criticality", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--functional-plan", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    meta = {row["pair_id"]: row for row in manifest["anchors"]}
    compact = {row["request_id"]: row for row in read_jsonl(args.compact)}
    criticality_payload = read_json(args.criticality)
    criticality = {
        (int(row["layer"]), int(row["expert_id"])): float(row["criticality"])
        for row in criticality_payload["records"]
    }
    groups = []
    capture_integrity = []
    for capture in args.capture:
        capture_groups, capture_audit = load_capture_groups(capture / "raw")
        groups.extend(capture_groups)
        capture_integrity.append({"capture": str(capture), **capture_audit})
    integrity = {
        "captures": capture_integrity,
        "complete_top8_groups": len(groups),
        "capture_count": len(capture_integrity),
    }
    rows = []
    available: dict[str, list] = defaultdict(list)
    for group in groups:
        if group.request_id not in meta or group.request_id not in compact:
            continue
        request = meta[group.request_id]
        layer_trace = compact[group.request_id]["prefill_layers"][group.layer]
        baseline = sum(
            (path.weight * path.output for path in group.paths),
            np.zeros_like(group.paths[0].output, dtype=np.float32),
        )
        baseline_norm = max(float(np.linalg.norm(baseline)), 1e-12)
        weights = [path.weight for path in group.paths]
        for slot, path in enumerate(group.paths):
            denominator = 1.0 - path.weight
            if denominator <= 0:
                continue
            modified = (baseline - path.weight * path.output) / denominator
            delta = modified - baseline
            contribution_norm = float(np.linalg.norm(path.weight * path.output))
            next_weight = weights[slot + 1] if slot + 1 < len(weights) else 0.0
            previous_weight = weights[slot - 1] if slot else weights[slot]
            vision = normalized(layer_trace["vision_mass"], path.expert_id)
            text = normalized(layer_trace["text_mass"], path.expert_id)
            vision_count = normalized(layer_trace["vision_count"], path.expert_id)
            text_count = normalized(layer_trace["text_count"], path.expert_id)
            rows.append({
                "request_id": group.request_id,
                "domain": request["domain"],
                "split": request["damage_split"],
                "layer": group.layer,
                "decode_step": group.step,
                "hidden_hash": group.hidden_hash,
                "expert_id": path.expert_id,
                "slot_rank": slot + 1,
                "router_weight": path.weight,
                "weight_margin_above": previous_weight - path.weight,
                "weight_margin_below": path.weight - next_weight,
                "layer_fraction": group.layer / 47.0,
                "step_fraction": group.step / 128.0,
                "vision_mass_affinity": vision,
                "text_mass_affinity": text,
                "full_mass_affinity": vision + text,
                "vision_count_affinity": vision_count,
                "text_count_affinity": text_count,
                "vision_text_mass_ratio": (vision + 1e-8) / (text + 1e-8),
                "frozen_visual_criticality": criticality.get((group.layer, path.expert_id), 0.0),
                "dropped_weighted_branch_norm": contribution_norm,
                "fractional_contribution_norm": contribution_norm / baseline_norm,
                "local_mixture_relative_l2": float(np.linalg.norm(delta)) / baseline_norm,
                "local_mixture_cosine": cosine(modified, baseline),
                "label_evidence": "MEASURED_EXACT_BRANCH_OUTPUT_LOCAL_DROP_RENORMALIZE",
            })
        available[group.request_id].append(group)
    rows.sort(key=lambda row: (
        row["split"], row["domain"], row["request_id"], row["decode_step"],
        row["layer"], row["slot_rank"],
    ))
    write_jsonl_gz(args.labels, rows)

    plan = []
    for request_id in sorted(meta):
        candidates = sorted(available[request_id], key=lambda group: (group.step, group.layer))
        if not candidates:
            raise RuntimeError(f"no captured group for {request_id}")
        digest = hashlib.sha256(request_id.encode()).digest()
        target_steps = sorted({group.step for group in candidates})
        chosen_step = target_steps[digest[0] % len(target_steps)]
        step_groups = [group for group in candidates if group.step == chosen_step]
        chosen = step_groups[digest[1] % len(step_groups)]
        # A captured structural step can outlive the natural continuation used
        # by the functional replay (the source trace may have a longer forced
        # control).  Functional damage must stay inside the frozen natural
        # continuation.  Preserve the original hash choice whenever valid;
        # otherwise use the latest predeclared captured step that is valid,
        # with the same layer hash.  This fallback depends only on sequence
        # length, never on damage outcomes.
        forced_token_ids = meta[request_id]["forced_token_ids"]
        length_fallback = False
        if chosen.step >= len(forced_token_ids):
            valid_steps = sorted({group.step for group in candidates if group.step < len(forced_token_ids)})
            if not valid_steps:
                raise RuntimeError(f"no natural-continuation functional step for {request_id}")
            chosen_step = valid_steps[-1]
            step_groups = [group for group in candidates if group.step == chosen_step]
            chosen = step_groups[digest[1] % len(step_groups)]
            length_fallback = True
        plan.append({
            "case_id": f"case{len(plan):02d}",
            "request_id": request_id,
            "domain": meta[request_id]["domain"],
            "split": meta[request_id]["damage_split"],
            "question": meta[request_id]["question"],
            "image_path": meta[request_id]["original_image_path"],
            "forced_token_ids": forced_token_ids[: chosen.step + 1],
            "probe_step": chosen.step,
            "target_layer": chosen.layer,
            "target_hidden_hash": chosen.hidden_hash,
            "paths": [{
                "slot_rank": slot + 1,
                "expert_id": path.expert_id,
                "weight": path.weight,
            } for slot, path in enumerate(chosen.paths)],
            "selection_rule": (
                "SHA256_REQUEST_ID_OVER_PREDECLARED_CAPTURED_STEP_LAYER_GRID"
                if not length_fallback else
                "NATURAL_LENGTH_VALIDITY_FALLBACK_TO_LATEST_PREDECLARED_STEP_THEN_SHA256_LAYER"
            ),
        })
    atomic_json(args.functional_plan, {
        "schema_version": 1,
        "frozen_before_functional_damage": True,
        "future_damage_used_for_selection": False,
        "case_count": len(plan),
        "request_executions": len(plan) * 9,
        "cases": plan,
    })
    values = [row["local_mixture_relative_l2"] for row in rows]
    by_split = {split: sum(row["split"] == split for row in rows) for split in ("calibration", "heldout")}
    summary = {
        "status": "MEASURED_LOCAL_DAMAGE",
        "capture_integrity": integrity,
        "path_labels": len(rows),
        "groups": len(rows) // 8,
        "requests": len(available),
        "by_split": by_split,
        "local_mixture_relative_l2": {
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p99": float(np.percentile(values, 99)),
        },
        "functional_plan_cases": len(plan),
        "functional_plan_request_executions": len(plan) * 9,
    }
    atomic_json(args.summary, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
