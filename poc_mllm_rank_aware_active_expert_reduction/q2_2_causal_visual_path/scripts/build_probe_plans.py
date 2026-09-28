from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

from ..contract import MODEL_PATH, TOP_K, atomic_json, read_json, write_jsonl_gz


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def load_condition(raw: Path, capture_id: str, layer: int, step: int) -> dict[str, Any]:
    router_files = sorted(raw.glob(f"router.{capture_id}.rank*.layer{layer}.step{step}.call*.npz"))
    candidates = []
    for shard in router_files:
        with np.load(shard) as payload:
            if len(payload["fingerprints"]) != 1:
                continue
            candidates.append({
                "fingerprint": str(payload["fingerprints"][0]),
                "hidden": payload["hidden_states"][0].astype(np.float32),
                "ids": payload["topk_ids"][0].astype(int),
                "weights": payload["topk_weights"][0].astype(np.float32),
            })
    if len(candidates) != 1:
        raise RuntimeError(f"{capture_id}: expected one source router row, got {len(candidates)}")
    route = candidates[0]
    outputs: dict[int, np.ndarray] = {}
    for shard in sorted(raw.glob(f"experts.{capture_id}.rank*.layer{layer}.step*.call*.npz")):
        with np.load(shard) as payload:
            for row, fingerprint in enumerate(payload["fingerprints"]):
                if str(fingerprint) != route["fingerprint"]:
                    continue
                for slot, expert in enumerate(payload["expert_ids"][row]):
                    if bool(payload["local_mask"][row, slot]):
                        outputs.setdefault(int(expert), payload["raw_outputs"][row, slot].astype(np.float32))
    # Receiver-side DeepEP rows are packed and cannot be joined to the logical
    # source row solely by receiver row position. Exact-fingerprint outputs are
    # retained only to validate the current-weight logical reference below;
    # completeness is deliberately not inferred from packed receiver arrays.
    route["outputs"] = outputs
    return route


def condition_case(case: dict[str, Any], condition: str, route: dict[str, Any], paths: list[dict]) -> dict[str, Any]:
    image_path = case["original_image_path"] if condition == "orig" else case["counterfactual_image_path"]
    selected = []
    weights = {int(expert): float(weight) for expert, weight in zip(route["ids"], route["weights"])}
    for path in paths:
        expert = int(path["expert_id"])
        if expert not in weights:
            continue
        selected.append({
            "slot_rank": int(path["slot_rank"]),
            "expert_id": expert,
            "weight": weights[expert],
            "original_slot_rank": int(path["slot_rank"]),
        })
    return {
        "case_id": f"{case['case_id']}.{condition}",
        "base_case_id": case["case_id"],
        "condition": condition,
        "request_id": case["request_id"],
        "domain": case["domain"],
        "split": case["split"],
        "question": case["question"],
        "image_path": image_path,
        "forced_token_ids": case["forced_token_ids"],
        "probe_step": int(case["probe_step"]),
        "target_layer": int(case["target_layer"]),
        "target_hidden_hash": route["fingerprint"],
        "paths": selected,
    }


def forced_expert_output(
    model_path: Path,
    layer: int,
    expert: int,
    hidden: np.ndarray,
    index: dict[str, str],
    weight_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]],
) -> np.ndarray:
    key = (layer, expert)
    if key not in weight_cache:
        gate_key = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
        down_key = f"model.language_model.layers.{layer}.mlp.experts.down_proj"
        with safe_open(model_path / index[gate_key], framework="pt", device="cpu") as handle:
            gate_up = handle.get_slice(gate_key)[expert]
        with safe_open(model_path / index[down_key], framework="pt", device="cpu") as handle:
            down = handle.get_slice(down_key)[expert]
        weight_cache[key] = (gate_up, down)
    gate_up, down = weight_cache[key]
    value = torch.from_numpy(hidden).to(torch.bfloat16) @ gate_up
    gate, up = value[:768], value[768:]
    return ((torch.nn.functional.silu(gate) * up) @ down).float().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--signal-summary", type=Path, required=True)
    parser.add_argument("--orig-plan", type=Path, required=True)
    parser.add_argument("--cf-plan", type=Path, required=True)
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    raw = args.capture / "raw"
    signal_rows = []
    orig_cases, cf_cases = [], []
    index = read_json(args.model_path / "model.safetensors.index.json")["weight_map"]
    weight_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    logical_expert_evaluations = 0
    forced_eval_validation = []
    cases = manifest["cases"][: args.limit_cases] if args.limit_cases else manifest["cases"]
    for case in cases:
        layer, step = int(case["target_layer"]), int(case["probe_step"])
        original = load_condition(raw, f"{case['case_id']}.orig", layer, step)
        counterfactual = load_condition(raw, f"{case['case_id']}.cf", layer, step)
        hidden_norm = max(float(np.linalg.norm(original["hidden"])), 1e-12)
        hidden_delta = float(np.linalg.norm(original["hidden"] - counterfactual["hidden"])) / hidden_norm
        hidden_cosine = cosine(original["hidden"], counterfactual["hidden"])
        cf_by_expert = {
            int(expert): (slot, float(counterfactual["weights"][slot]))
            for slot, expert in enumerate(counterfactual["ids"])
        }
        paths = []
        for slot, (expert_raw, go_raw) in enumerate(zip(original["ids"], original["weights"]), start=1):
            expert, go = int(expert_raw), float(go_raw)
            fo = forced_expert_output(
                args.model_path, layer, expert, original["hidden"], index, weight_cache
            )
            fc = forced_expert_output(
                args.model_path, layer, expert, counterfactual["hidden"], index, weight_cache
            )
            logical_expert_evaluations += 2
            if expert in original["outputs"]:
                forced_eval_validation.append(
                    float(np.linalg.norm(original["outputs"][expert] - fo))
                    / max(float(np.linalg.norm(original["outputs"][expert])), 1e-12)
                )
            retained = expert in cf_by_expert
            cf_slot, gc = cf_by_expert.get(expert, (-1, 0.0))
            co = go * fo
            cc = gc * fc
            co_norm = max(float(np.linalg.norm(co)), 1e-12)
            signal_rows.append({
                "case_id": case["case_id"],
                "request_id": case["request_id"],
                "domain": case["domain"],
                "split": case["split"],
                "layer": layer,
                "decode_step": step,
                "expert_id": expert,
                "slot_rank": slot,
                "router_weight_orig": go,
                "router_weight_cf": gc,
                "router_shift": abs(go - gc),
                "router_rank_orig": slot,
                "router_rank_cf": None if not retained else cf_slot + 1,
                "counterfactual_topk_retained": retained,
                "topk_membership_flip": not retained,
                "hidden_cosine": hidden_cosine,
                "hidden_delta": hidden_delta,
                "expert_output_shift": float(np.linalg.norm(fo - fc)) / max(float(np.linalg.norm(fo)), 1e-12),
                "path_contribution_shift": float(np.linalg.norm(co - cc)) / co_norm,
                "original_hidden_hash": original["fingerprint"],
                "counterfactual_hidden_hash": counterfactual["fingerprint"],
                "forced_prefix_sha256": case["forced_prefix_sha256"],
                "same_domain_swap": True,
                "counterfactual_nonactive_convention": "g_cf=0;c_cf=0;D_cf=0",
                "expert_output_shift_evidence": "BF16_CURRENT_WEIGHT_LOGICAL_EXPERT_REFERENCE",
                "evidence": "MEASURED_TRUE_EP4_IDENTICAL_TEACHER_FORCED_PREFIX",
            })
            paths.append({"slot_rank": slot, "expert_id": expert, "weight": go})
        orig_cases.append(condition_case(case, "orig", original, paths))
        cf_cases.append(condition_case(case, "cf", counterfactual, paths))

    calibration = [row["path_contribution_shift"] for row in signal_rows if row["split"] == "calibration"]
    low, high = (float(np.percentile(calibration, q)) for q in (33.333333, 66.666667))
    write_jsonl_gz(args.signals, signal_rows)
    summary = {
        "status": "MEASURED_CAUSAL_PATH_SIGNALS",
        "requests": len(cases),
        "paths": len(signal_rows),
        "calibration_paths": sum(row["split"] == "calibration" for row in signal_rows),
        "fresh_heldout_paths": sum(row["split"] == "heldout" for row in signal_rows),
        "counterfactual_topk_retention": float(np.mean([row["counterfactual_topk_retained"] for row in signal_rows])),
        "logical_same_expert_current_weight_evaluations": logical_expert_evaluations,
        "offline_current_weight_eval_validation": {
            "comparisons": len(forced_eval_validation),
            "mean_relative_l2": float(np.mean(forced_eval_validation)),
            "max_relative_l2": float(np.max(forced_eval_validation)),
            "pass_under_1pct": bool(max(forced_eval_validation) < 0.01),
        },
        "primary_signal": "path_contribution_shift",
        "frozen_calibration_thresholds": {"low_upper": low, "high_lower": high},
        "thresholds_use_heldout": False,
    }
    atomic_json(args.signal_summary, summary)
    for path, cases, condition in (
        (args.orig_plan, orig_cases, "orig"),
        (args.cf_plan, cf_cases, "cf"),
    ):
        atomic_json(path, {
            "schema_version": 1,
            "condition": condition,
            "frozen_before_functional_damage": True,
            "future_damage_used_for_selection": False,
            "case_count": len(cases),
            "intervention_count": sum(len(case["paths"]) for case in cases),
            "cases": cases,
        })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
