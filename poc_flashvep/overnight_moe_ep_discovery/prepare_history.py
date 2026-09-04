"""Build fixed-route history-conditioning cases for H4/H5."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path("/home/esjung/MLLM-EP-github/poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609/routes")


def _route(name: str, m: int) -> np.ndarray:
    with np.load(ROOT / name) as a:
        return np.asarray(a["routed_experts"][:m, 24, :], dtype=np.int64)


def _case(case_id: str, route: np.ndarray, role: str) -> dict:
    return {
        "case_id": case_id, "request_id": "history_control", "category": "controlled",
        "modality": "diagnostic", "layer": 24, "M": int(len(route)),
        "routes": route.tolist(), "token_count": int(len(route)),
        "total_assignments": int(route.size), "history_role": role,
    }


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--condition", choices=("steady", "alternating", "similar", "disjoint"), required=True)
    args = ap.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    target = _route("vision.deep_field.npz", 256)
    small = _route("text.text_00_coins.npz", 32)
    similar = _route("vision.model_card.npz", 256)
    disjoint = _route("text.text_22_retina.npz", 256)
    cases: list[dict] = []
    for i in range(5):
        if args.condition == "steady":
            if i == 0: cases.append(_case("B_steady_prime", target, "target_B"))
            cases.append(_case(f"B_steady_{i}", target, "target_B"))
        elif args.condition == "alternating":
            cases.extend([_case(f"A_prime_{i}", small, "prime_A"), _case(f"B_after_A_{i}", target, "target_B")])
        elif args.condition == "similar":
            cases.extend([_case(f"S_prime_{i}", similar, "prime_similar"), _case(f"B_after_S_{i}", target, "target_B")])
        else:
            cases.extend([_case(f"D_prime_{i}", disjoint, "prime_disjoint"), _case(f"B_after_D_{i}", target, "target_B")])
    (args.output / "cases.json").write_text(json.dumps(cases, separators=(",", ":")) + "\n")
    (args.output / "experiment_manifest.json").write_text(json.dumps({
        "hypotheses": ["H04_route_shape_transition_penalty", "H05_temporal_expert_warmth"],
        "condition": args.condition, "target_route": "vision.deep_field.npz layer24 first 256",
        "prime_routes": {"A": "text.text_00_coins.npz first 32", "similar": "vision.model_card.npz first 256", "disjoint": "text.text_22_retina.npz first 256"},
        "M_target": 256, "top_k": 8, "EP": 4, "placement": "expert_id // 32",
        "same_current_target": True, "physical_gpus": [1, 2, 3, 4],
        "interpretation": "target-B medians are conditioned on immediately preceding case; scheduler-free diagnostic replay",
    }, indent=2) + "\n")


if __name__ == "__main__": main()
