from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..contract import LAYERS, STEPS, atomic_json, read_json


MAPPING = {
    "H1_REDUNDANCY": {
        "policy": "redundancy", "rank_policy": "redundancy_rank",
        "structural_metric": "output_effective_rank", "direction": -1.0,
        "mechanism": "concurrent top-8 expert-output redundancy",
    },
    "H2_PERSISTENCE": {
        "policy": "temporal_persistence", "rank_policy": "temporal_rank",
        "structural_metric": "step_topk_jaccard", "direction": 1.0,
        "mechanism": "sampled-step route persistence",
    },
    "H3_CIRCUIT": {
        "policy": "cross_layer_circuit", "rank_policy": "circuit_rank",
        "structural_metric": "motif_persistence_jaccard", "direction": 1.0,
        "mechanism": "cross-layer transition-motif persistence",
    },
    "H4_RANK_OCCUPANCY": {
        "policy": "rank_closure", "rank_policy": "rank_closure",
        "structural_metric": "singleton_experts", "direction": 1.0,
        "mechanism": "B8 rank-local singleton closure opportunity",
    },
}


def positive(result: dict) -> bool:
    return result["ci95"][0] > 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--h1-functional", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    discovery, functional = read_json(args.discovery), read_json(args.h1_functional)
    h1_modality = any(
        positive(discovery["h1"][key])
        for key in (
            "mean_output_cosine", "effective_rank_lower_is_redundant",
            "largest_cluster", "reconstruction_residual_lower_is_redundant",
        )
    )
    h1_skip = any(
        positive(functional["comparisons"][f"redundancy_k{keep}"]["vs_router_tail"])
        for keep in (6, 4)
    )
    if h1_modality and h1_skip:
        selected = "H1_REDUNDANCY"
        rationale = "H1 passed both discovery modality and exact functional-damage gates"
    elif h1_modality:
        # H1 is still the primary modality mechanism when its structural
        # effect is reproducible but the skip link is negative.  Freezing it
        # lets the untouched cohort distinguish a real-but-useless modality
        # effect from a discovery-only fluctuation; it does not promote H1.
        selected = "H1_REDUNDANCY"
        rationale = "H1 structural modality effect passed; exact skip gate failed and is frozen for negative confirmation"
    else:
        selected = discovery["strongest_discovery_candidate"]
        rationale = "strongest predeclared H1-H4 discovery score after H1 exact gate"
    frozen = {
        "schema_version": 1,
        "frozen_before_confirmation_trace": True,
        "confirmation_data_inspected": False,
        "primary_hypothesis": selected,
        **MAPPING[selected],
        "layers": list(LAYERS), "steps": list(STEPS),
        "functional_site_rule": "PREASSIGNED_BALANCED_LAYER_STEP_GRID",
        "primary_keep_k": 6, "secondary_keep_k": 4,
        "generic_controls": ["router_tail", "random"],
        "h1_discovery_modality_pass": h1_modality,
        "h1_discovery_exact_skip_pass": h1_skip,
        "discovery_candidate_score": discovery["candidate_conservative_scores"][selected],
        "credible_discovery_gate": bool(
            selected == "H1_REDUNDANCY" and h1_modality
            or discovery["candidate_conservative_scores"][selected] > 0
        ),
        "selection_rationale": rationale,
        "derived_hypotheses": [],
    }
    atomic_json(args.output, frozen)
    print(json.dumps(frozen, indent=2))


if __name__ == "__main__": main()
