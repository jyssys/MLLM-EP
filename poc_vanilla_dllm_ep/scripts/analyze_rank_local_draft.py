#!/usr/bin/env python3
"""Summarize rank-local draft quality and conservative request-level oracles."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def weighted(rows, key: str) -> float:
    pairs = [
        (float(row[key]), int(row["masked_positions"]))
        for row in rows if row.get(key) is not None and int(row["masked_positions"]) > 0
    ]
    if not pairs:
        return float("nan")
    return sum(value * weight for value, weight in pairs) / sum(weight for _, weight in pairs)


def analyze(path: Path) -> dict:
    payload = json.loads(path.read_text())
    rows = payload["local_draft_rows"]
    local_forward_ms = [float(row["critical_local_draft_ms"]) for row in rows]
    local_ms = [
        float(row["critical_local_draft_ms"])
        + float(row["critical_local_proposal_postprocess_ms"])
        + float(row["critical_proposal_exchange_ms"])
        for row in rows
    ]
    global_ms = [float(row["critical_global_exact_ms"]) for row in rows]
    local_sum = sum(local_ms)
    global_sum = sum(global_ms)
    cost_ratio = local_sum / global_sum
    all_precision = weighted(rows, "all_rank_agreement_precision")
    all_coverage = weighted(rows, "all_rank_agreement_coverage")
    majority_precision = weighted(rows, "three_quarter_agreement_precision")
    majority_coverage = weighted(rows, "three_quarter_agreement_coverage")
    if all_coverage == 0:
        all_precision = 0.0
    if majority_coverage == 0:
        majority_precision = 0.0
    union_hit = weighted(rows, "proposal_union_hit_fraction")
    accepted_position_match = []
    for row in rows:
        exact_position = int(row["exact_best_confidence_position"])
        accepted_position_match.extend(
            int(position) == exact_position
            for position in row["per_rank_best_confidence_position"]
        )
    k_oracles = {}
    # This intentionally grants a free perfect-future validity signal.  It is
    # still charged K local forwards and one global trajectory verifier.  The
    # agreement coverage is part of the valid-segment probability; precision
    # alone would incorrectly count rare consensus as universal coverage.
    for depth in (1, 2, 4, 8):
        valid_probability = (all_coverage * all_precision) ** depth
        raw_saving = (depth - 1) - depth * cost_ratio
        perfect_fraction = valid_probability * max(raw_saving, 0.0) / depth
        k_oracles[str(depth)] = {
            "assumed_all_step_valid_probability": valid_probability,
            "perfect_model_forward_reduction_fraction": perfect_fraction,
            "raw_saving_global_forward_equivalents": raw_saving,
        }
    return {
        "path": str(path),
        "local_draft_k": int(rows[0]["local_draft_k"]),
        "refinement_forwards": len(rows),
        "local_draft_forward_sum_ms": sum(local_forward_ms),
        "local_draft_total_sum_ms": local_sum,
        "global_exact_sum_ms": global_sum,
        "local_to_global_cost_ratio": cost_ratio,
        "proposal_exchange_sum_ms": sum(
            float(row["proposal_exchange_ms"]) for row in rows
        ),
        "rank0_logits_cosine_mean": statistics.mean(
            float(row["local_global_logits_cosine_mean_rank0"]) for row in rows
        ),
        "rank0_top5_overlap_mean": statistics.mean(
            float(row["local_global_top5_overlap_mean_rank0"]) for row in rows
        ),
        "proposal_union_hit_fraction": union_hit,
        "all_rank_agreement_coverage": all_coverage,
        "all_rank_agreement_precision": all_precision,
        "three_quarter_agreement_coverage": majority_coverage,
        "three_quarter_agreement_precision": majority_precision,
        "per_rank_next_accept_position_match_fraction": statistics.mean(
            accepted_position_match
        ),
        "perfect_future_multistep_oracles": k_oracles,
        "oracle_warning": (
            "Independent one-step agreement is used as an optimistic trajectory-survival "
            "proxy; this is not evidence of a valid bidirectional multi-step verifier."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = [analyze(path) for path in args.inputs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
