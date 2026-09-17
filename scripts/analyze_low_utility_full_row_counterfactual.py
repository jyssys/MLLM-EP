#!/usr/bin/env python3
"""Exact-route full-physical counterfactual on a stratified GSM8K-32 sample.

Unlike the GSM8K-128 aggregate trace, the existing threshold-.95 heavy trace
retains all prefix/prior/current token routes.  This script selects evenly
spaced layer/refinement invocations, pads them only for vectorized CPU
analysis, and reuses the same grouped-mm and communication projector.  It
never scales latency by remaining route count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_h1_straggler_deepdive import BatchComputeModel, load_discovery_trace
from scripts.analyze_low_utility_straggler import (
    CounterfactualProjector, MASS_BUDGETS, build_current_fanout,
    build_current_unique, effective_slot_ranks, selection_mask,
    subtract_removed_hist,
)
from virtual_ep.comm_model import CommunicationScenario


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heavy", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--ep4-compute", type=Path, required=True)
    parser.add_argument("--ep8-compute", type=Path, required=True)
    parser.add_argument("--communication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--invocations", type=int, default=2048)
    parser.add_argument("--calibration-cases", type=Path)
    args = parser.parse_args()

    aggregate = load_discovery_trace(args.aggregate).arrays
    first32 = aggregate["request_id"] < 32
    source_indices = np.flatnonzero(first32)
    physical = aggregate["physical_rows"][first32].astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(physical)))
    count = min(args.invocations, len(physical))
    chosen = np.unique(np.linspace(0, len(physical) - 1, count).round().astype(np.int64))
    maximum = int(physical[chosen].max())
    ids = np.zeros((len(chosen), maximum, 8), dtype=np.int16)
    weights = np.zeros((len(chosen), maximum, 8), dtype=np.float32)
    valid = np.zeros((len(chosen), maximum), dtype=np.bool_)
    with np.load(args.heavy, allow_pickle=False) as heavy:
        all_ids = heavy["row__expert_ids"]
        all_weights = heavy["row__router_weights"]
        if offsets[-1] != len(all_ids):
            raise ValueError("aggregate/heavy invocation alignment mismatch")
        for target, invocation in enumerate(chosen):
            begin, end = offsets[invocation], offsets[invocation + 1]
            length = end - begin
            ids[target, :length] = all_ids[begin:end]
            weights[target, :length] = all_weights[begin:end]
            valid[target, :length] = True
    normalized = np.divide(
        weights, weights.sum(axis=-1, keepdims=True),
        out=np.zeros_like(weights), where=valid[..., None],
    )
    arrays = {
        "expert_counts_by_class": aggregate["expert_counts_by_class"][first32][chosen].copy(),
        "physical_rows": physical[chosen].astype(np.int32),
    }
    no_removed = np.zeros(ids.shape, dtype=np.bool_)
    for ep in (4, 8):
        source = np.broadcast_to(
            (np.arange(maximum, dtype=np.int32) % ep)[None, :], valid.shape
        ).astype(np.int8).copy()
        unique = build_current_unique(ids, valid, source, ep, no_removed)
        arrays[f"current_source_rank_ep{ep}"] = source
        arrays[f"unique_matrix_ep{ep}"] = unique
        arrays[f"mean_fanout_ep{ep}"] = (
            build_current_fanout(ids, valid, ep, no_removed) / physical[chosen]
        ).astype(np.float32)

    models = {
        4: BatchComputeModel(args.ep4_compute),
        8: BatchComputeModel(args.ep8_compute),
    }
    projector = CounterfactualProjector(
        arrays, ids, normalized, effective_slot_ranks(weights), valid, models,
        CommunicationScenario.load(args.communication, "ep2_calibrated_base"),
    )
    ones = {ep: np.ones((len(ids), ep), dtype=np.float64) for ep in (4, 8)}
    rows = []
    for budget in MASS_BUDGETS:
        generic = selection_mask(
            normalized, projector.slot_ranks, ids, valid, 4, ones[4],
            budget, 0, 4, no_removed, False,
        )
        for ep in (4, 8):
            item = projector.evaluate(generic, ep, f"P2_utility_budget_{budget:g}")
            item.update(policy="P2_utility", budget=budget)
            rows.append(item)
            if ep == 8 and args.calibration_cases is not None:
                post_hist = subtract_removed_hist(projector.full_hist, ids, valid, generic)
                local_assignments = post_hist.reshape(len(post_hist), 8, 32).sum(axis=2)
                low = np.argwhere(local_assignments < 221)
                # Lowest states plus deterministic spread, bounded per budget.
                low = low[np.argsort(local_assignments[low[:, 0], low[:, 1]])[:24]]
                args.calibration_cases.mkdir(parents=True, exist_ok=True)
                for ordinal, (invocation, rank) in enumerate(low):
                    length = int(physical[chosen[invocation]])
                    begin, end = rank * 32, (rank + 1) * 32
                    owner = ids[invocation, :length] // 32
                    kept = ~generic[invocation, :length]
                    local_ids = np.where(
                        (owner == rank) & kept,
                        ids[invocation, :length] - begin,
                        -1,
                    ).astype(np.int16)
                    received = np.any(local_ids >= 0, axis=1)
                    local_weights = np.where(
                        local_ids >= 0, weights[invocation, :length], 0
                    ).astype(np.float32)
                    np.savez_compressed(
                        args.calibration_cases / f"budget{budget:g}_{ordinal:02d}.npz",
                        recv_ids=local_ids[received],
                        recv_weights=local_weights[received],
                        counts=post_hist[invocation, begin:end].astype(np.int32),
                        budget=np.asarray(budget),
                        source_invocation=np.asarray(int(chosen[invocation])),
                        virtual_rank=np.asarray(int(rank)),
                    )
            removed = no_removed.copy()
            for fraction in (.25, .50, .75, 1.0):
                pressure = projector.pressures(removed, ep, calibrated=True)
                removed = selection_mask(
                    normalized, projector.slot_ranks, ids, valid, ep, pressure,
                    budget * fraction, 0, 4, removed, False,
                )
            item = projector.evaluate(removed, ep, f"P3_calibrated_oracle_budget_{budget:g}")
            item.update(policy="P3_calibrated_oracle", budget=budget, selection_rounds=4)
            rows.append(item)
            pressure = projector.pressures(no_removed, ep, calibrated=False)
            removed = selection_mask(
                normalized, projector.slot_ranks, ids, valid, ep, pressure,
                budget, 0, 4, no_removed, False,
            )
            item = projector.evaluate(removed, ep, f"P4_rank_pressure_budget_{budget:g}")
            item.update(policy="P4_rank_pressure", budget=budget)
            rows.append(item)
    increments = {}
    for ep in (4, 8):
        p2 = {row["budget"]: row for row in rows if row["ep"] == ep and row["policy"] == "P2_utility"}
        p3 = {row["budget"]: row for row in rows if row["ep"] == ep and row["policy"] == "P3_calibrated_oracle"}
        increments[f"ep{ep}"] = {
            str(budget): p3[budget]["stage_gain_percent"] - p2[budget]["stage_gain_percent"]
            for budget in MASS_BUDGETS
        }
    document = {
        "source": str(args.heavy),
        "sample": {
            "strategy": "evenly spaced layer/refinement invocations across GSM8K requests 0-31",
            "invocations": int(len(chosen)),
            "physical_token_rows": int(physical[chosen].sum()),
            "expert_slots": int(physical[chosen].sum() * 8),
            "indices": chosen.tolist(),
        },
        "scope": "all physical prompt/prefix/prior/current rows in selected invocations",
        "quality_rollout": False,
        "baseline": projector.base,
        "matched_mass": rows,
        "ep_specific_increment_pp": increments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
