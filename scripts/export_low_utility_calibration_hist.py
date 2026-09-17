#!/usr/bin/env python3
"""Export representative post-pruning histograms for grouped-mm calibration.

This performs no generation and no GPU work.  It deterministically rebuilds
the most aggressive EP8 P3 counterfactual from the discovery trace, then
stores actual post-pruning full expert histograms near the lower assignment
envelope for single-H100 replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_h1_straggler_deepdive import BatchComputeModel, load_discovery_trace
from scripts.analyze_low_utility_straggler import (
    CounterfactualProjector, effective_slot_ranks, selection_mask,
    subtract_removed_hist,
)
from virtual_ep.comm_model import CommunicationScenario


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--ep4-compute", type=Path, required=True)
    parser.add_argument("--ep8-compute", type=Path, required=True)
    parser.add_argument("--communication", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    trace = load_discovery_trace(args.trace)
    arrays = trace.arrays
    ids = arrays["current_expert_ids"].astype(np.int16, copy=False)
    weights = arrays["current_router_weights"].astype(np.float32)
    normalized = weights / np.maximum(weights.sum(axis=-1, keepdims=True), 1e-20)
    valid = arrays["current_position_class"] != 0
    models = {
        4: BatchComputeModel(args.ep4_compute),
        8: BatchComputeModel(args.ep8_compute),
    }
    projector = CounterfactualProjector(
        arrays, ids, normalized, effective_slot_ranks(weights), valid, models,
        CommunicationScenario.load(args.communication, "ep2_calibrated_base"),
    )
    removed = projector.no_removed.copy()
    for fraction in (.25, .50, .75, 1.0):
        pressure = projector.pressures(removed, 8, calibrated=True)
        removed = selection_mask(
            projector.normalized, projector.slot_ranks, projector.ids,
            projector.valid, 8, pressure, .10 * fraction, 0, 4,
            removed, False,
        )
    hist = subtract_removed_hist(projector.full_hist, ids, valid, removed)
    local = hist.reshape(len(hist), 8, 32)
    assignments = local.sum(axis=2)
    # Select full invocations producing the lowest-rank shapes plus quantiles.
    selected = set(np.argsort(assignments.min(axis=1))[:16].tolist())
    minima = assignments.min(axis=1)
    order = np.argsort(minima)
    for quantile in np.linspace(0, 1, 16):
        selected.add(int(order[round(quantile * (len(order) - 1))]))
    chosen = np.asarray(sorted(selected), dtype=np.int64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "request_0000.npz", hist_s1=hist[chosen])
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "source": str(args.trace),
        "policy": "P3 calibrated-pressure, 10% current-block mass budget, min_k=4",
        "target": "EP8 grouped-mm calibration envelope extension",
        "selected_invocations": chosen.tolist(),
        "selected_assignment_min": int(assignments[chosen].min()),
        "selected_assignment_max": int(assignments[chosen].max()),
        "generation_run": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
