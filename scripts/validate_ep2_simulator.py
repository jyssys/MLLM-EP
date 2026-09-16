#!/usr/bin/env python3
"""Held-out request-level EP2 simulator validation and predictor ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from virtual_ep.comm_model import CommunicationScenario
from virtual_ep.compute_model import ComputeModel
from virtual_ep.event_simulator import EventPolicy
from virtual_ep.schema import TraceBundle
from virtual_ep.simulator import simulate_trace
from virtual_ep.validation import error_summary


def _fit_predict(train_x, train_y, test_x, ridge: float = 1e-6):
    train_x = np.asarray(train_x, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    mean = train_x.mean(axis=0)
    scale = np.maximum(train_x.std(axis=0), 1e-9)
    train = (train_x - mean) / scale
    test = (test_x - mean) / scale
    train = np.column_stack([np.ones(len(train)), train])
    test = np.column_stack([np.ones(len(test)), test])
    penalty = np.eye(train.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(train.T @ train + penalty, train.T @ train_y)
    return test @ coefficients


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--communication-model", type=Path, required=True)
    parser.add_argument("--compute-model", type=Path, required=True)
    parser.add_argument("--measured-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = TraceBundle.load(args.trace)
    communication = CommunicationScenario.load(
        args.communication_model, "ep2_calibrated_base"
    )
    compute = ComputeModel.from_csv(args.compute_model, allowed_splits=("train",))
    predictions = simulate_trace(
        trace, 2, communication=communication, compute=compute,
        event_policy=EventPolicy(),
    )
    predicted = {
        (row.request_id, row.block_id, row.iteration_id, row.nfe, row.layer_id): row
        for row in predictions
    }
    measured_document = json.loads(args.measured_replay.read_text())
    rows = []
    for measured in measured_document["cases"]:
        key = (
            measured["request_id"], measured["block_id"],
            measured["iteration_id"], measured["nfe"], measured["layer_id"],
        )
        if key not in predicted:
            raise KeyError(f"measured replay key missing from simulator: {key}")
        simulation = predicted[key]
        rows.append(
            {
                "split": measured["split"],
                "measured_dispatch": measured["EP_dispatch"]["median_ms"],
                "measured_expert": measured["routed_expert_compute"]["median_ms"],
                "measured_combine": measured["EP_combine"]["median_ms"],
                "measured_stage": measured["EP_stage_excluding_bridge"]["median_ms"],
                "predicted_dispatch": simulation.dispatch_ms,
                "predicted_expert": simulation.critical_expert_ms,
                "predicted_combine": simulation.combine_ms,
                "predicted_stage": simulation.moe_stage_ms,
                "nfe": simulation.nfe,
                "physical_rows": simulation.physical_rows,
                "active_experts": simulation.active_experts,
                "remote_unique": simulation.remote_unique_activations,
                "remote_bytes": simulation.remote_dispatch_bytes,
                "max_rank_load": simulation.max_rank_load,
                "rank_cv": simulation.rank_load_cv,
                "fanout": simulation.mean_token_fanout,
            }
        )
    heldout = [row for row in rows if row["split"] == "heldout"]
    train = [row for row in rows if row["split"] == "train"]
    if not heldout or not train:
        raise RuntimeError("validation requires request-level train and heldout cases")
    mechanistic = {
        component: error_summary(
            [row[f"predicted_{component}"] for row in heldout],
            [row[f"measured_{component}"] for row in heldout],
        )
        for component in ("dispatch", "expert", "combine", "stage")
    }
    feature_sets = {
        "A_NFE_only": ("nfe",),
        "B_NFE_plus_routed_shape": ("nfe", "physical_rows", "active_experts"),
        "C_EP_aware": (
            "nfe", "physical_rows", "active_experts", "remote_unique",
            "remote_bytes", "max_rank_load", "rank_cv", "fanout",
        ),
    }
    predictors = {}
    for name, features in feature_sets.items():
        estimated = _fit_predict(
            [[row[field] for field in features] for row in train],
            [row["measured_stage"] for row in train],
            [[row[field] for field in features] for row in heldout],
        )
        predictors[name] = error_summary(
            estimated, [row["measured_stage"] for row in heldout]
        )
    stage_error = mechanistic["stage"]
    verdict = (
        "EP2-CALIBRATED"
        if stage_error["median_ape_percent"] <= 10.0
        and stage_error["p90_ape_percent"] <= 20.0
        else "EP2-STRUCTURAL-ONLY"
        if stage_error["median_ape_percent"] <= 20.0
        else "EP2-VALIDATION-FAILED"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "split": "request_id % 5 == 0 held out",
                "no_random_row_leakage": True,
                "heldout_cases": len(heldout),
                "train_cases": len(train),
                "mechanistic_model": mechanistic,
                "predictor_ablation": predictors,
                "verdict": verdict,
                "cost_scope": [
                    "EP dispatch", "routed expert compute", "EP combine"
                ],
                "replicated_state_bridge_allgather": "excluded",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
