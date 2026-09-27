from __future__ import annotations

import argparse
import json
from pathlib import Path

from poc_mllm_vision_aware_decode_coscheduling.contract import atomic_json
from poc_mllm_vision_aware_decode_coscheduling.stage_a import (
    TraceView,
    event_metrics,
    filter_pool_trace,
    freeze_pool,
    make_assignments,
    paired_layer_bootstrap,
    read_jsonl,
    summarize_metrics,
    write_jsonl_gz,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--compact-trace", type=Path, required=True)
    parser.add_argument("--prior-analysis", type=Path, required=True)
    parser.add_argument("--expert-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    pool = freeze_pool(json.loads(args.cohort.read_text()))
    atomic_json(args.output / "frozen_request_pool.json", pool)
    pool_ids = {row["pair_id"] for row in pool["anchors"]}
    filtered = filter_pool_trace(read_jsonl(args.compact_trace), pool_ids)
    write_jsonl_gz(args.output / "source_pool_trace.jsonl.gz", filtered)
    trace = TraceView(filtered)
    pair_ids = sorted(pool_ids)
    steps = trace.validate(pair_ids)
    prior = json.loads(args.prior_analysis.read_text())
    predictor = prior.get("predictor", prior)
    alphas = {
        key: float(value)
        for key, value in predictor["frozen_alphas"].items()
    }
    assignments = make_assignments(pool, trace, steps, alphas)
    atomic_json(
        args.output / "frozen_wave_assignments.json",
        {
            "group_size": 4,
            "trace_steps": steps,
            "separated_alphas_reused_from_prior_discovery": alphas,
            "policies": assignments,
        },
    )
    events = event_metrics(assignments, trace, steps)
    expert_contract = json.loads(args.expert_contract.read_text())
    metrics = summarize_metrics(events, int(expert_contract["bytes_per_expert_per_layer"]))
    atomic_json(args.output / "working_set_metrics.json", metrics)
    comparisons = {
        "S4_vs_S3": paired_layer_bootstrap(events, "S4_SEPARATED", "S3_FULL"),
        "S4_vs_S2": paired_layer_bootstrap(events, "S4_SEPARATED", "S2_TEXT"),
        "S4_vs_S0": paired_layer_bootstrap(events, "S4_SEPARATED", "S0_FIFO_RANDOM"),
    }
    atomic_json(args.output / "paired_comparisons.json", comparisons)

    primary = metrics["heldout"]["1-128"]
    oracle_fifo = float(primary["oracle_reduction_vs_fifo"])
    oracle_length = float(primary["oracle_reduction_vs_length"])
    separated = primary["S4_SEPARATED"]
    full = primary["S3_FULL"]
    gate = {
        "oracle_at_least_10pct_vs_fifo": oracle_fifo >= 0.10,
        "oracle_at_least_10pct_vs_length": oracle_length >= 0.10,
        "separated_recovers_at_least_30pct": float(
            separated["recovered_oracle_headroom"] or 0.0
        ) >= 0.30,
        "separated_beats_full": separated["mean_active_expert_union"]
        < full["mean_active_expert_union"],
        "not_explained_by_length": separated["mean_active_expert_union"]
        < primary["S1_LENGTH"]["mean_active_expert_union"],
    }
    gate["pass"] = all(gate.values())
    verdict = "STAGE_B_REQUIRED" if gate["pass"] else "INSUFFICIENT_BATCHING_HEADROOM"
    headroom = {
        "heldout_primary_window": "1-128",
        "oracle_reduction_vs_fifo": oracle_fifo,
        "oracle_reduction_vs_length": oracle_length,
        "gate": gate,
        "verdict": verdict,
    }
    atomic_json(args.output / "oracle_headroom.json", headroom)
    atomic_json(args.output / "stage_a_result.json", {"verdict": verdict, "gate": gate})


if __name__ == "__main__":
    main()
