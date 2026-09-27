#!/usr/bin/env python3
"""Generate shared structural baselines and freeze discovery criticality."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from poc_mllm_vision_conditioned_decode_reduction.shared import (
    batch_regime_sweep,
    build_discovery_vision_criticality,
    read_jsonl,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("poc_mllm_vision_conditioned_decode_reduction/artifacts"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    pool_path = repo / "poc_mllm_vision_aware_decode_coscheduling/artifacts/frozen_request_pool.json"
    pool_trace_path = repo / "poc_mllm_vision_aware_decode_coscheduling/artifacts/source_pool_trace.jsonl.gz"
    compact_path = repo / "poc_mllm_long_reasoning_decode_vision_signal/artifacts/compact_routing_summary.jsonl"
    expert_contract_path = repo / "poc_mllm_vision_aware_decode_coscheduling/artifacts/expert_tensor_contract.json"
    pool = json.loads(pool_path.read_text())
    expert_contract = json.loads(expert_contract_path.read_text())
    pool_rows = read_jsonl(pool_trace_path)
    pair_ids = [row["pair_id"] for row in pool["anchors"]]
    batch = batch_regime_sweep(
        pool_rows,
        pair_ids,
        tp2_dummy_copies=1,
        expert_weight_bytes=int(expert_contract["bytes_per_expert_per_layer"]),
    )
    batch["source_assets"] = {
        "pool": str(pool_path),
        "trace": str(pool_trace_path),
        "expert_weight_contract": str(expert_contract_path),
    }
    compact_rows = read_jsonl(compact_path)
    criticality = build_discovery_vision_criticality(compact_rows)
    criticality["source_asset"] = str(compact_path)
    criticality["evidence_boundary"] = (
        "Offline protection/risk feature only; not model input and not a quality certificate."
    )
    write_json(args.output_dir / "shared_batch_regime_baseline.json", batch)
    write_json(args.output_dir / "frozen_vision_criticality.json", criticality)
    write_json(
        args.output_dir / "shared_ep_scope.json",
        {
            "schema_version": 1,
            "owner_maps": {
                "EP2": {"experts_per_rank": 64, "rule": "expert_id // 64"},
                "EP4": {"experts_per_rank": 32, "rule": "expert_id // 32"},
                "EP8": {"experts_per_rank": 16, "rule": "expert_id // 16"},
            },
            "EP2": "PHYSICAL_AVAILABLE_TRUE_DEEPEP",
            "EP4": "PHYSICAL_AVAILABLE_TRUE_DEEPEP",
            "EP8": "STRUCTURAL_EP8_PROJECTION_ONLY",
            "EP8_latency": None,
            "EP8_tpot": None,
            "EP8_e2e": None,
            "remote_metric_contract": "requires explicit source_rank in the target topology",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
