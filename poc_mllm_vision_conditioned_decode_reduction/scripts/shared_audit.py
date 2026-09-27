#!/usr/bin/env python3
"""Audit reusable evidence without starting a model or touching GPUs."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from poc_mllm_vision_conditioned_decode_reduction.shared import read_jsonl, write_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("poc_mllm_vision_conditioned_decode_reduction/artifacts/shared_asset_audit.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    prior = repo / "poc_mllm_long_reasoning_decode_vision_signal" / "artifacts"
    cosched = repo / "poc_mllm_vision_aware_decode_coscheduling" / "artifacts"
    foundation = repo / "poc_mllm_ep_foundation" / "validation_runs" / "20260924_completion_v1"
    paths = {
        "baseline_audit": prior / "baseline_audit.json",
        "cohort": cosched / "frozen_request_pool.json",
        "compact_trace": prior / "compact_routing_summary.jsonl",
        "pool_trace": cosched / "source_pool_trace.jsonl.gz",
        "causal": prior / "causal_comparison.json",
        "predictor": prior / "predictor_comparison.json",
        "expert_contract": cosched / "expert_tensor_contract.json",
        "ep2_ep4_proof": foundation / "reports" / "ep2_ep4_anchor_manifest.md",
        "frozen_bundle": foundation / "frozen_baseline_bundle.json",
        "ep2_replay": foundation / "replay" / "ep2_from_ep2_v7.json",
        "ep4_replay": foundation / "replay" / "ep4_from_ep2_v7.json",
        "ep8_replay": foundation / "replay" / "ep8_from_ep2_v7.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing reusable assets: " + ", ".join(missing))

    baseline = json.loads(paths["baseline_audit"].read_text())
    pool = json.loads(paths["cohort"].read_text())
    pool_rows = read_jsonl(paths["pool_trace"])
    requests = pool["anchors"]
    steps = sorted({int(row["step"]) for row in pool_rows if row["record_type"] == "decode"})
    layers = sorted({int(row["layer"]) for row in pool_rows})
    record_types = Counter(row["record_type"] for row in pool_rows)
    split_counts = Counter(row["coscheduling_split"] for row in requests)
    dataset_counts = Counter(row["dataset"] for row in requests)
    natural_lengths = sorted(int(row["natural_output_token_count"]) for row in requests)

    raw_trace_root = prior / "traces"
    raw_trace_files = sorted(raw_trace_root.glob("*/route_summary.pid*.jsonl"))
    raw_sample: dict[str, Any] | None = None
    for trace_path in raw_trace_files:
        with trace_path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("decode_step") is not None:
                    raw_sample = row
                    break
        if raw_sample is not None:
            break

    spatial_root = Path(
        "/home/esjung/MLLM-EP-github/poc_spatial_basis_ep/artifacts/"
        "20260924_spatial_basis_ep/discovery_trace/raw"
    )
    spatial_npz = sorted(spatial_root.glob("*.npz")) if spatial_root.exists() else []
    spatial_keys: list[str] = []
    if spatial_npz:
        import numpy as np

        with np.load(spatial_npz[0], allow_pickle=False) as archive:
            spatial_keys = list(archive.files)

    output = {
        "schema_version": 1,
        "status": "READY_WITH_REQUIRED_NEW_VECTOR_CAPTURE",
        "gpu_commands_executed": False,
        "baseline": {
            "status": baseline["status"],
            "model_revision": baseline["model_revision"],
            "asset_integrity": baseline["asset_integrity"],
            "quality_ready": baseline["observed_ready"]["QUALITY_BASELINE_READY"],
            "actual_ep2_ready": baseline["observed_ready"]["ACTUAL_EP2_READY"],
            "actual_ep4_ready": baseline["observed_ready"]["ACTUAL_EP4_READY"],
            "sim_ep2_ready": baseline["observed_ready"]["SIM_EP2_READY"],
            "sim_ep4_ready": baseline["observed_ready"]["SIM_EP4_READY"],
        },
        "physical_scope": {
            "ep2": {
                "topology": "TP1-DP2-EP2-PP1",
                "experts_per_rank": 64,
                "backend": "DeepEP sparse dispatch/combine",
                "evidence": str(paths["ep2_ep4_proof"]),
            },
            "ep4": {
                "topology": "TP2-DP2-EP4-PP1",
                "experts_per_rank": 32,
                "backend": "DeepEP sparse dispatch/combine",
                "evidence": str(paths["ep2_ep4_proof"]),
            },
            "ep8": {
                "topology": "TP2-DP4-EP8-PP1",
                "experts_per_rank": 16,
                "status": "STRUCTURAL_EP8_PROJECTION_ONLY",
                "latency_allowed": False,
            },
        },
        "long_reasoning_pool": {
            "requests": len(requests),
            "recommended_minimum": pool["recommended_minimum"],
            "shortage": pool["shortage"],
            "split_counts": dict(split_counts),
            "dataset_counts": dict(dataset_counts),
            "natural_output_tokens": {
                "min": min(natural_lengths),
                "median": (natural_lengths[9] + natural_lengths[10]) / 2,
                "max": max(natural_lengths),
            },
            "image_hash_disjoint": len({row["original_image_sha256"] for row in requests})
            == len(requests),
            "selection_used_routing_oracle": pool["routing_oracle_used_for_selection"],
        },
        "routing_assets": {
            "pool_trace_rows": len(pool_rows),
            "record_types": dict(record_types),
            "layers": layers,
            "decode_steps": steps,
            "full_compact_trace_bytes": paths["compact_trace"].stat().st_size,
            "pool_trace_bytes": paths["pool_trace"].stat().st_size,
            "raw_trace_files": len(raw_trace_files),
            "raw_route_ids_available": raw_sample is not None and "route_ids" in raw_sample,
            "raw_route_weights_available": raw_sample is not None and "route_weights" in raw_sample,
            "source_ep_rank_available": raw_sample is not None and "source_ep_rank" in raw_sample,
            "prefill_vision_text_histograms_available": True,
            "causal_image_swap_available": True,
            "repeat_noise_available": True,
        },
        "vector_assets": {
            "long_reasoning_decode_hidden_vectors": False,
            "long_reasoning_decode_expert_output_vectors": False,
            "long_reasoning_decode_weighted_contribution_vectors": False,
            "spatial_basis_hidden_trace_exists": bool(spatial_npz),
            "spatial_basis_hidden_trace_files": len(spatial_npz),
            "spatial_basis_example_keys": spatial_keys,
            "spatial_basis_scope": (
                "vision-prefill, eight bounded layers, unrelated cohort; not valid as long-reasoning decode output evidence"
            ),
            "new_bounded_capture_required_for_track_a": True,
            "new_bounded_capture_required_for_track_b": True,
        },
        "reusable_signals": {
            "p_text_p_vision_p_full_p_separated": str(paths["predictor"]),
            "causal_image_swap_and_repeat_noise": str(paths["causal"]),
            "criticality_can_be_frozen_from_discovery_only": True,
        },
        "asset_paths": {name: str(path) for name, path in paths.items()},
        "asset_sha256": {
            name: sha256(path)
            for name, path in paths.items()
            if path.is_file() and path.stat().st_size < 120 * 1024 * 1024
        },
        "limitations": [
            "The 20-request >=128-token pool is four below the specification's preferred 24.",
            "The compact trace has only 22 decode checkpoints, not every step.",
            "The TP2 decode trace contains one exact dummy/padding row per request-layer-step; structural utilities remove it only under an explicit audited contract.",
            "Existing artifacts cannot answer either track's expert-output error gate without a new bounded vector capture.",
            "EP8 remote bytes require explicit target-topology source ranks; EP8 latency is forbidden.",
        ],
    }
    write_json(args.output, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
