#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from safetensors import safe_open

from poc_mllm_rank_aware_active_expert_reduction.contract import (
    DTYPE, EXPERTS, HIDDEN, INTERMEDIATE, LAYERS, MODEL, MODEL_REVISION, TOP_K,
    atomic_json,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--foundation-bundle", type=Path, default=Path(
        "poc_mllm_ep_foundation/artifacts/frozen_baseline_bundle.json"))
    args = parser.parse_args()
    model = Path(MODEL)
    config = json.loads((model / "config.json").read_text())
    text = config["text_config"]
    expected = {
        "hidden_size": HIDDEN, "moe_intermediate_size": INTERMEDIATE,
        "num_experts": EXPERTS, "num_experts_per_tok": TOP_K,
        "num_hidden_layers": LAYERS,
    }
    observed = {key: text.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError({"expected": expected, "observed": observed})
    index = json.loads((model / "model.safetensors.index.json").read_text())
    keys = [
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
    ]
    tensors = {}
    for key in keys:
        shard = model / index["weight_map"][key]
        with safe_open(shard, framework="pt", device="cpu") as handle:
            tensor = handle.get_slice(key)
            shape = list(tensor.get_shape())
        tensors[key] = {"shape": shape, "dtype": "BF16", "shard": shard.name}
    gate = tensors[keys[0]]["shape"]
    down = tensors[keys[1]]["shape"]
    # Checkpoint layout is [E,input,gate+up] and [E,intermediate,input].
    # vLLM transforms these to its fused runtime layout during load.
    if gate != [EXPERTS, HIDDEN, 2 * INTERMEDIATE] or down != [EXPERTS, INTERMEDIATE, HIDDEN]:
        raise RuntimeError(tensors)
    per_expert = (2 * INTERMEDIATE * HIDDEN + HIDDEN * INTERMEDIATE) * 2
    bundle = json.loads(args.foundation_bundle.read_text())
    git = subprocess.run(["git", "rev-parse", "HEAD"], text=True,
                         capture_output=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], text=True,
                            capture_output=True, check=True).stdout.splitlines()
    payload = {
        "status": "PASS_REUSED_VALIDATED_FOUNDATION",
        "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "model_path": str(model), "revision": MODEL_REVISION, "dtype": DTYPE,
        "config_expected": expected, "config_observed": observed,
        "expert_tensors": tensors, "per_expert_weight_bytes": per_expert,
        "per_expert_weight_mib": per_expert / (1 << 20),
        "foundation_bundle": str(args.foundation_bundle.resolve()),
        "foundation_bundle_sha256": sha256(args.foundation_bundle),
        "foundation_status": bundle.get("status", bundle.get("overall_status")),
        "topologies": {
            "ep2": {"physical_gpus": [2, 3], "tp": 1, "dp": 2, "ep": 2,
                    "experts_per_rank": 64},
            "ep4": {"physical_gpus": [2, 3, 6, 7], "tp": 2, "dp": 2, "ep": 4,
                    "experts_per_rank": 32},
            "ep8": {"status": "STRUCTURAL_EP8_PROJECTION_ONLY", "experts_per_rank": 16},
        },
        "git_head": git, "worktree_status_at_audit": status,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
