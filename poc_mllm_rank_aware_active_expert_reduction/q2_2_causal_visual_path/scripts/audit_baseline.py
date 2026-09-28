from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from ..contract import MODEL_PATH, MODEL_REVISION, atomic_json, read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(Path(MODEL_PATH) / "config.json")["text_config"]
    expected = {
        "hidden_size": 2048, "moe_intermediate_size": 768, "num_experts": 128,
        "num_experts_per_tok": 8, "num_hidden_layers": 48, "dtype": "bfloat16",
    }
    mismatches = {key: {"expected": value, "actual": config.get(key)}
                  for key, value in expected.items() if config.get(key) != value}
    prior = read_json(args.prior)
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    uuids = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"
    ], text=True).splitlines()
    selected = [line for line in uuids if int(line.split(",", 1)[0]) in {2, 3, 6, 7}]
    payload = {
        "status": "READY" if not mismatches and prior.get("overall_verdict") == "GENERIC_RANK_AWARE_ONLY" else "BLOCKED",
        "git_sha_at_audit": git_sha, "model_path": MODEL_PATH, "model_revision": MODEL_REVISION,
        "model_shape_contract": expected, "shape_mismatches": mismatches,
        "physical_gpu_records": selected, "allowed_gpus": [2, 3, 6, 7],
        "topology": "TRUE_TP2_DP2_EP4_PP1", "backend": "deepep_high_throughput",
        "prior_q1": prior.get("q1"), "prior_q2": prior.get("q2"), "prior_q3": prior.get("q3"),
        "scope": "CAUSAL_FUNCTIONAL_DISCOVERY_NO_LATENCY_CLAIM",
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
