from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from ..contract import BASE_COMMIT, MODEL_PATH, MODEL_REVISION, PHYSICAL_GPUS, TOPOLOGY, atomic_json, read_json


def command(*args: str) -> str:
    return subprocess.run(args, text=True, capture_output=True, check=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--prior-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(Path(MODEL_PATH) / "config.json")
    text = config["text_config"]
    gpu_rows = []
    for line in command(
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader"
    ).splitlines():
        fields = [value.strip() for value in line.split(",")]
        if int(fields[0]) in PHYSICAL_GPUS:
            gpu_rows.append({"physical_index": int(fields[0]), "uuid": fields[1], "name": fields[2], "memory": fields[3]})
    head = command("git", "-C", str(args.repo), "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "-C", str(args.repo), "merge-base", "--is-ancestor", BASE_COMMIT, head]
    ).returncode == 0
    prior = read_json(args.prior_baseline)
    checks = {
        "base_commit_is_ancestor": ancestor,
        "model_revision": Path(MODEL_PATH).name == MODEL_REVISION,
        "hidden_size": text["hidden_size"] == 2048,
        "intermediate_size": text["moe_intermediate_size"] == 768,
        "experts": text["num_experts"] == 128,
        "top_k": text["num_experts_per_tok"] == 8,
        "layers": text["num_hidden_layers"] == 48,
        "prior_true_ep4_ready": prior.get("status") == "READY",
        "gpu_set_exact": [row["physical_index"] for row in gpu_rows] == list(PHYSICAL_GPUS),
    }
    payload = {
        "status": "READY" if all(checks.values()) else "BLOCKED_BASELINE",
        "checks": checks,
        "git_head": head,
        "base_commit": BASE_COMMIT,
        "model_path": MODEL_PATH,
        "model_revision": MODEL_REVISION,
        "dtype": "BF16",
        "shape": {"d": 2048, "m": 768, "experts": 128, "top_k": 8, "layers": 48},
        "topology": TOPOLOGY,
        "physical_gpus": gpu_rows,
        "prior_evidence": str(args.prior_baseline.resolve()),
        "vllm_version": command(
            "/home/esjung/.venvs/flashvep-deepep-v020/bin/python", "-c",
            "import vllm; print(vllm.__version__)",
        ),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
