#!/usr/bin/env python3
"""Read-only hardware/software provenance capture; creates no CUDA context."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path


def _run(*command: str) -> str:
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return completed.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    commands = {
        "nvidia_smi_L.txt": ("nvidia-smi", "-L"),
        "nvidia_smi_topo_m.txt": ("nvidia-smi", "topo", "-m"),
        "nvidia_smi_nvlink_s.txt": ("nvidia-smi", "nvlink", "-s"),
        "nvidia_smi_q_gpu0_1.txt": ("nvidia-smi", "-q", "-i", "0,1"),
    }
    for filename, command in commands.items():
        (args.output / filename).write_text(_run(*command))
    model_config = json.loads(args.model_config.read_text())
    summary = {
        "capture_is_read_only": True,
        "cuda_contexts_created": False,
        "allowed_physical_gpus_for_experiments": [0, 1],
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_commit": _run("git", "rev-parse", "HEAD").strip(),
        "model": "inclusionAI/LLaDA2.0-mini",
        "revision": "dad945cac317da394b390f82c7b40691d8a881ed",
        "model_config": {
            "hidden_size": model_config.get("hidden_size"),
            "moe_intermediate_size": model_config.get("moe_intermediate_size"),
            "num_hidden_layers": model_config.get("num_hidden_layers"),
            "num_experts": model_config.get("num_experts"),
            "num_experts_per_tok": model_config.get("num_experts_per_tok"),
            "num_shared_experts": model_config.get("num_shared_experts"),
            "torch_dtype": model_config.get("torch_dtype"),
        },
        "topology_interpretation": (
            "NV18/all-active links are connectivity evidence, not independent "
            "per-peer application bandwidth; simulation uses measured endpoint calibration"
        ),
    }
    (args.output / "environment.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
