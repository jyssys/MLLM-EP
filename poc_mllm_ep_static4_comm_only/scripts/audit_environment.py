#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path

import torch

from poc_mllm_ep_static4_comm_only.contract import (
    ALLOWED_GPU_STRING,
    MODEL,
    MODEL_REVISION,
    TOPOLOGY,
    atomic_json,
)


def command(*args: str) -> dict[str, object]:
    completed = subprocess.run(args, text=True, capture_output=True, check=False)
    return {
        "argv": list(args),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ALLOWED_GPU_STRING:
        raise RuntimeError("audit must use the authorized physical GPU mask")
    payload = {
        "status": "AUDITED",
        "physical_gpus": [2, 3, 6, 7],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "topology": TOPOLOGY,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": version("vllm"),
            "deep_ep": version("deep-ep"),
            "transformers": version("transformers"),
        },
        "nvidia_smi_query": command(
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ),
        "nvidia_smi_processes": command(
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ),
        "nvidia_smi_topology": command("nvidia-smi", "topo", "-m"),
        "git": command("git", "rev-parse", "HEAD"),
        "git_status": command("git", "status", "--short"),
        "frozen_runtime": {
            "dtype": "bfloat16",
            "experts": 128,
            "top_k": 8,
            "expert_placement": "linear",
            "deep_ep_backend": "deepep_high_throughput",
            "eplb": False,
            "redundant_experts": 0,
            "dbo": False,
            "prefix_cache": False,
            "enforce_eager": True,
        },
    }
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
