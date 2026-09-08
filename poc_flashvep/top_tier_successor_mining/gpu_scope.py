"""Validate the authorized physical mapping before GPU initialization."""
import json
import os
from pathlib import Path


def allowed_devices():
    policy = json.loads(Path(__file__).with_name("GPU_EXECUTION_POLICY.json").read_text())
    if not policy["gpu_runs_allowed"]:
        raise RuntimeError("GPU work is paused by the user")
    devices = policy["physical_gpus"]
    actual = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    expected = ",".join(map(str, devices))
    if actual != expected:
        raise RuntimeError(f"Required physical mapping {expected}; got {actual!r}")
    return devices


def physical_gpu(logical_rank):
    return allowed_devices()[logical_rank]
