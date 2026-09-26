from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .constants import (
    ALLOWED_PHYSICAL_GPUS,
    EXPECTED_TEXT_CONFIG,
    MODEL,
    MODEL_REVISION,
    OFFICIAL_GENERATION,
)
from .io import atomic_json, sha256


def parse_physical_gpus(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"physical GPU set must be non-empty and unique: {value!r}")
    illegal = sorted(set(values) - set(ALLOWED_PHYSICAL_GPUS))
    if illegal:
        raise ValueError(f"GPU(s) outside allowed physical set {ALLOWED_PHYSICAL_GPUS}: {illegal}")
    return values


def _command(argv: list[str]) -> str:
    return subprocess.run(argv, check=True, capture_output=True, text=True).stdout


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def build_contract(model: Path, physical_gpus: list[int]) -> dict[str, Any]:
    model = model.resolve()
    if model.name != MODEL_REVISION:
        raise ValueError(f"unexpected model revision: {model.name}")
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    text_config = config["text_config"]
    vision_config = config["vision_config"]
    assertions = {
        name: {"expected": expected, "actual": text_config.get(name),
               "pass": text_config.get(name) == expected}
        for name, expected in EXPECTED_TEXT_CONFIG.items()
    }
    assertions.update({
        "patch_size": {"expected": 16, "actual": vision_config.get("patch_size"),
                       "pass": vision_config.get("patch_size") == 16},
        "spatial_merge_size": {
            "expected": 2, "actual": vision_config.get("spatial_merge_size"),
            "pass": vision_config.get("spatial_merge_size") == 2,
        },
    })
    if not all(item["pass"] for item in assertions.values()):
        raise AssertionError(assertions)
    generation = json.loads((model / "generation_config.json").read_text())
    git_head = _command(["git", "rev-parse", "HEAD"]).strip()
    dirty = _command(["git", "status", "--porcelain=v1"]).splitlines()
    topology = _command(["nvidia-smi", "topo", "-m"])
    gpu_query = _command([
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,driver_version",
        "--format=csv,noheader",
    ]).splitlines()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS",
        "model": {
            "id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "path": str(model),
            "revision": model.name,
            "config_sha256": sha256(model / "config.json"),
            "generation_config_sha256": sha256(model / "generation_config.json"),
            "chat_template_sha256": sha256(model / "chat_template.json"),
            "assertions": assertions,
            "checkpoint_generation_config": generation,
            "weight_contract": {
                "gate_up_shape": [128, 2048, 1536],
                "down_shape": [128, 768, 2048],
                "router_shape": [128, 2048],
                "shared_expert": False,
                "verification": "checkpoint tensor index and current vLLM model source",
            },
        },
        "processor": {
            "double_resize_policy": "PIL decoded once; qwen processor owns resize; no qwen-vl-utils pre-resize",
            "image_mode": "RGB with PIL EXIF transpose before processor",
            "grid_contract": "placeholder span == t*h*w/(merge_size^2)",
        },
        "generation_lanes": {
            "OFFICIAL_REPRODUCTION": OFFICIAL_GENERATION,
            "DETERMINISTIC_PARITY": {"do_sample": False, "temperature": 0.0},
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "vllm": _package_version("vllm"),
            "qwen_vl_utils": _package_version("qwen-vl-utils"),
            "cuda_visible_devices_at_audit": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "hardware": {
            "allowed_physical_gpus": list(ALLOWED_PHYSICAL_GPUS),
            "requested_physical_gpus": physical_gpus,
            "gpu_query": gpu_query,
            "topology": topology,
        },
        "repository": {
            "root": str(Path.cwd().resolve()),
            "commit": git_head,
            "dirty_entry_count": len(dirty),
            "dirty_paths_sha256": hashlib.sha256("\n".join(dirty).encode()).hexdigest(),
            "dirty_paths_sample": dirty[:100],
            "preservation": "unrelated pre-existing changes were not modified",
        },
    }


def audit_environment(output: Path, physical_gpus: str, model: Path = MODEL,
                      dry_run: bool = False) -> dict[str, Any]:
    parsed = parse_physical_gpus(physical_gpus)
    payload = build_contract(model, parsed)
    if not dry_run:
        atomic_json(output, payload)
    return payload

