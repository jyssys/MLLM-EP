from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ALLOWED_GPU_STRING = "2,3,6,7"
MODEL_REVISION = "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
MODEL = (
    "/home/esjung/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/"
    f"{MODEL_REVISION}"
)
TOPOLOGY = "TP2-DP2-EP4-PP1"
NUM_EXPERTS = 128
EXPERTS_PER_RANK = 32
TOP_K = 8
HIDDEN_SIZE = 2048
DTYPE_BYTES = 2


def require_gpu_contract() -> None:
    actual = os.environ.get("CUDA_VISIBLE_DEVICES")
    if actual != ALLOWED_GPU_STRING:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES must be {ALLOWED_GPU_STRING}, got {actual!r}"
        )


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
