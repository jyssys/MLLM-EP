from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
MODEL_REVISION = "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
DTYPE = "bfloat16"
NUM_LAYERS = 48
NUM_EXPERTS = 128
TOP_K = 8
EP_SIZE = 4
EXPERTS_PER_RANK = 32
GROUP_SIZE = 4
PREDICTED_SET_SIZE = 32
RANDOM_SEED = 3407
POLICIES = ("S0_FIFO_RANDOM", "S1_LENGTH", "S2_TEXT", "S3_FULL", "S4_SEPARATED", "S5_ORACLE")
WINDOWS = {
    "1-32": (1, 32),
    "33-64": (33, 64),
    "65-96": (65, 96),
    "97-128": (97, 128),
    "1-128": (1, 128),
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def owner_rank(expert: int) -> int:
    if expert < 0 or expert >= NUM_EXPERTS:
        raise ValueError(expert)
    return expert // EXPERTS_PER_RANK
