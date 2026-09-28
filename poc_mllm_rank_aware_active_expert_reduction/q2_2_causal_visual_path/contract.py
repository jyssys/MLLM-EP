from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable

MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
MODEL_REVISION = "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
MODEL_PATH = (
    "/home/esjung/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/"
    + MODEL_REVISION
)
ALLOWED_GPUS = (2, 3, 6, 7)
TOPOLOGY = "TRUE_TP2_DP2_EP4_PP1"
LAYERS = (4, 12, 24, 36, 44, 47)
STEPS = (8, 16, 32, 48)
TOP_K = 8
SEED = 3407


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)
