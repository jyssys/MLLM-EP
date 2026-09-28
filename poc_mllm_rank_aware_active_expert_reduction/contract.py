from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable


MODEL = (
    "/home/esjung/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/"
    "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
)
MODEL_REVISION = "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
ALLOWED_GPUS = {2, 3, 6, 7}
EXPERTS = 128
TOP_K = 8
LAYERS = 48
HIDDEN = 2048
INTERMEDIATE = 768
DTYPE = "bfloat16"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def expert_owner(expert: int, ep: int) -> int:
    if ep not in (2, 4, 8) or not 0 <= expert < EXPERTS:
        raise ValueError((expert, ep))
    return expert // (EXPERTS // ep)
