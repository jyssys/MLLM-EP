from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from poc_mllm_rank_aware_active_expert_reduction.contract import MODEL as MODEL


MODEL_REVISION = "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
ALLOWED_GPUS = (2, 3, 6, 7)
TOPOLOGY = "TP2-DP2-EP4-PP1"
HIDDEN_SIZE = 2048
DTYPE_BYTES = 2
NUM_EXPERTS = 128
TOP_K = 8
EXPERTS_PER_RANK = 32


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
