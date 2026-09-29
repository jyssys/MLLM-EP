from __future__ import annotations

from pathlib import Path
from typing import Any

from poc_mllm_ep_communication_motivation.contract import (
    ALLOWED_GPUS,
    DTYPE_BYTES,
    EXPERTS_PER_RANK,
    HIDDEN_SIZE,
    MODEL,
    MODEL_REVISION,
    NUM_EXPERTS,
    TOPOLOGY,
    TOP_K,
    atomic_json,
    percentile,
    read_json,
)

PROJECT = Path(__file__).resolve().parent
UPSTREAM_REPO = "https://github.com/scale-snu/layered-prefill"
UPSTREAM_COMMIT = "053f80e5201a7c0ab56e468e3d578907a2ca9cc3"
CHUNK_SIZES = (0, 2048, 1024, 512, 256)


def chunk_label(chunk_size: int) -> str:
    return "FULL" if chunk_size == 0 else f"CP{chunk_size}"


def engine_token_budget(chunk_size: int, decode_concurrency_per_dp: int = 4) -> int:
    if chunk_size == 0:
        return 20_000
    return chunk_size + decode_concurrency_per_dp


def layered_stage_count(num_batched_tokens: int, stage_cap: int = 16) -> int:
    """Exact public Layered-Prefill stage policy for the supported 16 cap."""
    if stage_cap != 16:
        raise ValueError("this PoC pins the upstream 16-stage-cap policy")
    if num_batched_tokens <= 512:
        return 1
    if num_batched_tokens <= 1024:
        return 2
    if num_batched_tokens <= 2048:
        return 4
    if num_batched_tokens <= 4096:
        return 8
    return 16


__all__ = [
    "ALLOWED_GPUS",
    "CHUNK_SIZES",
    "DTYPE_BYTES",
    "EXPERTS_PER_RANK",
    "HIDDEN_SIZE",
    "MODEL",
    "MODEL_REVISION",
    "NUM_EXPERTS",
    "PROJECT",
    "TOPOLOGY",
    "TOP_K",
    "UPSTREAM_COMMIT",
    "UPSTREAM_REPO",
    "Any",
    "atomic_json",
    "chunk_label",
    "engine_token_budget",
    "layered_stage_count",
    "percentile",
    "read_json",
]
