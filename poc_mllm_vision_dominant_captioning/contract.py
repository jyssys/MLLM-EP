from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
MODEL_REVISION = "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
DTYPE = "bfloat16"
NUM_LAYERS = 48
NUM_EXPERTS = 128
TOP_K = 8
HIDDEN_SIZE = 2048
MOE_INTERMEDIATE_SIZE = 768
DOMAINS = ("docci", "textcaps", "nocaps", "dense_relation", "scene_layout")
UNIVERSAL_PROMPT = (
    "Describe this image in detail. Mention the main visible entities, their attributes, "
    "spatial relationships, visible text when relevant, and notable actions or context. "
    "Do not add facts that are not visually supported."
)
DECODE_STEPS = (
    1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112,
    120, 128, 144, 160, 176, 192, 208, 224, 240, 256,
)
WINDOWS = ("1-32", "33-64", "65-96", "97-128", "129-192", "193-256", "1-256")
SPEC_PATH = Path(
    "/home/esjung/MLLM-EP-github/poc_flashvep/reports/"
    "MLLM_Vision_Dominant_MultiDomain_Detailed_Captioning_EP2_EP4_EP8_Video_PoC.md"
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "wt"
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with opener(temporary, mode) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(*values: object) -> str:
    text = "\0".join(str(value) for value in values)
    return hashlib.sha256(text.encode()).hexdigest()


def owner_rank(expert_id: int, ep_size: int) -> int:
    if ep_size not in (2, 4, 8):
        raise ValueError(f"unsupported EP size: {ep_size}")
    if not 0 <= expert_id < NUM_EXPERTS:
        raise ValueError(expert_id)
    return expert_id // (NUM_EXPERTS // ep_size)


def decode_window(step: int) -> str:
    if 1 <= step <= 32:
        return "1-32"
    if step <= 64:
        return "33-64"
    if step <= 96:
        return "65-96"
    if step <= 128:
        return "97-128"
    if step <= 192:
        return "129-192"
    if step <= 256:
        return "193-256"
    raise ValueError(step)
