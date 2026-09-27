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
DOMAINS = (
    "visual_math_geometry",
    "chart_plot_reasoning",
    "structured_diagram_layout",
    "general_visual_reasoning",
)
WARMUP_RATIOS = (0.0, 0.01, 0.05, 0.10, 0.20)
DECODE_STEPS = (1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128)
WINDOWS = ("1-32", "33-64", "65-96", "97-128", "1-128")
REASONING_PROMPT = (
    "Use the image to solve the problem. Give concise but explicit image-grounded "
    "reasoning, then end with a separate line in the form `Final answer: ...`."
)
SPEC_PATH = Path(
    "/home/esjung/MLLM-EP-github/poc_flashvep/reports/"
    "MLLM_Mixed_Domain_Online_Vision_Signature_Calibration_EP2_EP4_EP8_PoC.md"
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
    raise ValueError(step)


def warmup_count(ratio: float, domain_size: int) -> int:
    return 0 if ratio == 0 else max(1, math.ceil(ratio * domain_size))
