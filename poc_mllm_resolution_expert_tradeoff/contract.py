from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MODEL_REVISION = "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
MODEL = Path(
    "/home/esjung/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/"
    + MODEL_REVISION
)
FOUNDATION_RUN = Path(
    "/home/esjung/MLLM-EP-github/poc_mllm_ep_foundation/"
    "validation_runs/20260924_completion_v1"
)
FOUNDATION_BUNDLE = FOUNDATION_RUN / "frozen_baseline_bundle.json"
IMAGE_TOKEN_ID = 151655
REFERENCE_MIN_PIXELS = 65_536
REFERENCE_MAX_PIXELS = 16_777_216
RESOLUTION_RATIOS = (1.0, 0.5, 0.25)
VISION_K = (8, 6, 4)
TEXT_K = 8


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def verify_foundation_bundle() -> dict[str, Any]:
    bundle = load_json(FOUNDATION_BUNDLE)
    expected = {
        "QUALITY_BASELINE_READY": True,
        "ACTUAL_EP4_READY": True,
        "READY_FOR_PART2_PHYSICAL_EP4": True,
    }
    mismatches = {
        key: {"expected": value, "actual": bundle.get("status", {}).get(key)}
        for key, value in expected.items()
        if bundle.get("status", {}).get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"BLOCKED_BASELINE: {mismatches}")
    return bundle


def resolution_label(ratio: float) -> str:
    return {1.0: "R0", 0.5: "R05", 0.25: "R025"}[float(ratio)]


def configured_max_pixels(reference_processed_pixels: int, ratio: float) -> int | None:
    """Return a legitimate processor area cap for a target token ratio.

    R0 deliberately returns ``None`` so that the frozen processor contract is
    byte-for-byte unchanged.  Lower resolutions start from the original image
    and cap its processed area; they never resize an already resized image.
    """
    if ratio == 1.0:
        return None
    target = int(round(reference_processed_pixels * float(ratio)))
    return max(REFERENCE_MIN_PIXELS, min(REFERENCE_MAX_PIXELS, target))


def factorial_request_id(base_id: str, ratio: float, k_visual: int) -> str:
    return f"{base_id}__{resolution_label(ratio)}__K{k_visual}"

