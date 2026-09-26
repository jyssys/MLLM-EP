from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from ..constants import REPO
from ..io import atomic_json


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_sanity_manifest(output: Path) -> dict[str, Any]:
    ski = Path("/home/esjung/anaconda3/lib/python3.14/site-packages/skimage/data")
    mode = Path("/home/esjung/MLLM-EP/external/MODE/assets")
    tui = Path("/home/esjung/MLLM-EP/external/lmms-eval/docs/images")
    blank = output.parent / "generated_blank_white.png"
    blank.parent.mkdir(parents=True, exist_ok=True)
    if not blank.exists():
        Image.new("RGB", (512, 512), "white").save(blank)
    specs = [
        ("ocr_text", ski / "text.png", "Transcribe the visible text exactly."),
        ("ocr_page", ski / "page.png", "Read the heading and the first visible line."),
        ("ocr_ui", tui / "tui-main.png", "What is the most prominent visible interface title or label?"),
        ("number_clock", ski / "clock_motion.png", "Describe the clock-like object and any visible numbers."),
        ("count_coins", ski / "coins.png", "Count the visible coins as carefully as possible and give the number."),
        ("count_board", ski / "chessboard_RGB.png", "How many alternating squares are visible along one side?"),
        ("scene_astronaut", ski / "astronaut.png", "Who or what is in the image and where is the subject?"),
        ("scene_cat", ski / "chelsea.png", "What animal is shown, and what color is it?"),
        ("scene_coffee", ski / "coffee.png", "What drink and surrounding objects are visible?"),
        ("chart", mode / "bit-allocate.png", "Read the chart title or main labels and summarize its comparison."),
        ("wrong_image_control", ski / "rocket.jpg", "What animal is shown, and what color is it?"),
        ("blank_control", blank, "What animal is shown, and what color is it?"),
    ]
    requests = []
    for request_id, path, question in specs:
        if not path.exists():
            raise FileNotFoundError(path)
        with Image.open(path) as raw:
            oriented = ImageOps.exif_transpose(raw)
            mode_before = raw.mode
            size = list(oriented.size)
        requests.append({
            "request_id": request_id,
            "task": "manual_image_plumbing",
            "image_path": str(path.resolve()),
            "image_sha256": _hash(path),
            "decoded_mode_before_rgb": mode_before,
            "exif_transposed_size": size,
            "question": question,
            "expected_kind": "image-dependent full answer",
        })
    missing = str((output.parent / "definitely_missing.png").resolve())
    payload = {
        "schema_version": "foundation.request_manifest.v1",
        "kind": "image_plumbing_sanity",
        "requests": requests,
        "negative_controls": {
            "missing_image": {"path": missing, "expected": "fail loud before model invocation"},
            "wrong_image_pair": ["scene_cat", "wrong_image_control"],
            "blank_pair": ["scene_cat", "blank_control"],
        },
    }
    atomic_json(output, payload)
    return payload


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    ids = [str(row["request_id"]) for row in manifest["requests"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate request IDs")
    return manifest

