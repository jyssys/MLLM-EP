from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from ..contract import atomic_json, read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    records = []
    for case in manifest["cases"]:
        source = Path(case["original_image_path"])
        target = Path(case["neutral_image_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8)
            color = np.median(array.reshape(-1, 3), axis=0).astype(np.uint8)
            neutral = np.broadcast_to(color, array.shape).copy()
            Image.fromarray(neutral, mode="RGB").save(target, format="PNG", optimize=True)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        records.append({
            "case_id": case["case_id"],
            "source_sha256": case["original_image_sha256"],
            "neutral_sha256": digest,
            "width": int(neutral.shape[1]),
            "height": int(neutral.shape[0]),
            "transform": "PER_CHANNEL_GLOBAL_MEDIAN_CONSTANT_IMAGE_SAME_PIXEL_GEOMETRY",
        })
    atomic_json(args.output, {"schema_version": 1, "records": records})
    print({"neutral_images": len(records)})


if __name__ == "__main__":
    main()
