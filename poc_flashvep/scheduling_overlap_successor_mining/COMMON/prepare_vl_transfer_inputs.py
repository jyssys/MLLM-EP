"""Frozen real-image transfer inputs; reuses source data, not old measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import random

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(s) for s in args.source.read_text().splitlines()]
    selected = []
    rng = random.Random(20260908)
    for dataset in ("gqa", "chartqa"):
        pool = [r for r in rows if r["dataset"] == dataset and r["split"] != "calibration"]
        rng.shuffle(pool)
        seen = set()
        for row in pool:
            if row["image_id"] in seen:
                continue
            seen.add(row["image_id"])
            selected.append(row)
            if len(seen) == 16:
                break
        assert len(seen) == 16
    frozen = []
    for i, row in enumerate(selected):
        common = {"source_request_id": row["id"], "source_dataset": row["dataset"],
                  "source_image_id": row["image_id"], "source_answers": row["answers"],
                  "question": row["question"], "max_new_tokens": 64, "ignore_eos": True}
        images = row["images"]
        frozen.append({**common, "request_id": f"single_{i:03d}",
                       "family": "single_real_image", "images": images})
        frozen.append({**common, "request_id": f"text_{i:03d}",
                       "family": "text_only_no_image_control", "images": [],
                       "quality_scope": "image removed; not a task-quality comparison"})
        other = selected[(i + 1) % len(selected)]
        frozen.append({**common, "request_id": f"multi_{i:03d}",
                       "family": "two_real_images", "images": images + other["images"],
                       "question": "Describe the first image, then the second image. "
                                   "Keep the descriptions separate.",
                       "source_answers": [], "quality_scope": "descriptive workload, no exact-answer score"})
        frozen.append({**common, "request_id": f"longtext_{i:03d}",
                       "family": "long_text_plus_image", "images": images,
                       "question": ("Inspect the image carefully. Consider objects, labels, counts, "
                                    "spatial relationships, and visible evidence. Do not invent details. ") * 16
                                   + row["question"]})
    image_metadata = {}
    for row in frozen:
        for name in row["images"]:
            if name not in image_metadata:
                path = Path(name)
                with Image.open(path) as image:
                    dimensions = list(image.size)
                image_metadata[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                        "original_dimensions": dimensions}
    path = args.out / "requests.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in frozen))
    manifest = {"source": str(args.source.resolve()),
                "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "requests": len(frozen), "unique_images": len(image_metadata),
                "images": image_metadata, "seed": 20260908,
                "request_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "scope": "Fresh execution inputs only; no old latency reused",
                "controls": ["image-stripped controls are not quality evidence",
                             "actual processed/vision tokens must be recorded at runtime",
                             "token-matched inference must not be inferred from image count"]}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("requests", "unique_images", "request_sha256")}))


if __name__ == "__main__":
    main()
