#!/usr/bin/env python3
"""Prepare a frozen, real-image request pool for hierarchical rebatching.

The same request multiset is reused by every schedule.  This makes the central
control exact: only the partition/order changes, never the one-dimensional
workload marginals.  Route-dependent plans are built later from a fresh
single-request profile rather than guessed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-family", type=int, default=8)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.source.read_text().splitlines()]
    families = [
        "text_only_no_image_control",
        "single_real_image",
        "two_real_images",
        "long_text_plus_image",
    ]
    selected: list[dict] = []
    for family in families:
        family_rows = [row for row in rows if row["family"] == family]
        if len(family_rows) < args.per_family:
            raise ValueError(f"{family}: need {args.per_family}, found {len(family_rows)}")
        selected.extend(family_rows[: args.per_family])

    enriched = []
    for ordinal, row in enumerate(selected):
        images = []
        total_pixels = 0
        for name in row["images"]:
            path = Path(name)
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                width, height = image.size
            images.append({"path": str(path.resolve()), "width": width, "height": height,
                           "pixels": width * height})
            total_pixels += width * height
        enriched.append({
            **row,
            "pool_ordinal": ordinal,
            "image_metadata": images,
            "image_count": len(images),
            "total_pixels": total_pixels,
            "question_chars": len(row["question"]),
            "question_words": len(row["question"].split()),
        })

    # Interleave families for P0.  P1/P2 and causal partitions are generated
    # after the fresh profile, but all reuse exactly this multiset.
    by_family = {family: [row for row in enriched if row["family"] == family]
                 for family in families}
    natural = []
    for index in range(args.per_family):
        natural.extend(by_family[family][index] for family in families)
    assert sorted(row["request_id"] for row in natural) == sorted(row["request_id"] for row in enriched)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as sink:
        for row in natural:
            sink.write(json.dumps(row) + "\n")
    manifest = {
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "pool_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "requests": len(natural),
        "families": {family: args.per_family for family in families},
        "distinct_image_paths": len({image["path"] for row in natural for image in row["image_metadata"]}),
        "control": "All policies use the exact same request/image/prompt multiset; only partition/order changes.",
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
