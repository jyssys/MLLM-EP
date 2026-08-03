"""Download a 512-image ShareGPT4V calibration subset.

The public ShareGPT4V HF dataset stores image paths such as
``coco/train2017/000000000009.jpg`` rather than embedding all source images.
For Phase 1 calibration assets, this script materializes the first 512 COCO
train2017-backed records from the GPT-4V caption metadata.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve

from huggingface_hub import hf_hub_download


REPO_ID = "Lin-Chen/ShareGPT4V"
METADATA_FILE = "sharegpt4v_instruct_gpt4-vision_cap100k.json"
COCO_BASE_URL = "http://images.cocodataset.org"


def _download_one(record: dict, output_root: Path) -> dict:
    rel = record["image"]
    if not rel.startswith("coco/"):
        raise ValueError(f"only COCO-backed records are supported, got {rel}")
    coco_rel = rel[len("coco/") :]
    url = f"{COCO_BASE_URL}/{coco_rel}"
    local_path = output_root / "images" / rel
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if not local_path.exists() or local_path.stat().st_size == 0:
        urlretrieve(url, local_path)
    return {
        "id": record.get("id"),
        "image": rel,
        "image_url": url,
        "local_image": str(local_path),
        "conversations": record.get("conversations", []),
    }


def main() -> None:
    output_root = Path("data/sharegpt4v_512")
    output_root.mkdir(parents=True, exist_ok=True)
    meta_path = hf_hub_download(
        REPO_ID,
        repo_type="dataset",
        filename=METADATA_FILE,
        local_dir="data/sharegpt4v_meta",
    )
    with open(meta_path, encoding="utf-8") as f:
        records = json.load(f)

    coco_records = [row for row in records if str(row.get("image", "")).startswith("coco/train2017/")][:512]
    if len(coco_records) != 512:
        raise RuntimeError(f"expected 512 COCO records, found {len(coco_records)}")

    manifest_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(_download_one, row, output_root) for row in coco_records]
        for future in as_completed(futures):
            manifest_rows.append(future.result())

    manifest_rows.sort(key=lambda row: row["id"])
    manifest_path = output_root / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    bad = [row for row in manifest_rows if not Path(row["local_image"]).exists() or Path(row["local_image"]).stat().st_size == 0]
    if bad:
        raise RuntimeError(f"{len(bad)} image downloads failed")
    print(f"Wrote {manifest_path} with {len(manifest_rows)} images")


if __name__ == "__main__":
    main()
