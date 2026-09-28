from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image, ImageOps

from poc_mllm_vision_dominant_captioning.contract import (
    DOMAINS,
    UNIVERSAL_PROMPT,
    atomic_json,
    sha256_file,
    stable_key,
)

ROOT = Path(__file__).resolve().parents[1]
MIXED = ROOT.parent / "poc_mllm_mixed_domain_online_calibration/artifacts/manifests/frozen_domain_pool.json"
SOURCES = {
    "docci": (
        "not-lain/docci-small",
        Path.home() / ".cache/huggingface/hub/datasets--not-lain--docci-small/snapshots/"
        "d3709dca05c2a9c1e16f09e513e943b25047be9d/data/train-00000-of-00001.parquet",
    ),
    "textcaps": (
        "lmms-lab-encoder/TextCaps",
        Path.home() / ".cache/huggingface/hub/datasets--lmms-lab-encoder--TextCaps/snapshots/"
        "e9e223338832318a6161d04648871c18a024ce85/data/val-00000-of-00002.parquet",
    ),
    "nocaps": (
        "lmms-lab-encoder/NoCaps",
        Path.home() / ".cache/huggingface/hub/datasets--lmms-lab-encoder--NoCaps/snapshots/"
        "a26b3fe1e0021164ec430c57c48085d58f2fe922/data/"
        "validation-00000-of-00003-55b5edbf01d47b39.parquet",
    ),
}


def _caption(row: dict[str, Any], domain: str) -> str:
    if domain == "docci":
        return str(row.get("answer", ""))
    if domain == "textcaps":
        values = row.get("reference_strs") or row.get("caption_str") or []
        return " ".join(str(value) for value in values)
    values = row.get("annotations_captions") or []
    return " ".join(str(value) for value in values)


def _source_id(row: dict[str, Any], domain: str, index: int) -> str:
    if domain == "textcaps":
        return str(row["image_id"])
    if domain == "nocaps":
        return str(row["image_open_images_id"])
    return f"docci-{index:04d}"


def _parquet_candidates(domain: str) -> list[dict[str, Any]]:
    repo, path = SOURCES[domain]
    if not path.is_file():
        raise FileNotFoundError(path)
    candidates = []
    index = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=32):
        for row in batch.to_pylist():
            source_id = _source_id(row, domain, index)
            image = row["image"]
            blob = image.get("bytes") if isinstance(image, dict) else None
            if not blob:
                index += 1
                continue
            try:
                with Image.open(io.BytesIO(blob)) as opened:
                    rgb = ImageOps.exif_transpose(opened).convert("RGB")
                    width, height = rgb.size
            except Exception:
                index += 1
                continue
            candidates.append({
                "domain": domain,
                "source_dataset": repo,
                "source_id": source_id,
                "selection_key": stable_key("caption-cohort-v1", domain, source_id),
                "blob": blob,
                "image_width": width,
                "image_height": height,
                "reference_description_length": len(_caption(row, domain).split()),
            })
            index += 1
    candidates.sort(key=lambda row: row["selection_key"])
    return candidates


def _write_embedded(rows: list[dict[str, Any]], image_root: Path, seen: set[str], limit: int) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        blob = row.pop("blob")
        digest = hashlib.sha256(blob).hexdigest()
        if digest in seen:
            continue
        with Image.open(io.BytesIO(blob)) as opened:
            rgb = ImageOps.exif_transpose(opened).convert("RGB")
            path = image_root / row["domain"] / f"{row['source_id']}_{digest[:12]}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            rgb.save(path, format="JPEG", quality=95)
        digest = sha256_file(path)
        if digest in seen:
            path.unlink(missing_ok=True)
            continue
        seen.add(digest)
        output.append({**row, "image_path": str(path.resolve()), "image_sha256": digest})
        if len(output) == limit:
            break
    if len(output) != limit:
        raise RuntimeError(f"only {len(output)} unique images for {rows[0]['domain']}")
    return output


def _reuse_domain(source_domain: str, target_domain: str, seen: set[str], limit: int) -> list[dict[str, Any]]:
    pool = json.loads(MIXED.read_text())["requests"]
    rows = [row for row in pool if row["domain"] == source_domain]
    rows.sort(key=lambda row: stable_key("caption-cohort-v1", target_domain, row["source_id"]))
    output = []
    for row in rows:
        path = Path(row["image_path"])
        if not path.is_file():
            continue
        digest = sha256_file(path)
        if digest in seen:
            continue
        seen.add(digest)
        with Image.open(path) as opened:
            width, height = ImageOps.exif_transpose(opened).size
        output.append({
            "domain": target_domain,
            "source_dataset": f"MMStar/{source_domain}",
            "source_id": str(row["source_id"]),
            "selection_key": stable_key("caption-cohort-v1", target_domain, row["source_id"]),
            "image_path": str(path.resolve()),
            "image_sha256": digest,
            "image_width": width,
            "image_height": height,
            "reference_description_length": None,
        })
        if len(output) == limit:
            break
    if len(output) != limit:
        raise RuntimeError(f"only {len(output)} rows for substitute {target_domain}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-domain", type=int, default=40)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/manifests/frozen_image_cohort.json")
    parser.add_argument("--stream", type=Path, default=ROOT / "artifacts/manifests/frozen_iid_stream.json")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/images")
    args = parser.parse_args()
    if args.per_domain < 40:
        raise ValueError("spec requires at least 40 images/domain")
    seen: set[str] = set()
    by_domain = {
        domain: _write_embedded(_parquet_candidates(domain), args.image_root, seen, args.per_domain)
        for domain in ("docci", "textcaps", "nocaps")
    }
    by_domain["dense_relation"] = _reuse_domain("structured_diagram_layout", "dense_relation", seen, args.per_domain)
    by_domain["scene_layout"] = _reuse_domain("general_visual_reasoning", "scene_layout", seen, args.per_domain)
    requests = []
    prompt_hash = hashlib.sha256(UNIVERSAL_PROMPT.encode()).hexdigest()
    for domain in DOMAINS:
        for index, row in enumerate(by_domain[domain]):
            requests.append({
                **row,
                "request_id": f"{domain}::{row['source_id']}",
                "domain_index": index,
                "analysis_split": "discovery" if index < 24 else "heldout",
                "question": UNIVERSAL_PROMPT,
                "universal_prompt_sha256": prompt_hash,
                "visual_token_count": None,
                "generated_output_length": None,
                "selection_uses_routing": False,
                "selection_uses_generated_output": False,
            })
    if len({row["image_sha256"] for row in requests}) != len(requests):
        raise AssertionError("image hash duplicate")
    atomic_json(args.output, {
        "schema_version": 1,
        "selection_frozen_before_routing": True,
        "selection_uses_routing": False,
        "selection_uses_generated_output": False,
        "prompt": UNIVERSAL_PROMPT,
        "prompt_sha256": prompt_hash,
        "domains": list(DOMAINS),
        "per_domain": args.per_domain,
        "request_count": len(requests),
        "source_deviations": {
            "dense_relation": "MMStar structured-diagram/layout public images substitute for unavailable paragraph-caption image assets",
            "scene_layout": "MMStar general visual reasoning public images substitute for unavailable ADE20K Localized Narratives pixels",
        },
        "requests": requests,
    })
    order = []
    for index in range(args.per_domain):
        for domain in DOMAINS:
            order.append(f"{domain}::{by_domain[domain][index]['source_id']}")
    atomic_json(args.stream, {"schema_version": 1, "seed": 3407, "request_ids": order})


if __name__ == "__main__":
    main()
