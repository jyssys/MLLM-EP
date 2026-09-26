from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from poc_mllm_ep_foundation.io import atomic_json
from poc_mllm_resolution_expert_tradeoff.contract import (
    FOUNDATION_RUN,
    RESOLUTION_RATIOS,
    VISION_K,
    factorial_request_id,
    resolution_label,
    stable_key,
    verify_foundation_bundle,
)


OCR_TERMS = (
    "read", "written", "text", "letter", "label", "sign", "logo", "brand",
    "caption", "title", "poster", "word", "number", "digit", "sequence",
)
SCENE_CATEGORIES = {
    "image scene and topic", "recognition", "single-instance reasoning",
    "cross-instance attribute reasoning", "cross-instance relation reasoning",
    "localization", "object counting", "image emotion", "image style & quality",
}


def _load(name: str) -> list[dict[str, Any]]:
    return json.loads((FOUNDATION_RUN / "data" / name).read_text())["requests"]


def _one_per_image(rows: list[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["image_sha256"])].append(row)
    picked = []
    for image_hash, members in groups.items():
        picked.append(min(members, key=lambda row: stable_key(salt + str(row["request_id"]))))
    return sorted(picked, key=lambda row: stable_key(salt + str(row["image_sha256"])))


def _strata() -> dict[str, list[dict[str, Any]]]:
    mmstar = _load("mmstar_full_manifest.json")
    chartqa = _load("chartqa_full_manifest.json")
    ocr = [
        row for row in mmstar
        if any(term in str(row.get("raw_question", "")).lower() for term in OCR_TERMS)
    ]
    scene = [row for row in mmstar if row.get("l2_category") in SCENE_CATEGORIES]
    return {
        # The Part-1 registry has no DocVQA/TextVQA asset.  This is explicitly
        # a public-MMStar text-sensitive proxy, never reported as OCRBench.
        "ocr_document_proxy": _one_per_image(ocr, "ocr"),
        "chart_numeric": _one_per_image(chartqa, "chart"),
        "scene_spatial": _one_per_image(scene, "scene"),
    }


def _annotate(row: dict[str, Any], stratum: str, split: str) -> dict[str, Any]:
    return {
        **row,
        "base_request_id": str(row["request_id"]),
        "part2_stratum": stratum,
        "part2_split": split,
        "selection_key": stable_key(f"{stratum}:{row['image_sha256']}"),
        "selection_uses_answer": False,
    }


def _factorial(rows: list[dict[str, Any]], k_filter: int | None = None) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        for ratio in RESOLUTION_RATIOS:
            for k_visual in VISION_K:
                if k_filter is not None and k_visual != k_filter:
                    continue
                result.append({
                    **row,
                    "request_id": factorial_request_id(
                        str(row["base_request_id"]), ratio, k_visual
                    ),
                    "resolution_ratio_target": ratio,
                    "resolution_label": resolution_label(ratio),
                    "k_visual": k_visual,
                    "k_text": 8,
                    "k_decode": 8,
                })
    return result


def _manifest(requests: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return {
        "schema_version": "mllm_resolution_expert_tradeoff.v1",
        "name": name,
        "selection": "IMAGE_HASH_DISJOINT_STABLE_HASH_NO_ANSWER",
        "ocr_scope": "MMSTAR_TEXT_SENSITIVE_PROXY_NOT_OCRBENCH",
        "requests": requests,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_foundation_bundle()
    args.output.mkdir(parents=True, exist_ok=True)

    discovery: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    counts = {}
    used_hashes: set[str] = set()
    for stratum, candidates in _strata().items():
        if len(candidates) < 96:
            raise RuntimeError(f"insufficient {stratum}: {len(candidates)}")
        selected = []
        for row in candidates:
            image_hash = str(row["image_sha256"])
            if image_hash in used_hashes:
                continue
            used_hashes.add(image_hash)
            selected.append(row)
            if len(selected) == 96:
                break
        if len(selected) != 96:
            raise RuntimeError(f"cross-stratum image shortage {stratum}: {len(selected)}")
        discovery.extend(_annotate(row, stratum, "discovery") for row in selected[:32])
        heldout.extend(_annotate(row, stratum, "heldout") for row in selected[32:96])
        counts[stratum] = {"discovery": 32, "heldout": 64}

    atomic_json(args.output / "discovery96.json", _manifest(discovery, "discovery96"))
    atomic_json(args.output / "heldout192.json", _manifest(heldout, "heldout192"))
    atomic_json(
        args.output / "discovery96_factorial864.json",
        _manifest(_factorial(discovery), "discovery96_factorial864"),
    )
    for k_visual in VISION_K:
        atomic_json(
            args.output / f"discovery96_k{k_visual}.json",
            _manifest(_factorial(discovery, k_visual), f"discovery96_k{k_visual}"),
        )
    atomic_json(args.output / "split_audit.json", {
        "counts": counts,
        "discovery": len(discovery),
        "heldout": len(heldout),
        "unique_images": len(used_hashes),
        "intersection": sorted(
            {row["image_sha256"] for row in discovery}
            & {row["image_sha256"] for row in heldout}
        ),
        "answer_used_for_selection": False,
        "part1_public_eval_exposure": "ALL_SOURCE_BENCHMARK_ROWS_PREVIOUSLY_EVALUATED; NOT_PART2_TUNED",
    })


if __name__ == "__main__":
    main()
