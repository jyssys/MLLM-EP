from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from poc_mllm_mixed_domain_online_calibration.contract import (
    DOMAINS,
    REASONING_PROMPT,
    WARMUP_RATIOS,
    atomic_json,
    sha256_file,
    stable_key,
    warmup_count,
)

SEED = "mixed-domain-calibration-v1-20260928"
ROOT = Path("/home/esjung/MLLM-EP-long-reasoning-signal")
FOUNDATION = ROOT / "poc_mllm_ep_foundation/validation_runs/20260924_completion_v1/data"
LONG = ROOT / "poc_mllm_long_reasoning_decode_vision_signal/artifacts/screening_manifest.json"


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _existing_image(row: dict[str, Any], *keys: str) -> Path:
    for key in keys:
        value = row.get(key)
        if value and Path(value).is_file():
            return Path(value)
    raise FileNotFoundError({key: row.get(key) for key in keys})


def _dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        return image.size


def _wrap_question(raw: str) -> str:
    return f"{REASONING_PROMPT}\n\nProblem:\n{raw.strip()}"


def _canonical(
    row: dict[str, Any], *, domain: str, source_dataset: str, source_id: str,
    image_path: Path, raw_question: str, answer: Any,
) -> dict[str, Any]:
    image_hash = row.get("image_sha256") or row.get("original_image_sha256")
    actual_hash = sha256_file(image_path)
    if image_hash and image_hash != actual_hash:
        raise ValueError(f"image hash mismatch: {image_path}")
    width, height = _dimensions(image_path)
    request_id = f"{domain}::{source_dataset}::{source_id}"
    return {
        "request_id": request_id,
        "domain": domain,
        "source_dataset": source_dataset,
        "source_id": str(source_id),
        "image_path": str(image_path),
        "image_sha256": actual_hash,
        "image_width": width,
        "image_height": height,
        "image_area": width * height,
        "raw_question": raw_question,
        "question": _wrap_question(raw_question),
        "answer": answer,
        "selection_key": stable_key(SEED, domain, actual_hash, source_id),
        "selection_uses_routing": False,
        "selection_uses_answer": False,
    }


def _math_candidates() -> list[dict[str, Any]]:
    payload = _load(LONG)
    output = []
    for row in payload["anchors"]:
        path = _existing_image(row, "original_image_path")
        output.append(_canonical(
            row,
            domain=DOMAINS[0],
            source_dataset=str(row.get("dataset", "visual_math")),
            source_id=str(row.get("dataset_sample_id", row.get("pair_id"))),
            image_path=path,
            raw_question=str(row.get("question_raw", row["question"])),
            answer=row.get("answer"),
        ))
    return output


def _chart_candidates() -> list[dict[str, Any]]:
    requests = _load(FOUNDATION / "chartqa_full_manifest.json")["requests"]
    return [
        _canonical(
            row,
            domain=DOMAINS[1],
            source_dataset="chartqa",
            source_id=str(row["source_row"]),
            image_path=_existing_image(row, "image_path"),
            raw_question=str(row.get("raw_question", row["question"])),
            answer=row.get("answer"),
        )
        for row in requests
    ]


def _mmstar_candidates() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requests = _load(FOUNDATION / "mmstar_full_manifest.json")["requests"]
    structured_l2 = {"diagram reasoning", "code & sequence reasoning"}
    structured, general = [], []
    for row in requests:
        target = structured if row.get("l2_category") in structured_l2 else general
        domain = DOMAINS[2] if target is structured else DOMAINS[3]
        if target is general and row.get("category") in {"math", "logical reasoning"}:
            continue
        target.append(_canonical(
            row,
            domain=domain,
            source_dataset="mmstar",
            source_id=str(row["source_row"]),
            image_path=_existing_image(row, "image_path"),
            raw_question=str(row.get("raw_question", row["question"])),
            answer=row.get("answer"),
        ))
    return structured, general


def _freeze_domain(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    unique = {}
    for row in rows:
        unique.setdefault(row["image_sha256"], row)
    selected = sorted(unique.values(), key=lambda row: row["selection_key"])[:count]
    if len(selected) < count:
        raise RuntimeError(f"only {len(selected)} unique images available, need {count}")
    result = []
    for index, row in enumerate(selected):
        item = copy.deepcopy(row)
        item["domain_index"] = index
        item["analysis_split"] = "discovery" if index < count // 2 else "heldout"
        result.append(item)
    return result


def _iid(domains: dict[str, list[dict[str, Any]]]) -> list[str]:
    output = []
    for index in range(max(map(len, domains.values()))):
        order = sorted(DOMAINS, key=lambda domain: stable_key(SEED, "iid", index, domain))
        for domain in order:
            if index < len(domains[domain]):
                output.append(domains[domain][index]["request_id"])
    return output


def _domain_shift(domains: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    heldout = {domain: [row for row in rows if row["analysis_split"] == "heldout"] for domain, rows in domains.items()}
    phases = (
        ("math_heavy", (DOMAINS[0], DOMAINS[0], DOMAINS[1], DOMAINS[3])),
        ("chart_structured_heavy", (DOMAINS[1], DOMAINS[2], DOMAINS[1], DOMAINS[2])),
        ("general_heavy", (DOMAINS[3], DOMAINS[3], DOMAINS[2], DOMAINS[0])),
    )
    cursor = Counter()
    output = []
    used = set()
    for phase, pattern in phases:
        for domain in pattern * 4:
            rows = heldout[domain]
            while cursor[domain] < len(rows) and rows[cursor[domain]]["request_id"] in used:
                cursor[domain] += 1
            if cursor[domain] >= len(rows):
                continue
            request = rows[cursor[domain]]["request_id"]
            cursor[domain] += 1
            used.add(request)
            output.append({"phase": phase, "request_id": request})
    for domain in DOMAINS:
        for row in heldout[domain]:
            if row["request_id"] not in used:
                output.append({"phase": "tail_balanced", "request_id": row["request_id"]})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-domain", type=int, default=40)
    args = parser.parse_args()

    structured, general = _mmstar_candidates()
    candidates = {
        DOMAINS[0]: _math_candidates(),
        DOMAINS[1]: _chart_candidates(),
        DOMAINS[2]: structured,
        DOMAINS[3]: general,
    }
    domains = {domain: _freeze_domain(rows, args.per_domain) for domain, rows in candidates.items()}
    hashes = [row["image_sha256"] for rows in domains.values() for row in rows]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("cross-domain image-hash overlap")

    args.output.mkdir(parents=True, exist_ok=True)
    for domain, rows in domains.items():
        atomic_json(args.output / f"domain_{domain}.json", {
            "schema_version": 1,
            "domain": domain,
            "request_count": len(rows),
            "requests": rows,
        })
    all_rows = [row for domain in DOMAINS for row in domains[domain]]
    iid = _iid(domains)
    shift = _domain_shift(domains)
    warmups = {}
    for ratio in WARMUP_RATIOS:
        key = f"{ratio:.2f}"
        warmups[key] = {}
        for domain in DOMAINS:
            count = warmup_count(ratio, len(domains[domain]))
            selected = domains[domain][:count]
            warmups[key][domain] = {
                "nominal_ratio": ratio,
                "count": count,
                "realized_ratio": count / len(domains[domain]),
                "request_ids": [row["request_id"] for row in selected],
            }
    atomic_json(args.output / "frozen_domain_pool.json", {
        "schema_version": 1,
        "seed": SEED,
        "selection_frozen_before_routing": True,
        "routing_oracle_used": False,
        "answer_used": False,
        "domains": list(DOMAINS),
        "request_count": len(all_rows),
        "reasoning_prompt": REASONING_PROMPT,
        "requests": all_rows,
        "source_deviation": (
            "No local DocVQA/OCRBench asset was available; D3 is the semantically "
            "distinct MMStar structured-diagram/layout regime, not an OCR benchmark."
        ),
    })
    atomic_json(args.output / "frozen_iid_stream.json", {
        "schema_version": 1,
        "seed": SEED,
        "request_ids": iid,
    })
    atomic_json(args.output / "frozen_domain_shift_stream.json", {
        "schema_version": 1,
        "seed": SEED,
        "events": shift,
    })
    atomic_json(args.output / "warmup_indices.json", {
        "schema_version": 1,
        "formula": "K_d(0)=0; K_d(r>0)=max(1,ceil(r*N_d))",
        "ratios": warmups,
    })
    atomic_json(args.output / "cohort_audit.json", {
        "schema_version": 1,
        "counts": {domain: len(rows) for domain, rows in domains.items()},
        "candidate_counts": {domain: len(rows) for domain, rows in candidates.items()},
        "image_hash_unique": len(hashes) == len(set(hashes)),
        "split_counts": dict(Counter(row["analysis_split"] for row in all_rows)),
    })


if __name__ == "__main__":
    main()
