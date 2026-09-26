"""Frozen public benchmark manifests and local, judge-free scorers.

The builders deliberately run outside the vLLM environment: ``pyarrow`` is
only required while materialising the public data.  Runtime manifests contain
ordinary image paths and JSON metadata so inference never depends on a
datasets package or an online service.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from ..io import atomic_json, sha256
from .metrics import chartqa_relaxed_correct, extract_mcq_letter


MMSTAR_REVISION = "bc98d668301da7b14f648724866e57302778ab27"
MMSTAR_PARQUET_SHA256 = "29afd74b0134cfab083a8909b5358577ab18fd41c1e612031577cfb3635531c2"
CHARTQA_REVISION = "9e63b7df1592a1c2158e735cc1725454aef0d6d9"
LMMS_EVAL_REVISION = "42296f8fd2f0929d88a6ac4f7a1f2d784537f8d8"
QWEN3_VL_REPO_REVISION = "96588727e44c78b25ba03ea03b8e12f7e64fd0da"
VLMEVALKIT_REVISION = "6f0370704932cec0d9aa1e73f01954cd786c57ea"


def _image_hash(path: Path) -> str:
    return sha256(path)


def build_mmstar_manifest(parquet: Path, output: Path, image_dir: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    if sha256(parquet) != MMSTAR_PARQUET_SHA256:
        raise ValueError("MMStar parquet does not match the frozen public revision")
    table = pq.read_table(parquet)
    if table.num_rows != 1500:
        raise ValueError(f"expected full MMStar val=1500, got {table.num_rows}")
    image_dir.mkdir(parents=True, exist_ok=True)
    requests: list[dict[str, Any]] = []
    for row in table.to_pylist():
        index = int(row["index"])
        image_path = image_dir / f"{index:04d}.jpg"
        if not image_path.exists():
            with Image.open(io.BytesIO(row["image"])) as raw:
                raw.convert("RGB").save(image_path, format="JPEG", quality=100, subsampling=0)
        question = str(row["question"]).strip().replace(" Please answer yes or no.", "")
        requests.append({
            "request_id": f"mmstar_{index:04d}",
            "task": "mmstar",
            "split": "val",
            "source_row": index,
            "image_path": str(image_path.resolve()),
            "image_sha256": _image_hash(image_path),
            "image_group": f"mmstar:{index}",
            "question": f"Question: {question}Answer with the option letter only.",
            "raw_question": question,
            "answer": str(row["answer"]).strip().upper(),
            "category": row["category"],
            "l2_category": row["l2_category"],
        })
    payload = {
        "schema_version": "foundation.public_manifest.v2",
        "task": "mmstar",
        "protocol_status": "PUBLIC_PROTOCOL_PARTIAL",
        "protocol_note": (
            "Full Lin-Chen/MMStar val split and lmms-eval mmstar_qwen prompt/scorer; "
            "the Qwen technical-report private execution wrapper is not fully published."
        ),
        "source": {
            "repo": "Lin-Chen/MMStar",
            "revision": MMSTAR_REVISION,
            "file": "mmstar.parquet",
            "sha256": MMSTAR_PARQUET_SHA256,
            "rows": 1500,
        },
        "evaluation_source": {
            "lmms_eval_revision": LMMS_EVAL_REVISION,
            "qwen3_vl_repo_revision": QWEN3_VL_REPO_REVISION,
            "vlmevalkit_revision": VLMEVALKIT_REVISION,
            "prompt_contract": (
                "lmms-eval mmstar _default_template_yaml qwen3_vl: "
                "pre_prompt='Question: ', post_prompt='Answer with the option letter only.'"
            ),
        },
        "requests": requests,
    }
    atomic_json(output, payload)
    return payload


def build_chartqa_manifest(source_jsonl: Path, output: Path) -> dict[str, Any]:
    rows = []
    for line in source_jsonl.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("dataset") == "chartqa" and row.get("split") == "heldout":
            rows.append(row)
    rows.sort(key=lambda row: int(row["source_row"]))
    if len(rows) != 2500 or [int(row["source_row"]) for row in rows] != list(range(2500)):
        raise ValueError("ChartQA full public test split is incomplete")
    requests: list[dict[str, Any]] = []
    for row in rows:
        path = Path(row["images"][0])
        if not path.exists():
            raise FileNotFoundError(path)
        index = int(row["source_row"])
        requests.append({
            "request_id": f"chartqa_{index:04d}",
            "task": "chartqa",
            "split": "test",
            "source_row": index,
            "image_path": str(path.resolve()),
            "image_sha256": _image_hash(path),
            "image_group": f"chartqa:{row['image_id']}",
            "question": f"{str(row['question']).strip()}\nAnswer the question with a single word.",
            "raw_question": row["question"],
            "answer": str(row["answers"][0]),
            "annotation_type": "human_test" if int(row["human_or_machine"]) == 0 else "augmented_test",
        })
    payload = {
        "schema_version": "foundation.public_manifest.v2",
        "task": "chartqa",
        "protocol_status": "PUBLIC_PROTOCOL_PARTIAL",
        "protocol_note": (
            "Full 2500-row public test split and lmms-eval relaxed scorer/default prompt; "
            "the Qwen technical-report wrapper details are not fully published."
        ),
        "source": {
            "repo": "lmms-lab/ChartQA",
            "revision": CHARTQA_REVISION,
            "rows": 2500,
            "annotation_counts": {"human_test": 1250, "augmented_test": 1250},
            "image_count": len({row["image_id"] for row in rows}),
            "materialized_from": str(source_jsonl.resolve()),
        },
        "evaluation_source": {
            "lmms_eval_revision": LMMS_EVAL_REVISION,
            "qwen3_vl_repo_revision": QWEN3_VL_REPO_REVISION,
            "vlmevalkit_revision": VLMEVALKIT_REVISION,
            "prompt_contract": (
                "lmms-eval chartqa default: post_prompt='Answer the question with a single word.'"
            ),
        },
        "requests": requests,
    }
    atomic_json(output, payload)
    return payload


def score_public_results(manifest_path: Path, results_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {str(row["request_id"]): row for row in manifest["requests"]}
    observed: dict[str, dict[str, Any]] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            observed[str(row["request_id"])] = row
    scored = []
    for request_id, item in expected.items():
        result = observed.get(request_id)
        if result is None:
            continue
        prediction = str(result.get("output_text", "")).strip()
        if manifest["task"] == "mmstar":
            parsed = extract_mcq_letter(prediction)
            correct = parsed == item["answer"]
        else:
            parsed = prediction
            correct = chartqa_relaxed_correct(parsed, item["answer"])
        scored.append({
            "request_id": request_id,
            "prediction": prediction,
            "parsed_prediction": parsed,
            "answer": item["answer"],
            "correct": bool(correct),
            "image_group": item["image_group"],
            "l2_category": item.get("l2_category"),
            "annotation_type": item.get("annotation_type"),
        })
    aggregate: dict[str, Any] = {
        "task": manifest["task"],
        "expected_n": len(expected),
        "observed_n": len(observed),
        "scored_n": len(scored),
        "missing_n": len(expected) - len(scored),
        "extra_n": len(set(observed) - set(expected)),
        "micro_accuracy": sum(row["correct"] for row in scored) / len(scored) if scored else None,
        "protocol_status": manifest["protocol_status"],
        "manifest_sha256": sha256(manifest_path),
        "results_sha256": sha256(results_path),
        "per_sample": scored,
    }
    if manifest["task"] == "mmstar" and scored:
        groups: dict[str, list[bool]] = defaultdict(list)
        for row in scored:
            groups[str(row["l2_category"])].append(row["correct"])
        aggregate["l2_accuracy"] = {
            key: sum(values) / len(values) for key, values in sorted(groups.items())
        }
        aggregate["official_macro_accuracy"] = sum(aggregate["l2_accuracy"].values()) / len(groups)
    elif scored:
        groups = defaultdict(list)
        for row in scored:
            groups[str(row["annotation_type"])].append(row["correct"])
        aggregate["split_accuracy"] = {
            key: sum(values) / len(values) for key, values in sorted(groups.items())
        }
        aggregate["official_relaxed_accuracy"] = aggregate["micro_accuracy"]
    return aggregate
