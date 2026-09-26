"""Audit the frozen 1-GPU/2-GPU semantic and target-EP acceptance run.

This intentionally keeps host execution and virtual target latency separate:
the former is an actual BF16 full-model generation result, while the latter is
the same frozen replay/cost model evaluated with a different host annotation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from poc_mllm_ep_foundation.evaluation.metrics import (
    chartqa_relaxed_correct,
    extract_mcq_letter,
)
from poc_mllm_ep_foundation.io import atomic_json


def _read_jsonl(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    rows: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        request_id = str(row["request_id"])
        if request_id in rows:
            duplicates.append(request_id)
        rows[request_id] = row
    return rows, sorted(set(duplicates))


def _parsed(row: dict[str, Any]) -> str:
    prediction = str(row.get("output_text", "")).strip()
    if row["task"] == "mmstar":
        return str(extract_mcq_letter(prediction) or "")
    return prediction


def _correct(row: dict[str, Any]) -> bool:
    prediction = str(row.get("output_text", "")).strip()
    if row["task"] == "mmstar":
        return extract_mcq_letter(prediction) == str(row["answer"])
    return chartqa_relaxed_correct(prediction, str(row["answer"]))


def _first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if left == right else min(len(left), len(right))


def _prediction_invariance(host1: Path, host2: Path) -> dict[str, Any]:
    left = json.loads(host1.read_text(encoding="utf-8"))
    right = json.loads(host2.read_text(encoding="utf-8"))
    fields = ("target_ep", "target_topology", "evidence", "in_domain_rows",
              "total_rows", "rows", "request_routed_ms", "ep8_warning")
    differing = [name for name in fields if left.get(name) != right.get(name)]
    return {
        "identical_target_prediction": not differing,
        "differing_fields": differing,
        "host1_annotation": left.get("host_physical_gpus"),
        "host2_annotation": right.get("host_physical_gpus"),
        "evidence": left.get("evidence"),
        "in_domain_rows": left.get("in_domain_rows"),
        "total_rows": left.get("total_rows"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--host1-results", type=Path, required=True)
    parser.add_argument("--host2-results", type=Path, required=True)
    parser.add_argument("--host1-prediction-pattern", required=True)
    parser.add_argument("--host2-prediction-pattern", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = {str(row["request_id"]): row for row in manifest["requests"]}
    host1, duplicate1 = _read_jsonl(args.host1_results)
    host2, duplicate2 = _read_jsonl(args.host2_results)
    common = sorted(set(expected) & set(host1) & set(host2))

    exact = []
    token_exact = []
    parsed_equal = []
    metadata_equal = []
    deltas = []
    divergence = []
    task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request_id in common:
        left, right = host1[request_id], host2[request_id]
        exact.append(left["output_text"] == right["output_text"])
        token_exact.append(left["output_token_ids"] == right["output_token_ids"])
        left_parsed, right_parsed = _parsed(left), _parsed(right)
        if left["task"] == "chartqa":
            parsed_equal.append(
                chartqa_relaxed_correct(left_parsed, right_parsed)
                and chartqa_relaxed_correct(right_parsed, left_parsed)
            )
        else:
            parsed_equal.append(left_parsed == right_parsed)
        metadata_equal.append(all(
            left.get(key) == right.get(key)
            for key in ("input_ids_sha256", "processed_pixel_values_sha256",
                        "image_grid_thw", "prompt_token_count", "vision_token_count")
        ))
        left_correct, right_correct = _correct(left), _correct(right)
        deltas.append(float(right_correct) - float(left_correct))
        task_rows[str(left["task"])].append({
            "host1_correct": left_correct, "host2_correct": right_correct,
            "parsed_equal": parsed_equal[-1], "exact": exact[-1],
        })
        where = _first_divergence(left["output_token_ids"], right["output_token_ids"])
        if where is not None:
            divergence.append({
                "request_id": request_id, "task": left["task"],
                "first_token_divergence": where,
                "host1_output_token_count": len(left["output_token_ids"]),
                "host2_output_token_count": len(right["output_token_ids"]),
                "host1_parsed": left_parsed, "host2_parsed": right_parsed,
            })

    clusters: dict[str, list[float]] = defaultdict(list)
    for request_id, delta in zip(common, deltas):
        clusters[str(expected[request_id]["image_group"])].append(delta)
    cluster_delta = np.asarray([np.mean(value) for value in clusters.values()], dtype=float)
    rng = np.random.default_rng(args.seed)
    samples = cluster_delta[
        rng.integers(0, len(cluster_delta), size=(args.bootstrap, len(cluster_delta)))
    ].mean(1)

    by_task = {}
    for task, rows in sorted(task_rows.items()):
        by_task[task] = {
            "n": len(rows),
            "host1_accuracy": float(np.mean([row["host1_correct"] for row in rows])),
            "host2_accuracy": float(np.mean([row["host2_correct"] for row in rows])),
            "host2_minus_host1_pp": float(np.mean([
                float(row["host2_correct"]) - float(row["host1_correct"]) for row in rows
            ]) * 100),
            "parsed_answer_agreement": float(np.mean([row["parsed_equal"] for row in rows])),
            "exact_output_agreement": float(np.mean([row["exact"] for row in rows])),
        }

    prediction = {
        f"target_ep{ep}": _prediction_invariance(
            Path(args.host1_prediction_pattern.format(ep=ep)),
            Path(args.host2_prediction_pattern.format(ep=ep)),
        )
        for ep in (2, 4, 8)
    }
    complete = (
        len(host1) == len(expected) == len(host2)
        and not duplicate1 and not duplicate2
        and not (set(expected) - set(host1)) and not (set(expected) - set(host2))
    )
    payload = {
        "schema_version": "mllm_ep_low_resource_acceptance.v1",
        "manifest": str(args.manifest.resolve()),
        "expected_n": len(expected), "host1_n": len(host1), "host2_n": len(host2),
        "common_n": len(common), "duplicate_host1": duplicate1,
        "duplicate_host2": duplicate2,
        "missing_host1": sorted(set(expected) - set(host1)),
        "missing_host2": sorted(set(expected) - set(host2)),
        "input_metadata_exact_fraction": float(np.mean(metadata_equal)),
        "exact_output_agreement": float(np.mean(exact)),
        "exact_token_sequence_agreement": float(np.mean(token_exact)),
        "parsed_answer_agreement": float(np.mean(parsed_equal)),
        "host2_minus_host1_accuracy_pp": float(np.mean(deltas) * 100),
        "host2_minus_host1_cluster_bootstrap_95ci_pp": [
            float(np.quantile(samples, 0.025) * 100),
            float(np.quantile(samples, 0.975) * 100),
        ],
        "by_task": by_task,
        "divergent_outputs": divergence,
        "virtual_prediction_host_invariance": prediction,
        "semantic_execution_complete": complete,
        "semantic_quality_status": (
            "LOW_RESOURCE_SEMANTIC_READY_NUMERIC_DRIFT_OBSERVED" if complete
            else "INCOMPLETE"
        ),
        "target_prediction_status": (
            "HOST_INVARIANT" if all(row["identical_target_prediction"]
                                    for row in prediction.values()) else "FAILED"
        ),
        "interpretation": (
            "TP1 and TP2 BF16 reduction order is not bit exact. Task accuracy and parsed-answer "
            "agreement are reported separately; this is not a proof of universal semantic identity."
        ),
    }
    atomic_json(args.output, payload)
    print(json.dumps({key: value for key, value in payload.items()
                      if key != "divergent_outputs"}, indent=2))


if __name__ == "__main__":
    main()
