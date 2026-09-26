from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from poc_mllm_ep_foundation.evaluation.metrics import (
    chartqa_relaxed_correct,
    extract_mcq_letter,
)
from poc_mllm_ep_foundation.io import atomic_json


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parsed(row: dict[str, Any]) -> str:
    output = str(row.get("output_text", "")).strip()
    if row["task"] == "mmstar":
        return str(extract_mcq_letter(output))
    return output


def _correct(row: dict[str, Any]) -> bool:
    output = str(row.get("output_text", "")).strip()
    answer = str(row.get("answer", "")).strip()
    if row["task"] == "mmstar":
        return extract_mcq_letter(output) == answer.upper()
    return chartqa_relaxed_correct(output, answer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part1-ep4", type=Path, required=True)
    parser.add_argument("--part2-k8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = {str(row["request_id"]): row for row in _rows(args.part1_ep4)}
    candidates = [
        row for row in _rows(args.part2_k8)
        if row.get("resolution_label") == "R0" and int(row["k_visual"]) == 8
    ]
    comparisons = []
    for row in candidates:
        base_id = str(row["base_request_id"])
        source = reference[base_id]
        comparisons.append({
            "request_id": base_id,
            "input_ids_equal": row["input_ids_sha256"] == source["input_ids_sha256"],
            "processed_pixels_equal": (
                row["processed_pixel_values_sha256"]
                == source["processed_pixel_values_sha256"]
            ),
            "image_grid_equal": row["image_grid_thw"] == source["image_grid_thw"],
            "output_tokens_exact": row["output_token_ids"] == source["output_token_ids"],
            "parsed_answer_equal": _parsed(row) == _parsed(source),
            "score_outcome_equal": _correct(row) == _correct(source),
        })
    keys = [
        "input_ids_equal", "processed_pixels_equal", "image_grid_equal",
        "output_tokens_exact", "parsed_answer_equal", "score_outcome_equal",
    ]
    result = {
        "schema_version": "mllm_resolution_expert_tradeoff.noop_parity.v1",
        "scope": "R0_K8_PART2_ADAPTER_VS_PART1_TRUE_EP4_OFFICIAL_LANE",
        "n": len(comparisons),
        "counts": {key: sum(bool(row[key]) for row in comparisons) for key in keys},
        "fractions": {
            key: sum(bool(row[key]) for row in comparisons) / len(comparisons)
            for key in keys
        },
        "interpretation": (
            "Sampling sequence identity is diagnostic; canonical input, parsed answer, "
            "and score-outcome identity are the no-op semantic checks."
        ),
        "comparisons": comparisons,
    }
    atomic_json(args.output, result)
    print(json.dumps({"n": result["n"], "counts": result["counts"]}, indent=2))


if __name__ == "__main__":
    main()
