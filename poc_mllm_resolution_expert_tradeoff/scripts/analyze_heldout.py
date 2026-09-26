from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from poc_mllm_ep_foundation.io import atomic_json
from poc_mllm_resolution_expert_tradeoff.scripts.analyze_grid import (
    _base_id,
    _config,
    _read_jsonl,
    _score,
)


def _bootstrap(delta: np.ndarray, repeats: int = 10_000) -> list[float]:
    rng = np.random.default_rng(3407)
    values = delta[rng.integers(0, len(delta), size=(repeats, len(delta)))].mean(1)
    return [float(np.quantile(values, 0.025) * 100),
            float(np.quantile(values, 0.975) * 100)]


def _optional_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def _optional_quantile(values: list[float | None], quantile: float) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return float(np.quantile(finite, quantile)) if finite else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--frozen-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(args.frozen_policy.read_text(encoding="utf-8"))
    rows = _read_jsonl(args.results)
    for row in rows:
        row["config"] = _config(row)
        row["base_request_id"] = _base_id(row)
        row["parsed_answer"], row["correct"] = _score(row)
    by_config = {}
    for row in rows:
        by_config.setdefault(row["config"], {})[row["base_request_id"]] = row
    baseline = by_config.get("R0_K8")
    if baseline is None:
        raise RuntimeError("heldout baseline R0_K8 missing")
    expected = {str(row["base_request_id"]) for row in baseline.values()}
    results = {}
    for config, values in sorted(by_config.items()):
        if set(values) != expected:
            raise RuntimeError(f"heldout IDs differ for {config}")
        ids = sorted(expected)
        delta = np.asarray([
            float(values[key]["correct"]) - float(baseline[key]["correct"])
            for key in ids
        ])
        ci = _bootstrap(delta)
        tasks = sorted({str(values[key]["task"]) for key in ids})
        strata = sorted({str(values[key]["part2_stratum"]) for key in ids})
        results[config] = {
            "n": len(ids),
            "accuracy": float(np.mean([values[key]["correct"] for key in ids])),
            "baseline_accuracy": float(np.mean([baseline[key]["correct"] for key in ids])),
            "baseline_delta_pp": float(delta.mean() * 100),
            "cluster_bootstrap_95ci_pp": ci,
            "noninferiority_margin_pp": -1.0,
            "noninferiority_proven": bool(ci[0] > -1.0),
            "baseline_correct_to_wrong": int(sum(
                baseline[key]["correct"] and not values[key]["correct"] for key in ids
            )),
            "baseline_wrong_to_correct": int(sum(
                not baseline[key]["correct"] and values[key]["correct"] for key in ids
            )),
            "mean_ttft_ms": _optional_mean([
                values[key].get("ttft_ms") for key in ids
            ]),
            "mean_fresh_image_ttft_ms": _optional_mean([
                None if values[key].get("processor_ms") is None
                or values[key].get("ttft_ms") is None
                else float(values[key]["processor_ms"]) + float(values[key]["ttft_ms"])
                for key in ids
            ]),
            "mean_natural_engine_ms": _optional_mean([
                values[key].get("request_engine_ms") for key in ids
            ]),
            "mean_fresh_image_natural_ms": _optional_mean([
                None if values[key].get("processor_ms") is None
                or values[key].get("request_engine_ms") is None
                else float(values[key]["processor_ms"])
                + float(values[key]["request_engine_ms"])
                for key in ids
            ]),
            "mean_output_tokens": _optional_mean([
                values[key].get("output_token_count") for key in ids
            ]),
            "p50_output_tokens": _optional_quantile([
                values[key].get("output_token_count") for key in ids
            ], 0.5),
            "p90_output_tokens": _optional_quantile([
                values[key].get("output_token_count") for key in ids
            ], 0.9),
            "mean_output_token_delta_vs_baseline": _optional_mean([
                float(values[key]["output_token_count"])
                - float(baseline[key]["output_token_count"])
                for key in ids
                if values[key].get("output_token_count") is not None
                and baseline[key].get("output_token_count") is not None
            ]),
            "generation_timeout_count": int(sum(
                values[key].get("finish_reason") == "generation_timeout"
                for key in ids
            )),
            "normal_termination_fraction": float(np.mean([
                values[key].get("finish_reason") == "stop" for key in ids
            ])),
            "finish_reason_counts": {
                reason: int(sum(values[key].get("finish_reason") == reason for key in ids))
                for reason in sorted({
                    str(values[key].get("finish_reason")) for key in ids
                })
            },
            "task_accuracy": {
                task: float(np.mean([
                    values[key]["correct"] for key in ids
                    if values[key]["task"] == task
                ]))
                for task in tasks
            },
            "stratum_accuracy": {
                stratum: float(np.mean([
                    values[key]["correct"] for key in ids
                    if values[key]["part2_stratum"] == stratum
                ]))
                for stratum in strata
            },
            "regression_ids": [
                key for key in ids
                if baseline[key]["correct"] and not values[key]["correct"]
            ],
            "improvement_ids": [
                key for key in ids
                if not baseline[key]["correct"] and values[key]["correct"]
            ],
        }
    atomic_json(args.output, {
        "schema_version": "mllm_resolution_expert_tradeoff.heldout.v1",
        "split": "heldout192_image_hash_disjoint_frozen_once",
        "frozen_policy": str(args.frozen_policy.resolve()),
        "policy_candidates": policy["candidates"],
        "results": results,
        "evidence": "MEASURED_TRUE_EP4_OFFICIAL_REPRODUCTION_PUBLIC_PROTOCOL_PARTIAL",
    })
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
