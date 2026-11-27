"""Reduce the InternVL3 native representation controls without relabeling their scope."""

import argparse
import json
from pathlib import Path

from .workload import file_digest, write_frozen


def service(row):
    language = row["language_calls"]
    decode_steps = len(row["output_token_ids"]) - 1
    prefill_steps = len(language) - decode_steps
    if prefill_steps < 1:
        breakdown = {"language_stage_evidence": "NOT_MEASURED", "instrumented_language_calls": len(language)}
    else:
        prefill = language[:prefill_steps]
        decode = language[prefill_steps:]
        breakdown = {
            "language_stage_evidence": "MEASURED_PHYSICAL",
            "prefill_calls": prefill_steps,
            "prefill_rows": sum(item["position_rows"] for item in prefill),
            "decode_rows": sum(item["position_rows"] for item in decode),
            "prefill_gpu_ms": sum(item["gpu_event_ms"] for item in prefill),
            "decode_gpu_ms": sum(item["gpu_event_ms"] for item in decode),
        }
    return {
        "prompt_rows": row["model_prompt_tokens"], "stock_apc_rows": row["num_cached_tokens"],
        **breakdown,
        "pixel_embedding_gpu_ms": sum(item["gpu_event_ms"] for item in row["vision_calls"] if item["input"] == "pixels"),
        "injected_embedding_gpu_ms": sum(item["gpu_event_ms"] for item in row["vision_calls"] if item["input"] == "embedding"),
        "native_pixel_embedding_calls": sum(item["input"] == "pixels" for item in row["vision_calls"]),
        "wall_generation_ms_with_hooks": row["wall_ms"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ragged-failure-log", type=Path)
    args = parser.parse_args()
    raw = json.loads(args.raw.read_bytes())
    if raw["evidence"] != "MEASURED_PHYSICAL" or not raw["embedding_round_trip"]:
        parser.error("A completed physical InternVL control is required")
    cases = []
    for row in raw["embedding_round_trip"]:
        match = {key: value for key, value in row.items() if key.endswith("_match")}
        cases.append({
            "request_id": row["request_id"], "images": len(row["representations"]),
            "native_visual_rows": sum(rep["shape"][0] for rep in row["representations"]),
            "lm_ready_embedding_bytes": row["embedding_bytes"],
            "full_prompt_logical_kv_bytes": row["full_prompt_logical_kv_bytes"],
            "KV_to_embedding_byte_ratio": row["full_prompt_logical_kv_bytes"] / row["embedding_bytes"],
            "parity": match, "injection_status": row["injection_status"],
            **{key: service(row[key]) for key in ("native", "native_repeated", "injected") if key in row},
        })
    apc = raw.get("native_document_apc")
    if apc:
        apc = {**{key: value for key, value in apc.items() if key not in ("cold", "warm")},
               **{key: service(apc[key]) for key in ("cold", "warm")}}
    failure = None
    if args.ragged_failure_log:
        lines = args.ragged_failure_log.read_text().splitlines()
        failure = {"evidence": "DIAGNOSTIC", "log_sha256": file_digest(args.ragged_failure_log),
                   "errors": [line for line in lines if "ValueError: data contains inconsistent shapes" in line]}
        if not failure["errors"]:
            parser.error("The supplied log lacks the observed ragged TensorSchema failure")
    write_frozen(args.out, {
        "raw_sha256": file_digest(args.raw),
        "evidence": "MEASURED_PHYSICAL",
        "checkpoint": raw["model"], "revision": raw["revision"], "runtime": raw["runtime"],
        "resident_weights": raw["resident_weights"], "geometry": raw["geometry"],
        "per_row_KV_bytes": raw["bf16_full_layer_kv_bytes_per_row"], "cases": cases,
        "greedy_exact_native_AA_cases": sum(case["parity"]["native_AA_greedy_match"] for case in cases),
        "injected_cases": sum(case["injection_status"] == "EXECUTED" for case in cases),
        "greedy_exact_embedding_round_trip_cases": sum(case["parity"].get("injected_greedy_match", False) for case in cases),
        "native_document_apc": apc,
        "ragged_stock_failure": failure,
        "scope": raw["scope"],
        "interpretation": "When LM hooks observe calls, the prefill/decode split counts the final N-1 single-token "
                          "calls as decode for N generated tokens in the isolated serial request. The empty V2 "
                          "language hook is reported separately from observed native APC counters and wall time. "
                          "Embedding call enclosures exclude H2D transport outside the method. The raw file records "
                          "all observed calls and generated token sequences.",
        "not_run": ["InternVL pinned peer tier", "InternVL EPD online", "InternVL natural-EOS task-quality noninferiority"],
    })


if __name__ == "__main__":
    main()
