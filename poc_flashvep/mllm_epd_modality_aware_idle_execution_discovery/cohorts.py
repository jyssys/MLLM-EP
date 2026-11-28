"""Freeze original Very-long and held-out short Natural EOS workload contracts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from random import Random
from statistics import median

from ..mllm_epd_prefill_heavy_modality_discovery.datasets import file_digest, freeze, safe_rows


ARRIVAL_SEED = 2026112809
COHORT_SEED = 2026112807
PRIMARY_QPS = 1.465
HOLDOUT_MIN_INPUT = 16384
HOLDOUT_MAX_INPUT = 32768 - 256
HOLDOUT_MAX_PREVIOUS_OUTPUT = 24
PRIMARY_INPUT = Path("dataset/natural_very_long_n24.json")
POPULATION = Path("dataset/natural_media_n400.json")
PRIOR_OUTPUTS = Path("native-population/raw/full_main400_eos_requests.jsonl")
BASELINE_TRACES = {
    24: Path("nixl-ready/configs/nixl_readiness_n24_clean_before_arrival.json"),
    64: Path("observer-long64/configs/long_primed_n64_clean_before_arrival.json"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def index(rows):
    records = {row["request_id"]: row for row in rows}
    require(len(records) == len(rows), "Repeated original request IDs")
    return records


def image_hashes(rows):
    return {image["SHA256"] for row in rows for image in row["images"]}


def documents(rows):
    return {(row["source"], row["document_id"]) for row in rows if row.get("document_id")}


def verify_assets(rows):
    checked, missing = safe_rows(rows, check_hashes=True)
    require(not missing and len(checked) == len(rows), "Missing original images or processor records")
    for row in rows:
        actual = hashlib.sha256(row["prompt"].encode()).hexdigest()
        require(actual == row["prompt_SHA256"], f"Original prompt mismatch in {row['request_id']}")


def reference_outputs(source, population):
    observed = {}
    with source.open("rb") as stream:
        for line in stream:
            row = json.loads(line)
            key = row["request_id"]
            require(key in population and key not in observed, f"Unexpected prior native result {key}")
            original = population[key]
            for name in ("prompt_SHA256", "source"):
                require(row[name] == original[name], f"Mismatched native result {name} for {key}")
            require(row["input_tokens_manifest"] == original["total_input_tokens"],
                    f"Mismatched native context length for {key}")
            observed[key] = {name: row[name] for name in ("output_tokens", "status", "finish_reason")}
    require(observed.keys() == population.keys(), "Prior native Main400 measurement does not cover full population")
    return observed


def heldout(primary, population, previous):
    primary_ids = index(primary)
    prompt_digests = {row["prompt_SHA256"] for row in primary}
    duplicate_images = image_hashes(primary)
    duplicate_documents = documents(primary)
    selected = []
    for row in population:
        key = row["request_id"]
        if key in primary_ids or row["prompt_SHA256"] in prompt_digests or not row["images"]:
            continue
        if not HOLDOUT_MIN_INPUT <= row["total_input_tokens"] <= HOLDOUT_MAX_INPUT:
            continue
        old = previous[key]
        if (old["status"] != "MEASURED_ONLINE" or old["finish_reason"] != "stop"
                or not 1 <= old["output_tokens"] <= HOLDOUT_MAX_PREVIOUS_OUTPUT):
            continue
        if image_hashes([row]) & duplicate_images or documents([row]) & duplicate_documents:
            continue
        selected.append(row)
    return sorted(selected, key=lambda r: hashlib.sha256(
        f"{COHORT_SEED}:{r['request_id']}".encode()).hexdigest())


def arrival(rows, size, qps):
    rng = Random(ARRIVAL_SEED)
    offsets, offset_seconds = [], 0.
    for i in range(size):
        if i:
            offset_seconds += rng.expovariate(qps)
        row = rows[i % len(rows)]
        suffix = f"--cycle{i // len(rows)}" if i >= len(rows) else ""
        offsets.append({"index": i, "request_id": row["request_id"] + suffix,
                        "offset_seconds": offset_seconds, "replica": 0})
    return {"seed": ARRIVAL_SEED, "QPS": qps, "serial": False, "trace": offsets,
            "definition": "Prospective open-loop Poisson. Client TTFT includes dispatch lateness."}


def describe(rows):
    return {
        "requests": len(rows),
        "original_documents_known": len(documents(rows)),
        "image_occurrences": sum(row["image_count"] for row in rows),
        "unique_image_byte_hashes": len(image_hashes(rows)),
        "median_input_tokens": median(row["total_input_tokens"] for row in rows),
        "median_images": median(row["image_count"] for row in rows),
        "median_visual_ratio": median(row["visual_tokens"] / row["total_input_tokens"] for row in rows),
        "sources": Counter(row["source"] for row in rows),
    }


def identifier(row):
    return {name: row[name] for name in (
        "request_id", "source", "source_revision", "prompt_SHA256", "total_input_tokens", "visual_tokens",
        "processor_revision", "token_metadata_SHA256",
    )} | {
        "ordered_image_SHA256": [item["SHA256"] for item in row["images"]],
    }


def without_asset_paths(row):
    return {key: value for key, value in row.items() if key not in {"token_metadata_file", "images"}} | {
        "images": [{key: value for key, value in image.items() if key != "path"} for image in row["images"]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="External workload directory")
    parser.add_argument("--receipts", type=Path, required=True, help="Media-free protocol and request receipts")
    args = parser.parse_args()
    root, out = args.prior_root.resolve(), args.out.resolve()
    primary_source = root / PRIMARY_INPUT
    primary_data = read(primary_source)
    primary = primary_data["requests"]
    population = read(root / POPULATION)["requests"]
    require(len(primary) == 24 and len(population) == 400, "Wrong original sealed cohort size")
    population_by_id = index(population)
    for row in primary:
        require(without_asset_paths(row) == without_asset_paths(population_by_id[row["request_id"]]),
                f"Main400 payload differs for {row['request_id']}")
    previous = reference_outputs(root / PRIOR_OUTPUTS, population_by_id)
    independent = heldout(primary, population, previous)
    require(len(independent) >= 24, "The blinded eligibility contract has fewer than 24 held-out requests")
    verify_assets(primary + independent)
    out.mkdir(parents=True, exist_ok=True)
    primary_path = out / "primary_very_long_n24.json"
    if not primary_path.exists():
        primary_path.write_bytes(primary_source.read_bytes())
    require(file_digest(primary_path) == file_digest(primary_source), "Primary manifest bytes changed")
    holdout_manifests = {}
    for rows in (independent[:24], independent):
        name = f"heldout_long_short_natural_n{len(rows)}"
        path = out / (name + ".json")
        digest = freeze(path, {
            "name": name, "seed": COHORT_SEED, "actual_count": len(rows), "requests": rows,
            "workload_kind": "Independent unaltered Main400 requests selected using previous native EOS lengths",
        })
        lengths = [previous[row["request_id"]]["output_tokens"] for row in rows]
        holdout_manifests[name] = describe(rows) | {
            "manifest": path.name, "manifest_SHA256": digest,
            "previous_native_output_tokens": {"min": min(lengths), "median": median(lengths), "max": max(lengths)},
            "frozen_requests": [identifier(row) | {"previous_native_output_tokens": length}
                                for row, length in zip(rows, lengths, strict=True)],
        }
    schedules = []
    for size in (24, 64, 128, 256):
        schedule = arrival(primary, size, PRIMARY_QPS)
        path = out / f"primary_very_long_n{size}_arrival.json"
        digest = freeze(path, schedule)
        baseline = BASELINE_TRACES.get(size)
        original_digest = None
        if baseline and (root / baseline).is_file():
            old = read(root / baseline)
            require((old["seed"], old["QPS"], old["trace"]) == (ARRIVAL_SEED, PRIMARY_QPS, schedule["trace"]),
                    f"Frozen schedule differs from prior N{size}")
            original_digest = file_digest(root / baseline)
        schedules.append({
            "requests": size, "distinct_original_requests": len(primary), "output_length": 16,
            "manifest": primary_path.name, "filename": path.name, "sha256": digest,
            "prior_native_schedule_source": str(baseline) if baseline else None,
            "prior_native_schedule_sha256": original_digest,
            "last_arrival_seconds": schedule["trace"][-1]["offset_seconds"],
        })
    freeze(args.receipts, {
        "evidence": "FROZEN_WORKLOAD_CONTRACT",
        "source_files": {
            str(path): file_digest(root / path) for path in (PRIMARY_INPUT, POPULATION, PRIOR_OUTPUTS)
        },
        "model_revision": sorted({row["model_revision"] for row in primary + independent}),
        "selection_contract": {
            "minimum_original_input_tokens": HOLDOUT_MIN_INPUT, "maximum_original_input_tokens": HOLDOUT_MAX_INPUT,
            "previous_native_finish_reason": "stop", "previous_native_minimum_output_tokens": 1,
            "previous_native_maximum_output_tokens": HOLDOUT_MAX_PREVIOUS_OUTPUT,
            "exclusions": ["same request ID", "same prompt SHA-256", "same known document ID within source",
                           "any shared original image byte SHA-256"],
            "stable_order": "Ascending SHA-256 of dataset seed followed by original request ID",
            "cohort_seed": COHORT_SEED, "no_candidate_run_used": True,
            "prospective_output_mode": "Natural EOS, fixed decoding configuration across arms",
        },
        "primary": describe(primary) | {"manifest_SHA256": file_digest(primary_path),
                                        "frozen_requests": [identifier(row) for row in primary]},
        "primary_schedules": schedules, "heldout": holdout_manifests,
    })


if __name__ == "__main__":
    main()
