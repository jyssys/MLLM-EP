"""Freeze original-image audit pairs and reduce native encoder checksums and counters."""

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re

from .workload import file_digest, make_manifest, write_frozen


SAME_DOCUMENT = "same_ordered_images_new_question"
COMPANIONS = "shared_image_with_different_companions"
METRIC = re.compile(r"^([^ {]+)(?:\{([^}]*)\})? (\S+)")
ENCODER_COUNTERS = ("acquired_image_hits", "acquired_image_misses", "computed_image_saves",
                    "saved_count", "released_count", "late_image_saves", "cache_evictions",
                    "host_loads", "host_stores", "host_evictions", "host_load_bytes", "host_store_bytes", "host_store_batches",
                    "host_load_seconds", "host_store_seconds")
ENCODER_GAUGES = ("active_request_leases", "outstanding_exports", "live_storage_bytes",
                  "idle_storage_bytes", "retained_storage_bytes", "diagnostic_tensor_hashes",
                  "retained_cache_device", "host_retained_images", "host_retained_storage_bytes",
                  "host_peak_storage_bytes", "host_pending_storage_bytes")


def load(path):
    return json.loads(path.read_text())


def records(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def images(row):
    if "content_order" in row:
        indices = [part["image_index"] for part in row["content_order"] if part["type"] == "image"]
        return [row["images"][index] for index in indices]
    return row["images"]


def image_sequence(row):
    return tuple(item["SHA256"] for item in images(row))


def make_fixture(smoke, cohort):
    base = smoke["requests"]
    source = cohort["requests"]
    by_image = defaultdict(list)
    source_index = {row["request_id"]: index for index, row in enumerate(source)}
    for row in source:
        for key in sorted(set(image_sequence(row))):
            by_image[key].append(row)
    natural = {}
    for key, related in by_image.items():
        for a, b in combinations(related, 2):
            if set(image_sequence(a)) == set(image_sequence(b)):
                continue
            pair = (a["request_id"], b["request_id"])
            natural.setdefault(pair, set()).add(key)
    same = [(a["request_id"], b["request_id"]) for a, b in combinations(base, 2)
            if image_sequence(a) == image_sequence(b) and a["question"] != b["question"]]
    if not same or not natural:
        raise ValueError("Missing a natural audit contrast in the supplied frozen cohorts")
    by_id = {row["request_id"]: row for row in source}
    picked = list(base)
    existing = {row["request_id"] for row in picked}
    for pair in sorted(natural, key=lambda pair: tuple(source_index[rid] for rid in pair)):
        for rid in pair:
            if rid not in existing:
                picked.append(by_id[rid])
                existing.add(rid)
    fixture = make_manifest("native_encoder_geometry_audit", picked, cohort)
    fixture["audit_contract"] = {
        "evidence": "CORRECTNESS_ONLY",
        "scope": "Original public questions, image ordering and full image arrays are preserved. Different companions are naturally observed in distinct Mantis questions without source document IDs.",
        "same_ordered_images_new_question_pairs": same,
        "different_original_companions_pairs": [
            {"request_ids": pair, "shared_file_sha256": sorted(shared)} for pair, shared in natural.items()
        ],
    }
    return fixture


def compare_counts(before, after, keys):
    result = {}
    for key in keys:
        if key in before and key in after:
            value = after[key] - before[key]
            if value < 0:
                raise ValueError(f"Counter reset inside one cohort: {key}")
            result[key] = value
    return result


def stage_metrics(snapshot):
    metrics = {}
    for role, lines in snapshot["roles"].items():
        values = Counter()
        for line in lines:
            match = METRIC.match(line)
            if not match:
                continue
            name, labels, value = match.groups()
            if not name.endswith("_total"):
                continue
            if name == "vllm:prompt_tokens_by_source_total":
                source = re.search(r'\bsource="([^"]+)"', labels or "")
                if source:
                    name += "/" + source[1]
            if any(part in name for part in ("prefix_cache", "prompt_tokens_by_source", "kv_offload", "mm_cache", "nixl")):
                values[name] += float(value)
        metrics[role] = dict(values)
    return metrics


def counter_snapshot(data):
    before, after = data["before"], data["after"]
    result = {"stages": {}}
    for role, initial in stage_metrics(before).items():
        terminal = stage_metrics(after)[role]
        result["stages"][role] = compare_counts(initial, terminal, sorted(initial))
    if before.get("encoder_cache") and after.get("encoder_cache"):
        initial, terminal = before["encoder_cache"][0], after["encoder_cache"][0]
        if initial["producer_pid"] != terminal["producer_pid"]:
            raise ValueError("Encoder producer restarted inside one cohort")
        result["encoder_registry"] = compare_counts(initial, terminal, ENCODER_COUNTERS)
        result["encoder_start"] = {key: initial[key] for key in ENCODER_GAUGES if key in initial}
        result["encoder_end"] = {key: terminal[key] for key in ENCODER_GAUGES if key in terminal}
    return result


def pair_kind(a, b):
    if a["request_id"] == b["request_id"]:
        return "same_request_recompute"
    if a["sequence"] == b["sequence"]:
        return SAME_DOCUMENT if a["question"] != b["question"] else "same_ordered_images_same_question"
    if set(a["sequence"]) != set(b["sequence"]):
        return COMPANIONS
    return "same_companions_changed_order"


def summarize_hashes(audit, owner_request, manifest):
    acquisitions = {}
    sample_rows = defaultdict(list)
    sample_geometry = Counter()
    unknown_measured_owner = 0
    unresolved = 0
    for event in audit:
        owner = event["owner"]
        if event["kind"] == "acquire":
            if owner in acquisitions:
                raise ValueError("Duplicate request owner in native audit")
            acquisitions[owner] = event["ordered_image_keys"]
            continue
        if event["kind"] != "encoder_compute" or owner not in acquisitions:
            raise ValueError("Native encoder event has no prior owner acquisition")
        runtime = owner_request.get(owner)
        if runtime is None or runtime["request_id"] not in manifest:
            unresolved += 1
            unknown_measured_owner += runtime is not None and runtime["runtime_request_id"].startswith("measured_")
            continue
        row = manifest[runtime["request_id"]]
        source_images = images(row)
        ordered_keys = acquisitions[owner]
        if len(source_images) != len(ordered_keys):
            raise ValueError(f"Original request image roster disagrees with native acquisition: {owner}")
        occurrences = [(index, item) for index, (key, item) in enumerate(zip(ordered_keys, source_images, strict=True))
                       if key == event["key"]]
        if not occurrences or len({item["SHA256"] for _, item in occurrences}) != 1:
            raise ValueError(f"Native encoder image identity has no unique content provenance: {owner}")
        index, item = occurrences[0]
        if event["shape"][0] != item["visual_tokens"]:
            raise ValueError(f"Native encoder row count differs from frozen processor shape: {owner}")
        sequence = image_sequence(row)
        sample = {"runtime_request_id": runtime["runtime_request_id"], "request_id": row["request_id"],
                  "question": row["question"], "sequence": sequence, "image_index": index, "owner": owner,
                  "sha256": event["sha256"], "shape": event["shape"], "dtype": event["dtype"],
                  "ordered_grids": [entry["grid_thw"] for entry in source_images]}
        sample_rows[item["SHA256"]].append(sample)
        sample_geometry[(tuple(event["shape"]), event["dtype"])] += 1
    counts = defaultdict(Counter)
    examples = []
    images_compared = defaultdict(set)
    for key, samples in sorted(sample_rows.items()):
        for a, b in combinations(samples, 2):
            kind = pair_kind(a, b)
            same_geometry = (a["shape"], a["dtype"]) == (b["shape"], b["dtype"])
            matched = same_geometry and a["sha256"] == b["sha256"]
            counts[kind]["pairs"] += 1
            counts[kind]["byte_identical"] += matched
            counts[kind]["shape_or_dtype_mismatch"] += not same_geometry
            images_compared[kind].add(key)
            if not matched or kind in (SAME_DOCUMENT, COMPANIONS):
                examples.append({"kind": kind, "image_file_sha256": key, "byte_identical": matched,
                                 "observations": [{k: v for k, v in row.items() if k not in ("question", "sequence")}
                                                  for row in (a, b)]})
    return {
        "evidence": "CORRECTNESS_ONLY",
        "source": "Native E worker, real BF16 LM-ready tensors. Diagnostic GPU-to-CPU copies are excluded from performance claims.",
        "scope": "Pairwise repeated tensor observations are correlated. Frozen request image grids identify logical geometry. Cross-request GPU batching and underlying operator launch geometry are unobserved.",
        "acquisitions": len(acquisitions), "matched_compute_observations": sum(map(len, sample_rows.values())),
        "observations_without_manifest_row": unresolved, "unresolved_measured_observations": unknown_measured_owner,
        "actual_shapes": [{"shape": shape, "dtype": dtype, "count": count}
                          for (shape, dtype), count in sorted(sample_geometry.items())],
        "comparisons": {kind: {**values, "distinct_image_files": len(images_compared[kind]),
                               "not_byte_identical": values["pairs"] - values["byte_identical"]}
                        for kind, values in sorted(counts.items())},
        "observations": examples,
    }


def summarize(run):
    raw = run / "raw"
    cohorts = {}
    owner_request = {}
    roster = {}
    for file in sorted(raw.glob("*_prometheus.json")):
        name = file.name.removesuffix("_prometheus.json")
        requests = records(raw / f"{name}_requests.jsonl")
        counts = Counter(row["status"] for row in requests)
        roster[name] = [row["request_id"] for row in requests]
        for row in requests:
            owner = hashlib.sha256(row["runtime_request_id"].encode()).hexdigest()
            if owner in owner_request:
                raise ValueError(f"Repeated runtime request ID: {row['runtime_request_id']}")
            owner_request[owner] = row
        cohorts[name] = {"statuses": dict(counts), **counter_snapshot(load(file))}
    config = load(run / "state" / "environment.json")
    result = {"run": run.name, "config": {k: config["args"].get(k) for k in
              ("identity", "tier", "encoder_cache", "apc_roles", "cycles", "cache_warm_count", "qps", "output")},
              "manifest_sha256": config["manifest_SHA256"], "cohorts": cohorts, "warm_cycle_comparisons": {}}
    first = cohorts.get("measured_1")
    if first:
        for name in sorted(cohorts):
            if name == "measured_1" or not name.startswith("measured_"):
                continue
            if Counter(roster[name]) != Counter(roster["measured_1"]):
                raise ValueError("Warm cycle does not replay the initial request roster")
            later = cohorts[name]
            initial = first.get("encoder_registry", {})
            terminal = later.get("encoder_registry", {})
            result["warm_cycle_comparisons"][name] = {
                "request_count": len(roster[name]),
                "encoder_warm_minus_first": {k: terminal[k] - value for k, value in initial.items() if k in terminal},
                "stage_warm_minus_first": {role: {key: later["stages"][role][key] - value
                                          for key, value in stage.items() if key in later["stages"][role]}
                                          for role, stage in first["stages"].items()},
                "scope": "First measured cycle follows runtime warmup and may hit within the same cycle. Warm delta describes counters in repeated arrivals, without attribution to a new method or speedup.",
            }
    audit = raw / "stable_ec_encoder_audit.jsonl"
    if audit.exists():
        manifest = load(run / "data" / "frozen_manifest.json")
        result["hash_audit"] = summarize_hashes(records(audit), owner_request, {r["request_id"]: r for r in manifest["requests"]})
        result["hash_audit"]["source_sha256"] = file_digest(audit)
    else:
        result["hash_audit"] = {"evidence": "NOT_RUN", "scope": "No diagnostic native tensor digest trace."}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("fixture")
    fixture.add_argument("--smoke4", type=Path, required=True)
    fixture.add_argument("--cohort", type=Path, required=True)
    fixture.add_argument("--out", type=Path, required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--run", type=Path, action="append", required=True)
    summary.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fixture":
        result = make_fixture(load(args.smoke4), load(args.cohort))
    else:
        result = {"runs": [summarize(run) for run in args.run]}
    write_frozen(args.out, result)


if __name__ == "__main__":
    main()
