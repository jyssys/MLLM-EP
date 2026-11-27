"""Relocate immutable Main400 inputs and audit image and causal-prefix reuse."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


ORIGINAL_TREE = Path("/home/esjung/MLLM-EP-github")
DATA = Path("poc_flashvep/reports/mllm_epd_prefill_heavy_token_split_20261124/data")
MAIN400 = DATA / "main_manifest_template_v2.json"
MAIN400_SHA256 = "7ac7d28ae1678499bee35ec20e317760555944cac7b4ca29eb46aeb3957b9592"
MAIN100_LOCAL_SHA256 = "1d7221b2391c5239ca0a85a109d3b93d46ab1a03c4a42f4caaf44d0a9a437151"
BASE = "07892cbe7193366ec422f1df82006c6e69f91cec"
BLOCKS = (16, 32, 128)
MIB = 2 ** 20


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_frozen(path, content):
    data = (json.dumps(content, indent=2, ensure_ascii=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Frozen output differs at {path}")
    else:
        path.write_bytes(data)


def read_canonical(repository):
    data = subprocess.check_output(["git", "-C", str(repository), "show", f"{BASE}:{MAIN400}"])
    if hashlib.sha256(data).hexdigest() != MAIN400_SHA256:
        raise ValueError("Unexpected canonical Main400 manifest hash")
    return json.loads(data)


def canonical_path(original, assets):
    return assets / Path(original).relative_to(ORIGINAL_TREE)


def relocate(records, assets):
    """Keep the original prompt and image list, and verify every retained byte."""
    hashes = {}

    def verify(path, expected):
        if path not in hashes:
            hashes[path] = file_digest(path)
        if hashes[path] != expected:
            raise ValueError(f"Frozen media or tensor hash mismatch at {path}")

    rows = []
    for row in records:
        metadata = canonical_path(row["token_metadata_file"], assets)
        locations = [canonical_path(image["path"], assets) for image in row["images"]]
        if not metadata.is_file() or not all(path.is_file() for path in locations):
            continue
        verify(metadata, row["token_metadata_SHA256"])
        for path, image in zip(locations, row["images"], strict=True):
            verify(path, image["SHA256"])
        rows.append({**row, "token_metadata_file": str(metadata), "images": [
            {**image, "path": str(path)} for path, image in zip(locations, row["images"], strict=True)
        ]})
    return rows


def image_key(row, image):
    return (row["model_revision"], row["processor_revision"], image["pixel_SHA256"],
            tuple(image["grid_thw"]), tuple(image["processed_resolution"]))


def ordered_images(row):
    return tuple(image_key(row, image) for image in row["images"])


def distinct_images(rows):
    chosen, seen = [], set()
    for row in rows:
        keys = ordered_images(row)
        unique = set(keys)
        if len(keys) == len(unique) and not seen.intersection(unique):
            chosen.append(row)
            seen.update(unique)
    return chosen


def make_manifest(name, rows, metadata):
    ids = {r["request_id"] for r in rows}
    manifest = {} if metadata is None else {k: v for k, v in metadata.items() if k != "requests"}
    for entry in ("calibration_ids", "discovery_ids", "heldout_ids"):
        if entry in manifest:
            manifest[entry] = [rid for rid in manifest[entry] if rid in ids]
    # Preserve the canonical request objects. The new cohort is availability scoped.
    return {**manifest, "name": name, "actual_count": len(rows), "target_count": len(rows),
            "canonical_source": str(MAIN400), "canonical_source_SHA256": MAIN400_SHA256,
            "requests": rows}


def counter_metrics(rows):
    ordered, docs, images, byte_hashes = Counter(), Counter(), Counter(), Counter()
    seen_images = set()
    seen_questions = defaultdict(set)
    request_with_image_hit, repeat_with_new_question = 0, 0
    hit_images, within_request_duplicates = 0, 0
    image_reuse_distance, ordered_reuse_distance = [], []
    last_image, last_ordered = {}, {}
    for index, row in enumerate(rows):
        keys = ordered_images(row)
        cached = [item for item in keys if item in seen_images]
        request_with_image_hit += bool(cached)
        hit_images += len(cached)
        within_request_duplicates += len(keys) - len(set(keys))
        for key in cached:
            image_reuse_distance.append(index - last_image[key])
        for key in keys:
            last_image[key] = index
        seen_images.update(keys)
        if keys in last_ordered:
            ordered_reuse_distance.append(index - last_ordered[keys])
        last_ordered[keys] = index
        question = row["question"]
        repeat_with_new_question += bool(seen_questions[keys] - {question}) and question not in seen_questions[keys]
        seen_questions[keys].add(question)
        images.update(keys)
        byte_hashes.update(image["SHA256"] for image in row["images"])
        ordered[keys] += 1
        if row.get("document_id") is not None:
            docs[(row["source"], row["document_id"])] += 1
    return {
        "requests": len(rows), "total_image_occurrences": images.total(),
        "unique_derivative_byte_hashes": len(byte_hashes), "unique_processor_image_keys": len(images),
        "image_hits_from_earlier_requests": hit_images, "requests_with_image_hit": request_with_image_hit,
        "within_request_duplicate_occurrences": within_request_duplicates,
        "ordered_image_sequences": len(ordered), "repeat_ordered_image_requests": ordered.total() - len(ordered),
        "ordered_image_hit_with_new_question": repeat_with_new_question,
        "repeated_ordered_image_groups": sum(size > 1 for size in ordered.values()),
        "same_ordered_images_group_histogram": dict(sorted(Counter(ordered.values()).items())),
        "document_id_coverage": docs.total(), "source_scoped_distinct_documents": len(docs),
        "repeat_document_id_requests": docs.total() - len(docs),
        "image_hit_request_distance": distribution(image_reuse_distance),
        "ordered_hit_request_distance": distribution(ordered_reuse_distance),
    }


def distribution(values):
    if not values:
        return {"count": 0}
    return {"count": len(values), "min": float(min(values)), "p50": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "max": float(max(values))}


def block_keys(row, asset_root, image_numbers):
    path = Path(row["token_metadata_file"])
    if path.is_relative_to(ORIGINAL_TREE):
        path = canonical_path(path, asset_root)
    if file_digest(path) != row["token_metadata_SHA256"]:
        raise ValueError(f"Corrupt token metadata for {row['request_id']}")
    with np.load(path, allow_pickle=False) as metadata:
        ids = metadata["input_ids"].reshape(-1)
        positions = metadata["positions"].reshape(3, -1)
        types = metadata["mm_token_type_ids"].reshape(-1)
    tokens = len(ids)
    image_slots = np.zeros(tokens, dtype="<i8")
    for image in row["images"]:
        begin, end = image["visual_span"]
        key = image_key(row, image)
        image_slots[begin:end] = image_numbers.setdefault(key, len(image_numbers) + 1)
    if tokens != row["total_input_tokens"] or np.count_nonzero(image_slots) != row["visual_tokens"]:
        raise ValueError(f"Processor metadata or visual span mismatch for {row['request_id']}")
    streams = np.ascontiguousarray(np.column_stack((ids, positions.T, types, image_slots)), dtype="<i8")
    salt = hashlib.sha256(json.dumps([
        row["model_revision"], row["processor_revision"], row["prompt_template_version"], "bf16", "causal", "public-study"
    ]).encode()).digest()
    result = {}
    for block in BLOCKS:
        chain = salt
        signatures = []
        for start in range(0, tokens - block + 1, block):
            chain = hashlib.sha256(chain + streams[start:start + block].tobytes()).digest()
            signatures.append(chain)
        result[block] = signatures
    return result


def prefix_metrics(rows, asset_root):
    seen = {block: set() for block in BLOCKS}
    reusable = {block: [] for block in BLOCKS}
    image_numbers = {}
    for row in rows:
        tokens = row["total_input_tokens"]
        signatures = block_keys(row, asset_root, image_numbers)
        for block, keys in signatures.items():
            # vLLM must recompute at least one prompt token for generation logits.
            max_blocks = (tokens - 1) // block
            matched = next((i for i, key in enumerate(keys[:max_blocks]) if key not in seen[block]),
                           min(max_blocks, len(keys)))
            reusable[block].append(matched * block)
            seen[block].update(keys)
    result = {}
    total_prompt_tokens = sum(row["total_input_tokens"] for row in rows)
    for block, counts in reusable.items():
        hit = [count for count in counts if count]
        result[block] = {"requests_with_full_block_hit": len(hit), "prompt_tokens": total_prompt_tokens,
                         "reusable_prompt_tokens": sum(counts), "full_block_hit_token_rows": distribution(hit)}
    return result


def representation_sizes(rows):
    """Exact logical stored payloads, excluding allocator and cache metadata."""
    # Qwen2.5-VL-7B BF16. The LM width is 3,584. Each of 28 layers has
    # 4 KV heads of width 128, with both K and V stored for each token.
    embedding_per_row, kv_per_row = 3584 * 2, 28 * 2 * 4 * 128 * 2
    raw, projected, prefix = [], [], []
    for row in rows:
        raw.append(sum(Path(image["path"]).stat().st_size for image in row["images"]) / MIB)
        projected.append(row["visual_tokens"] * embedding_per_row / MIB)
        prefix.append(row["total_input_tokens"] * kv_per_row / MIB)
    return {"raw_derivative_file_MiB": distribution(raw), "lm_ready_embedding_MiB": distribution(projected),
            "full_prompt_KV_MiB": distribution(prefix), "embedding_bytes_per_visual_token": embedding_per_row,
            "kv_bytes_per_prompt_token": kv_per_row}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-repo", type=Path, required=True)
    parser.add_argument("--main100", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    if file_digest(args.main100) != MAIN100_LOCAL_SHA256:
        raise ValueError("Unexpected relocated Main100 hash")
    canonical = read_canonical(Path.cwd())
    full = canonical["requests"]
    local100 = json.loads(args.main100.read_bytes())["requests"]
    assets = args.asset_repo.absolute()
    online = relocate(full, assets)
    if not {r["request_id"] for r in local100} <= {r["request_id"] for r in online}:
        raise ValueError("Main400 local-media subset lost original Main100 inputs")
    unique = distinct_images(online)
    ordered = defaultdict(list)
    for row in online:
        ordered[ordered_images(row)].append(row)
    grouped = [row for group in ordered.values() for row in group]
    args.results.mkdir(parents=True, exist_ok=True)
    args.run_root.mkdir(parents=True, exist_ok=True)
    cohort_entries = {}
    for name, rows in (("natural149", online), ("natural149_grouped", grouped), ("unique_images", unique)):
        manifest = args.run_root / f"{name}_manifest.json"
        write_frozen(manifest, make_manifest(name, rows, canonical))
        cohort_entries[name] = {"requests": len(rows), "manifest_filename": manifest.name,
                                "sha256": file_digest(manifest)}
    cohort_rows = {"main100": local100, "main400": full, "natural149": online,
                   "natural149_grouped": grouped, "unique_images": unique}
    result = {"evidence": "OFFLINE_ORACLE", "main400_manifest_sha256": MAIN400_SHA256,
              "main100_manifest_sha256": MAIN100_LOCAL_SHA256, "rows_with_local_assets": len(online),
              "online_cohorts": cohort_entries, "cohorts": {}}
    for name, rows in cohort_rows.items():
        sources = sorted({row["source"] for row in rows})
        result["cohorts"][name] = {"image_and_document": counter_metrics(rows),
                                    "per_source": {source: counter_metrics([r for r in rows if r["source"] == source])
                                                   for source in sources},
                                    "causal_full_block_prefix": prefix_metrics(rows, assets)}
        if name != "main400":
            result["cohorts"][name]["representation_sizes"] = representation_sizes(rows)
    write_frozen(args.results / "workload_reuse.json", result)
    write_frozen(args.run_root / "workload_manifest_receipt.json", cohort_entries)
    print(json.dumps(cohort_entries, indent=2))


if __name__ == "__main__":
    main()
