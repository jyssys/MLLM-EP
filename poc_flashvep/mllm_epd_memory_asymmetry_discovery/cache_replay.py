"""Causal capacity screening on frozen multimodal traces without GPU timing."""

import argparse
from collections import OrderedDict
import json
from pathlib import Path

from .workload import block_keys, file_digest, image_key, read_canonical, write_frozen


GIB = 2 ** 30
BLOCK = 128
EMBEDDING_BYTES = 7168
KV_BYTES = 57344


class SizeBoundLRU:
    def __init__(self, budget):
        self.budget = budget
        self.entries = OrderedDict()
        self.used = 0
        self.peak = 0
        self.evictions = 0

    def touch(self, key):
        if key not in self.entries:
            return False
        self.entries.move_to_end(key)
        return True

    def insert(self, key, size):
        if self.touch(key) or size > self.budget:
            return
        while self.used + size > self.budget:
            _, evicted = self.entries.popitem(last=False)
            self.used -= evicted
            self.evictions += 1
        self.entries[key] = size
        self.used += size
        self.peak = max(self.peak, self.used)


def prepare(rows, assets):
    image_numbers = {}
    records = []
    for row in rows:
        length = row["total_input_tokens"]
        blocks = tuple(block_keys(row, assets, image_numbers)[BLOCK])
        images = tuple((image_key(row, image), image["visual_tokens"], image["visual_span"][1])
                       for image in row["images"])
        records.append((row["request_id"], length, images, blocks))
    return records


def replay(records, budget_gib, embedding_fraction):
    total = int(budget_gib * GIB)
    embedding_budget = int(total * embedding_fraction)
    embeddings = SizeBoundLRU(embedding_budget)
    prefix = SizeBoundLRU(total - embedding_budget)
    images = image_before_cache = image_hits = prefix_skipped_images = 0
    visual_rows = encode_saved_rows = prompt_rows = prefill_saved_rows = requests_prefix_hit = 0
    image_requests_hit = image_requests_masked = 0
    for _, tokens, visual, keys in records:
        prompt_rows += tokens
        matchable = keys[:(tokens - 1) // BLOCK]
        hits = 0
        for key in matchable:
            if not prefix.touch(key):
                break
            hits += 1
        reusable_rows = hits * BLOCK
        requests_prefix_hit += bool(hits)
        prefill_saved_rows += reusable_rows
        fill_embeddings = []
        used_embedding_cache = False
        used_prefix_for_encoding = False
        for key, rows, end in visual:
            images += 1
            visual_rows += rows
            if end <= reusable_rows:
                prefix_skipped_images += 1
                encode_saved_rows += rows
                used_prefix_for_encoding = True
                continue
            image_before_cache += 1
            if embeddings.touch(key):
                image_hits += 1
                encode_saved_rows += rows
                used_embedding_cache = True
            else:
                fill_embeddings.append((key, rows * EMBEDDING_BYTES))
        image_requests_hit += used_embedding_cache
        image_requests_masked += used_prefix_for_encoding
        for key, size in fill_embeddings:
            embeddings.insert(key, size)
        for key in keys:
            prefix.insert(key, BLOCK * KV_BYTES)
    return {
        "budget_GiB": budget_gib, "embedding_fraction": embedding_fraction,
        "requests": len(records), "prompt_rows": prompt_rows, "visual_rows": visual_rows, "image_occurrences": images,
        "prefix_hit_requests": requests_prefix_hit, "prefill_avoided_rows": prefill_saved_rows,
        "prefix_completely_covered_image_occurrences": prefix_skipped_images,
        "requests_with_images_covered_by_prefix": image_requests_masked,
        "required_image_occurrences_after_prefix": image_before_cache, "embedding_hit_occurrences": image_hits,
        "embedding_hit_requests": image_requests_hit, "encode_avoided_visual_rows": encode_saved_rows,
        "embedding_evictions": embeddings.evictions, "prefix_block_evictions": prefix.evictions,
        "embedding_peak_bytes": embeddings.peak, "prefix_peak_bytes": prefix.peak,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-repo", type=Path, required=True)
    parser.add_argument("--manifest", action="append", default=[], type=Path)
    parser.add_argument("--main400", action="store_true", help="Include the Git-pinned parent using metadata only")
    parser.add_argument("--budget-gib", type=float, nargs="+", default=[0, .25, .5, 1, 2, 4, 8, 16, 32, 64, 80])
    parser.add_argument("--embedding-fraction", type=float, nargs="+", default=[0, .25, .5, .75, 1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.main400 and not args.manifest:
        parser.error("At least one trace is required")
    if any(value < 0 for value in args.budget_gib) or any(not 0 <= value <= 1 for value in args.embedding_fraction):
        parser.error("Budgets must be nonnegative and fractions within [0, 1]")
    cohorts = [(p.stem, json.loads(p.read_bytes())["requests"], file_digest(p)) for p in args.manifest]
    if args.main400:
        manifest = read_canonical(Path.cwd())
        from .workload import MAIN400_SHA256
        cohorts.append(("main400", manifest["requests"], MAIN400_SHA256))
    results = []
    for name, rows, digest in cohorts:
        records = prepare(rows, args.asset_repo.absolute())
        for budget in args.budget_gib:
            for fraction in args.embedding_fraction:
                results.append({"cohort": name, "manifest_sha256": digest, **replay(records, budget, fraction)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_frozen(args.out, {
        "evidence": "OFFLINE_ORACLE",
        "scope": "Causal sequential complete-before-next cache replay with an ideal connector. No GPU timing, "
                 "request queue, active page reservation, or concurrent transfer. Prefix-covered images contribute "
                 "to hypothetical encode avoidance only when an EPD implementation can bypass Encode on those images.",
        "attention": "Qwen2.5-VL BF16, 128-token physical blocks, final prompt row reserved for logits",
        "results": results,
    })


if __name__ == "__main__":
    main()
