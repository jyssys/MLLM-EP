"""Compare multi-representation cache policies on the same exact payloads."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import json
from pathlib import Path

from .cache_replay import BLOCK, EMBEDDING_BYTES, GIB, KV_BYTES, prepare
from .workload import file_digest, write_frozen


class Costs:
    def __init__(self, summary=None):
        self.summary = summary
        if summary:
            cases = summary["economics"]
            self.encode_points = sorted((row["embedding_bytes"] / EMBEDDING_BYTES, row["encode_ms"]) for row in cases)
            self.prefill_points = sorted((row["prefix_kv_bytes"] / KV_BYTES, row["image_prefix_prefill_ms"]) for row in cases)
            self.transfer_points = {
                (src, dst): sorted((row["bytes"], row["gpu_ms_per_copy"]) for row in summary["transfers"]
                                   if (row["source"], row["destination"]) == (src, dst))
                for src, dst in (("E", "P"), ("P", "E"))
            }

    @staticmethod
    def estimate(points, value):
        if value == 0:
            return 0.0
        index = bisect_left(points, (value,))
        if index == 0 or index == len(points):
            x, y = points[0 if index == 0 else -1]
            return y * value / x
        low_x, low_y = points[index - 1]
        high_x, high_y = points[index]
        scale = (value - low_x) / (high_x - low_x)
        return low_y + scale * (high_y - low_y)

    def encode(self, rows):
        return self.estimate(self.encode_points, rows) if self.summary else rows

    def prefill(self, rows):
        return self.estimate(self.prefill_points, rows) if self.summary else rows

    def transfer(self, src, dst, size):
        return self.estimate(self.transfer_points[src, dst], size) if self.summary else 0


class Pool:
    def __init__(self, budget, policy):
        self.budget = budget
        self.policy = policy
        self.resident = {}
        self.uses = Counter()
        self.scopes = defaultdict(set)
        self.base = 0.0
        self.used = 0
        self.peak = 0
        self.evictions = Counter()
        self.inserts = Counter()
        self.inserted_bytes = Counter()

    def touch(self, item, tick):
        if item not in self.resident:
            return False
        size, density, _ = self.resident[item]
        self.resident[item] = size, density, tick
        return True

    def insert(self, item, size, benefit, scope, tick):
        self.uses[item] += 1
        self.scopes[item].add(scope)
        if size > self.budget:
            return
        present = self.touch(item, tick)
        if self.policy == "lru":
            priority = float(tick)
        else:
            factor = 1.0
            if self.policy == "scope" and item[0] == "embedding":
                factor += 0.5 * min(2, len(self.scopes[item]) - 1)
            priority = self.base + max(0, benefit) * self.uses[item] * factor / size
        if present:
            self.resident[item] = size, priority, tick
            return False
        while self.used + size > self.budget:
            victim = min(self.resident, key=lambda key: self.resident[key][1:])
            evicted, victim_priority, _ = self.resident.pop(victim)
            self.used -= evicted
            self.evictions[victim[0]] += 1
            if self.policy != "lru":
                self.base = victim_priority
        if self.policy != "lru":
            priority = self.base + max(0, benefit) * self.uses[item] * factor / size
        self.resident[item] = size, priority, tick
        self.used += size
        self.peak = max(self.peak, self.used)
        self.inserts[item[0]] += 1
        self.inserted_bytes[item[0]] += size
        return True


def image_prefix_blocks(prompt_rows, images):
    image_prefix_end = max(end for _, _, end in images)
    return min((prompt_rows - 1) // BLOCK, (image_prefix_end + 1) // BLOCK)


def run(records, budget_gib, policy, embedding_fraction, costs, prefix_encode_bypass, *, unified_pool=Pool):
    total = int(budget_gib * GIB)
    if embedding_fraction is None:
        embeddings = prefix = unified_pool(total, policy)
        pools = [embeddings]
    else:
        embedding_bytes = int(total * embedding_fraction)
        embeddings = Pool(embedding_bytes, "lru")
        prefix = Pool(total - embedding_bytes, "lru")
        pools = [embeddings, prefix]

    s = Counter()
    for tick, (_, prompt_rows, images, blocks) in enumerate(records, 1):
        block_count = image_prefix_blocks(prompt_rows, images)
        scope = blocks[block_count - 1] if block_count else None
        match = 0
        for index in range(block_count - 1, -1, -1):
            if prefix.touch(("prefix", blocks[index]), tick):
                match = (index + 1) * BLOCK
                break
        prefix_bytes = match * KV_BYTES
        visual_rows = sum(rows for _, rows, _ in images)
        encode_cost = costs.encode(visual_rows)
        prefill_cost = costs.prefill(prompt_rows)
        s.update(requests=1, baseline_visual_rows=visual_rows, baseline_prompt_rows=prompt_rows,
                 baseline_encode=encode_cost, baseline_prefill=prefill_cost,
                 baseline_embedding_transfer=costs.transfer("E", "P", visual_rows * EMBEDDING_BYTES),
                 prefix_hit_requests=bool(match), prefill_avoided_rows=match,
                 prefill_avoided=costs.prefill(match), prefix_load_bytes=prefix_bytes,
                 prefix_load_transfer=costs.transfer("E", "P", prefix_bytes))
        visual_avoided = transfer_bytes = 0
        observations = {}
        for key, rows, end in images:
            item = ("embedding", key)
            if prefix_encode_bypass and end <= match:
                s["prefix_encode_covered_visual_rows"] += rows
                s["prefix_encode_covered_image_occurrences"] += 1
                visual_avoided += rows
                continue
            size = rows * EMBEDDING_BYTES
            transfer_bytes += size
            if embeddings.touch(item, tick):
                s["embedding_hit_occurrences"] += 1
                s["embedding_avoided_visual_rows"] += rows
                visual_avoided += rows
            scoped_benefit = encode_cost * rows / visual_rows
            observations[item] = size, scoped_benefit
        for item, (size, benefit) in observations.items():
            embeddings.insert(item, size, benefit, scope, tick)
        s["actual_embedding_transfer"] += costs.transfer("E", "P", transfer_bytes)
        s["embedding_transfer_bytes"] += transfer_bytes
        s["encode_avoided"] += encode_cost * visual_avoided / visual_rows
        if scope is not None:
            rows = block_count * BLOCK
            size = rows * KV_BYTES
            avoided_encode = sum(visual for _, visual, end in images if end <= rows) * encode_cost / visual_rows
            transfer = costs.transfer("P", "E", size)
            benefit = costs.prefill(rows) - costs.transfer("E", "P", size) - transfer
            if prefix_encode_bypass:
                benefit += avoided_encode
            if prefix.insert(("prefix", scope), size, benefit, scope, tick):
                s["prefix_insert_transfer"] += transfer
    assert all(pool.peak <= pool.budget for pool in pools)
    stages = {
        "encode_baseline": s["baseline_encode"], "encode_after_hits": s["baseline_encode"] - s["encode_avoided"],
        "prefill_baseline": s["baseline_prefill"], "prefill_after_hits": s["baseline_prefill"] - s["prefill_avoided"],
        "native_embedding_copy": s["baseline_embedding_transfer"],
        "remaining_embedding_copy": s["actual_embedding_transfer"],
        "prefix_load_copy": s["prefix_load_transfer"], "prefix_insert_copy": s["prefix_insert_transfer"],
    }
    if costs.summary:
        stages["modeled_E_P_compute_stage_max_baseline"] = max(stages["encode_baseline"], stages["prefill_baseline"])
        stages["modeled_E_P_compute_stage_max_after_hits"] = max(stages["encode_after_hits"], stages["prefill_after_hits"])
    return {
        "policy": policy, "budget_GiB": budget_gib, "embedding_fraction": embedding_fraction,
        "allow_prefix_to_bypass_encode": prefix_encode_bypass, "stages": stages,
        "counts": dict(s),
        "residency": [{
            "budget_bytes": pool.budget, "peak_bytes": pool.peak, "end_bytes": pool.used,
            "evictions": dict(pool.evictions), "inserts": dict(pool.inserts), "inserted_bytes": dict(pool.inserted_bytes),
        } for pool in pools],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-repo", type=Path, required=True)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--economics", type=Path)
    parser.add_argument("--budget-gib", type=float, nargs="+", default=[1, 4, 8, 16, 32, 42, 64])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if any(budget < 0 for budget in args.budget_gib):
        parser.error("Cache sizes must be nonnegative")
    costs = Costs(json.loads(args.economics.read_bytes()) if args.economics else None)
    policies = [("lru", fraction) for fraction in (0, .25, .5, .75, 1)] + [("generic", None), ("scope", None)]
    cohorts = {}
    for manifest in args.manifest:
        rows = json.loads(manifest.read_bytes())["requests"]
        records = prepare(rows, args.asset_repo)
        cohorts[manifest.stem] = {
            "manifest_sha256": file_digest(manifest),
            "runs": [run(records, budget, name, fraction, costs, bypass)
                     for budget in args.budget_gib for name, fraction in policies for bypass in (False, True)],
        }
    write_frozen(args.out, {
        "evidence": "OFFLINE_ORACLE",
        "timing_calibration_sha256": file_digest(args.economics) if args.economics else None,
        "stage_units": "Modeled stage milliseconds" if args.economics else "Visual-row and prompt-row units, separately",
        "scope": {
            "geometry": "Each arm uses the same whole-image embedding items and atomic, full-block visual-prefix checkpoints. "
                        "Each request's checkpoint ends at the last whole 128-row block at or before the image-closing marker. "
                        "Prefix hits preserve at least one final prompt row for logits. All items use identical content and positional key support.",
            "policy": "Fixed LRU partitions compare with frequency-cost-per-byte unified GreedyDual. The scope policy uses the same "
                      "score and appends a factor of up to 2x for embeddings observed in multiple exact prefix scopes. "
                      "Insertion and weights depend only on history. Event order is sequential complete-before-next.",
            "timing": "When available, isolated CUDA-event calibration is interpolated by query rows or contiguous transfer bytes "
                      "and linearly scaled outside its support. Copy and compute estimates remain separate additive stages. "
                      "The maximum of aggregate E/P compute demands is a service-floor proxy with ideal overlap. "
                      "This replay does not represent per-request delay, actual bottleneck, decode, contention, or goodput.",
            "encode": "The encode-bypass sensitivity checks both connector contracts. In the bypass mode an image "
                      "is skipped only when its entire span is covered by the exact checkpoint. A prefix hit and "
                      "embedding hit never book overlapping encode cost.",
        },
        "cohorts": cohorts,
    })


if __name__ == "__main__":
    main()
