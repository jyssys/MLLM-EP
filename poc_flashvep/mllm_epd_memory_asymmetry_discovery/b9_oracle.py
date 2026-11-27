"""Evaluate byte-bounded future-aware retention over frozen multimodal traces."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path

from .cache_replay import BLOCK, prepare
from .policy_screen import Costs, Pool, image_prefix_blocks, run
from .workload import file_digest, write_frozen


def future_uses(records):
    matchable = [blocks[:image_prefix_blocks(rows, images)] for _, rows, images, blocks in records]
    checkpoints = {keys[-1] for keys in matchable if keys}
    uses = defaultdict(list)
    for tick, (record, keys) in enumerate(zip(records, matchable, strict=True), 1):
        for image in dict.fromkeys(image[0] for image in record[2]):
            uses[("embedding", image)].append(tick)
        for key in keys:
            if key in checkpoints:
                uses[("prefix", key)].append(tick)
    return uses


class FuturePool(Pool):
    def __init__(self, budget, policy, uses):
        super().__init__(budget, policy)
        self.future = uses

    def priority(self, item, bytes_used, saved, tick):
        uses = self.future.get(item, ())
        upcoming = bisect_right(uses, tick)
        if upcoming == len(uses):
            return 0.0
        return max(0.0, saved) / (bytes_used * (uses[upcoming] - tick))

    def insert(self, item, size, benefit, _scope, tick):
        if size > self.budget:
            return False
        rank = self.priority(item, size, benefit, tick)
        if item in self.resident:
            self.resident[item] = size, benefit, tick
            return False
        if rank <= 0:
            return False
        if self.used + size > self.budget:
            ranked = sorted((self.priority(key, mem, saved, tick), last, key)
                            for key, (mem, saved, last) in self.resident.items())
            needed = self.used + size - self.budget
            victims = []
            for priority, _, key in ranked:
                if priority >= rank:
                    break
                victims.append(key)
                needed -= self.resident[key][0]
                if needed <= 0:
                    break
            if needed > 0:
                return False
            for victim in victims:
                evicted, _, _ = self.resident.pop(victim)
                self.used -= evicted
                self.evictions[victim[0]] += 1
        self.resident[item] = size, benefit, tick
        self.used += size
        self.peak = max(self.peak, self.used)
        self.inserts[item[0]] += 1
        self.inserted_bytes[item[0]] += size
        return True


def capacity_relaxed(records, bypass):
    images_seen = set()
    checkpoints = set()
    stats = Counter()
    for _, rows, images, blocks in records:
        candidates = blocks[:image_prefix_blocks(rows, images)]
        reuse = next(((i + 1) * BLOCK for i in range(len(candidates) - 1, -1, -1) if candidates[i] in checkpoints), 0)
        stats.update(requests=1, prefix_hit_requests=bool(reuse), prompt_rows=rows, prefill_avoided_rows=reuse)
        for key, num_rows, end in images:
            stats.update(image_occurrences=1, visual_rows=num_rows)
            if bypass and end <= reuse:
                stats.update(prefix_covered_image_occurrences=1, encode_avoided_rows=num_rows)
            elif key in images_seen:
                stats.update(embedding_hit_occurrences=1, encode_avoided_rows=num_rows)
        images_seen.update(key for key, _, _ in images)
        if candidates:
            checkpoints.add(candidates[-1])
    return {"allow_prefix_to_bypass_encode": bypass, **stats}


def online_cross_check(path, workload):
    frozen = json.loads(workload.read_bytes())
    reference = frozen["cohorts"]["natural149"]
    sha256 = frozen["online_cohorts"]["natural149"]["sha256"]
    online = json.loads(path.read_bytes())
    run_rows = (row for run in online["runs"] for row in run["measurements"]
                if row["manifest_sha256"] == sha256 and row["cohort"] == "measured_1")
    measured = next(row for row in run_rows if row["params"]["encoder_cache"] and "prefill" in row["params"]["apc_roles"])
    e = reference["image_and_document"]
    p = reference["causal_full_block_prefix"][str(BLOCK)]
    return {
        "online_pilot_receipt_sha256": file_digest(path),
        "workload_audit_sha256": file_digest(workload),
        "manifest_sha256": sha256,
        "arrival_sha256": measured["arrival_sha256"],
        "whole_block_prior_order_reference": {
            "embedding_occurrences": e["image_hits_from_earlier_requests"],
            "prefix_hit_requests": p["requests_with_full_block_hit"],
            "prefix_hit_rows": p["reusable_prompt_tokens"],
        },
        "online_first_sequence": {
            "embedding_occurrences": measured["cache_counters"]["acquired_image_hits"],
            "prefix_hit_requests": measured["host_stage"]["P_usage_positive_cache_entries"],
            "prefix_hit_rows": measured["host_stage"]["P_usage_reported_cached_tokens"],
        },
        "scope": "Independent counters in the Natural149 QPS2.5 pilot. Full-block native APC and the reference allow "
                 "shorter-than-checkpoint prefixes, whereas the B9 policy admits atomic visual-prefix checkpoints. "
                 "Concurrent overlap and runtime insertion semantics differ from the sequential reference.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--economics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--online-pilot", type=Path)
    parser.add_argument("--workload-receipt", type=Path)
    parser.add_argument("--budget-gib", type=float, nargs="+", default=[4, 16, 64])
    args = parser.parse_args()
    if any(b <= 0 for b in args.budget_gib) or bool(args.online_pilot) != bool(args.workload_receipt):
        parser.error("Use positive budgets and supply both optional comparison receipts together")
    costs = Costs(json.loads(args.economics.read_bytes()))
    cohorts = {}
    for manifest in args.manifest:
        rows = json.loads(manifest.read_bytes())["requests"]
        records = prepare(rows, args.asset_repo)
        uses = future_uses(records)

        def pool_factory(budget, policy):
            return FuturePool(budget, policy, uses)

        cohorts[manifest.stem] = {
            "manifest_sha256": file_digest(manifest),
            "bounded_future_aware": [run(records, budget, "future_weighted_next_use", None, costs, bypass,
                                        unified_pool=pool_factory)
                                     for budget in args.budget_gib for bypass in (False, True)],
            "capacity_relaxed_stage_coverage": [capacity_relaxed(records, bypass) for bypass in (False, True)],
        }
    cross_check = online_cross_check(args.online_pilot, args.workload_receipt) if args.online_pilot else None
    write_frozen(args.out, {
        "evidence": "OFFLINE_ORACLE",
        "timing_calibration_sha256": file_digest(args.economics),
        "stage_units": "Modeled stage milliseconds",
        "scope": {
            "bounded": "Past stores only. Future occurrence indices only rank an otherwise causal admission/eviction. "
                       "Rank is calibrated storage benefit divided by bytes and request distance to the next compatible use. "
                       "An incoming item may displace enough lower-ranked resident items. This is a hindsight heuristic, "
                       "with no finite-capacity optimality or online claim.",
            "upper_bound": "Independent per-stage coverage bound for the same finite trace when atomic checkpoint "
                           "and image objects fit in unlimited storage, with exact identities and no transport constraints.",
            "comparability": "Shares B4–B8 checkpoint geometry, costs, byte accounting and encode-coverage logic. "
                             "Complete-before-next simulator, actual stage copy and queuing require online measurement. "
                             "Prefix and embedding hits cannot both credit one image.",
        },
        "cohorts": cohorts,
        "natural149_online_cross_check": cross_check,
    })


if __name__ == "__main__":
    main()
