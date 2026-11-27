"""Replay bounded APC and passive GPU tier occupancy using exact 128-row keys."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .cache_replay import BLOCK, GIB, KV_BYTES, SizeBoundLRU, prepare
from .workload import file_digest, write_frozen


BLOCK_BYTES = BLOCK * KV_BYTES


def simulate(records, waves, local_gib, peer_gib, threshold=1, delay=0, active_gib=0):
    local = SizeBoundLRU(int(max(0, local_gib - active_gib) * GIB))
    peer = SizeBoundLRU(int(peer_gib * GIB))
    observations = Counter()
    pending = set()
    writes = defaultdict(list)
    result = {}
    for index, ((_, length, _, keys), wave) in enumerate(zip(records, waves, strict=True)):
        stat = result.setdefault(wave, Counter())
        for key in writes.pop(index, []):
            pending.remove(key)
            peer.insert(key, BLOCK_BYTES)
            stat["peer_durable_bytes"] += BLOCK_BYTES

        stat["requests"] += 1
        local_blocks = peer_blocks = 0
        matchable = keys[:(length - 1) // BLOCK]
        for key in matchable:
            if not local.touch(key):
                break
            local_blocks += 1
        for key in matchable[local_blocks:]:
            if not peer.touch(key):
                stat["first_remote_missing_is_pending"] += key in pending
                break
            peer_blocks += 1
        stat["local_hit_requests"] += bool(local_blocks)
        stat["peer_hit_requests"] += bool(peer_blocks)
        stat["any_hit_requests"] += bool(local_blocks or peer_blocks)
        stat["local_avoided_rows"] += local_blocks * BLOCK
        stat["peer_avoided_rows"] += peer_blocks * BLOCK
        stat["peer_load_bytes"] += peer_blocks * BLOCK_BYTES

        for key in keys:
            local.insert(key, BLOCK_BYTES)
            if peer.budget < BLOCK_BYTES:
                continue
            observations[key] += 1
            if observations[key] < threshold or peer.touch(key) or key in pending:
                continue
            stat["peer_write_admitted_bytes"] += BLOCK_BYTES
            if delay:
                pending.add(key)
                writes[index + delay + 1].append(key)
            else:
                peer.insert(key, BLOCK_BYTES)
                stat["peer_durable_bytes"] += BLOCK_BYTES
        stat["local_end_bytes"] = local.used
        stat["peer_end_durable_bytes"] = peer.used
        stat["peer_end_pending_bytes"] = len(pending) * BLOCK_BYTES
        stat["local_evictions_cumulative"] = local.evictions
        stat["peer_evictions_cumulative"] = peer.evictions

    return {
        "local_GiB": local_gib, "peer_GiB": peer_gib, "local_active_reserve_GiB": active_gib,
        "admit_after_observations": threshold, "durability_delay_completed_requests": delay,
        "local_peak_bytes": local.peak, "peer_peak_durable_bytes": peer.peak,
        "peer_pending_bytes_after_last_request": len(pending) * BLOCK_BYTES,
        "waves": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    configurations = [
        (42, 0, 1, 0, 0), (42, 0, 1, 0, 2), (8, 0, 1, 0, 0),
        (8, 8, 1, 0, 0), (8, 24, 1, 0, 0), (8, 34, 1, 0, 0),
        (8, 48, 1, 0, 0), (8, 64, 1, 0, 0),
        (8, 64, 1, 4, 2), (8, 64, 2, 0, 0), (8, 64, 2, 4, 2),
        (42, 8, 1, 0, 0), (42, 64, 1, 0, 0),
    ]
    result = {}
    for manifest in args.manifest:
        rows = json.loads(manifest.read_bytes())["requests"]
        records = prepare(rows, args.asset_repo)
        waves = [row.get("cache_wave", 0) for row in rows]
        result[manifest.stem] = {
            "manifest_sha256": file_digest(manifest),
            "configurations": [simulate(records, waves, *config) for config in configurations],
        }
    write_frozen(args.out, {
        "evidence": "OFFLINE_ORACLE",
        "block_bytes": BLOCK_BYTES,
        "scope": {
            "policy": "Independent inclusive bounded LRU tiers. Compute missing suffix after contiguous P hit and peer extension. "
                      "Fill the completed prompt into hot P and conditionally into the peer; all writes precede the next request "
                      "with delay 0. Tier writes on a hit touch the existing block without charging another transfer.",
            "admission": "Observed request-level frequency per block before tier store, including prior local accesses. "
                         "Stock vLLM offload counts actual connector lookups, which may stop at the first missing block.",
            "durability": "Peer key remains unreadable for the prescribed number of subsequent completed requests. "
                          "Pending bytes tracked separately and assume independent staging capacity.",
            "memory": "Capacities are usable logical payload budgets. Positive active reserve is deducted from P. "
                      "GPU allocation, scheduler pinning, staging interference, execution speed, and queues require runtime evidence.",
        },
        "cohorts": result,
    })


if __name__ == "__main__":
    main()
