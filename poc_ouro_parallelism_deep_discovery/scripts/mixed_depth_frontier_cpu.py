#!/usr/bin/env python3
"""Check mixed-depth frontier identities on deterministic CPU-only state."""

from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Row:
    request: int
    position: int
    next_r: int
    completed_r: int
    gate_hazards: tuple
    layer: int = 0


def gate_state(hazards):
    survival = Fraction(1)
    cdf = Fraction(0)
    for hazard in hazards:
        cdf += survival * hazard
        survival *= 1 - hazard
    return cdf, survival


def check_frontier(rows, adaptive):
    assert len({row.request for row in rows}) == len(rows), "two live rows for one request"
    for row in rows:
        assert row.next_r == row.completed_r + 1, "uncomputed recurrent input"
        assert 1 <= row.next_r <= 4
        if adaptive:
            assert len(row.gate_hazards) == row.completed_r, "uncomputed original gate"
            cdf, _ = gate_state(row.gate_hazards)
            assert cdf < Fraction(1, 2), "request already exited"
    return True


def make_frontier(c, adaptive):
    result = []
    for request in range(c):
        next_r = request % 4 + 1
        hazards = (Fraction(1, 16),) * (next_r - 1) if adaptive else ()
        result.append(Row(request, 7, next_r, next_r - 1, hazards))
    return result


def groups(rows, depth_key):
    result = {}
    for row in rows:
        key = (row.layer, row.next_r) if depth_key else (row.layer, "same_operator_signature")
        result.setdefault(str(key), []).append(row.request)
    return result


def generic_unrolled_groups(rows):
    # Keep logical depth in each node, then resolve immutable weight aliases.
    logical_nodes = [(row.request, row.next_r, row.layer) for row in rows]
    weight_aliases = {(r, layer): f"physical_layer_{layer}"
                      for _, r, layer in logical_nodes}
    by_weight = {}
    for request, r, layer in logical_nodes:
        by_weight.setdefault(weight_aliases[(r, layer)], []).append(request)
    return list(by_weight.values())


def must_reject(rows, adaptive):
    try:
        check_frontier(rows, adaptive)
    except AssertionError as exc:
        return str(exc)
    raise AssertionError("negative control did not fail")


def arena_control():
    # One physical-layer allocation can contain all depth banks without aliasing.
    c, r_count, layer_count, positions = 8, 4, 2, 8
    addresses = {}
    values = {}
    for layer in range(layer_count):
        for r in range(1, r_count + 1):
            for request in range(c):
                for position in range(positions):
                    identity = (request, r, layer, position, "committed_v1")
                    offset = ((((r - 1) * c + request) * positions) + position)
                    address = (layer, offset)
                    assert address not in addresses
                    addresses[address] = identity
                    values[identity] = 10000 * r + 1000 * layer + 10 * request + position
    reads = 0
    for row in make_frontier(c, True):
        for layer in range(layer_count):
            for position in range(row.position):
                identity = (row.request, row.next_r, layer, position, "committed_v1")
                offset = (((row.next_r - 1) * c + row.request) * positions) + position
                assert addresses[(layer, offset)] == identity
                reads += 1
    correct = values[(1, 2, 0, 0, "committed_v1")]
    wrong_depth = values[(1, 1, 0, 0, "committed_v1")]
    assert correct != wrong_depth
    return {
        "identities": len(addresses),
        "matched_historical_reads": reads,
        "wrong_depth_values": [correct, wrong_depth],
        "scope": "address identity only, not a paged-attention kernel or model test",
    }


def dense_control():
    matrix = ((2, -1, 3), (0, 4, -2))
    rows = [(i + 1, 2 - i, i % 3) for i in range(8)]
    separate = [tuple(sum(a * b for a, b in zip(w, x)) for w in matrix) for x in rows]
    pooled = [tuple(sum(matrix[j][k] * rows[i][k] for k in range(3))
                    for j in range(2)) for i in range(8)]
    assert separate == pooled
    return {"exact_integer_rows": 8, "outputs_match": True}


def dag_excerpt():
    result = []
    for row in make_frontier(2, True):
        prefix = f"q{row.request}.t7.r{row.next_r}.l0"
        input_id = f"q{row.request}.ready_input_r{row.next_r}"
        qkv_id = prefix + ".qkv_rope"
        kv_key = f"q{row.request}/r{row.next_r}/l0/p7/committed_v1"
        common = {
            "request_id": row.request, "token_or_chunk_id": 7,
            "r": row.next_r, "l": 0, "owner": f"request_owner_{row.request % 2}",
            "layout": "complete_row_BF16", "scratch_version": "snapshot_v1",
            "estimated_cost": None, "resource_type": "symbolic_GPU_node_not_executed",
        }
        result.append({**common, "node_id": qkv_id, "operator": "QKV_then_RoPE",
                       "predecessors": [input_id], "weight_id": "physical_layer_0_QKV",
                       "kv_reads": [], "kv_writes": [kv_key]})
        result.append({**common, "node_id": prefix + ".attention",
                       "operator": "original_attention", "predecessors": [qkv_id],
                       "weight_id": None,
                       "kv_reads": [f"q{row.request}/r{row.next_r}/l0/p{j}/committed_v1"
                                    for j in range(8)], "kv_writes": []})
    return result


def main():
    start = utc_now()
    frontier_results = []
    for adaptive in (False, True):
        for c in (1, 2, 4, 8):
            rows = make_frontier(c, adaptive)
            assert check_frontier(rows, adaptive)
            # An alias-aware unrolled graph uses the same physical weight identifier.
            generic_groups = generic_unrolled_groups(rows)
            candidate_groups = groups(rows, False)
            assert generic_groups == list(candidate_groups.values())
            frontier_results.append({
                "mode": "original_gate_structural_control" if adaptive else "fixed_R4",
                "system_C": c,
                "next_depths": [row.next_r for row in rows],
                "ready_rows": len(rows),
                "depth_buckets": list(map(len, groups(rows, True).values())),
                "generic_union_sizes": list(map(len, generic_groups)),
                "candidate_union_equals_generic": True,
                "previous_gate_cdf": [float(gate_state(row.gate_hazards)[0]) for row in rows],
            })
    negatives = {
        "same_request_two_depths": must_reject([Row(0, 7, 1, 0, ()), Row(0, 7, 2, 1, ())], False),
        "future_recurrence_input": must_reject([Row(0, 7, 3, 1, ())], False),
        "future_original_gate": must_reject([Row(0, 7, 2, 1, ())], True),
        "continue_after_actual_exit": must_reject([Row(0, 7, 2, 1, (Fraction(3, 4),))], True),
    }
    ownership = []
    for p in (2, 4, 8):
        for c in (1, 2, 4, 8):
            counts = [sum(i % p == owner for i in range(c)) for owner in range(p)]
            remote = c - counts[0]
            ownership.append({
                "P": p, "system_C": c, "rows_per_owner": counts,
                "remote_rows_to_owner_0_pool": remote,
                "BF16_MLP_roundtrip_payload_bytes_per_layer": remote * 2 * 2048 * 2,
                "scope": "logical payload only, excludes headers, packing, sync and waiting",
            })
    artifact = {
        "candidate": "C16", "evidence_class": "CPU_ALGEBRA_AND_ADDRESS_CONTROLS",
        "start_utc": start, "finish_utc": utc_now(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "assertion_groups": 6,
        "frontier_cases": frontier_results, "negative_controls": negatives,
        "KV_arena": arena_control(), "dense_batch": dense_control(),
        "ownership_payload_screen": ownership, "DAG_excerpt": dag_excerpt(),
        "limitations": ["constructed state, no checkpoint weights", "no GPU execution",
                        "no model numerics, exit agreement or quality result",
                        "no latency estimate or resource-constrained full schedule"],
    }
    artifact["finish_utc"] = utc_now()
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
