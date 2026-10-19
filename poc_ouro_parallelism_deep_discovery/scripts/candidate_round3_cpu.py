#!/usr/bin/env python3
"""Deterministic CPU algebra controls for two provisional execution proposals."""

from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct


def now():
    return datetime.now(timezone.utc).isoformat()


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def r3a_controls():
    x = (1.0, 3.0, -2.0, 0.5)
    gamma = (2.0, 0.5, -1.0, 1.5)
    matrix = ((1.0, 2.0, -0.5, 3.0), (0.5, -1.0, 2.0, -2.0))
    eps = 1e-6
    scalar = 1 / math.sqrt(dot(x, x) / len(x) + eps)
    normalized = tuple(g * a * scalar for g, a in zip(gamma, x))
    ordinary = [dot(row, normalized) for row in matrix]
    partials = [[sum(row[k] * gamma[k] * x[k] for k in tile)
                 for row in matrix] for tile in ((0, 1), (2, 3))]
    delayed = [scalar * sum(partial[j] for partial in partials) for j in range(2)]
    error = max(abs(a - b) for a, b in zip(ordinary, delayed))
    assert error < 1e-12

    # These BF16 inputs and weights are exact. Only the norm output cast changes.
    x_bf16 = (1.0, 3.0)
    weights = (3.0, -1.0)
    mean_square = f32(f32(f32(1.0) + f32(9.0)) / 2)
    scale = f32(1 / math.sqrt(f32(mean_square + f32(eps))))
    norm_bf16 = tuple(bf16(f32(a * scale)) for a in x_bf16)
    source = bf16(f32(dot(weights, norm_bf16)))
    deferred = bf16(f32(f32(dot(weights, x_bf16)) * scale))
    assert source != deferred

    # The first input tile is identical. A different unfinished tail changes it.
    first_coordinate = []
    for tail in (0.0, 3.0):
        s = 1 / math.sqrt((1 + tail * tail) / 2 + eps)
        first_coordinate.append(bf16(s))
    assert first_coordinate[0] != first_coordinate[1]
    return {
        "float64_positive_error": error,
        "float64_outputs": ordinary,
        "BF16_inputs": x_bf16,
        "BF16_weights": weights,
        "FP32_norm_scalar": scale,
        "BF16_norm_output": norm_bf16,
        "source_projected_output": source,
        "deferred_projected_output": deferred,
        "identical_prefix_different_tail_normalized_prefix": first_coordinate,
        "interpretation": "generic real-arithmetic identity, not unrestricted BF16 identity",
    }


def attention(query, keys, values):
    scores = [dot(query, key) for key in keys]
    maximum = max(scores)
    probabilities = [math.exp(score - maximum) for score in scores]
    total = sum(probabilities)
    return sum(p * v for p, v in zip(probabilities, values)) / total


def read_key(hidden, keys, values, position=7, backend="pinned_scalar_control"):
    def exact_bytes(items):
        return b"".join(struct.pack("<d", float(value)) for value in items)
    return (exact_bytes(hidden), tuple(exact_bytes(key) for key in keys),
            exact_bytes(values), position,
            "same_physical_weights", "same_mask", "same_precision", backend)


def r3b_controls():
    hidden = (1.0, 0.0)
    keys = ((0.0, 0.0), (0.0, 0.0))
    values_a = (0.0, 2.0)
    values_b = (0.0, 4.0)
    output_a = attention(hidden, keys, values_a)
    output_b = attention(hidden, keys, values_b)
    assert output_a != output_b
    state_a = read_key(hidden, keys, values_a)
    state_b = read_key(hidden, keys, values_b)
    assert state_a != state_b

    # A hash only selects candidates. Full comparison must reject collisions.
    deliberately_bad_hash = lambda state: 0
    assert deliberately_bad_hash(state_a) == deliberately_bad_hash(state_b)
    assert not (state_a == state_b)
    memo = {state_a: output_a}
    assert memo[state_a] == attention(hidden, keys, values_a)
    assert state_b not in memo
    assert read_key(hidden, keys, values_a, position=8) not in memo
    assert read_key(hidden, keys, values_a, backend="different_backend") not in memo
    assert read_key((0.0,), ((0.0,),), (0.0,)) != read_key((-0.0,), ((0.0,),), (0.0,))

    positional_keys = ((1.0, 0.0), (0.0, 1.0))
    pos0 = attention((1.0, 0.0), positional_keys, values_a)
    pos1 = attention((0.0, 1.0), positional_keys, values_a)
    assert pos0 != pos1

    # A constructed stationary fixed point allows reuse, but still needs writes.
    fixed_hidden = (2.0,)
    fixed_keys = ((0.0,),)
    fixed_values = (2.0,)
    fixed_key = read_key(fixed_hidden, fixed_keys, fixed_values)
    cached = {}
    committed = {}
    hits = 0
    for r in range(1, 5):
        if fixed_key in cached:
            result = cached[fixed_key]
            hits += 1
        else:
            result = (bf16(0.5 * fixed_hidden[0] + 1.0),)
            cached[fixed_key] = result
        assert result == fixed_hidden
        # Reproduce the skipped transition's writes in every distinct depth bank.
        committed[(r, "request0", "layer0", 7)] = (fixed_hidden, fixed_hidden)
    assert hits == 3 and len(committed) == 4

    cdf = Fraction(0)
    survival = Fraction(1)
    exit_depth = None
    cdfs = []
    for r in range(1, 5):
        hazard = Fraction(1, 4)
        cdf += hazard * survival
        survival *= 1 - hazard
        cdfs.append(float(cdf))
        if r >= 2 and cdf >= Fraction(1, 2):
            exit_depth = r
            break
    assert exit_depth == 3

    # A real-arithmetic geometric shortcut need not preserve the policy's doubles.
    geometric_counterexample = None
    for candidate in (0.1, 0.2, 0.3, 0.6, 0.7):
        hazard = f32(candidate)
        cdf_float, survival_float = 0.0, 1.0
        for _ in range(4):
            cdf_float += hazard * survival_float
            survival_float *= 1 - hazard
        shortcut = 1 - (1 - hazard) ** 4
        if shortcut < cdf_float:
            geometric_counterexample = {
                "hazard_float32": hazard, "sequential_CDF": cdf_float,
                "geometric_CDF": shortcut, "threshold": cdf_float,
                "sequential_reaches_threshold": True,
                "shortcut_reaches_threshold": False,
            }
            break
    assert geometric_counterexample is not None
    return {
        "equal_hidden_different_V_outputs": [output_a, output_b],
        "equal_raw_hidden_different_query_position_outputs": [pos0, pos1],
        "full_readset_positive_hit": True,
        "bad_hash_collision_rejected_by_full_key": True,
        "position_or_backend_change_rejects_hit": True,
        "signed_zero_bytes_remain_distinct": True,
        "stationary_constructed_fixed_point": {
            "computed_transitions": 1, "reused_transitions": hits,
            "distinct_committed_depth_writes": len(committed),
        },
        "same_hazard_original_policy_CDFs": cdfs,
        "same_hazard_original_policy_exit_depth": exit_depth,
        "geometric_rounding_counterexample": geometric_counterexample,
        "interpretation": "readset and side-effect controls, not a model fixed-point claim",
    }


def resource_counts():
    result = []
    for layers in (24, 48):
        for context in (1024, 4096, 8192):
            one_readset = 2 * layers * context * 2048 * 2
            result.append({"physical_layers": layers, "context": context,
                           "one_pass_historical_KV_bytes": one_readset,
                           "two_bank_byte_comparison_bytes": 2 * one_readset,
                           "current_KV_replay_bytes_per_pass": 2 * layers * 2048 * 2})
    return result


def main():
    start = now()
    result = {
        "proposals": ["R3-A", "R3-B"],
        "evidence_class": "DETERMINISTIC_CPU_ALGEBRA_ONLY",
        "start_utc": start,
        "R3-A": r3a_controls(), "R3-B": r3b_controls(),
        "structural_resource_counts": resource_counts(),
        "limitations": ["no checkpoint tensors", "no GPU execution",
                        "software BF16 rounding is not a CUDA kernel emulator",
                        "no model hit rate, quality, or timing evidence"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "finish_utc": now(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
