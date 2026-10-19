#!/usr/bin/env python3
"""CPU controls for certified value readiness, not a GPU or model emulator."""

from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import struct


def now():
    return datetime.now(timezone.utc).isoformat()


def power2(exponent):
    return F(2 ** exponent) if exponent >= 0 else F(1, 2 ** -exponent)


def round_bf16_real(value):
    """Round a finite real fraction to BF16 RNE, within this control's range."""
    value = F(value)
    if value == 0:
        return F(0)
    sign = -1 if value < 0 else 1
    value = abs(value)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < power2(exponent):
        exponent -= 1
    assert exponent <= 126, "overflow cases are outside this control"
    step = power2(max(exponent - 7, -133))
    ratio = value / step
    lower = ratio.numerator // ratio.denominator
    remainder = ratio - lower
    upper = remainder > F(1, 2) or (remainder == F(1, 2) and lower % 2 == 1)
    return sign * (lower + int(upper)) * step


def certify_interval(lower, upper):
    assert lower <= upper
    lo, hi = round_bf16_real(lower), round_bf16_real(upper)
    # Signed-zero, NaN, infinity and hardware FTZ are outside these controls.
    return lo if lo == hi and lo != 0 else None


def f32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def main():
    start = now()
    prefix, tail_bound = F(1), F(1, 1024)
    certified = certify_interval(prefix - tail_bound, prefix + tail_bound)
    assert certified == F(1)
    completions = [prefix + tail for tail in (-tail_bound, F(0), tail_bound)]
    assert all(round_bf16_real(value) == certified for value in completions)

    # The prefix is at the midpoint between 1 and the next BF16 value.
    midpoint, tiny_tail = F(257, 256), F(1, 65536)
    around_midpoint = certify_interval(midpoint - tiny_tail, midpoint + tiny_tail)
    assert around_midpoint is None
    midpoint_outputs = [round_bf16_real(midpoint - tiny_tail),
                        round_bf16_real(midpoint + tiny_tail)]
    assert midpoint_outputs[0] != midpoint_outputs[1]

    # BF16 inputs can have this product stream. Sequential FP32 is not an exact sum.
    terms = (2 ** 24, 1, -(2 ** 24))
    exact = sum(map(F, terms))
    sequential = 0.0
    for term in terms:
        sequential = f32(sequential + f32(term))
    assert exact == 1 and sequential == 0
    real_only_certificate = certify_interval(exact, exact)
    assert real_only_certificate != round_bf16_real(F(sequential))
    gamma3 = F(3, 2 ** 24 - 3)
    scalar_error_envelope = gamma3 * sum(abs(F(term)) for term in terms)
    assert abs(F(sequential) - exact) <= scalar_error_envelope
    assert certify_interval(exact - scalar_error_envelope,
                            exact + scalar_error_envelope) is None

    # One output coordinate is exact, but an unfinished coordinate controls RMSNorm.
    normalized_first = []
    for tail in (0.0, 2.0):
        variance = f32(f32(1.0 + f32(tail * tail)) / 2)
        scale = f32(1.0 / math.sqrt(f32(variance + f32(1e-6))))
        normalized_first.append(round_bf16_real(F(f32(scale))))
    assert normalized_first[0] != normalized_first[1]

    tile_predicates = [certified is not None, around_midpoint is not None]
    assert not all(tile_predicates)
    metadata = []
    for name, rows, columns in (("gate", 5632, 2048),
                                ("up", 5632, 2048),
                                ("down", 2048, 5632)):
        bounds_bytes = rows * (columns // 64) * 4
        weight_bytes = rows * columns * 2
        metadata.append({"projection": name, "K_chunk": 64,
                         "FP32_suffix_bounds_bytes": bounds_bytes,
                         "BF16_weight_bytes": weight_bytes,
                         "bound_bytes_over_weight_bytes": bounds_bytes / weight_bytes})

    artifact = {
        "hypothesis": "R4-A certified value readiness",
        "evidence_class": "CONSTRUCTED_CPU_ALGEBRA_CONTROLS",
        "start_utc": start,
        "finite_real_BF16_positive_control": {
            "prefix": float(prefix), "tail_bound": float(tail_bound),
            "committed_value": float(certified),
            "different_valid_completions": list(map(float, completions)),
            "all_round_to_same_value": True,
        },
        "midpoint_rejection": {
            "prefix": float(midpoint), "tail_bound": float(tiny_tail),
            "possible_BF16_values": list(map(float, midpoint_outputs)),
            "certificate_rejected": True,
        },
        "real_bound_is_not_FP32_backend_certificate": {
            "BF16_exact_product_terms": terms,
            "exact_real_sum": float(exact), "sequential_FP32_sum": sequential,
            "real_only_commit_would_be_wrong_for_sequential_FP32": True,
            "gamma3_scalar_error_bound": float(scalar_error_envelope),
            "widened_scalar_interval_rejects_commit": True,
        },
        "norm_barrier": {
            "certified_first_raw_coordinate": 1,
            "possible_unfinished_second_coordinates": [0, 2],
            "possible_BF16_first_normalized_coordinates": list(map(float, normalized_first)),
            "raw_coordinate_identity_does_not_certify_normalized_coordinate": True,
        },
        "whole_tile_guard": {"per_row_predicates": tile_predicates,
                             "entire_tile_ready": all(tile_predicates)},
        "structural_metadata_screen": metadata,
        "limitations": ["No checkpoint weights or captured activations",
                        "No GPU execution or H100 arithmetic emulation",
                        "The scalar gamma bound does not qualify a Tensor Core kernel",
                        "No certified-hit rate, workload quality, or performance result",
                        "No NaN, infinity, signed-zero, overflow, or FTZ certificate"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "finish_utc": now(),
    }
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
