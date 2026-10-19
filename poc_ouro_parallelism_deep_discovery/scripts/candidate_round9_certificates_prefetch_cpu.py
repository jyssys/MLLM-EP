#!/usr/bin/env python3
"""Constructed CPU controls. This file does not import a model or CUDA."""

import argparse
from bisect import bisect_left
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path


def utc():
    return datetime.now(timezone.utc).isoformat()


def pow2(e):
    return F(2**e) if e >= 0 else F(1, 2**(-e))


def word_value(word):
    """Decode a nonnegative finite BF16 word exactly."""
    assert 0 <= word <= 0x7F7F
    exponent, fraction = (word >> 7) & 255, word & 127
    return fraction * pow2(-133) if exponent == 0 else (128 + fraction) * pow2(exponent - 134)


def rounded_word(value):
    """Round a nonnegative rational in the finite tested range, ties to even."""
    assert 0 <= value <= word_value(0x7F7F)
    low, high = 0, 0x7F7F
    while low < high:
        middle = (low + high) // 2
        if word_value(middle) < value:
            low = middle + 1
        else:
            high = middle
    upper = low
    if word_value(upper) == value or upper == 0:
        return upper
    lower = upper - 1
    distances = value - word_value(lower), word_value(upper) - value
    return lower if distances[0] < distances[1] or (distances[0] == distances[1] and lower % 2 == 0) else upper


def cell(word):
    assert 0 < word < 0x7F7F
    center = word_value(word)
    return (word_value(word - 1) + center) / 2, (center + word_value(word + 1)) / 2


def row_case(name, kept_weights, kept_values, tail_weights, tail_values):
    zs, zd = sum(kept_weights), sum(tail_weights)
    selected = sum(w * v for w, v in zip(kept_weights, kept_values)) / zs
    dense = (zs * selected + sum(w * v for w, v in zip(tail_weights, tail_values))) / (zs + zd)
    # Treat exact tail mass as its upper bound. This isolates value/rounding requirements.
    vmax = max(abs(v) for v in tail_values)
    error_cap = zd * (vmax + abs(selected)) / zs
    word = rounded_word(selected)
    left, right = cell(word)
    certified = left < selected - error_cap and selected + error_cap < right
    if certified:
        assert rounded_word(dense) == word
    assert abs(dense - selected) <= error_cap
    return {
        "name": name, "kept_weights": list(map(str, kept_weights)),
        "kept_values": list(map(str, kept_values)), "tail_weights": list(map(str, tail_weights)),
        "tail_values": list(map(str, tail_values)), "selected_exact": str(selected),
        "dense_exact": str(dense), "error_cap_exact": str(error_cap),
        "selected_bf16_word": word, "dense_bf16_word": rounded_word(dense),
        "selected_bf16_value": str(word_value(word)),
        "dense_bf16_value": str(word_value(rounded_word(dense))),
        "strict_cell_certificate": certified,
        "contract": "exact rational attention with positive unnormalized masses, then one BF16 RNE cast; not native attention",
    }


def stage_prefix(source, expected_identity, count):
    if source["identity"] != expected_identity:
        raise ValueError("The cache generation or logical identity does not match.")
    if not 0 <= count <= source["committed_count"]:
        raise ValueError("The requested range includes an uncommitted position.")
    return tuple(source["words"][:count])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("The output exists. Use a new path.")
    start = utc()
    cases = [
        row_case("small_tail_positive_control", [F(1)], [F(1)], [pow2(-12)], [F(2)]),
        row_case("small_mass_large_value", [F(1)], [F(1)], [pow2(-20)], [pow2(20)]),
        row_case("zero_value_still_changes_denominator", [F(1)], [F(1)], [pow2(-8)], [F(0)]),
        row_case("arbitrarily_small_tail_at_rounding_boundary", [F(1), F(1)], [F(1), F(129, 128)], [pow2(-20)], [F(2)]),
        row_case("flat_scores_need_full_scan_under_this_bound", [F(1)], [F(1)], [F(63)], [F(0)]),
    ]
    assert cases[0]["strict_cell_certificate"]
    assert all(not c["strict_cell_certificate"] for c in cases[1:])
    assert all(c["selected_bf16_word"] != c["dense_bf16_word"] for c in cases[1:])
    assert word_value(rounded_word(F(257, 256))) == 1
    assert word_value(rounded_word(F(257, 256) + pow2(-30))) == F(129, 128)
    keys, query = [(F(1), F(-1)), (F(-1), F(1))], [F(1), F(1)]
    actual_max = max(sum(q * k for q, k in zip(query, key)) for key in keys)
    upper = sum(max(query[j] * min(k[j] for k in keys), query[j] * max(k[j] for k in keys)) for j in range(2))
    assert actual_max == 0 and upper == 2
    identity = ("request-A", 7, 3, 5, "head-shard-0")
    source = {"identity": identity, "committed_count": 2, "words": [(10, 20), (11, 21), None]}
    staged = stage_prefix(source, identity, 2)
    source["words"][2] = (12, 22)
    assert staged + (source["words"][2],) == tuple(source["words"])
    refused = []
    for label, changed_identity, count in [
        ("uncommitted_current_position", identity, 3),
        ("reused_page_generation", ("request-A", 8, 3, 5, "head-shard-0"), 2),
        ("wrong_recurrent_depth", ("request-A", 7, 2, 5, "head-shard-0"), 2),
    ]:
        try:
            stage_prefix(source, changed_identity, count)
        except ValueError:
            refused.append(label)
    assert len(refused) == 3
    rows = []
    for layers in (24, 48):
        for n in (1024, 8192):
            for concurrency in (1, 2, 4, 8):
                for p in (2, 4, 8):
                    one_layer = 2 * n * 2048 * 2 * concurrency // p
                    rows.append({"physical_layers": layers, "history_tokens_per_request": n,
                                 "system_C": concurrency, "P": p,
                                 "one_future_layer_all_C_history_bytes_per_rank": one_layer,
                                 "two_staging_buffers_bytes_per_rank": 2 * one_layer,
                                 "all_R4_history_bytes_per_rank": 4 * layers * one_layer,
                                 "P8_status": "STRUCTURAL_ONLY" if p == 8 else "NO_NEW_DEVICE_RUN"})
    assert len(rows) == 48
    summaries = [{"page_tokens": b, "bf16_KV_bytes": 2 * b * 2048 * 2,
                  "K_min_max_and_V_absmax_bytes": 3 * 2048 * 2,
                  "metadata_fraction": str(F(3, 2 * b))} for b in (16, 64)]
    result = {"schema": "candidate_round9_cpu_v1", "start_utc": start, "end_utc": utc(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "evidence_class": "CONSTRUCTED_CPU_ALGEBRA_AND_STRUCTURAL_ACCOUNTING",
              "model_execution": False, "gpu_execution": False, "native_parity_claim": False,
              "assertion_groups_passed": 6, "attention_cases": cases,
              "box_bound_control": {"actual_max": str(actual_max), "box_upper": str(upper)},
              "prefetch_control": {"historical_copy_before_query_is_legal": True,
                                   "no_query_is_evaluated": True, "refused": refused,
                                   "prefix_plus_fresh_current_matches_source": True},
              "capacity_rows": rows, "metadata_rows": summaries,
              "limits": ["No native exp, GEMM, attention order, FTZ, or error-envelope emulation.",
                         "Positive rational masses isolate the attention certificate; logits need not be BF16.",
                         "No cache, network, copy-engine, task-quality, or timing measurement."]}
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "assertion_groups_passed": 6,
                      "capacity_rows": len(rows), "attention_cases": len(cases)}))


if __name__ == "__main__":
    main()
