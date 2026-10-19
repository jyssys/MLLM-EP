#!/usr/bin/env python3
"""Deterministic attention algebra controls. No model or GPU execution."""

from datetime import datetime, timezone
import json
import math
import struct


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def mv(matrix, vector):
    return [dot(row, vector) for row in matrix]


def transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def rope(vector, angle):
    """Use Ouro's split-half rotation convention for one frequency pair."""
    assert len(vector) == 2
    a, b = vector
    c, s = math.cos(angle), math.sin(angle)
    return [a * c - b * s, b * c + a * s]


def softmax(values):
    scale = max(values)
    exps = [math.exp(value - scale) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def weighted_sum(weights, vectors):
    return [sum(weight * vector[k] for weight, vector in zip(weights, vectors)) for k in range(len(vectors[0]))]


def error(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def bf16(value):
    raw = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    rounded = (raw + 0x7FFF + ((raw >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def attention_control(xs, wk, wv, raw_queries, positions, query_position):
    heads = []
    summaries = []
    for key_weight, value_weight, raw_query in zip(wk, wv, raw_queries):
        query = rope(raw_query, query_position)
        keys = [rope(mv(key_weight, x), pos) for x, pos in zip(xs, positions)]
        values = [mv(value_weight, x) for x in xs]
        scores = [dot(query, key) / math.sqrt(len(query)) for key in keys]
        alpha = softmax(scores)
        original = weighted_sum(alpha, values)
        summary = weighted_sum(alpha, xs)
        reassociated = mv(value_weight, summary)
        summaries.append(summary)
        position_specific = [
            dot(mv(transpose(key_weight), rope(query, -pos)), x) / math.sqrt(len(query))
            for x, pos in zip(xs, positions)
        ]
        static_query = mv(transpose(key_weight), query)
        invalid_static_scores = [dot(static_query, x) / math.sqrt(len(query)) for x in xs]
        assert error(original, reassociated) < 1e-12
        assert error(scores, position_specific) < 1e-12
        heads.append({
            "scores_cached_K": scores,
            "scores_position_specific_reassociation": position_specific,
            "scores_invalid_static_absorption": invalid_static_scores,
            "score_position_specific_max_abs_error": error(scores, position_specific),
            "score_static_max_abs_error": error(scores, invalid_static_scores),
            "alpha": alpha,
            "output_cached_V": original,
            "output_value_reassociation": reassociated,
            "value_reassociation_max_abs_error": error(original, reassociated),
        })
    invalid_shared = mv(wv[1], summaries[0])
    shared_error = error(heads[1]["output_cached_V"], invalid_shared)
    return {
        "heads": heads,
        "head_specific_X_summaries": summaries,
        "invalid_shared_summary_head1_output": invalid_shared,
        "invalid_shared_summary_head1_max_abs_error": shared_error,
    }


def main():
    started = datetime.now(timezone.utc)
    xs = [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, -1.0, 1.0]]
    wk = [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
          [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]
    wv = [[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
          [[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]]]
    queries = [[2.0, 0.5], [1.0, -1.0]]
    no_rope = attention_control(xs, wk, wv, queries, [0.0, 0.0, 0.0], 0.0)
    assert all(head["score_static_max_abs_error"] < 1e-12 for head in no_rope["heads"])
    assert no_rope["invalid_shared_summary_head1_max_abs_error"] > 1e-3
    with_rope = attention_control(xs, wk, wv, queries, [0.0, 1.0, 2.0], 3.0)
    assert any(head["score_static_max_abs_error"] > 1e-3 for head in with_rope["heads"])

    rotation = [[0.0, -1.0], [1.0, 0.0]]
    weight = [[2.0, 0.0], [0.0, 1.0]]
    query, x = [1.0, 0.0], [1.0, 1.0]
    direct = dot(query, mv(rotation, mv(weight, x)))
    valid = dot(mv(transpose(weight), mv(transpose(rotation), query)), x)
    invalid = dot(mv(transpose(rotation), mv(transpose(weight), query)), x)
    assert direct == valid == -1.0 and invalid == -2.0

    # Stored BF16 projected V introduces an intermediate rounding boundary.
    value_weight = [[1.0, 1.0]]
    value_inputs = [[1.0, 1.0 / 256], [1.0, 3.0 / 512]]
    alphas = [0.5, 0.5]
    cached_values = [[bf16(v) for v in mv(value_weight, vector)] for vector in value_inputs]
    value_reference = bf16(weighted_sum(alphas, cached_values)[0])
    weighted_input = weighted_sum(alphas, value_inputs)
    value_reassociated = bf16(mv(value_weight, weighted_input)[0])
    assert value_reference == 1.0 and value_reassociated == 1.0078125

    # Key projection rounding also breaks unrestricted BF16 reassociation.
    key_weight = [[1.0, 1.0]]
    key_input, key_query = [1.0, 1.0 / 256], [1.5]
    cached_key = [bf16(v) for v in mv(key_weight, key_input)]
    key_reference = bf16(dot(key_query, cached_key))
    absorbed = mv(transpose(key_weight), key_query)
    key_reassociated = bf16(dot(absorbed, key_input))
    assert key_reference == 1.5 and key_reassociated == 1.5078125

    D, H, d, width = 2048, 16, 128, 2
    assert D == H * d
    work_rows = []
    for N in (1024, 4096, 8192, 16384):
        baseline = 4 * D * D + 2 * N * D
        no_rope_work = 4 * D * D + 2 * H * N * D
        key_plus_x = 4 * D * D + (H + 1) * N * D
        position_specific = (N + 3) * D * D + 2 * H * N * D
        key_remat = (N + 3) * D * D + (H + 1) * N * D
        work_rows.append({
            "N": N,
            "baseline_attention_multiplications": baseline,
            "X_only_no_RoPE_multiplications": no_rope_work,
            "X_only_no_RoPE_ratio": no_rope_work / baseline,
            "K_plus_X_value_reassociation_multiplications": key_plus_x,
            "K_plus_X_ratio": key_plus_x / baseline,
            "X_only_direct_position_specific_multiplications": position_specific,
            "X_only_direct_position_specific_ratio": position_specific / baseline,
            "X_only_K_rematerialization_value_reassociation_multiplications": key_remat,
            "X_only_K_rematerialization_value_reassociation_ratio": key_remat / baseline,
            "feature_shard_score_allreduce_FP32_logical_bytes": H * N * 4,
        })
    memory_rows = []
    for L in (24, 48):
        for P in (2, 4, 8):
            for N in (1024, 16384):
                baseline = 2 * L * 4 * N * D * width / P
                X_sharded = L * 4 * N * D * width / P
                X_replicated = L * 4 * N * D * width
                K_plus_X = L * 4 * N * D * width * (1 + 1 / P)
                memory_rows.append({
                    "L": L, "P": P, "N": N,
                    "KV_head_TP_GiB_per_rank": baseline / 2**30,
                    "X_feature_sharded_GiB_per_rank": X_sharded / 2**30,
                    "X_replicated_GiB_per_rank": X_replicated / 2**30,
                    "K_head_TP_plus_X_replicated_GiB_per_rank": K_plus_X / 2**30,
                    "replicated_X_to_KV_ratio": P / 2,
                    "K_plus_replicated_X_to_KV_ratio": (P + 1) / 2,
                })
    output = {
        "schema": "residual_input_reassociation_cpu_v1",
        "evidence_class": "CPU_ALGEBRA_CONTROL_AND_STRUCTURAL_ACCOUNTING",
        "run_started_UTC": started.isoformat(),
        "inputs": {"X": xs, "Wk": wk, "Wv": wv, "raw_queries": queries},
        "no_RoPE_positive_control": no_rope,
        "RoPE_position_specific_positive_and_static_negative": with_rope,
        "noncommutation": {"direct": direct, "valid_WT_RT_q": valid, "invalid_RT_WT_q": invalid},
        "BF16_value_reassociation_counterexample": {"W": value_weight, "X": value_inputs, "alpha": alphas, "cached_V": cached_values, "reference_output": value_reference, "reassociated_output": value_reassociated},
        "BF16_key_reassociation_counterexample": {"W": key_weight, "X": key_input, "q": key_query, "cached_K": cached_key, "reference_score": key_reference, "reassociated_score": key_reassociated},
        "BF16_caveat": "Explicit software output rounding, not an H100 kernel measurement or model-quality test.",
        "assertion_groups_passed": 6,
        "work_rows": work_rows,
        "work_boundary": "One new query at one logical layer. Includes Q,K,V,O projections or their stated replacements. Excludes softmax, RoPE/scaling arithmetic, norm, metadata, and runtime costs. Direct counts are not lower bounds for all algorithms.",
        "memory_rows": memory_rows,
        "memory_boundary": "Full depth-indexed R4 state. No adaptive slot reduction, metadata, scratch, weights, or allocator overhead.",
        "GPU_used": False,
        "model_weights_used": False,
        "model_quality_test": "NOT_RUN",
        "run_finished_UTC": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
