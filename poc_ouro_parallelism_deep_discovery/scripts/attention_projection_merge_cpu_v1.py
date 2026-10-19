"""Deterministic non-model controls for projected attention-state merging.

This program uses Python binary64 arithmetic and a software BF16 rounding helper.
It imports no tensor or GPU library.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def mv(matrix, vector):
    return [math.fsum(a * b for a, b in zip(row, vector)) for row in matrix]


def mm(a, b):
    return [[math.fsum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def max_error(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def bf16(value):
    word = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = (word + 0x7FFF + ((word >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def head_preserving_control():
    weights = [[1, 2, -1, 0], [0, 1, 1, 2], [2, 0, 1, -1], [-1, 1, 0, 1]]
    local_values = [[[1, 2], [-1, 3]], [[4, -2], [2, 1]]]
    masses = [[3, 1], [1, 3]]
    beta = [[masses[s][h] / sum(row[h] for row in masses) for h in range(2)] for s in range(2)]
    merged = [sum(beta[s][h] * local_values[s][h][d] for s in range(2))
              for h in range(2) for d in range(2)]
    reference = mv(weights, merged)
    projected = []
    for s in range(2):
        projected.append([mv([row[h * 2:(h + 1) * 2] for row in weights], local_values[s][h])
                          for h in range(2)])
    actual = [math.fsum(beta[s][h] * projected[s][h][d] for s in range(2) for h in range(2))
              for d in range(4)]
    assert max_error(actual, reference) < 1e-12
    return {"local_values": local_values, "positive_softmax_masses": masses,
            "per_head_merge_coefficients": beta, "projection": weights,
            "reference": reference, "projected_head_state_merge": actual,
            "max_abs_error": max_error(actual, reference),
            "scope": "One key per shard and head can realize each positive mass. No model tensors."}


def scalar_merge_counterexample():
    weights = [[1, 2], [3, 5]]
    beta = [0.75, 0.25]
    value = [1, 1]
    projected = mv(weights, value)
    target = mv(weights, [b * v for b, v in zip(beta, value)])
    scalar = math.fsum(x * y for x, y in zip(projected, target)) / math.fsum(x * x for x in projected)
    scalar_output = [scalar * x for x in projected]
    assert max_error(target, scalar_output) > 0.01
    inverse = [[-5, 2], [3, -1]]
    diagonal = [[0.75, 0], [0, 0.25]]
    basis_merge = mm(mm(weights, diagonal), inverse)
    recovered = mv(basis_merge, projected)
    assert max_error(recovered, target) < 1e-12
    return {"projection": weights, "first_shard_values": value, "second_shard_values": [0, 0],
            "first_shard_head_coefficients": beta, "first_shard_projected_state": projected,
            "reference": target, "best_L2_single_scalar": scalar,
            "single_scalar_output": scalar_output, "single_scalar_max_abs_error": max_error(target, scalar_output),
            "invertible_basis_positive_control": {"inverse": inverse, "basis_merge_matrix": basis_merge,
                                                   "recovered": recovered, "max_abs_error": max_error(recovered, target)},
            "scope": "Rejects one scalar merge after head mixing. The inverse-basis control prevents a universal impossibility claim."}


def singular_information_loss():
    weights = [[1, 1], [2, 2]]
    first_a, first_b = [1, -1], [0, 0]
    beta = [0.75, 0.25]
    projected_a, projected_b = mv(weights, first_a), mv(weights, first_b)
    target_a = mv(weights, [a * b for a, b in zip(first_a, beta)])
    target_b = mv(weights, [a * b for a, b in zip(first_b, beta)])
    assert projected_a == projected_b == [0.0, 0.0]
    assert target_a == [0.5, 1.0] and target_b == [0.0, 0.0]
    return {"projection": weights, "first_shard_case_A": first_a, "first_shard_case_B": first_b,
            "second_shard_both_cases": [0, 0], "same_head_coefficients": beta,
            "same_projected_state": projected_a, "required_A": target_a, "required_B": target_b,
            "scope": "Constructed rank-deficient matrix. No rank claim about checkpoint matrices."}


def rounding_boundary():
    v0, v1 = [1.0, 1.0], [1.0078125, 1.0]
    weights = [[1.0, -1.0]]
    merged = [(a + b) / 2 for a, b in zip(v0, v1)]
    rounded = list(map(bf16, merged))
    baseline = bf16(mv(weights, rounded)[0])
    early = bf16((mv(weights, v0)[0] + mv(weights, v1)[0]) / 2)
    assert baseline == 0.0 and early == 0.00390625
    return {"head_count": 1, "values": [v0, v1], "equal_softmax_coefficients": [0.5, 0.5],
            "projection": weights, "complete_attention_before_BF16_cast": merged,
            "complete_attention_after_BF16_cast": rounded, "source_order_BF16_output": baseline,
            "project_before_merge_BF16_output": early,
            "scope": "Software BF16 boundary control. No CUDA accumulation-order or quality claim."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    start = utc()
    result = {"schema_version": 1, "start_UTC": start,
              "evidence_class": "CPU_ALGEBRA_CONTROL", "model_execution": False, "GPU_execution": False,
              "head_preserving": head_preserving_control(),
              "scalar_merge": scalar_merge_counterexample(),
              "rank_deficient_collision": singular_information_loss(),
              "BF16_boundary": rounding_boundary(),
              "Ouro_structural_accounting": {
                  "H": 16, "d": 128, "D": 2048,
                  "ordinary_FP32_state_vector_bytes_per_row_per_shard": 16 * 128 * 4,
                  "head_preserving_projected_FP32_state_bytes_per_row_per_shard": 16 * 2048 * 4,
                  "FP32_LSE_bytes_per_row_per_shard": 16 * 4,
                  "state_vector_expansion_factor": 16,
                  "full_BF16_Wo_bytes_per_physical_layer": 2048 * 2048 * 2,
                  "full_Wo_FLOPs_per_ready_row": 2 * 2048 * 2048,
                  "all_physical_Wo_bytes_Base_1p4B": 24 * 2048 * 2048 * 2,
                  "all_physical_Wo_bytes_Base_2p6B": 48 * 2048 * 2048 * 2,
                  "direct_full_Wo_per_KV_shard_work_factor": {str(P): P for P in [2, 4, 8]},
                  "condition": "Direct plan has all heads and all output coordinates at every sequence shard. Tiled or 2D layouts need separate accounting.",
                  "HBM_traffic_or_latency_measured": False},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "end_UTC": utc()}
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        output = args.output.resolve()
        assert output.is_relative_to(ROOT / "artifacts")
        with output.open("x") as stream:
            stream.write(text)
    print(text)


if __name__ == "__main__":
    main()
