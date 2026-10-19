#!/usr/bin/env python3
"""Deterministic algebra controls. No model, Torch, GPU, or random tensors."""

from datetime import datetime, timezone
from fractions import Fraction as Q
import json
import math
import struct


def dot(a, b):
    return sum((Q(x) * Q(y) for x, y in zip(a, b)), Q(0))


def squared_norm(a):
    return dot(a, a)


def argmax(values):
    return max(range(len(values)), key=lambda index: (values[index], -index))


def bf16(value):
    """Round a finite binary32 value to BF16 with round-to-nearest-even."""
    raw = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    rounded = (raw + 0x7FFF + ((raw >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def certificate(weights, hidden, split, incumbent):
    """Certify strict real argmax with exact rational squared comparisons."""
    full = [dot(row, hidden) for row in weights]
    float_full = [sum(float(x) * float(y) for x, y in zip(row, hidden)) for row in weights]
    assert float_full == [float(value) for value in full]
    prefix = [dot(row[:split], hidden[:split]) for row in weights]
    tail_query_sq = squared_norm(hidden[split:])
    radius_sq = [squared_norm(row[split:]) * tail_query_sq for row in weights]
    bound_valid = all(
        (full[i] - prefix[i]) ** 2 <= radius_sq[i]
        for i in range(len(weights))
    )
    assert bound_valid
    tests = []
    for i in range(len(weights)):
        if i == incumbent:
            continue
        margin = full[incumbent] - prefix[i]
        passes = margin > 0 and margin * margin > radius_sq[i]
        tests.append({
            "row": i,
            "prefix_exact": str(prefix[i]),
            "tail_radius_squared_exact": str(radius_sq[i]),
            "incumbent_minus_prefix_exact": str(margin),
            "strict_prune": passes,
        })
    certified = all(item["strict_prune"] for item in tests)
    if certified:
        assert argmax(full) == incumbent
    uncertain = sum(not item["strict_prune"] for item in tests)
    dimensions = len(hidden)
    vocabulary = len(weights)
    return {
        "weights_exact": [[str(Q(x)) for x in row] for row in weights],
        "hidden_exact": [str(Q(x)) for x in hidden],
        "split": split,
        "incumbent": incumbent,
        "full_logits_exact": [str(x) for x in full],
        "full_logits_float64": float_full,
        "full_argmax_lowest_index": argmax(full),
        "bounds_valid_exact": bound_valid,
        "rows": tests,
        "strict_certificate": certified,
        "uncertain_competitors": uncertain,
        "projection_multiplications_before_fallback": dimensions + (vocabulary - 1) * split,
        "projection_multiplications_with_reused_prefix_fallback": dimensions + (vocabulary - 1) * split + uncertain * (dimensions - split),
        "dense_projection_multiplications": vocabulary * dimensions,
        "cost_exclusions": "Norm arithmetic, metadata reads, comparisons, packing, dispatch, and communication",
        "cost_interpretation": "Structural count for a hypothetical reused-prefix schedule. This harness computes all full oracle dots to test its bounds.",
    }


def main():
    started = datetime.now(timezone.utc)
    positive = certificate(
        [[10, 0, 0, 0], [1, 1, 1, 1], [-1, 1, -1, 1], [0, 0, 0, 0]],
        [1, 1, 1, 1], 1, 0,
    )
    assert positive["strict_certificate"]
    assert positive["projection_multiplications_with_reused_prefix_fallback"] == 7

    reversal = certificate([[1, 0], [0, 2]], [1, 1], 1, 0)
    assert not reversal["strict_certificate"]
    assert reversal["full_argmax_lowest_index"] == 1

    tie = certificate([[1, 0], [0, 1]], [1, 1], 1, 1)
    assert not tie["strict_certificate"]
    assert tie["full_argmax_lowest_index"] == 0

    near = certificate([[1, 0], [1, Q(1, 512)]], [1, 1], 1, 1)
    rounded = [bf16(Q(value)) for value in near["full_logits_exact"]]
    assert near["strict_certificate"] and argmax(rounded) == 0
    near["software_BF16_output_logits"] = rounded
    near["software_BF16_argmax"] = argmax(rounded)
    near["caveat"] = "This constructed output cast is not a measurement of an H100 GEMM."

    epsilon = Q(1, 1024)
    early = [Q(1), Q(1) - epsilon]
    later = [Q(1) - epsilon, Q(1)]
    identity = [[1, 0], [0, 1]]
    old_cert = certificate(identity, early, 1, 0)
    later_logits = [dot(row, later) for row in identity]
    cosine = float(dot(early, later)) / math.sqrt(float(squared_norm(early) * squared_norm(later)))
    assert old_cert["strict_certificate"] and argmax(later_logits) == 1
    drift = {
        "earlier_certificate": old_cert,
        "later_hidden_exact": [str(x) for x in later],
        "later_argmax": argmax(later_logits),
        "equal_hidden_norms": squared_norm(early) == squared_norm(later),
        "hidden_difference_squared_exact": str(squared_norm([a - b for a, b in zip(later, early)])),
        "cosine_float64": cosine,
        "scope": "Small hidden drift and a certificate for an earlier hidden do not certify a later hidden. This is not an Ouro trajectory."
    }

    gate_weight = [Q(1), Q(1, 4)]
    gate_hidden = [Q(2), Q(1)]
    gate_prefix = dot(gate_weight[:1], gate_hidden[:1])
    gate_radius = Q(1, 4)
    gate_full = dot(gate_weight, gate_hidden)
    assert gate_prefix - gate_radius > 0 and gate_full > 0
    gate = {
        "weights_exact": [str(x) for x in gate_weight],
        "hidden_exact": [str(x) for x in gate_hidden],
        "prefix_exact": str(gate_prefix),
        "tail_radius_exact": str(gate_radius),
        "full_gate_logit_exact": str(gate_full),
        "threshold_logit_exact": "0",
        "certifies_sigmoid_at_least_half": True,
        "scope": "One current affine gate decision with an already-ready hidden. Not a certificate for an unfinished core."
    }

    p_low, p_high, p_next = Q(1, 10), Q(2, 5), Q(1, 5)
    cdf_low = p_low + (1 - p_low) * p_next
    cdf_high = p_high + (1 - p_high) * p_next
    assert cdf_low < Q(1, 2) <= cdf_high
    state = {
        "R1_hazards_exact": [str(p_low), str(p_high)],
        "R1_exit_decision_for_both": "Continue, including when minimum exit depth is R2",
        "R2_hazard_exact": str(p_next),
        "R2_CDFs_exact": [str(cdf_low), str(cdf_high)],
        "q_exact": "1/2",
        "R2_exit_decisions": [False, True],
        "scope": "A Boolean current decision does not determine the survival state required by later gates. Conservative interval propagation remains possible."
    }

    probability = lambda a, b: 1.0 / (1.0 + math.exp(b - a))
    sampling = {
        "logits_A": [2, 0],
        "logits_B": [2, -100],
        "argmax_for_both": 0,
        "probability_token0_A": probability(2, 0),
        "probability_token0_B": probability(2, -100),
        "scope": "Matching greedy argmax does not preserve the full sampling distribution."
    }

    clusters = [[Q(-100), Q(102)], [Q(2), Q(2)]]
    centroids = [sum(cluster) / len(cluster) for cluster in clusters]
    selected_cluster = argmax(centroids)
    cluster_maxima = [max(cluster) for cluster in clusters]
    assert selected_cluster == 1 and argmax(cluster_maxima) == 0
    centroid_control = {
        "one_dimensional_clusters_exact": [[str(x) for x in cluster] for cluster in clusters],
        "query_exact": "1",
        "centroids_exact": [str(x) for x in centroids],
        "selected_cluster_by_centroid": selected_cluster,
        "winning_cluster_by_full_dot": argmax(cluster_maxima),
        "safe_center_plus_radius_bounds_exact": [str(max(cluster)) for cluster in clusters],
        "scope": "Centroid ordering alone does not certify arbitrary clusters. This is not a test of any released model or a full retrieval implementation."
    }

    output = {
        "schema": "certified_decision_cpu_v1",
        "evidence_class": "CPU_ALGEBRA_CONTROL",
        "inputs": "Tiny deterministic constructed rational tensors. No model weights.",
        "run_started_UTC": started.isoformat(),
        "positive_partial_head_certificate": positive,
        "partial_head_reversal_caught": reversal,
        "tie_requires_fallback": tie,
        "real_certificate_not_BF16_reference": near,
        "earlier_hidden_not_later_hidden": drift,
        "current_gate_certificate": gate,
        "gate_state_counterexample": state,
        "greedy_not_sampling": sampling,
        "centroid_order_not_certificate": centroid_control,
        "assertion_groups_passed": 9,
        "GPU_used": False,
        "model_quality_test": "NOT_RUN",
        "run_finished_UTC": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
