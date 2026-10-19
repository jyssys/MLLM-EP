"""Analyze frozen component records. This program imports no GPU library.

The arithmetic control is a non-model BF16 projection. It is not a quality test.
"""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
TARGET_ROWS = {1, 2, 4, 8}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bf16_rne(value):
    """Round a finite binary32 value to BF16, with ties to even."""
    word = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = (word + 0x7FFF + ((word >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def numerical_control():
    activation = [1.0, 1.0, 1.0, 1.0]
    down_weights = [256.0, 1.0, -256.0, 0.0]
    products = [x * w for x, w in zip(activation, down_weights)]
    full = bf16_rne(sum(products))
    partials_real = [sum(products[:2]), sum(products[2:])]
    partials_bf16 = list(map(bf16_rne, partials_real))
    sharded = bf16_rne(sum(partials_bf16))
    assert sum(products) == sum(partials_real) == 1.0
    assert full == 1.0 and sharded == 0.0
    return {
        "evidence_class": "CPU_ALGEBRA_CONTROL",
        "scope": "Constructed non-model down projection. No model quality claim.",
        "activation": activation,
        "weights": down_weights,
        "real_full_and_sharded_sum": 1.0,
        "unrounded_shard_outputs": partials_real,
        "BF16_shard_outputs": partials_bf16,
        "full_output_BF16": full,
        "sharded_output_BF16": sharded,
        "claim": "A BF16 cast at each shard can change the final projection value.",
        "native_kernel_order_reproduction": False,
    }


def structural_rows():
    result = []
    D, I, R, element_bytes = 2048, 5632, 4, 2
    for size, L in [("1.4B", 24), ("2.6B", 48)]:
        mlp = 3 * D * I * element_bytes * L
        attention = 4 * D * D * element_bytes * L
        kv_per_token = 2 * L * R * 16 * 128 * element_bytes
        for P in [2, 4, 8]:
            result.append({
                "model": size, "P": P, "L": L, "R": R,
                "evidence_class": "ANALYTIC_ACCOUNTING",
                "physical_MLP_bytes_per_replica": mlp,
                "MLP_extra_bytes_per_rank_vs_TP": mlp * (P - 1) // P,
                "TP_core_bytes_per_rank": (mlp + attention) // P,
                "replicated_MLP_TP_attention_core_bytes_per_rank": mlp + attention // P,
                "four_materialized_MLP_copies_per_rank_bytes": R * mlp,
                "fixed_MLP_reductions_removed_per_token": L * R,
                "fixed_attention_reductions_retained_per_token": L * R,
                "full_MLP_logical_weight_operand_bytes_per_rank_per_R4_batch": R * mlp,
                "aggregate_MLP_dense_work_multiple_vs_TP": P,
                "full_MLP_FLOPs_per_ready_row_per_layer": 6 * D * I,
                "KV_bytes_per_cached_token_total_unchanged": kv_per_token,
                "KV_bytes_per_cached_token_per_rank_unchanged": kv_per_token // P,
                "fixed_ready_row_upper_bounds": {str(C): C for C in [1, 2, 4, 8]},
                "adaptive_ready_row_upper_bounds": {str(C): C for C in [1, 2, 4, 8]},
                "adaptive_note": "Actual continuation decisions can reduce this frontier. No oracle exit trace is assumed.",
                "excluded_capacity_bytes": "Embedding, LM head, norms, gate, allocator, graphs, communication buffers, scratch.",
                "HBM_traffic_measured": False,
                "P8_performance": "NOT_RUN",
            })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    start = utc()
    inputs, rows = [], []
    for path in sorted((ROOT / "outputs").glob("components_*graph_v1/components.json")):
        source = json.loads(path.read_text())
        inputs.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                       "source_UTC": source["UTC"],
                       "checkpoint_revision": source["checkpoint_revision"]})
        for value in source["results"]:
            local, tp = value["local"], value["TP"]
            a, b = local["median_critical_stream_ms"], tp["median_critical_stream_ms"]
            assert local["graph"] and tp["graph"]
            assert local["inner_repetitions"] == tp["inner_repetitions"] == 16
            rows.append({
                "source": str(path.relative_to(ROOT)), "model": value["model"],
                "P": value["P"], "rows": value["rows"],
                "target_low_C_shape": value["rows"] in TARGET_ROWS,
                "evidence_class": "CPU_ANALYSIS_OF_EXISTING_GPU_MICROBENCH",
                "full_local_critical_stream_median_ms": a,
                "TP_plus_reduction_critical_stream_median_ms": b,
                "TP_over_full_local_latency_ratio": b / a,
                "full_local_minus_TP_ms": a - b,
                "full_local_faster_at_median": a < b,
                "trials_per_arm": len(local["critical_stream_ms"]),
                "source_numerical_TP": value["numerical_TP"],
                "residency": local["residency"],
            })
    target = [r for r in rows if r["target_low_C_shape"]]
    assert len(target) == 16
    assert all(r["full_local_minus_TP_ms"] > 0 for r in target)
    result = {
        "schema_version": 1, "start_UTC": start, "end_UTC": utc(),
        "command": "python scripts/candidate_round6_replica_screen.py --output artifacts/candidate_round6_replica_screen_v1.json",
        "GPU_execution_this_run": False, "model_execution_this_run": False,
        "script_sha256": sha256(Path(__file__)), "input_records": inputs,
        "numerical_control": numerical_control(), "structural_accounting": structural_rows(),
        "component_comparisons": rows,
        "target_screen": {
            "comparisons": len(target), "full_local_faster_at_median": 0,
            "TP_over_full_local_latency_ratio_range": [
                min(r["TP_over_full_local_latency_ratio"] for r in target),
                max(r["TP_over_full_local_latency_ratio"] for r in target)],
            "scope": "Rows1/2/4/8, both Base checkpoints, P2/P4, physical layer0, frozen isolated operator records.",
            "full_stack_or_native_fused_kernel_disproof": False,
        },
        "outside_target_full_local_faster_shapes": [
            {k:r[k] for k in ["model", "P", "rows", "TP_over_full_local_latency_ratio"]}
            for r in rows if not r["target_low_C_shape"] and r["full_local_faster_at_median"]],
    }
    result["end_UTC"] = utc()
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        output = args.output.resolve()
        assert output.is_relative_to(ROOT / "artifacts")
        with output.open("x") as stream:
            stream.write(rendered)
    print(json.dumps({"start_UTC": result["start_UTC"], "end_UTC": result["end_UTC"],
                      "target_screen": result["target_screen"],
                      "outside_target_full_local_faster_shapes": result["outside_target_full_local_faster_shapes"],
                      "BF16_control": result["numerical_control"]}, indent=2))


if __name__ == "__main__":
    main()
