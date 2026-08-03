"""Validate and summarize rank/layer CUDA-event records for Phase 1."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RANKS = tuple(range(4))
LAYERS = tuple(range(48))
REQUIRED_STAGES = {
    "decoder_layer",
    "attention_block",
    "qkv_projection",
    "attention_core",
    "attention_output_projection",
    "input_residual_rmsnorm",
    "post_attention_residual_rmsnorm",
    "router_projection",
    "router_topk",
    "dispatch_prepare_no_collective",
    "local_expert_execution",
    "local_finalize",
    "combine_tp_allreduce",
    "moe_layer",
}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p90": _percentile(values, 0.9),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            if record.get("duration_ms") is None or record.get("error") is not None:
                raise ValueError(f"invalid record at {path}:{line_number}: {record}")
            records.append(record)
    return records


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", required=True, type=Path)
    parser.add_argument("--layer-stages", type=Path)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    request_data = json.loads(args.requests.read_text(encoding="utf-8"))
    request_rows = request_data["iterations"]
    iterations = tuple(int(row["iteration_id"]) for row in request_rows)
    prompt_tokens = {
        int(row["iteration_id"]): int(row["prompt_token_count"])
        for row in request_rows
    }
    records = _read_jsonl(args.stages)

    grouped: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (
            int(record["iteration_id"]),
            int(record["layer"]),
            int(record["rank"]),
        )
        stage = str(record["stage"])
        if stage in grouped[key]:
            raise ValueError(f"duplicate stage {stage} for {key}")
        grouped[key][stage] = record

    expected_keys = {(iteration, layer, rank) for iteration in iterations for layer in LAYERS for rank in RANKS}
    if set(grouped) != expected_keys:
        missing = sorted(expected_keys - set(grouped))[:10]
        extra = sorted(set(grouped) - expected_keys)[:10]
        raise ValueError(f"rank/layer coverage mismatch; missing={missing}, extra={extra}")
    for key, stages in grouped.items():
        if set(stages) != REQUIRED_STAGES:
            raise ValueError(
                f"stage coverage mismatch for {key}; "
                f"missing={sorted(REQUIRED_STAGES - set(stages))}, "
                f"extra={sorted(set(stages) - REQUIRED_STAGES)}"
            )
        router = stages["router_topk"]
        expected_assignments = prompt_tokens[key[0]] * 8
        if int(router["hidden_tokens"]) != prompt_tokens[key[0]]:
            raise ValueError(f"router token count mismatch for {key}")
        if int(router["routed_token_assignments"]) != expected_assignments:
            raise ValueError(f"route assignment count mismatch for {key}")
        if sum(router["global_expert_token_counts"]) != expected_assignments:
            raise ValueError(f"global expert histogram mismatch for {key}")
        if len(router["local_expert_token_counts"]) != 32:
            raise ValueError(f"local expert histogram width mismatch for {key}")
        start = int(router["local_expert_start"])
        end = int(router["local_expert_end"])
        if router["local_expert_token_counts"] != router[
            "global_expert_token_counts"
        ][start:end]:
            raise ValueError(f"local expert histogram contents mismatch for {key}")

    for iteration in iterations:
        for layer in LAYERS:
            routers = [grouped[(iteration, layer, rank)]["router_topk"] for rank in RANKS]
            reference = routers[0]["global_expert_token_counts"]
            if any(router["global_expert_token_counts"] != reference for router in routers[1:]):
                raise ValueError(
                    f"rank routing disagreement for iteration={iteration}, layer={layer}"
                )
            reconstructed = [
                count
                for router in routers
                for count in router["local_expert_token_counts"]
            ]
            if reconstructed != reference:
                raise ValueError(
                    f"EP histogram reconstruction failed for iteration={iteration}, layer={layer}"
                )

    layer_grouped = grouped
    if args.layer_stages is not None:
        layer_grouped = defaultdict(dict)
        for record in _read_jsonl(args.layer_stages):
            key = (
                int(record["iteration_id"]),
                int(record["layer"]),
                int(record["rank"]),
            )
            stage = str(record["stage"])
            if stage in layer_grouped[key]:
                raise ValueError(f"duplicate lean stage {stage} for {key}")
            layer_grouped[key][stage] = record
        if set(layer_grouped) != expected_keys:
            raise ValueError("lean rank/layer coverage mismatch")
        for key, stages in layer_grouped.items():
            if set(stages) != {"decoder_layer", "router_topk"}:
                raise ValueError(f"lean stage coverage mismatch for {key}")
            if int(stages["router_topk"]["hidden_tokens"]) != prompt_tokens[key[0]]:
                raise ValueError(f"lean router token count mismatch for {key}")

    breakdown: list[dict[str, Any]] = []
    for iteration in iterations:
        for layer in LAYERS:
            by_rank = {rank: grouped[(iteration, layer, rank)] for rank in RANKS}
            layer_by_rank = {
                rank: layer_grouped[(iteration, layer, rank)] for rank in RANKS
            }

            def duration(rank: int, stage: str) -> float:
                return float(by_rank[rank][stage]["duration_ms"])

            decoder = {
                rank: float(layer_by_rank[rank]["decoder_layer"]["duration_ms"])
                for rank in RANKS
            }
            critical_rank = max(RANKS, key=decoder.__getitem__)
            attention = {rank: duration(rank, "attention_block") for rank in RANKS}
            norm_router = {
                rank: sum(
                    duration(rank, stage)
                    for stage in (
                        "input_residual_rmsnorm",
                        "post_attention_residual_rmsnorm",
                        "router_projection",
                        "router_topk",
                    )
                )
                for rank in RANKS
            }
            experts = {
                rank: duration(rank, "local_expert_execution") for rank in RANKS
            }
            combines = {
                rank: duration(rank, "combine_tp_allreduce") for rank in RANKS
            }

            t_layer = decoder[critical_rank]
            t_attention = max(attention.values())
            t_norm_router = max(norm_router.values())
            t_dispatch = 0.0
            t_expert = max(experts.values())
            expert_rank = max(RANKS, key=experts.__getitem__)
            t_combine = max(combines.values())
            nonexpert = t_attention + t_norm_router + t_dispatch
            t_optimistic = max(t_expert, nonexpert) + t_combine
            router = by_rank[critical_rank]["router_topk"]
            breakdown.append(
                {
                    "run_id": request_data["run_id"],
                    "iteration_id": iteration,
                    "layer": layer,
                    "critical_rank": critical_rank,
                    "critical_physical_gpu": layer_by_rank[critical_rank]["decoder_layer"]["physical_gpu"],
                    "t_layer_ms": t_layer,
                    "t_attention_ms": t_attention,
                    "t_qkv_max_ms": max(duration(rank, "qkv_projection") for rank in RANKS),
                    "t_attention_core_max_ms": max(duration(rank, "attention_core") for rank in RANKS),
                    "t_attention_output_max_ms": max(duration(rank, "attention_output_projection") for rank in RANKS),
                    "t_norm_router_ms": t_norm_router,
                    "t_input_norm_max_ms": max(duration(rank, "input_residual_rmsnorm") for rank in RANKS),
                    "t_post_attention_norm_max_ms": max(duration(rank, "post_attention_residual_rmsnorm") for rank in RANKS),
                    "t_router_projection_max_ms": max(duration(rank, "router_projection") for rank in RANKS),
                    "t_router_topk_max_ms": max(duration(rank, "router_topk") for rank in RANKS),
                    "t_dispatch_ms": t_dispatch,
                    "dispatch_basis": "structurally_absent_tp4_ep4_dp1",
                    "t_expert_max_ms": t_expert,
                    "expert_critical_rank": expert_rank,
                    "t_combine_ms": t_combine,
                    "t_moe_max_ms": max(duration(rank, "moe_layer") for rank in RANKS),
                    "expert_fraction": t_expert / t_layer,
                    "exposed_nonexpert_fraction": nonexpert / t_layer,
                    "t_optimistic_ms": t_optimistic,
                    "oracle_speedup": t_layer / t_optimistic,
                    "routed_token_assignments": int(router["routed_token_assignments"]),
                    "critical_rank_local_assignments": sum(router["local_expert_token_counts"]),
                    "max_local_expert_batch": int(router["max_local_expert_batch"]),
                }
            )

    metric_names = (
        "t_layer_ms",
        "t_attention_ms",
        "t_norm_router_ms",
        "t_dispatch_ms",
        "t_expert_max_ms",
        "t_combine_ms",
        "t_moe_max_ms",
        "expert_fraction",
        "exposed_nonexpert_fraction",
        "t_optimistic_ms",
        "oracle_speedup",
    )
    summary: list[dict[str, Any]] = []
    for layer in LAYERS:
        layer_rows = [row for row in breakdown if row["layer"] == layer]
        row: dict[str, Any] = {
            "scope": "layer",
            "layer": layer,
            "samples": len(layer_rows),
            "critical_rank_mode": Counter(
                int(value["critical_rank"]) for value in layer_rows
            ).most_common(1)[0][0],
        }
        for metric in metric_names:
            for stat_name, value in _stats(
                [float(item[metric]) for item in layer_rows]
            ).items():
                row[f"{metric}_{stat_name}"] = value
        summary.append(row)

    overall_by_iteration: list[dict[str, float]] = []
    for iteration in iterations:
        rows = [row for row in breakdown if row["iteration_id"] == iteration]
        t_layer = sum(float(row["t_layer_ms"]) for row in rows)
        t_optimistic = sum(float(row["t_optimistic_ms"]) for row in rows)
        nonexpert = sum(
            float(row["t_attention_ms"])
            + float(row["t_norm_router_ms"])
            + float(row["t_dispatch_ms"])
            for row in rows
        )
        overall_by_iteration.append(
            {
                "t_layer_ms": t_layer,
                "t_optimistic_ms": t_optimistic,
                "oracle_speedup": t_layer / t_optimistic,
                "exposed_nonexpert_fraction": nonexpert / t_layer,
            }
        )
    overall: dict[str, Any] = {
        "scope": "all_layers_per_request",
        "layer": "ALL",
        "samples": len(overall_by_iteration),
        "critical_rank_mode": "",
    }
    for metric in metric_names:
        values = [float(row[metric]) for row in overall_by_iteration if metric in row]
        for stat_name in ("median", "p90", "mean", "stdev"):
            overall[f"{metric}_{stat_name}"] = (
                _stats(values)[stat_name] if values else ""
            )
    summary.append(overall)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "layer_breakdown.csv", breakdown)
    _write_csv(args.output_dir / "summary.csv", summary)
    gate_path = args.output_dir / "gate.json"
    if gate_path.exists():
        raise FileExistsError(f"refusing to overwrite {gate_path}")
    speedup = float(overall["oracle_speedup_median"])
    exposed = float(overall["exposed_nonexpert_fraction_median"])
    gate = {
        "basis": "median of per-request sums across 48 transformer MoE layers; T_layer uses the optional lean pass when supplied",
        "oracle_speedup": speedup,
        "exposed_nonexpert_fraction": exposed,
        "speedup_threshold": 1.15,
        "exposed_nonexpert_threshold": 0.15,
        "numeric_gate_pass": speedup >= 1.15 and exposed >= 0.15,
        "dispatch_note": "TP4/EP4 with DP1 has no dispatch all-to-all; t_dispatch is structurally zero.",
        "estimate_note": "This is an optimistic oracle estimate, not measured overlap speedup.",
    }
    gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
