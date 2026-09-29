#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from poc_mllm_ep_communication_motivation.contract import (
    atomic_json,
    percentile,
    read_json,
)


def records(root: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted((root / "raw_stage").glob("stage.pid*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            result.extend(json.loads(line) for line in handle if line.strip())
    return result


def critical(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("wave"),
            row.get("condition"),
            row.get("case_key"),
            row.get("replicate"),
            row.get("iteration"),
            row.get("layer"),
        )
        grouped[key].append(row)
    return [max(group, key=lambda row: row["moe_ms"]) for group in grouped.values()]


def summary(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def stage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    critical_rows = critical(rows)
    totals = {
        name: sum(float(row[name]) for row in critical_rows)
        for name in ("dispatch_ms", "expert_execution_ms", "combine_ms", "moe_ms")
    }
    total = totals["moe_ms"]
    return {
        "critical_rank_invocations": len(critical_rows),
        "totals_ms": totals,
        "shares": {
            "dispatch": totals["dispatch_ms"] / total,
            "expert_execution": totals["expert_execution_ms"] / total,
            "combine": totals["combine_ms"] / total,
            "communication": (totals["dispatch_ms"] + totals["combine_ms"]) / total,
        },
        "per_invocation_ms": {
            name: summary([float(row[name]) for row in critical_rows])
            for name in totals
        },
        "aggregation": "component values on the rank with maximum measured MoE time per layer invocation",
    }


def paired_bootstrap(
    base: dict[int, float], candidate: dict[int, float], seed: int = 3407
) -> dict[str, Any]:
    keys = sorted(set(base) & set(candidate))
    deltas = [100.0 * (candidate[key] / base[key] - 1.0) for key in keys]
    if not deltas:
        return {"n": 0, "mean_percent": None, "ci95": [None, None]}
    rng = random.Random(seed)
    boot = []
    for _ in range(10000):
        boot.append(mean(rng.choice(deltas) for _ in deltas))
    return {
        "n": len(deltas),
        "mean_percent": mean(deltas),
        "ci95": [percentile(boot, 0.025), percentile(boot, 0.975)],
        "paired_percent_deltas": deltas,
    }


def q1_q2(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = records(root)
    full = [row for row in rows if row["condition"] == "FULL_REPLAY"]
    q1_prefill = stage_summary(full)
    by_case_condition: dict[str, dict[str, Any]] = {}
    manifest = read_json(root / "manifest.json")
    sizes = {
        f"q2::{r['size_class']}::{r['request_id']}": r for r in manifest["requests"]
    }
    for case_key, request in sizes.items():
        by_case_condition[case_key] = {"request": request, "conditions": {}}
        for condition in ("FULL_REPLAY", "VISUAL_ONLY", "TEXT_ONLY"):
            selected = [
                row
                for row in rows
                if row.get("case_key") == case_key and row["condition"] == condition
            ]
            by_case_condition[case_key]["conditions"][condition] = stage_summary(
                selected
            )
        # Sum the per-source-shard accounting over all full records of one
        # measured replicate/layer, then aggregate layers.  Pick replicate 0.
        direct = [
            row
            for row in full
            if row.get("case_key") == case_key
            and row.get("replicate") == 0
            and row.get("route_stats")
        ]
        totals = {
            name: defaultdict(int) for name in ("visual", "text", "special_other")
        }
        rank_totals = {
            name: {
                "pairs_per_rank": [0, 0, 0, 0],
                "dispatch_payload_bytes_per_rank": [0, 0, 0, 0],
                "combine_payload_bytes_per_rank": [0, 0, 0, 0],
            }
            for name in ("visual", "text", "special_other")
        }
        for row in direct:
            for name in totals:
                part = row["route_stats"][name]
                for field in (
                    "tokens",
                    "pairs",
                    "dispatch_payload_bytes",
                    "combine_payload_bytes",
                ):
                    totals[name][field] += int(part[field])
                for field in rank_totals[name]:
                    for rank, value in enumerate(part[field]):
                        rank_totals[name][field][rank] += int(value)
        vision = totals["visual"]
        text = totals["text"]
        other = totals["special_other"]
        total_tokens = vision["tokens"] + text["tokens"] + other["tokens"]
        total_dispatch = (
            vision["dispatch_payload_bytes"]
            + text["dispatch_payload_bytes"]
            + other["dispatch_payload_bytes"]
        )
        total_combine = (
            vision["combine_payload_bytes"]
            + text["combine_payload_bytes"]
            + other["combine_payload_bytes"]
        )
        by_case_condition[case_key]["direct_mixed_accounting"] = {
            "visual": dict(vision),
            "text": dict(text),
            "special_other": dict(other),
            "per_rank": rank_totals,
            "visual_token_fraction": vision["tokens"] / total_tokens,
            "visual_pair_fraction": vision["pairs"]
            / max(1, vision["pairs"] + text["pairs"] + other["pairs"]),
            "visual_dispatch_byte_fraction": vision["dispatch_payload_bytes"]
            / max(1, total_dispatch),
            "visual_combine_byte_fraction": vision["combine_payload_bytes"]
            / max(1, total_combine),
            "visual_pairs_per_token": vision["pairs"] / max(1, vision["tokens"]),
            "text_pairs_per_token": text["pairs"] / max(1, text["tokens"]),
            "accounting_scope": "48 layers x all TP/DP source shards; logical BF16 payload",
        }
    high_key = next(
        key for key, value in sizes.items() if value["size_class"] == "high"
    )
    high = by_case_condition[high_key]
    fractions = high["direct_mixed_accounting"]
    visual_frac = fractions["visual_dispatch_byte_fraction"]
    q2_verdict = (
        "VISUAL_PREFILL_COMMUNICATION_DOMINANT"
        if visual_frac >= 0.70
        else "VISUAL_PREFILL_COMMUNICATION_SUBSTANTIAL"
        if visual_frac >= 0.40
        else "NO_VISUAL_COMMUNICATION_DOMINANCE"
    )
    q2 = {
        "status": "MEASURED_TRUE_EP4",
        "cases": by_case_condition,
        "primary_high_case": high_key,
        "verdict": q2_verdict,
        "non_additivity_warning": "isolated visual/text latency is not summed to reconstruct full latency",
    }
    q1 = {"prefill": q1_prefill}
    return q1, q2


def paired_overhead(root: Path) -> dict[str, Any]:
    """Compare wrapper bookkeeping with a bypass in one loaded engine.

    Keeping both arms in the same process avoids the clock/cache confounding of
    cross-process hook/no-hook runs.  Iteration zero is prompt prefill and the
    last three external-DP padding iterations are excluded by using 1..31.
    """
    workers = [read_json(path) for path in sorted(root.glob("worker.dp*.json"))]
    walls: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for worker in workers:
        for wave in worker["waves"]:
            replicate = int(wave["replicate"])
            if replicate < 0:
                continue
            for step in wave["steps"]:
                iteration = int(step["iteration"])
                if 1 <= iteration <= 31:
                    walls[(wave["condition"], replicate, iteration)].append(
                        float(step["step_wall_ms"])
                    )
    means: dict[str, dict[int, float]] = defaultdict(dict)
    for condition in ("AUDIT_BYPASS", "DECODE_ONLY"):
        for replicate in (0, 1):
            means[condition][replicate] = mean(
                max(walls[(condition, replicate, iteration)])
                for iteration in range(1, 32)
            )
    comparison = paired_bootstrap(means["AUDIT_BYPASS"], means["DECODE_ONLY"])
    return {
        "bypass_tpot_ms": means["AUDIT_BYPASS"],
        "bookkeeping_tpot_ms": means["DECODE_ONLY"],
        "paired_overhead": comparison,
        "status": "PASS" if abs(comparison["mean_percent"]) <= 5 else "OVER_5_PERCENT",
        "evidence": "same-process paired/interleaved wrapper-bypass audit; fixed32 iterations 1..31",
    }


def q3(
    clean_root: Path, trace_root: Path, overhead_root: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    workers = [read_json(path) for path in sorted(clean_root.glob("worker.dp*.json"))]
    # Max of the two DP caller walls represents global wave progress.
    walls: dict[tuple[str, int, bool, int], list[float]] = defaultdict(list)
    injections: dict[tuple[str, int, bool], int | None] = {}
    for worker in workers:
        for wave in worker["waves"]:
            key0 = (wave["condition"], int(wave["replicate"]), bool(wave["trace"]))
            injections[key0] = wave["injection_prompt_tokens"]
            for step in wave["steps"]:
                walls[key0 + (int(step["iteration"]),)].append(
                    float(step["step_wall_ms"])
                )
    global_walls = {key: max(values) for key, values in walls.items()}
    clean_by_condition: dict[str, dict[int, float]] = defaultdict(dict)
    clean_details = {}
    for condition in (
        "DECODE_ONLY",
        "TEXT_PREFILL",
        "VISUAL_PREFILL",
        "TOKEN_MATCHED_TEXT",
    ):
        for replicate in range(3):
            sequence = [
                global_walls[(condition, replicate, False, iteration)]
                for iteration in range(1, 64)
            ]
            clean_by_condition[condition][replicate] = mean(sequence)
            clean_details[f"{condition}::{replicate}"] = {
                "tpot_ms": summary(sequence),
                # Injection is submitted at caller iteration 16. External-DP
                # async scheduling makes the physical mixed collective appear
                # at iteration 18 (text) or 19 (visual). Use the same bounded
                # response window, 16..20 inclusive, for every clean arm.
                "injection_window_ms": summary(sequence[15:20]),
                "injection_step_ms": global_walls[(condition, replicate, False, 16)],
                "maximum_stall_ms": max(sequence),
                "injection_prompt_tokens": injections[(condition, replicate, False)],
            }
    comparisons = {
        condition: paired_bootstrap(clean_by_condition["DECODE_ONLY"], values)
        for condition, values in clean_by_condition.items()
        if condition != "DECODE_ONLY"
    }
    itl_distributions = {}
    for condition in (
        "DECODE_ONLY",
        "TEXT_PREFILL",
        "VISUAL_PREFILL",
        "TOKEN_MATCHED_TEXT",
    ):
        values = [
            global_walls[(condition, replicate, False, iteration)]
            for replicate in range(3)
            for iteration in range(1, 64)
        ]
        itl_distributions[condition] = summary(values)
    peak_by_condition: dict[str, dict[int, float]] = defaultdict(dict)
    for key, detail in clean_details.items():
        condition, replicate = key.split("::")
        peak_by_condition[condition][int(replicate)] = float(
            detail["injection_window_ms"]["max"]
        )
    peak_comparisons = {
        condition: paired_bootstrap(peak_by_condition["DECODE_ONLY"], values)
        for condition, values in peak_by_condition.items()
        if condition != "DECODE_ONLY"
    }
    trace_rows = records(trace_root)
    stage_by_condition = {}
    mixed_iteration = {}
    for condition in (
        "DECODE_ONLY",
        "TEXT_PREFILL",
        "VISUAL_PREFILL",
        "TOKEN_MATCHED_TEXT",
    ):
        condition_rows = [row for row in trace_rows if row["condition"] == condition]
        if condition == "DECODE_ONLY":
            selected = [row for row in condition_rows if int(row["iteration"]) == 18]
            mixed_iteration[condition] = None
        else:
            iterations = sorted(
                {
                    int(row["iteration"])
                    for row in condition_rows
                    if row.get("phase") == "mixed"
                }
            )
            if not iterations:
                raise RuntimeError(
                    f"no physically observed mixed iteration for {condition}"
                )
            mixed_iteration[condition] = iterations
            selected = [
                row for row in condition_rows if int(row["iteration"]) in iterations
            ]
        stage_by_condition[condition] = stage_summary(selected)
    inflations = {}
    baseline_reference_stage = {}
    injected_route_summary = {}
    for condition, value in stage_by_condition.items():
        if condition == "DECODE_ONLY":
            continue
        target_iterations = set(mixed_iteration[condition])
        base_summary = stage_summary(
            [
                row
                for row in trace_rows
                if row["condition"] == "DECODE_ONLY"
                and int(row["iteration"]) in target_iterations
            ]
        )
        baseline_reference_stage[condition] = base_summary
        base = base_summary["totals_ms"]
        totals = value["totals_ms"]
        inflations[condition] = {
            name: 100.0 * (totals[name] / base[name] - 1.0) for name in base
        }
        inflations[condition]["communication_ms"] = (
            totals["dispatch_ms"] + totals["combine_ms"]
        )
        inflations[condition]["communication_inflation_percent"] = 100.0 * (
            (totals["dispatch_ms"] + totals["combine_ms"])
            / (base["dispatch_ms"] + base["combine_ms"])
            - 1.0
        )
        mixed_rows = [
            row
            for row in trace_rows
            if row["condition"] == condition
            and int(row["iteration"]) in target_iterations
            and row.get("route_stats")
        ]
        by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in mixed_rows:
            by_layer[int(row["layer"])].append(row)
        rank_dispatch = [0, 0, 0, 0]
        rank_combine = [0, 0, 0, 0]
        for row in mixed_rows:
            injected = row["route_stats"]["injected"]
            for rank, value in enumerate(injected["dispatch_payload_bytes_per_rank"]):
                rank_dispatch[rank] += int(value)
            for rank, value in enumerate(injected["combine_payload_bytes_per_rank"]):
                rank_combine[rank] += int(value)
        injected_route_summary[condition] = {
            "token_expert_pairs_across_48_layers": sum(
                int(row["route_stats"]["injected"]["pairs"]) for row in mixed_rows
            ),
            "logical_dispatch_payload_bytes_across_48_layers": sum(
                int(row["route_stats"]["injected"]["dispatch_payload_bytes"])
                for row in mixed_rows
            ),
            "logical_combine_payload_bytes_across_48_layers": sum(
                int(row["route_stats"]["injected"]["combine_payload_bytes"])
                for row in mixed_rows
            ),
            "injected_tokens_per_layer": max(
                (
                    sum(
                        int(row["route_stats"]["injected"]["tokens"])
                        for row in layer_rows
                    )
                    for layer_rows in by_layer.values()
                ),
                default=0,
            ),
            "visual_tokens_per_layer": max(
                (
                    sum(
                        int(row["route_stats"]["visual"]["tokens"])
                        for row in layer_rows
                    )
                    for layer_rows in by_layer.values()
                ),
                default=0,
            ),
            "logical_dispatch_payload_bytes_per_rank_across_48_layers": rank_dispatch,
            "logical_combine_payload_bytes_per_rank_across_48_layers": rank_combine,
            "peak_rank_dispatch_payload_bytes_across_48_layers": max(
                rank_dispatch, default=0
            ),
            "scope": "sum over 48 layers and all source shards; logical BF16 payload",
        }
    visual_cmp = comparisons["VISUAL_PREFILL"]
    visual_peak = peak_comparisons["VISUAL_PREFILL"]
    matched_peak = peak_comparisons["TOKEN_MATCHED_TEXT"]
    visual_comm = inflations["VISUAL_PREFILL"]["communication_inflation_percent"]
    visual_expert = inflations["VISUAL_PREFILL"]["expert_execution_ms"]
    # Interference is a one-iteration burst. Full fixed64 mean TPOT dilutes it
    # by ~63x, so the primary causal serving statistic is the frozen-window
    # peak ITL, with aggregate mean TPOT retained as a secondary observation.
    visual_supported = (
        visual_peak["mean_percent"] > 5
        and visual_peak["ci95"][0] > 0
        and visual_comm > 5
    )
    modality_specific = visual_peak["mean_percent"] > matched_peak["mean_percent"] + 3
    if visual_supported and modality_specific:
        verdict = "VISUAL_PREFILL_EP_COMM_INTERFERENCE_SUPPORTED"
    elif visual_supported:
        verdict = "PREFILL_EP_COMM_INTERFERENCE_GENERIC"
    elif visual_cmp["mean_percent"] > 5 and visual_expert > visual_comm:
        verdict = "PREFILL_INTERFERENCE_COMPUTE_DOMINATED"
    else:
        verdict = "NO_MEANINGFUL_PREFILL_DECODE_INTERFERENCE"
    q3_result = {
        "status": "MEASURED_TRUE_EP4",
        "clean_wave_details": clean_details,
        "mean_tpot_ms": {
            condition: mean(values.values())
            for condition, values in clean_by_condition.items()
        },
        "itl_ms": itl_distributions,
        "paired_tpot_inflation": comparisons,
        "paired_injection_window_peak_inflation": peak_comparisons,
        "trace_mixed_iteration": mixed_iteration,
        "trace_decode_only_matched_reference": baseline_reference_stage,
        "trace_injection_stage": stage_by_condition,
        "trace_injection_stage_inflation": inflations,
        "injected_prefill_route_summary": injected_route_summary,
        "timing_scope": "global step wall=max(DP0,DP1); iteration 0 prompt-prefill excluded",
        "injection_response_window": "caller iterations 16..20 inclusive; frozen from observed async pipeline lag, identical across clean conditions/repeats",
        "stage_scope": "shared DeepEP collective experienced by resident decode; no invalid row-wise time split",
        "verdict": verdict,
    }
    if overhead_root is not None:
        q3_result["instrumentation_overhead_audit"] = paired_overhead(overhead_root)
    return q3_result, {"visual_comm": visual_comm, "visual_expert": visual_expert}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q1-q2", type=Path, required=True)
    parser.add_argument("--q3", type=Path, required=True)
    parser.add_argument("--q3-trace", type=Path)
    parser.add_argument("--q3-paired-overhead", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    q1, q2 = q1_q2(args.q1_q2)
    trace_root = args.q3_trace or args.q3
    q3_result, _ = q3(args.q3, trace_root, args.q3_paired_overhead)
    decode_rows = [
        row
        for row in records(trace_root)
        if row["condition"] == "DECODE_ONLY" and row.get("phase") == "decode"
    ]
    q1["decode"] = stage_summary(decode_rows)
    max_comm = max(
        q1["prefill"]["shares"]["communication"],
        q1["decode"]["shares"]["communication"],
    )
    q1["verdict"] = (
        "COMMUNICATION_FIRST_CLASS_BOTTLENECK"
        if max_comm >= 0.35
        else "COMMUNICATION_MEANINGFUL_BUT_SECONDARY"
        if max_comm >= 0.20
        else "COMMUNICATION_MINOR"
    )
    p1 = q1["verdict"] != "COMMUNICATION_MINOR"
    p2 = q2["verdict"] != "NO_VISUAL_COMMUNICATION_DOMINANCE"
    p3 = q3_result["verdict"] in {
        "VISUAL_PREFILL_EP_COMM_INTERFERENCE_SUPPORTED",
        "PREFILL_EP_COMM_INTERFERENCE_GENERIC",
    }
    if (
        p1
        and p2
        and q3_result["verdict"] == "VISUAL_PREFILL_EP_COMM_INTERFERENCE_SUPPORTED"
    ):
        overall = "COMMUNICATION_ONLY_CHUNKING_MOTIVATED"
    elif p1 and p3:
        overall = "GENERIC_MOE_COMMUNICATION_MOTIVATION_ONLY"
    elif p1:
        overall = "COMMUNICATION_IMPORTANT_BUT_NO_INTERFERENCE"
    else:
        overall = "COMMUNICATION_DIRECTION_NOT_SUPPORTED"
    atomic_json(args.output / "q1_stage_breakdown.json", q1)
    atomic_json(args.output / "q2_modality_routes.json", q2)
    atomic_json(args.output / "q3_interference.json", q3_result)
    atomic_json(
        args.output / "final_verdict.json",
        {
            "q1": q1["verdict"],
            "q2": q2["verdict"],
            "q3": q3_result["verdict"],
            "promotion_gates": {"P1": p1, "P2": p2, "P3": p3},
            "overall": overall,
            "communication_only_chunking_implemented": False,
        },
    )
    for name, root in (
        ("q1_q2_execution_manifest", args.q1_q2),
        ("q3_clean_execution_manifest", args.q3),
        ("q3_trace_execution_manifest", trace_root),
        ("q3_overhead_execution_manifest", args.q3_paired_overhead),
    ):
        if root is not None and (root / "manifest.json").exists():
            atomic_json(args.output / f"{name}.json", read_json(root / "manifest.json"))


if __name__ == "__main__":
    main()
