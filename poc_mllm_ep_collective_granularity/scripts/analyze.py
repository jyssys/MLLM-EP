#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from poc_mllm_ep_collective_granularity.contract import atomic_json, read_json

EXPERT_WEIGHT_BYTES = 9_437_184  # exact BF16 gate_up + down tensors


def pctile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summary(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "p90": pctile(values, 0.90),
        "p95": pctile(values, 0.95),
        "p99": pctile(values, 0.99),
        "max": max(values) if values else None,
    }


def bootstrap(values: list[float], seed: int = 3407) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "ci95": [None, None], "values": []}
    rng = random.Random(seed)
    draws = [mean(rng.choice(values) for _ in values) for _ in range(10_000)]
    return {
        "n": len(values),
        "mean": mean(values),
        "ci95": [pctile(draws, 0.025), pctile(draws, 0.975)],
        "values": values,
    }


def load_stage(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "raw_stage").glob("stage.pid*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def timeline_index(row: dict[str, Any]) -> int:
    """Prefer the caller-visible scheduler iteration over engine-internal numbering."""
    value = row.get("outer_iteration")
    return int(value) if value is not None else int(row["iteration"])


def critical_map(
    rows: list[dict[str, Any]], *, outer: bool = False
) -> dict[tuple[Any, ...], dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["condition"],
                int(row["replicate"]),
                timeline_index(row) if outer else int(row["iteration"]),
                int(row["layer"]),
            )
        ].append(row)
    return {
        key: max(group, key=lambda row: float(row["moe_ms"]))
        for key, group in grouped.items()
    }


def component_map(
    rows: list[dict[str, Any]], component: str, *, outer: bool = False
) -> dict[tuple[Any, ...], dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["condition"],
                int(row["replicate"]),
                timeline_index(row) if outer else int(row["iteration"]),
                int(row["layer"]),
            )
        ].append(row)
    return {
        key: max(group, key=lambda row: float(row[component]))
        for key, group in grouped.items()
    }


def injected_iterations(
    rows: list[dict[str, Any]], condition: str, *, internal: bool = False
) -> dict[int, list[int]]:
    result: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        if row["condition"] != condition or not row.get("route_stats"):
            continue
        if int(row["route_stats"].get("injected", {}).get("pairs", 0)) > 0:
            index = int(row["iteration"]) if internal else timeline_index(row)
            result[int(row["replicate"])].add(index)
    return {rep: sorted(values) for rep, values in result.items() if rep >= 0}


def route_accounting(
    rows: list[dict[str, Any]], condition: str, iterations: dict[int, list[int]]
) -> dict[str, Any]:
    selected = []
    for row in rows:
        rep = int(row["replicate"])
        if (
            row["condition"] == condition
            and rep in iterations
            and int(row["iteration"]) in iterations[rep]
            and row.get("route_stats")
        ):
            selected.append(row)
    by_rep_layer_chunk: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    hist = [0] * 128
    rank_dispatch = [0] * 4
    rank_combine = [0] * 4
    dispatch_bytes = combine_bytes = pairs = 0
    for row in selected:
        part = row["route_stats"]["injected"]
        pairs += int(part["pairs"])
        dispatch_bytes += int(part["dispatch_payload_bytes"])
        combine_bytes += int(part["combine_payload_bytes"])
        for index, value in enumerate(part["expert_histogram"]):
            hist[index] += int(value)
            if value:
                by_rep_layer_chunk[
                    (int(row["replicate"]), int(row["layer"]), int(row["iteration"]))
                ].add(index)
        for index, value in enumerate(part["dispatch_payload_bytes_per_rank"]):
            rank_dispatch[index] += int(value)
        for index, value in enumerate(part["combine_payload_bytes_per_rank"]):
            rank_combine[index] += int(value)
    amps = []
    repeated_active = []
    unique_active = []
    by_rep_layer: dict[tuple[int, int], list[set[int]]] = defaultdict(list)
    for (rep, layer, _), experts in by_rep_layer_chunk.items():
        by_rep_layer[(rep, layer)].append(experts)
    for chunks in by_rep_layer.values():
        repeated = sum(len(chunk) for chunk in chunks)
        union = len(set().union(*chunks)) if chunks else 0
        if union:
            amps.append(repeated / union)
            repeated_active.append(repeated)
            unique_active.append(union)
    total_hist = sum(hist)
    hhi = sum((value / total_hist) ** 2 for value in hist) if total_hist else None
    reps = max(1, len(iterations))
    return {
        "expert_reload_amp": summary(amps),
        "repeated_active_experts_per_layer": summary(
            [float(value) for value in repeated_active]
        ),
        "unique_active_experts_per_layer": summary(
            [float(value) for value in unique_active]
        ),
        "estimated_expert_weight_bytes_proxy_per_layer": summary(
            [float(value * EXPERT_WEIGHT_BYTES) for value in repeated_active]
        ),
        "expert_weight_bytes_per_expert": EXPERT_WEIGHT_BYTES,
        "logical_dispatch_bytes_mean_replicate": dispatch_bytes / reps,
        "logical_combine_bytes_mean_replicate": combine_bytes / reps,
        "routed_pairs_mean_replicate": pairs / reps,
        "dispatch_bytes_per_rank_mean_replicate": [
            value / reps for value in rank_dispatch
        ],
        "combine_bytes_per_rank_mean_replicate": [
            value / reps for value in rank_combine
        ],
        "routing_hhi": hhi,
        "structural_bytes_are_not_measured_hbm_traffic": True,
    }


def trace_metrics(root: Path, condition: str = "VISUAL_PREFILL") -> dict[str, Any]:
    rows = load_stage(root)
    dispatch_map = component_map(rows, "dispatch_ms")
    combine_map = component_map(rows, "combine_ms")
    expert_map = component_map(rows, "expert_execution_ms")
    moe_map = critical_map(rows)
    baseline_dispatch_outer = component_map(rows, "dispatch_ms", outer=True)
    baseline_expert_outer = component_map(rows, "expert_execution_ms", outer=True)
    baseline_moe_outer = critical_map(rows, outer=True)
    internal_iterations = injected_iterations(rows, condition, internal=True)
    outer_iterations = injected_iterations(rows, condition)
    per_rep_cidd = []
    per_rep_expert = []
    per_rep_moe = []
    per_rep_candidate_expert = []
    per_rep_baseline_expert = []
    per_rep_candidate_moe = []
    per_rep_baseline_moe = []
    per_rep_dispatch_mncd = []
    per_rep_combine_mncd = []
    per_rep_dispatch_mean = []
    per_rep_combine_mean = []
    per_rep_max_iteration_cidd = []
    all_iteration_cidd = []
    isolated_dispatch = []
    per_layer_diffs: dict[int, list[float]] = defaultdict(list)
    for rep, active_iterations in sorted(internal_iterations.items()):
        cidd_values = []
        expert_total = moe_total = 0.0
        candidate_expert_total = baseline_expert_total = 0.0
        candidate_moe_total = baseline_moe_total = 0.0
        dispatch_values = []
        combine_values = []
        cidd_by_outer: dict[int, float] = defaultdict(float)
        for iteration in active_iterations:
            iteration_cidd = []
            iteration_outer = None
            for layer in range(48):
                key = (condition, rep, iteration, layer)
                candidate_dispatch = dispatch_map.get(key)
                candidate_combine = combine_map.get(key)
                candidate_expert = expert_map.get(key)
                candidate_moe = moe_map.get(key)
                if candidate_dispatch is None:
                    continue
                iteration_outer = timeline_index(candidate_dispatch)
                base_key = ("DECODE_ONLY", rep, iteration_outer, layer)
                baseline_dispatch = baseline_dispatch_outer.get(base_key)
                baseline_expert = baseline_expert_outer.get(base_key)
                baseline_moe = baseline_moe_outer.get(base_key)
                if any(
                    value is None
                    for value in (
                        candidate_dispatch,
                        baseline_dispatch,
                        candidate_combine,
                        candidate_expert,
                        baseline_expert,
                        candidate_moe,
                        baseline_moe,
                    )
                ):
                    continue
                delta = float(candidate_dispatch["dispatch_ms"]) - float(
                    baseline_dispatch["dispatch_ms"]
                )
                cidd_values.append(delta)
                iteration_cidd.append(delta)
                per_layer_diffs[layer].append(delta)
                isolated_dispatch.append(float(baseline_dispatch["dispatch_ms"]))
                dispatch_values.append(float(candidate_dispatch["dispatch_ms"]))
                combine_values.append(float(candidate_combine["combine_ms"]))
                expert_total += float(candidate_expert["expert_execution_ms"]) - float(
                    baseline_expert["expert_execution_ms"]
                )
                candidate_expert_total += float(candidate_expert["expert_execution_ms"])
                baseline_expert_total += float(baseline_expert["expert_execution_ms"])
                moe_total += float(candidate_moe["moe_ms"]) - float(
                    baseline_moe["moe_ms"]
                )
                candidate_moe_total += float(candidate_moe["moe_ms"])
                baseline_moe_total += float(baseline_moe["moe_ms"])
            if iteration_cidd:
                iteration_total = sum(iteration_cidd)
                all_iteration_cidd.append(iteration_total)
                assert iteration_outer is not None
                cidd_by_outer[iteration_outer] += iteration_total
        if cidd_values:
            per_rep_cidd.append(mean(cidd_values))
            per_rep_expert.append(expert_total)
            per_rep_moe.append(moe_total)
            per_rep_candidate_expert.append(candidate_expert_total)
            per_rep_baseline_expert.append(baseline_expert_total)
            per_rep_candidate_moe.append(candidate_moe_total)
            per_rep_baseline_moe.append(baseline_moe_total)
            per_rep_dispatch_mncd.append(max(dispatch_values))
            per_rep_combine_mncd.append(max(combine_values))
            per_rep_dispatch_mean.append(mean(dispatch_values))
            per_rep_combine_mean.append(mean(combine_values))
            per_rep_max_iteration_cidd.append(max(cidd_by_outer.values()))
    route = route_accounting(rows, condition, internal_iterations)
    request_walls = []
    for path in sorted(root.glob("worker.dp*.json")):
        worker = read_json(path)
        for wave in worker["waves"]:
            if wave["condition"] != condition or int(wave["replicate"]) < 0:
                continue
            begin = wave.get("injection_submit_ns")
            end = wave.get("injection_finish_ns")
            if begin is not None and end is not None:
                request_walls.append((int(end) - int(begin)) / 1e6)
    repeated_mean = route["repeated_active_experts_per_layer"]["mean"]
    pairs_per_rep = route["routed_pairs_mean_replicate"]
    route.update(
        {
            "injected_iterations": outer_iterations,
            "injected_outer_iterations": outer_iterations,
            "injected_internal_iterations": internal_iterations,
            "chunk_count": summary(
                [float(len(value)) for value in internal_iterations.values()]
            ),
            "decode_isolated_dispatch_ms_per_layer": summary(isolated_dispatch),
            "cidd_ms_per_layer": bootstrap(per_rep_cidd),
            "cidd_ms_per_victim_iteration": bootstrap(per_rep_max_iteration_cidd),
            "all_injected_iteration_cidd_ms": summary(all_iteration_cidd),
            "mncd_dispatch_ms": summary(per_rep_dispatch_mncd),
            "mncd_combine_ms": summary(per_rep_combine_mncd),
            "mixed_dispatch_ms_per_layer": bootstrap(per_rep_dispatch_mean),
            "mixed_combine_ms_per_layer": bootstrap(per_rep_combine_mean),
            "incremental_expert_execution_ms_per_request": bootstrap(per_rep_expert),
            "incremental_moe_ms_per_request": bootstrap(per_rep_moe),
            "mixed_expert_execution_ms_per_request": bootstrap(
                per_rep_candidate_expert
            ),
            "aligned_decode_expert_execution_ms_per_request": bootstrap(
                per_rep_baseline_expert
            ),
            "mixed_moe_ms_per_request": bootstrap(per_rep_candidate_moe),
            "aligned_decode_moe_ms_per_request": bootstrap(per_rep_baseline_moe),
            "injected_prefill_wall_ms": summary(request_walls),
            "expert_paths_per_active_expert": (
                pairs_per_rep / (48 * repeated_mean) if repeated_mean else None
            ),
            "hbm_dram_profiler_lane": "NOT_RUN_STRUCTURAL_WEIGHT_BYTES_ONLY",
            "cidd_by_layer_ms": {
                str(layer): summary(values) for layer, values in per_layer_diffs.items()
            },
            "cidd_definition": (
                "critical-rank DeepEP dispatch completion for the same resident decode "
                "wave in the physical mixed collective minus the aligned isolated-decode "
                "dispatch; reported per MoE layer"
            ),
            "cwt_status": "UNAVAILABLE_NO_RELIABLE_GPU_READY_TIMESTAMP",
        }
    )
    return route


def clean_walls(root: Path) -> dict[tuple[str, int, int], float]:
    values: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for path in sorted(root.glob("worker.dp*.json")):
        worker = read_json(path)
        for wave in worker["waves"]:
            rep = int(wave["replicate"])
            if rep < 0:
                continue
            for step in wave["steps"]:
                values[(wave["condition"], rep, int(step["outer_iteration"]))].append(
                    float(step["step_wall_ms"])
                )
    return {key: max(samples) for key, samples in values.items()}


def clean_metrics(
    root: Path,
    event_iterations: dict[int, list[int]],
    condition: str = "VISUAL_PREFILL",
) -> dict[str, Any]:
    walls = clean_walls(root)
    starts = [min(values) for values in event_iterations.values() if values]
    alignment_spread = max(starts) - min(starts) if starts else None
    alignment_status = (
        "PASS"
        if alignment_spread is not None and alignment_spread <= 4
        else "UNRESOLVED_ASYNC_SCHEDULER_DRIFT"
    )
    reps = sorted(
        rep
        for rep, values in event_iterations.items()
        if values and ("DECODE_ONLY", rep, values[0]) in walls
    )
    response_by_rep: dict[int, list[int]] = {}
    peak_vsel = []
    peak_vsel_pct = []
    peak_inflation = []
    base_peak = []
    candidate_peak = []
    all_vsel = []
    curve: dict[int, list[float]] = defaultdict(list)
    base_near = []
    condition_near = []
    for rep in reps:
        first = min(event_iterations[rep])
        last = max(event_iterations[rep])
        response = list(range(first, last + 3))
        response_by_rep[rep] = response
        vsel = []
        for iteration in response:
            key = (condition, rep, iteration)
            base_key = ("DECODE_ONLY", rep, iteration)
            if key not in walls or base_key not in walls:
                continue
            delta = walls[key] - walls[base_key]
            vsel.append((delta, walls[base_key], iteration))
            all_vsel.append(delta)
        if not vsel:
            continue
        delta, base_at_peak, _ = max(vsel, key=lambda row: row[0])
        peak_vsel.append(delta)
        peak_vsel_pct.append(100.0 * delta / base_at_peak)
        paired_iterations = [
            iteration
            for iteration in response
            if ("DECODE_ONLY", rep, iteration) in walls
            and (condition, rep, iteration) in walls
        ]
        bp = max(
            walls[("DECODE_ONLY", rep, iteration)] for iteration in paired_iterations
        )
        cp = max(walls[(condition, rep, iteration)] for iteration in paired_iterations)
        base_peak.append(bp)
        candidate_peak.append(cp)
        peak_inflation.append(100.0 * (cp / bp - 1.0))
        for relative in range(-5, 6):
            iteration = first + relative
            key = (condition, rep, iteration)
            base_key = ("DECODE_ONLY", rep, iteration)
            if key in walls and base_key in walls:
                curve[relative].append(walls[key] - walls[base_key])
                base_near.append(walls[base_key])
                condition_near.append(walls[key])
    base_median = median(base_near) if base_near else 0.0
    if base_peak and median(base_peak) < 5.0:
        alignment_status = "UNRESOLVED_ASYNC_LOW_BASELINE_DENOMINATOR"
    slo = {}
    for ratio in (1.10, 1.20, 1.30):
        threshold = base_median * ratio
        slo[f"{ratio:.2f}x"] = {
            "threshold_ms": threshold,
            "violation_fraction": (
                sum(value > threshold for value in condition_near) / len(condition_near)
                if condition_near
                else None
            ),
        }
    return {
        "condition": condition,
        "alignment_status": alignment_status,
        "first_mixed_iteration_spread": alignment_spread,
        "event_iterations": event_iterations,
        "response_window_iterations": response_by_rep,
        "peak_vsel_ms": bootstrap(peak_vsel),
        "peak_vsel_percent": bootstrap(peak_vsel_pct),
        "all_response_vsel_ms": summary(all_vsel),
        "decode_only_response_peak_ms": summary(base_peak),
        "condition_response_peak_ms": summary(candidate_peak),
        "response_peak_inflation_percent": bootstrap(peak_inflation),
        "event_aligned_vsel_curve_ms": {
            str(relative): summary(values) for relative, values in curve.items()
        },
        "decode_slo": slo,
        "global_step_wall_scope": "maximum of the two physical DP caller walls",
    }


def layered_replay(full_trace_root: Path, full_clean: dict[str, Any]) -> dict[str, Any]:
    rows = load_stage(full_trace_root)
    dispatch_map = component_map(rows, "dispatch_ms")
    moe_map = critical_map(rows)
    baseline_dispatch_outer = component_map(rows, "dispatch_ms", outer=True)
    baseline_moe_outer = critical_map(rows, outer=True)
    iterations = injected_iterations(rows, "VISUAL_PREFILL", internal=True)
    stage_cidd = []
    stage_moe_excess = []
    for rep, active in iterations.items():
        if not active:
            continue
        iteration = active[0]
        for start in range(0, 48, 12):
            dispatch_delta = moe_delta = 0.0
            valid = True
            for layer in range(start, start + 12):
                key = ("VISUAL_PREFILL", rep, iteration, layer)
                candidate_dispatch = dispatch_map.get(key)
                candidate_moe = moe_map.get(key)
                if candidate_dispatch is None:
                    valid = False
                    break
                outer_iteration = timeline_index(candidate_dispatch)
                base_key = ("DECODE_ONLY", rep, outer_iteration, layer)
                baseline_dispatch = baseline_dispatch_outer.get(base_key)
                baseline_moe = baseline_moe_outer.get(base_key)
                if any(
                    value is None
                    for value in (
                        candidate_dispatch,
                        baseline_dispatch,
                        candidate_moe,
                        baseline_moe,
                    )
                ):
                    valid = False
                    break
                dispatch_delta += float(candidate_dispatch["dispatch_ms"]) - float(
                    baseline_dispatch["dispatch_ms"]
                )
                moe_delta += float(candidate_moe["moe_ms"]) - float(
                    baseline_moe["moe_ms"]
                )
            if valid:
                stage_cidd.append(dispatch_delta)
                stage_moe_excess.append(moe_delta)
    baseline_wall = float(full_clean["decode_only_response_peak_ms"]["median"] or 0.0)
    peak_moe = max(stage_moe_excess, default=0.0)
    result = {
        "baseline_type": "LAYERED_EP_PHYSICAL_REPLAY",
        "stage_count": 4,
        "layers_per_stage": 12,
        "expert_reload_amp": 1.0,
        "cidd_ms_per_stage": summary(stage_cidd),
        "cidd_ms_primary": max(stage_cidd, default=None),
        "stage_moe_excess_ms": summary(stage_moe_excess),
        "response_peak_inflation_percent_replay": (
            100.0 * peak_moe / baseline_wall if baseline_wall else None
        ),
        "semantics": (
            "four contiguous twelve-layer groups; each active group reuses the measured "
            "full-prompt TRUE-EP4 DeepEP collective for those layers while the other "
            "layers retain isolated decode measurements"
        ),
        "native_e2e_claim": False,
    }
    prior = Path(
        "poc_mllm_ep_communication_motivation/artifacts/results/q2_modality_routes.json"
    )
    if prior.exists():
        cases = read_json(prior)["cases"]
        result["visual_size_sweep"] = {
            value["request"]["size_class"]: {
                "prompt_tokens": value["request"]["prompt_token_count"],
                "full_prompt_dispatch_ms_mean": value["conditions"]["FULL_REPLAY"][
                    "per_invocation_ms"
                ]["dispatch_ms"]["mean"],
                "evidence": "REUSED_MEASURED_TRUE_EP4",
            }
            for value in cases.values()
        }
    return result


def moe_only_replay_metrics(root: Path) -> dict[str, Any]:
    rows = load_stage(root)
    condition_rep_means: dict[str, dict[str, dict[int, float]]] = {}
    result: dict[str, Any] = {
        "status": "MEASURED_TRUE_EP4_EXACT_HIDDEN_ROUTE_REPLAY",
        "output_used_by_model": False,
    }
    for condition in (
        "VISUAL_PREFILL_MOE_REPLAY",
        "TOKEN_MATCHED_TEXT_MOE_REPLAY",
    ):
        selected = [
            row
            for row in rows
            if row.get("condition") == condition and int(row["replicate"]) >= 0
        ]
        per_rep: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            grouped[(int(row["replicate"]), int(row["layer"]))].append(row)
        for (rep, _), group in grouped.items():
            for component in (
                "dispatch_ms",
                "expert_execution_ms",
                "combine_ms",
                "moe_ms",
            ):
                per_rep[rep][component].append(
                    max(float(row[component]) for row in group)
                )
        condition_result: dict[str, Any] = {}
        condition_rep_means[condition] = {}
        for component in (
            "dispatch_ms",
            "expert_execution_ms",
            "combine_ms",
            "moe_ms",
        ):
            by_rep = {rep: mean(values[component]) for rep, values in per_rep.items()}
            means = list(by_rep.values())
            maxima = [max(values[component]) for values in per_rep.values()]
            condition_result[f"{component}_per_layer"] = bootstrap(means)
            condition_result[f"{component}_mncd"] = summary(maxima)
            condition_rep_means[condition][component] = by_rep
        condition_result["layers_per_replicate"] = {
            str(rep): len(values["moe_ms"]) for rep, values in per_rep.items()
        }
        result[condition] = condition_result
    visual = result["VISUAL_PREFILL_MOE_REPLAY"]
    text = result["TOKEN_MATCHED_TEXT_MOE_REPLAY"]
    result["visual_over_text"] = {
        component: (
            visual[f"{component}_per_layer"]["mean"]
            / text[f"{component}_per_layer"]["mean"]
        )
        for component in (
            "dispatch_ms",
            "expert_execution_ms",
            "combine_ms",
            "moe_ms",
        )
    }
    result["visual_minus_text_paired_ms_per_layer"] = {}
    for component in (
        "dispatch_ms",
        "expert_execution_ms",
        "combine_ms",
        "moe_ms",
    ):
        visual_by_rep = condition_rep_means["VISUAL_PREFILL_MOE_REPLAY"][component]
        text_by_rep = condition_rep_means["TOKEN_MATCHED_TEXT_MOE_REPLAY"][component]
        paired = [
            visual_by_rep[rep] - text_by_rep[rep]
            for rep in sorted(set(visual_by_rep) & set(text_by_rep))
        ]
        result["visual_minus_text_paired_ms_per_layer"][component] = bootstrap(paired)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    full_trace = args.artifacts / "full_trace_aligned"
    if not (full_trace / "run.json").exists():
        full_trace = args.artifacts / "full_trace"
    cp2048_trace = args.artifacts / "cp2048_trace_aligned"
    if not (cp2048_trace / "run.json").exists():
        cp2048_trace = args.artifacts / "cp2048_trace"
    cp256_trace = args.artifacts / "cp256_trace_aligned"
    if not (cp256_trace / "run.json").exists():
        cp256_trace = args.artifacts / "cp256_trace"
    chunks = {
        "FULL": (args.artifacts / "full_clean", full_trace),
        "CP2048": (args.artifacts / "cp2048_clean", cp2048_trace),
        "CP1024": (args.artifacts / "cp1024_clean", args.artifacts / "cp1024_trace"),
        "CP512": (args.artifacts / "cp512_clean", args.artifacts / "cp512_trace"),
        "CP256": (args.artifacts / "cp256_clean", cp256_trace),
    }
    results = {}
    for label, (clean_root, trace_root) in chunks.items():
        visual_trace = trace_metrics(trace_root)
        visual_clean = clean_metrics(
            clean_root, visual_trace["injected_outer_iterations"]
        )
        results[label] = {"trace": visual_trace, "clean": visual_clean}
        if label == "FULL":
            text_trace = trace_metrics(trace_root, "TOKEN_MATCHED_TEXT")
            results[label]["matched_text_trace"] = text_trace
            results[label]["matched_text_clean"] = clean_metrics(
                clean_root,
                text_trace["injected_outer_iterations"],
                "TOKEN_MATCHED_TEXT",
            )
    layered = layered_replay(full_trace, results["FULL"]["clean"])
    replay_root = args.artifacts / "moe_only_replay"
    moe_replay = (
        moe_only_replay_metrics(replay_root)
        if (replay_root / "run.json").exists()
        else {"status": "NOT_RUN"}
    )
    full = results["FULL"]
    cp512 = results["CP512"]
    cp256 = results["CP256"]
    g1 = bool(
        full["clean"]["peak_vsel_ms"]["ci95"][0] > 0
        and full["trace"]["cidd_ms_per_layer"]["ci95"][0] > 0
        and full["trace"]["mncd_dispatch_ms"]["max"]
        > 2 * full["trace"]["decode_isolated_dispatch_ms_per_layer"]["median"]
    )
    g2_candidates = {"CP512": cp512, "CP256": cp256}
    g2_witnesses = [
        label
        for label, arm in g2_candidates.items()
        if arm["trace"]["mncd_dispatch_ms"]["median"]
        < 0.90 * full["trace"]["mncd_dispatch_ms"]["median"]
        and arm["trace"]["cidd_ms_per_layer"]["mean"]
        < 0.50 * full["trace"]["cidd_ms_per_layer"]["mean"]
        and arm["trace"]["expert_reload_amp"]["median"] > 1.5
        and arm["trace"]["mixed_expert_execution_ms_per_request"]["mean"]
        > 1.05 * full["trace"]["mixed_expert_execution_ms_per_request"]["mean"]
    ]
    g2 = bool(g2_witnesses)
    g3 = bool(
        layered["expert_reload_amp"] < cp512["trace"]["expert_reload_amp"]["median"]
        and layered["cidd_ms_primary"]
        > 1.10 * cp512["trace"]["cidd_ms_per_victim_iteration"]["mean"]
        and full["trace"]["mncd_dispatch_ms"]["median"]
        > cp512["trace"]["mncd_dispatch_ms"]["median"]
    )
    visual_peak = full["clean"]["response_peak_inflation_percent"]["mean"]
    text_peak = full["matched_text_clean"]["response_peak_inflation_percent"]["mean"]
    visual_cidd = full["trace"]["cidd_ms_per_layer"]["mean"]
    text_cidd = full["matched_text_trace"]["cidd_ms_per_layer"]["mean"]
    replay_dispatch_delta = moe_replay.get(
        "visual_minus_text_paired_ms_per_layer", {}
    ).get("dispatch_ms", {"ci95": [None, None]})
    replay_dispatch_ci_low = replay_dispatch_delta["ci95"][0]
    g4 = bool(
        visual_cidd > 1.10 * text_cidd
        and visual_peak > text_peak + 3.0
        and replay_dispatch_ci_low is not None
        and replay_dispatch_ci_low > 0
    )
    if g4:
        root_cause = "VISUAL_EP_COMMUNICATION_SPECIFIC_CANDIDATE"
        g4_label = "MLLM_SPECIFIC_EP_INTERFERENCE_CONFIRMED"
    elif visual_peak > text_peak + 3.0:
        root_cause = "MULTIMODAL_COMPUTE_OR_SCHEDULING_DOMINANT"
        g4_label = "MLLM_STRESS_REGIME_ONLY"
    else:
        root_cause = "PREVIOUS_GAP_WAS_ALIGNMENT_OR_NOISE_DOMINATED"
        g4_label = "MLLM_SPECIFICITY_INCONCLUSIVE"
    if not g1:
        verdict = "DIRECT_COMM_STALL_NOT_CONFIRMED"
    elif g2 and g3:
        verdict = "PROCEED_COMMUNICATION_ONLY_CHUNKING"
    elif g2 and not g3:
        verdict = "LAYERED_PREFILL_CLOSES_EP_GAP"
    else:
        verdict = "CHUNKED_PREFILL_SUFFICIENT_UNDER_EP"
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "chunk_results.json", results)
    atomic_json(args.output / "layered_physical_replay.json", layered)
    atomic_json(args.output / "moe_only_physical_replay.json", moe_replay)
    runtime_audit = Path(
        "poc_mllm_ep_communication_motivation/artifacts/manifests/runtime_audit.json"
    )
    atomic_json(
        args.output / "execution_manifest.json",
        {
            "status": "COMPLETE",
            "physical_gpus": [2, 3, 6, 7],
            "topology": "TP2-DP2-EP4-PP1",
            "model_revision": "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c",
            "runtime_audit_reused": read_json(runtime_audit),
            "chunk_arms": list(chunks),
            "full_clean_repeats": 10,
            "trace_repeats_per_arm": 3,
            "layered_upstream_commit": "053f80e5201a7c0ab56e468e3d578907a2ca9cc3",
            "layered_baseline_type": "LAYERED_EP_PHYSICAL_REPLAY",
            "moe_only_replay_status": moe_replay["status"],
            "communication_only_chunking_implemented": False,
        },
    )
    atomic_json(
        args.output / "final_verdict.json",
        {
            "gates": {"G1": g1, "G2": g2, "G3": g3, "G4": g4},
            "g2_witness_arms": g2_witnesses,
            "g2_operational_definition": (
                "at least one CP512/CP256 arm has >=10% MNCD reduction, >=50% "
                "CIDD/layer reduction, ExpertReloadAmp>1.5, and >=5% measured "
                "mixed expert-execution increase versus FULL"
            ),
            "g4_label": g4_label,
            "rq4_root_cause": root_cause,
            "visual_peak_inflation_percent": visual_peak,
            "matched_text_peak_inflation_percent": text_peak,
            "visual_cidd_ms_per_layer": visual_cidd,
            "matched_text_cidd_ms_per_layer": text_cidd,
            "moe_only_replay_status": moe_replay["status"],
            "moe_only_visual_minus_text_dispatch_ms_per_layer": replay_dispatch_delta,
            "overall_verdict": verdict,
            "communication_only_chunking_implemented": False,
        },
    )


if __name__ == "__main__":
    main()
