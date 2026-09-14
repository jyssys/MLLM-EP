#!/usr/bin/env python3
"""Analyze future-dormant token geometry and routed-EP cost/oracles."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median


HORIZONS = (0, 1, 2, 4, 8)
PERIODS = (2, 4, 8)
HIDDEN_BYTES = 4096 * 2
DISPATCH_FLOOR_MS = 0.099776
COMBINE_FLOOR_MS = 0.152912
DENOISE_PREFIX = "[LLADA_DENOISE]"


def read_jsonl(path: Path):
    with path.open(errors="replace") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_denoise(log: Path):
    records = []
    with log.open(errors="replace") as handle:
        for line in handle:
            where = line.find(DENOISE_PREFIX)
            if where < 0:
                continue
            record = json.loads(line[where + len(DENOISE_PREFIX):])
            if not str(record.get("request_id", "")).startswith("warmup"):
                records.append(record)
    return records


def clean_median_ms(log_dir: Path, task: str):
    values = []
    pattern = re.compile(r"Forward:\s+\d+, Time:\s+([0-9.]+)")
    for path in sorted(log_dir.glob(f"clean_{task}_*.log")):
        text = path.read_text(errors="replace")
        matches = pattern.findall(text)
        if matches:
            values.append(float(matches[-1]) * 1000.0)
    if not values:
        raise RuntimeError(f"no clean times for {task}")
    return median(values), values


def acceptance_map(path: Path):
    document = json.loads(path.read_text())
    return {
        tuple(int(part) for part in key.split(":")): int(value)
        for key, value in document["acceptance_iteration_by_key"].items()
    }


def state_for(masked: bool, future_accept: int | None, t: int, horizon: int):
    if not masked:
        return "DEAD"
    if future_accept is None:
        return "ACTIVE"
    return "ACTIVE" if future_accept - t <= horizon else "DORMANT"


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = fields or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def timing_critical(raw_root: Path):
    grouped = defaultdict(lambda: defaultdict(list))
    for path in sorted(raw_root.glob("ep_rank*.jsonl")):
        for record in read_jsonl(path):
            if not record.get("sequence_ids"):
                continue
            key = (int(record["layer"]), int(record["invocation"]))
            for stage in ("router", "dispatch", "expert", "combine"):
                grouped[key][stage].append(float(record[f"{stage}_ms"]))
    return {
        key: {stage: max(values) for stage, values in stage_rows.items()}
        for key, stage_rows in grouped.items()
    }


def policy_names():
    names = ["perfect_remove"]
    for period in PERIODS:
        names.extend((f"periodic_k{period}", f"route_trigger_k{period}"))
    return names


def analyze_task(campaign: Path, task: str, task_root: Path):
    clean_ms, clean_values = clean_median_ms(campaign / "logs", task)
    denoise = read_denoise(
        campaign / "logs" / f"shape_{task}_ep4_b32_mini32_r1_g32.log"
    )
    denoise_by_wave = {int(record["iteration"]): record for record in denoise}
    accept = acceptance_map(campaign / "analysis" / f"{task}_acceptance_plan.json")
    timing = timing_critical(campaign / "raw" / "timing" / task / "r1")

    token_rows = []
    census = defaultdict(int)
    for record in denoise:
        for seq, block, t, masks, confidence, margin in zip(
            record["sequence_ids"], record["block_starts"],
            record["block_iterations"], record["mask_before"],
            record.get("token_confidence") or [[None] * 32] * len(record["sequence_ids"]),
            record.get("confidence_margin") or [[None] * 32] * len(record["sequence_ids"]),
        ):
            for position, masked in enumerate(masks):
                key = (int(seq), int(block), position)
                future_accept = accept.get(key)
                for horizon in HORIZONS:
                    state = state_for(bool(masked), future_accept, int(t), horizon)
                    census[(horizon, record["phase"], state)] += 1
                    token_rows.append({
                        "task": task,
                        "request_id": record["request_id"],
                        "sequence_id": seq,
                        "block_start": block,
                        "iteration": record["iteration"],
                        "block_iteration": t,
                        "position": position,
                        "horizon": horizon,
                        "state": state,
                        "future_accept_iteration": future_accept,
                        "future_distance": None if future_accept is None else future_accept - int(t),
                        "phase": record["phase"],
                        "confidence": confidence[position] if confidence else None,
                        "margin": margin[position] if margin else None,
                    })

    # Aggregate global state geometry per layer/wave across the four source ranks.
    wave_metrics = defaultdict(lambda: defaultdict(float))
    wave_experts = defaultdict(set)
    route_dynamics = defaultdict(lambda: defaultdict(float))
    policy_metrics = defaultdict(lambda: defaultdict(float))
    previous_route = {}
    policy_cache = defaultdict(dict)

    shape_root = campaign / "raw" / "shape" / task / "r1"
    for path in sorted(shape_root.glob("shape_rank*.jsonl")):
        for record in read_jsonl(path):
            wave = int(record.get("wave", -1))
            if wave < 0:
                continue
            rank = int(record["rank"])
            layer = int(record["layer"])
            invocation = int(record["invocation"])
            sequence_ids = record["sequence_ids"]
            block_starts = record["block_starts"]
            block_iterations = record["block_iterations"]
            masks = record["mask_before"]
            global_rows = [
                ((int(seq), int(block), position), int(t), bool(mask_row[position]))
                for seq, block, t, mask_row in zip(
                    sequence_ids, block_starts, block_iterations, masks
                )
                for position in range(len(mask_row))
            ]
            start = int(record["global_row_start"])
            ids_rows = record["topk_ids_local_source"]
            local_rows = global_rows[start:start + len(ids_rows)]
            key_base = (layer, invocation)
            for row_index, ((logical_key, t, masked), expert_ids) in enumerate(
                zip(local_rows, ids_rows)
            ):
                owners = [int(expert) // 64 for expert in expert_ids]
                remote_assignments = sum(owner != rank for owner in owners)
                remote_destinations = len({owner for owner in owners if owner != rank})
                for horizon in HORIZONS:
                    state = state_for(masked, accept.get(logical_key), t, horizon)
                    metric_key = (key_base, horizon, state, record["phase"])
                    values = wave_metrics[metric_key]
                    values["rows"] += 1
                    values["assignments"] += len(expert_ids)
                    values["remote_assignments"] += remote_assignments
                    values["remote_destinations"] += remote_destinations
                    wave_experts[metric_key].update(int(value) for value in expert_ids)

                    prior = previous_route.get((rank, layer, logical_key))
                    if prior is not None:
                        prior_ids, prior_owners = prior
                        current_ids = set(int(value) for value in expert_ids)
                        current_owners = set(owners)
                        dyn = route_dynamics[(horizon, state)]
                        dyn["matched"] += 1
                        dyn["exact_set"] += current_ids == prior_ids
                        dyn["topk_overlap_sum"] += len(current_ids & prior_ids) / max(
                            len(current_ids | prior_ids), 1
                        )
                        dyn["destination_exact"] += current_owners == prior_owners
                        dyn["destination_overlap_sum"] += len(
                            current_owners & prior_owners
                        ) / max(len(current_owners | prior_owners), 1)

                    for policy in policy_names():
                        if state != "DORMANT":
                            continue
                        policy_key = (horizon, policy, key_base, record["phase"])
                        cache_key = (rank, layer, logical_key)
                        cache = policy_cache[(horizon, policy)]
                        prior_policy = cache.get(cache_key)
                        if policy == "perfect_remove":
                            defer = True
                        else:
                            period = int(policy.rsplit("k", 1)[1])
                            refresh_due = (
                                prior_policy is None or t - prior_policy[0] >= period
                            )
                            route_changed = (
                                prior_policy is not None
                                and set(int(value) for value in expert_ids) != prior_policy[1]
                            )
                            destination_changed = (
                                prior_policy is not None
                                and set(owners) != prior_policy[2]
                            )
                            defer = not refresh_due
                            if policy.startswith("route_trigger") and (
                                route_changed or destination_changed
                            ):
                                defer = False
                        if defer and key_base in timing:
                            pm = policy_metrics[policy_key]
                            pm["rows"] += 1
                            pm["assignments"] += len(expert_ids)
                            pm["remote_assignments"] += remote_assignments
                            pm["remote_destinations"] += remote_destinations
                        else:
                            cache[cache_key] = (
                                t, set(int(value) for value in expert_ids), set(owners)
                            )
                previous_route[(rank, layer, logical_key)] = (
                    set(int(value) for value in expert_ids), set(owners)
                )

    state_totals = defaultdict(lambda: defaultdict(float))
    policy_totals = defaultdict(lambda: defaultdict(float))
    all_states = ("ACTIVE", "DORMANT", "DEAD")
    for (key_base, horizon, state, phase), values in wave_metrics.items():
        critical = timing.get(key_base)
        if critical is None:
            continue
        totals = {
            metric: sum(
                wave_metrics.get((key_base, horizon, candidate, phase), {}).get(metric, 0)
                for candidate in all_states
            )
            for metric in ("rows", "assignments", "remote_assignments", "remote_destinations")
        }
        row_fraction = values["rows"] / max(totals["rows"], 1)
        assignment_fraction = values["assignments"] / max(totals["assignments"], 1)
        payload_fraction = values["remote_destinations"] / max(totals["remote_destinations"], 1)
        target = state_totals[(horizon, phase, state)]
        for metric, amount in values.items():
            target[metric] += amount
        target["expert_touch_events"] += len(wave_experts[(key_base, horizon, state, phase)])
        target["router_ms"] += critical["router"] * row_fraction
        target["dispatch_ms"] += critical["dispatch"] * payload_fraction
        target["expert_ms"] += critical["expert"] * assignment_fraction
        target["combine_ms"] += critical["combine"] * payload_fraction
        target["feasible_dispatch_ms"] += max(
            critical["dispatch"] - DISPATCH_FLOOR_MS, 0
        ) * payload_fraction
        target["feasible_expert_ms"] += critical["expert"] * assignment_fraction
        target["feasible_combine_ms"] += max(
            critical["combine"] - COMBINE_FLOOR_MS, 0
        ) * payload_fraction

    # Convert saved-policy row counts into timing using matching wave totals.
    for (horizon, policy, key_base, phase), values in policy_metrics.items():
        critical = timing.get(key_base)
        if critical is None:
            continue
        totals = {
            metric: sum(
                wave_metrics.get((key_base, horizon, candidate, phase), {}).get(metric, 0)
                for candidate in all_states
            )
            for metric in ("rows", "assignments", "remote_assignments", "remote_destinations")
        }
        assignment_fraction = values["assignments"] / max(totals["assignments"], 1)
        payload_fraction = values["remote_destinations"] / max(totals["remote_destinations"], 1)
        target = policy_totals[(horizon, policy)]
        for metric, amount in values.items():
            target[metric] += amount
        target["dispatch_ms"] += critical["dispatch"] * payload_fraction
        target["expert_ms"] += critical["expert"] * assignment_fraction
        target["combine_ms"] += critical["combine"] * payload_fraction
        target["feasible_dispatch_ms"] += max(
            critical["dispatch"] - DISPATCH_FLOOR_MS, 0
        ) * payload_fraction
        target["feasible_expert_ms"] += critical["expert"] * assignment_fraction
        target["feasible_combine_ms"] += max(
            critical["combine"] - COMBINE_FLOOR_MS, 0
        ) * payload_fraction

    # Per-row hidden/routed drift from the independent five-layer detail pass.
    drift = defaultdict(lambda: defaultdict(list))
    detail_root = campaign / "raw" / "detail" / task / "r1"
    for path in sorted(detail_root.glob("shape_rank*.jsonl")):
        for record in read_jsonl(path):
            if int(record.get("wave", -1)) < 0:
                continue
            sequence_ids = record["sequence_ids"]
            block_starts = record["block_starts"]
            block_iterations = record["block_iterations"]
            masks = record["mask_before"]
            global_rows = [
                ((int(seq), int(block), position), int(t), bool(mask_row[position]))
                for seq, block, t, mask_row in zip(
                    sequence_ids, block_starts, block_iterations, masks
                )
                for position in range(len(mask_row))
            ]
            start = int(record["global_row_start"])
            nrows = int(record["physical_m_local"])
            local_rows = global_rows[start:start + nrows]
            arrays = {
                "hidden_cosine": record.get("hidden_cosine_lag1_local_source"),
                "hidden_rel_l2": record.get("hidden_rel_l2_lag1_local_source"),
                "routed_cosine": record.get("routed_output_cosine_lag1_local_source"),
                "routed_rel_l2": record.get("routed_output_rel_l2_lag1_local_source"),
            }
            if not arrays["hidden_cosine"]:
                continue
            for row_index, (logical_key, t, masked) in enumerate(local_rows):
                for horizon in HORIZONS:
                    state = state_for(masked, accept.get(logical_key), t, horizon)
                    for metric, array in arrays.items():
                        value = array[row_index]
                        if value is not None and math.isfinite(float(value)):
                            drift[(horizon, state)][metric].append(float(value))

    return {
        "clean_ms": clean_ms,
        "clean_values": clean_values,
        "token_rows": token_rows,
        "census": census,
        "state_totals": state_totals,
        "policy_totals": policy_totals,
        "route_dynamics": route_dynamics,
        "drift": drift,
        "denoise_records": len(denoise),
        "timing_records": len(timing),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    args = parser.parse_args()
    results = {
        task: analyze_task(args.campaign, task, args.task_root)
        for task in ("gsm8k", "humaneval")
    }

    # The fresh all-layer event trace deliberately synchronizes every layer
    # and therefore inflates wall time.  Preserve its within-trace state
    # allocation, but normalize absolute stage mass to the prior low-overhead
    # CUDA-event run on the identical model/topology/configuration.
    import pandas as pd
    low_overhead = pd.read_csv(
        args.task_root.parent
        / "poc_dllm_layer_sensitivity"
        / "LOW_OVERHEAD_LAYER_PHASE_COST.csv"
    )
    stage_scales = {}
    for task, result in results.items():
        reference = low_overhead[low_overhead["dataset"] == task]
        old_clean = float(reference["clean_request_median_ms"].iloc[0])
        for stage in ("router", "dispatch", "expert", "combine"):
            target = float(reference[f"{stage}_sum_ms"].sum()) * (
                result["clean_ms"] / old_clean
            )
            observed = sum(
                values.get(f"{stage}_ms", 0)
                for (horizon, _phase, _state), values in result["state_totals"].items()
                if horizon == 0
            )
            stage_scales[(task, stage)] = target / max(observed, 1e-12)

    token_rows = [row for result in results.values() for row in result["token_rows"]]
    write_csv(args.task_root / "DORMANT_TOKEN_TRACE.csv", token_rows)

    census_rows = []
    cost_rows = []
    dynamics_rows = []
    oracle_rows = []
    for task, result in results.items():
        clean_ms = result["clean_ms"]
        for horizon in HORIZONS:
            for phase in ("early", "middle", "late"):
                phase_total = sum(
                    result["census"].get((horizon, phase, state), 0)
                    for state in ("ACTIVE", "DORMANT", "DEAD")
                )
                for state in ("ACTIVE", "DORMANT", "DEAD"):
                    count = result["census"].get((horizon, phase, state), 0)
                    census_rows.append({
                        "task": task, "horizon": horizon, "phase": phase,
                        "state": state, "token_rows": count,
                        "state_fraction_pct": 100 * count / max(phase_total, 1),
                    })
                    values = result["state_totals"].get(
                        (horizon, phase, state), {}
                    )
                    routed_ms = sum(
                        values.get(f"{stage}_ms", 0) * stage_scales[(task, stage)]
                        for stage in ("dispatch", "expert", "combine")
                    )
                    cost_rows.append({
                        "task": task, "horizon": horizon, "phase": phase,
                        "state": state,
                        "token_rows": values.get("rows", 0),
                        "token_fraction": 100 * count / max(phase_total, 1),
                        "expert_assignments": values.get("assignments", 0),
                        "remote_assignments": values.get("remote_assignments", 0),
                        "remote_activation_payload_bytes": values.get("remote_destinations", 0) * HIDDEN_BYTES,
                        "combine_payload_bytes": values.get("remote_destinations", 0) * HIDDEN_BYTES,
                        "active_experts": values.get("expert_touch_events", 0),
                        "router_e2e_share_pct": 100 * values.get("router_ms", 0) * stage_scales[(task, "router")] / clean_ms,
                        "dispatch_e2e_share_pct": 100 * values.get("dispatch_ms", 0) * stage_scales[(task, "dispatch")] / clean_ms,
                        "expert_e2e_share_pct": 100 * values.get("expert_ms", 0) * stage_scales[(task, "expert")] / clean_ms,
                        "combine_e2e_share_pct": 100 * values.get("combine_ms", 0) * stage_scales[(task, "combine")] / clean_ms,
                        "total_routed_e2e_share_pct": 100 * routed_ms / clean_ms,
                        "evidence_boundary": "CUDA-event attribution normalized to clean E2E; analytical state allocation",
                    })

            for state in ("ACTIVE", "DORMANT", "DEAD"):
                dyn = result["route_dynamics"].get((horizon, state), {})
                count = dyn.get("matched", 0)
                dr = result["drift"].get((horizon, state), {})
                dynamics_rows.append({
                    "task": task, "horizon": horizon, "state": state,
                    "matched_rows": count,
                    "exact_topk_set_pct": 100 * dyn.get("exact_set", 0) / max(count, 1),
                    "topk_jaccard": dyn.get("topk_overlap_sum", 0) / max(count, 1),
                    "exact_destination_set_pct": 100 * dyn.get("destination_exact", 0) / max(count, 1),
                    "destination_jaccard": dyn.get("destination_overlap_sum", 0) / max(count, 1),
                    "hidden_cosine_p50": median(dr.get("hidden_cosine", [math.nan])),
                    "hidden_rel_l2_p50": median(dr.get("hidden_rel_l2", [math.nan])),
                    "routed_output_cosine_p50": median(dr.get("routed_cosine", [math.nan])),
                    "routed_output_rel_l2_p50": median(dr.get("routed_rel_l2", [math.nan])),
                    "detail_layers": "1|8|16|24|31",
                })

            totals_all = defaultdict(float)
            for phase in ("early", "middle", "late"):
                for state in ("ACTIVE", "DORMANT", "DEAD"):
                    for metric, value in result["state_totals"].get(
                        (horizon, phase, state), {}
                    ).items():
                        totals_all[metric] += value
            for policy in policy_names():
                values = result["policy_totals"].get((horizon, policy), {})
                optimistic_ms = sum(
                    values.get(f"{stage}_ms", 0) * stage_scales[(task, stage)]
                    for stage in ("dispatch", "expert", "combine")
                )
                feasible_ms = sum(
                    values.get(f"feasible_{stage}_ms", 0) * stage_scales[(task, stage)]
                    for stage in ("dispatch", "expert", "combine")
                )
                feasible_pct = 100 * feasible_ms / clean_ms
                gate = "KILL" if feasible_pct < 5 else (
                    "CHARACTERIZATION" if feasible_pct < 8 else (
                        "HOLD" if feasible_pct < 12 else (
                            "STRONG" if feasible_pct <= 20 else "VERY_STRONG"
                        )
                    )
                )
                oracle_rows.append({
                    "task": task, "horizon": horizon, "policy": policy,
                    "period": 0 if policy == "perfect_remove" else int(policy.rsplit("k", 1)[1]),
                    "trigger": "future-dormant" if policy == "perfect_remove" else (
                        "future-dormant+route/destination-change" if policy.startswith("route_trigger") else "future-dormant+periodic"
                    ),
                    "raw_rows_saved_pct": 100 * values.get("rows", 0) / max(totals_all.get("rows", 0), 1),
                    "remote_assignments_saved_pct": 100 * values.get("remote_assignments", 0) / max(totals_all.get("remote_assignments", 0), 1),
                    "remote_payload_saved_pct": 100 * values.get("remote_destinations", 0) / max(totals_all.get("remote_destinations", 0), 1),
                    "expert_rows_saved_pct": 100 * values.get("assignments", 0) / max(totals_all.get("assignments", 0), 1),
                    "optimistic_e2e_pct": 100 * optimistic_ms / clean_ms,
                    "post_epoch_e2e_pct": 100 * optimistic_ms / clean_ms,
                    "feasible_e2e_pct": feasible_pct,
                    "quality_constraint": "not yet tested",
                    "evidence_boundary": "offline oracle; fixed DeepEP floor retained in feasible estimate",
                    "gate": gate,
                })

    write_csv(args.task_root / "DORMANT_TOKEN_CENSUS.csv", census_rows)
    write_csv(args.task_root / "DORMANT_EP_COST.csv", cost_rows)
    write_csv(args.task_root / "DORMANT_ROUTING_DYNAMICS.csv", dynamics_rows)
    write_csv(args.task_root / "DORMANT_ORACLES.csv", oracle_rows)
    summary = {
        task: {
            "clean_median_ms": result["clean_ms"],
            "clean_restart_ms": result["clean_values"],
            "denoise_records": result["denoise_records"],
            "timing_layer_invocations": result["timing_records"],
        }
        for task, result in results.items()
    }
    (args.campaign / "analysis" / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
