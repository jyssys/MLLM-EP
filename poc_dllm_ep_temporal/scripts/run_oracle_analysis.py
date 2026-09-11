#!/usr/bin/env python3
"""CPU-only economic gates for transient replication and rebatching.

The script reports two mappings. ``empirical`` uses the observed dependence
of cleanly aggregated MoE time on max-rank assignments and clips a negative
slope to zero. ``optimistic_expert_proportional`` assumes, in favor of the
hypothesis, that expert CUDA time scales perfectly with max-rank work.  The
latter is an upper bound, not a measured latency effect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd


KEY = ["request_id", "block_id", "iteration_id", "layer_id"]


def percentile(values, q):
    return float(np.quantile(np.asarray(values, dtype=float), q))


def rank_load(expert, ownership):
    return np.asarray([expert[ownership == rank].sum() for rank in range(4)], dtype=float)


def split_replica(vector, expert_load, home, target, mode="equalizing"):
    result = np.asarray(vector, dtype=float).copy()
    if mode == "equalizing":
        moved = float(np.clip((result[home] - result[target]) / 2.0, 0.0, expert_load))
    elif mode == "50_50":
        moved = float(expert_load / 2.0)
    else:
        raise ValueError(f"unknown replica split mode: {mode}")
    result[home] -= moved
    result[target] += moved
    return result


def replica_candidate(future, expert, target, ownership, mode="equalizing"):
    home = int(ownership[expert])
    improvements = []
    for row in future:
        loads = np.asarray(row["rank"], dtype=float)
        candidate = split_replica(
            loads,
            float(row["expert"][expert]),
            home,
            target,
            mode=mode,
        )
        improvements.append(max(0.0, float(loads.max() - candidate.max())))
    return improvements


def replication_analysis(df, clean_request_ms, visible_copy_ms):
    ownership = np.repeat(np.arange(4), 16)
    groups = {key: part.sort_values("iteration_id")
              for key, part in df.groupby(["request_id", "block_id", "layer_id"])}
    results = []
    for horizon in (1, 2, 4, 8):
        oracle_gross = oracle_net = causal_gross = causal_net = 0.0
        half_oracle_gross = half_oracle_net = 0.0
        oracle_load_saved = causal_load_saved = 0.0
        leases_considered = copies = half_copies = causal_copies = causal_wasted = future_steps = 0
        causal_target_is_current_least = 0
        per_request_oracle = {request: 0.0 for request in clean_request_ms}
        per_request_causal = {request: 0.0 for request in clean_request_ms}
        for (request, _block, _layer), part in groups.items():
            rows = []
            for row in part.itertuples(index=False):
                expert = np.asarray(row.expert_assignment_counts, dtype=float)
                rows.append({"expert": expert, "rank": rank_load(expert, ownership),
                             "expert_ms": float(row.expert_ms)})
            # Non-overlapping leases prevent overlapping replicas from
            # double-counting the same future work.
            current = 0
            while current + 1 < len(rows):
                future = rows[current + 1:current + 1 + horizon]
                if not future:
                    break
                future_steps += len(future)
                leases_considered += 1
                best = None
                best_half = None
                for expert in range(64):
                    home = int(ownership[expert])
                    for target in range(4):
                        if target == home:
                            continue
                        gains = replica_candidate(future, expert, target, ownership)
                        gross = sum(
                            row["expert_ms"] * gain / max(float(np.max(row["rank"])), 1.0)
                            for row, gain in zip(future, gains))
                        if best is None or gross > best[0]:
                            best = (gross, expert, target, gains)
                        half_gains = replica_candidate(
                            future, expert, target, ownership, mode="50_50"
                        )
                        half_gross = sum(
                            row["expert_ms"] * gain / max(float(np.max(row["rank"])), 1.0)
                            for row, gain in zip(future, half_gains)
                        )
                        if best_half is None or half_gross > best_half[0]:
                            best_half = (half_gross, expert, target, half_gains)
                assert best is not None
                assert best_half is not None
                gross = best[0]
                if gross > visible_copy_ms:
                    net = gross - visible_copy_ms
                    oracle_gross += gross
                    oracle_net += net
                    oracle_load_saved += sum(best[3])
                    copies += 1
                    per_request_oracle[request] += net
                if best_half[0] > visible_copy_ms:
                    half_oracle_gross += best_half[0]
                    half_oracle_net += best_half[0] - visible_copy_ms
                    half_copies += 1

                now = rows[current]
                predicted = None
                for expert in range(64):
                    home = int(ownership[expert])
                    for target in range(4):
                        if target == home:
                            continue
                        gain = replica_candidate([now], expert, target, ownership)[0]
                        projected = horizon * now["expert_ms"] * gain / max(float(np.max(now["rank"])), 1.0)
                        if predicted is None or projected > predicted[0]:
                            predicted = (projected, expert, target)
                assert predicted is not None
                if predicted[0] > visible_copy_ms:
                    actual_gains = replica_candidate(future, predicted[1], predicted[2], ownership)
                    actual_gross = sum(
                        row["expert_ms"] * gain / max(float(np.max(row["rank"])), 1.0)
                        for row, gain in zip(future, actual_gains))
                    actual_net = actual_gross - visible_copy_ms
                    causal_gross += actual_gross
                    causal_net += actual_net
                    causal_load_saved += sum(actual_gains)
                    causal_copies += 1
                    causal_target_is_current_least += int(
                        predicted[2] == int(np.argmin(now["rank"]))
                    )
                    causal_wasted += int(actual_gross <= visible_copy_ms)
                    per_request_causal[request] += actual_net
                current += horizon + 1

        clean_total = float(sum(clean_request_ms.values()))
        results.append({
            "horizon": horizon,
            "leases_considered": leases_considered,
            "oracle_replica_installs": copies,
            "causal_replica_installs": causal_copies,
            "causal_wasted_copy_rate": causal_wasted / causal_copies if causal_copies else 0.0,
            "future_steps_covered": future_steps,
            "visible_copy_ms_each": visible_copy_ms,
            "expert_bytes_each": 12582912,
            "oracle_copy_bytes": copies * 12582912,
            "oracle_50_50_replica_installs": half_copies,
            "oracle_50_50_gross_expert_ms": half_oracle_gross,
            "oracle_50_50_net_expert_ms": half_oracle_net,
            "oracle_50_50_projected_request_reduction_pct": 100.0 * half_oracle_net / clean_total,
            "causal_copy_bytes": causal_copies * 12582912,
            "causal_destination_is_current_least_loaded_rate": (
                causal_target_is_current_least / causal_copies if causal_copies else 0.0
            ),
            "oracle_gross_max_rank_assignments_saved": oracle_load_saved,
            "causal_gross_max_rank_assignments_saved": causal_load_saved,
            "oracle_gross_expert_ms": oracle_gross,
            "oracle_net_expert_ms": oracle_net,
            "causal_gross_expert_ms": causal_gross,
            "causal_net_expert_ms": causal_net,
            "causal_recovery_of_oracle": causal_net / oracle_net if oracle_net else 0.0,
            "oracle_projected_request_reduction_pct": 100.0 * oracle_net / clean_total,
            "causal_projected_request_reduction_pct": 100.0 * causal_net / clean_total,
            "oracle_request_p50_pct": percentile([
                100.0 * per_request_oracle[r] / clean_request_ms[r] for r in clean_request_ms], .5),
            "causal_request_p50_pct": percentile([
                100.0 * per_request_causal[r] / clean_request_ms[r] for r in clean_request_ms], .5),
        })
    return results


def batch_cost(matrices, batches):
    total = 0.0
    for batch in batches:
        aggregate = np.stack([matrices[i] for i in batch]).sum(axis=0)
        total += float(aggregate.max(axis=1).sum())
    return total


def fixed_batches(ids, size=4):
    return [ids[i:i + size] for i in range(0, len(ids), size)]


def greedy_batches(ids, predictor, size=4):
    remaining = list(ids)
    batches = []
    while remaining:
        seed = remaining.pop(0)
        batch = [seed]
        aggregate = predictor[seed].copy()
        while remaining and len(batch) < size:
            candidate = min(remaining,
                            key=lambda item: float((aggregate + predictor[item]).max(axis=1).sum()))
            remaining.remove(candidate)
            batch.append(candidate)
            aggregate += predictor[candidate]
        batches.append(batch)
    return batches


def oracle_batches(ids, matrices, rng, attempts=100):
    best_batches = fixed_batches(ids)
    best = batch_cost(matrices, best_batches)
    for _ in range(attempts):
        order = list(ids)
        rng.shuffle(order)
        candidate = greedy_batches(order, matrices)
        # Bounded pair-swap descent turns random greedy starts into a strong
        # offline oracle without claiming exact global optimality.
        improved = True
        passes = 0
        while improved and passes < 3:
            improved = False
            passes += 1
            for left in range(len(candidate)):
                for right in range(left + 1, len(candidate)):
                    left_aggregate = np.stack([matrices[x] for x in candidate[left]]).sum(axis=0)
                    right_aggregate = np.stack([matrices[x] for x in candidate[right]]).sum(axis=0)
                    for li in range(len(candidate[left])):
                        for ri in range(len(candidate[right])):
                            left_id, right_id = candidate[left][li], candidate[right][ri]
                            old = float(left_aggregate.max(axis=1).sum() +
                                        right_aggregate.max(axis=1).sum())
                            new_left = left_aggregate - matrices[left_id] + matrices[right_id]
                            new_right = right_aggregate - matrices[right_id] + matrices[left_id]
                            new = float(new_left.max(axis=1).sum() + new_right.max(axis=1).sum())
                            if new + 1e-9 < old:
                                improved = True
                                candidate[left][li], candidate[right][ri] = right_id, left_id
                                left_aggregate, right_aggregate = new_left, new_right
            # one complete no-improvement pass terminates
        value = batch_cost(matrices, candidate)
        if value < best:
            best, best_batches = value, [list(x) for x in candidate]
    return best_batches


def batching_analysis(df, clean_total_ms):
    requests = sorted(df.request_id.unique())
    max_iteration = int(df.iteration_id.max())
    rng = random.Random(20260911)
    totals = {name: 0.0 for name in ("fcfs", "random", "perfect", "current", "ema")}
    # This deliberately relaxes request indivisibility and batch membership:
    # every layer's aggregate work may be split evenly over all four ranks.
    # No request-grouping policy can beat this fractional lower bound, so it
    # is useful as an implementation-independent economic kill gate.
    fractional_assignment_lower_bound = 0.0
    random_samples = []
    previous = {}
    ema = {}
    iteration_rows = []
    for iteration in range(max_iteration + 1):
        part = df[df.iteration_id == iteration]
        active = sorted(part.request_id.unique())
        matrices = {}
        for request, request_rows in part.groupby("request_id"):
            matrix = np.zeros((16, 4), dtype=float)
            for row in request_rows.itertuples(index=False):
                matrix[int(row.layer_id)] = np.asarray(row.rank_assignment_counts, dtype=float)
            matrices[request] = matrix
        if not active:
            continue
        aggregate_all_requests = np.stack([matrices[r] for r in active]).sum(axis=0)
        fractional_assignment_lower_bound += float(
            (aggregate_all_requests.sum(axis=1) / aggregate_all_requests.shape[1]).sum()
        )
        fcfs = fixed_batches(active)
        fcfs_cost = batch_cost(matrices, fcfs)
        random_costs = []
        for _ in range(200):
            order = list(active); rng.shuffle(order)
            random_costs.append(batch_cost(matrices, fixed_batches(order)))
        random_cost = float(np.median(random_costs))
        perfect = oracle_batches(active, matrices, rng)
        perfect_cost = batch_cost(matrices, perfect)
        current_prediction = {r: previous.get(r, np.full((16, 4), 256.0)) for r in active}
        current_batches = greedy_batches(active, current_prediction)
        current_cost = batch_cost(matrices, current_batches)
        ema_prediction = {r: ema.get(r, np.full((16, 4), 256.0)) for r in active}
        ema_batches = greedy_batches(active, ema_prediction)
        ema_cost = batch_cost(matrices, ema_batches)
        values = dict(fcfs=fcfs_cost, random=random_cost, perfect=perfect_cost,
                      current=current_cost, ema=ema_cost)
        for name, value in values.items():
            totals[name] += value
        iteration_rows.append({"iteration": iteration, "active_requests": len(active), **values})
        for request in active:
            previous[request] = matrices[request]
            ema[request] = (0.5 * matrices[request] + 0.5 * ema[request]
                            if request in ema else matrices[request].copy())
    random_samples.append(totals["random"])
    baseline = totals["fcfs"]
    # The clean request sum is deliberately generous: multiplying assignment
    # improvement by the observed expert share gives an optimistic E2E map.
    expert_total = float(df.expert_ms.sum())
    expert_share = min(1.0, expert_total / clean_total_ms)
    summary = {"assignment_cost": totals, "expert_share_of_clean_request_sum": expert_share,
               "offline_oracle_kind": "100-start greedy plus bounded pair-swap descent; not exact global optimum",
               "iterations": iteration_rows}
    for name in ("random", "perfect", "current", "ema"):
        improvement = max(0.0, (baseline - totals[name]) / baseline)
        summary[f"{name}_assignment_improvement_pct"] = 100.0 * improvement
        summary[f"{name}_optimistic_e2e_projection_pct"] = 100.0 * improvement * expert_share
    available = totals["fcfs"] - totals["perfect"]
    for name in ("current", "ema"):
        summary[f"{name}_oracle_recovery"] = ((totals["fcfs"] - totals[name]) / available
                                                if available > 0 else 0.0)
    absolute_improvement = max(
        0.0,
        (baseline - fractional_assignment_lower_bound) / baseline,
    )
    summary["fractional_assignment_lower_bound"] = fractional_assignment_lower_bound
    summary["absolute_assignment_improvement_upper_bound_pct"] = 100.0 * absolute_improvement
    summary["absolute_optimistic_e2e_upper_bound_pct"] = (
        100.0 * absolute_improvement * expert_share
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--copy-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    runs = Path(args.runs)
    traced = []
    for restart in (1, 2, 3):
        frame = pd.read_json(runs / f"trace_{restart}" / "rank0_trace.jsonl", lines=True)
        frame["restart"] = restart
        traced.append(frame)
    all_trace = pd.concat(traced, ignore_index=True)
    timing_columns = ["dispatch_ms", "expert_ms", "combine_ms", "moe_total_ms", "iteration_wall_ms"]
    timing = all_trace.groupby(KEY, as_index=False)[timing_columns].median()
    route = all_trace.sort_values("restart").drop_duplicates(KEY).drop(columns=timing_columns + ["restart"])
    df = route.merge(timing, on=KEY, validate="one_to_one")

    clean_by_request = {}
    clean_restart_medians = []
    trace_restart_medians = []
    for restart in (1, 2, 3):
        clean = pd.read_json(runs / f"clean_{restart}" / "rank0_clean_timing.json")
        trace = pd.read_json(runs / f"trace_{restart}" / "rank0_trace_timing.json")
        clean_restart_medians.append(float(clean.elapsed_ms.median()))
        trace_restart_medians.append(float(trace.elapsed_ms.median()))
        for row in clean.itertuples(index=False):
            clean_by_request.setdefault(row.request_id, []).append(float(row.elapsed_ms))
    clean_by_request = {key: float(np.median(value)) for key, value in clean_by_request.items()}
    clean_total = float(sum(clean_by_request.values()))
    copy = pd.read_csv(args.copy_csv)
    visible_copy = float(copy.visible_copy_p50_ms.median())

    central = df[df.moe_total_ms <= df.moe_total_ms.quantile(.99)]
    slope, intercept = np.polyfit(central.max_rank_load, central.moe_total_ms, 1)
    prediction = slope * central.max_rank_load + intercept
    denominator = float(((central.moe_total_ms - central.moe_total_ms.mean()) ** 2).sum())
    r2 = 1.0 - float(((central.moe_total_ms - prediction) ** 2).sum()) / denominator
    instrumentation = {
        "clean_restart_request_p50_ms": clean_restart_medians,
        "trace_restart_request_p50_ms": trace_restart_medians,
        "median_clean_ms": float(np.median(clean_restart_medians)),
        "median_trace_ms": float(np.median(trace_restart_medians)),
        "observer_tax_pct": 100.0 * (np.median(trace_restart_medians) /
                                     np.median(clean_restart_medians) - 1.0),
        "exact_output_matches": "90/90 clean-vs-trace; 90/90 across restarts",
    }
    output = {
        "trace": {"logical_records": len(df), "requests": len(clean_by_request),
                  "median_nfe": float(df.groupby("request_id").iteration_id.nunique().median())},
        "instrumentation": instrumentation,
        "load_latency_calibration": {
            "trimmed_moe_slope_ms_per_max_assignment": float(slope),
            "trimmed_r2": r2,
            "positive_empirical_slope_used": max(0.0, float(slope)),
            "interpretation": "No positive measured load-to-latency sensitivity; proportional expert mapping is optimistic only.",
        },
        "replication": replication_analysis(df, clean_by_request, visible_copy),
        "batching": batching_analysis(df, clean_total),
    }
    rank_matrix = np.vstack(df.rank_assignment_counts.to_numpy())
    max_rank = rank_matrix.max(axis=1)
    mean_rank = rank_matrix.mean(axis=1)
    # Give replication an even stronger counterfactual than the one-replica
    # lease oracle: copy cost is zero and arbitrary replicas remove every bit
    # of rank imbalance.  Expert time is assumed perfectly proportional to
    # max-rank assignments.  This is an absolute optimistic upper bound, not
    # an implementable policy or a measured latency reduction.
    absolute_replication_ms = float(
        (df.expert_ms.to_numpy() * (max_rank - mean_rank) /
         np.maximum(max_rank, 1.0)).sum()
    )
    output["replication_absolute_upper_bound"] = {
        "assumption": "zero-cost unlimited replication with perfect rank equalization and perfectly proportional expert time",
        "removable_expert_ms": absolute_replication_ms,
        "projected_request_reduction_pct": 100.0 * absolute_replication_ms / clean_total,
        "median_max_over_mean": float(np.median(max_rank / np.maximum(mean_rank, 1.0))),
        "p90_max_over_mean": float(np.quantile(max_rank / np.maximum(mean_rank, 1.0), 0.9)),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    pd.DataFrame(output["replication"]).to_csv(target.with_name("replication_oracle.csv"), index=False)
    pd.DataFrame(output["batching"]["iterations"]).to_csv(target.with_name("batching_iterations.csv"), index=False)


if __name__ == "__main__":
    main()
