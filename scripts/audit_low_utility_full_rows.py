#!/usr/bin/env python3
"""Audit low-utility enrichment on an existing GSM8K-32 full-row trace.

The 2 GiB compressed heavy trace contains nearly 99M physical token rows.
For a bounded audit, this script samples 64 evenly spaced physical rows from
every layer/refinement invocation (3.82M token rows, about 30.6M expert
slots).  The primary GSM8K-128 current-block analysis remains exact; this
audit checks whether prefix/prior rows reverse its conclusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit, prange

from scripts.analyze_h1_straggler_deepdive import load_discovery_trace


def request_cluster_bootstrap(counts, request_ids, excess, per_inv_fraction, seed=20260918):
    """Bootstrap complete request trajectories rather than route rows."""
    unique, inverse = np.unique(request_ids, return_inverse=True)
    grouped_counts = np.zeros((len(unique), 4), dtype=np.float64)
    grouped_excess = np.zeros(len(unique), dtype=np.float64)
    grouped_attribution = np.zeros(len(unique), dtype=np.float64)
    for request in range(len(unique)):
        selected = inverse == request
        grouped_counts[request] = counts[selected].sum(axis=0)
        grouped_excess[request] = excess[selected].sum()
        grouped_attribution[request] = np.sum(excess[selected] * per_inv_fraction[selected])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(unique), size=(2_000, len(unique)))
    totals = grouped_counts[samples].sum(axis=1)
    critical_fraction = totals[:, 0] / totals[:, 1]
    noncritical_fraction = totals[:, 2] / totals[:, 3]
    enrichment = critical_fraction / noncritical_fraction
    difference = 100 * (critical_fraction - noncritical_fraction)
    excess_share = grouped_attribution[samples].sum(axis=1) / grouped_excess[samples].sum(axis=1)
    return {
        "cluster_unit": f"request ({len(unique)} independent generation trajectories)",
        "replicates": 2_000,
        "enrichment_95ci": np.quantile(enrichment, [.025, .975]).tolist(),
        "percentage_point_difference_95ci": np.quantile(difference, [.025, .975]).tolist(),
        "attributed_excess_share_95ci": np.quantile(excess_share, [.025, .975]).tolist(),
    }


@njit(parallel=True, cache=True)
def audit(ids, weights, offsets, critical, ep, sample_rows):
    # metrics: slot7-8, global bottom25 threshold, <=10% mass
    counts = np.zeros((len(critical), 3, 4), dtype=np.int64)
    per_rank = 256 // ep
    for invocation in prange(len(critical)):
        begin, end = offsets[invocation], offsets[invocation + 1]
        length = end - begin
        take = min(sample_rows, length)
        for ordinal in range(take):
            row = begin + ((2 * ordinal + 1) * length) // (2 * take)
            total = 0.0
            for slot in range(8):
                total += weights[row, slot]
            for slot in range(8):
                mass = weights[row, slot] / max(total, 1e-20)
                rank = 1
                for other in range(8):
                    if weights[row, other] > weights[row, slot]:
                        rank += 1
                is_critical = ids[row, slot] // per_rank == critical[invocation]
                for metric in range(3):
                    low = ((metric == 0 and rank >= 7)
                           or (metric == 1 and mass <= 0.09272008389234543)
                           or (metric == 2 and mass <= 0.10))
                    if is_critical:
                        counts[invocation, metric, 1] += 1
                        if low:
                            counts[invocation, metric, 0] += 1
                    else:
                        counts[invocation, metric, 3] += 1
                        if low:
                            counts[invocation, metric, 2] += 1
    return counts


@njit(cache=True)
def sampled_excess_curve(ids, weights, offsets, critical, excess, ep, sample_rows, bins):
    """Estimate the low-mass-first excess curve on the same bounded sample.

    Every invocation's modeled critical-rank excess is divided uniformly over
    its sampled critical-rank assignments, matching H1's expert-row attribution
    convention.  Router mass is Horvitz-style expanded by physical_rows/take
    so its denominator remains the full physical token mass, not merely the
    sampled critical-rank mass.
    """
    router_mass = np.zeros(bins, dtype=np.float64)
    attributed_excess = np.zeros(bins, dtype=np.float64)
    per_rank = 256 // ep
    for invocation in range(len(critical)):
        begin, end = offsets[invocation], offsets[invocation + 1]
        length = end - begin
        take = min(sample_rows, length)
        critical_routes = 0
        for ordinal in range(take):
            row = begin + ((2 * ordinal + 1) * length) // (2 * take)
            for slot in range(8):
                if ids[row, slot] // per_rank == critical[invocation]:
                    critical_routes += 1
        if critical_routes == 0:
            continue
        attribution = excess[invocation] / critical_routes
        scale = length / take
        for ordinal in range(take):
            row = begin + ((2 * ordinal + 1) * length) // (2 * take)
            total = 0.0
            for slot in range(8):
                total += weights[row, slot]
            for slot in range(8):
                if ids[row, slot] // per_rank != critical[invocation]:
                    continue
                mass = weights[row, slot] / max(total, 1e-20)
                index = min(int(mass * bins), bins - 1)
                router_mass[index] += mass * scale
                attributed_excess[index] += attribution
    return router_mass, attributed_excess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heavy", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--rank-times-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rows", type=int, default=64)
    args = parser.parse_args()

    aggregate = load_discovery_trace(args.aggregate).arrays
    selected = aggregate["request_id"] < 32
    request_ids = aggregate["request_id"][selected]
    physical = aggregate["physical_rows"][selected].astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(physical)))
    result = {
        "source": str(args.heavy),
        "population": "GSM8K requests 0-31, all physical row classes",
        "sampling": f"{args.sample_rows} evenly spaced rows per layer/refinement invocation",
        "invocations": int(len(physical)),
        "sampled_token_rows": int(np.minimum(physical, args.sample_rows).sum()),
        "quality_rollout": False,
        "targets": {},
    }
    with np.load(args.heavy, allow_pickle=False) as source:
        ids = source["row__expert_ids"]
        # Numba has no CPU float16 array data model; widen only for this
        # bounded audit after reading the existing artifact.
        weights = source["row__router_weights"].astype(np.float32)
        if offsets[-1] != len(ids):
            raise ValueError("aggregate/heavy invocation alignment mismatch")
        for ep in (4, 8):
            times = np.load(args.rank_times_root / f"rank_times_ep{ep}.npy", mmap_mode="r")[selected]
            critical = np.argmax(times, axis=1).astype(np.int8)
            counts = audit(ids, weights, offsets, critical, ep, args.sample_rows)
            excess = np.asarray(times.max(axis=1) - times.mean(axis=1), dtype=np.float64)
            mass_hist, excess_hist = sampled_excess_curve(
                ids, weights, offsets, critical, excess, ep, args.sample_rows, 10_000,
            )
            target = {}
            for metric, name in enumerate(("slots_7_8", "bottom_25_global", "mass_le_0.1")):
                summed = counts[:, metric].sum(axis=0).astype(np.float64)
                critical_fraction = summed[0] / summed[1]
                noncritical_fraction = summed[2] / summed[3]
                per_inv_fraction = np.divide(
                    counts[:, metric, 0], counts[:, metric, 1],
                    out=np.zeros(len(counts), dtype=np.float64),
                    where=counts[:, metric, 1] > 0,
                )
                target[name] = {
                    "critical_fraction": float(critical_fraction),
                    "noncritical_fraction": float(noncritical_fraction),
                    "enrichment": float(critical_fraction / noncritical_fraction),
                    "percentage_point_difference": float(100 * (critical_fraction - noncritical_fraction)),
                    # H1 row-share attribution makes this fraction the sampled
                    # estimate of full critical excess carried by the bin.
                    "sampled_attributed_excess_share": float(
                        np.sum(excess * per_inv_fraction) / np.sum(excess)
                    ),
                    "request_cluster_bootstrap": request_cluster_bootstrap(
                        counts[:, metric], request_ids, excess, per_inv_fraction,
                    ),
                }
            cumulative_mass = np.cumsum(mass_hist)
            cumulative_excess = np.cumsum(excess_hist)
            mass_denominator = float(physical.sum())
            target["low_mass_first_curve"] = {
                "sampling_estimator": (
                    "modeled invocation excess divided over sampled critical assignments; "
                    "router mass expanded by physical_rows/sample_rows"
                ),
                "router_mass_percent_needed_for_attributed_excess": {},
            }
            for fraction in (.25, .50, .75, .90):
                wanted = fraction * cumulative_excess[-1]
                index = int(np.searchsorted(cumulative_excess, wanted, side="left"))
                target["low_mass_first_curve"]["router_mass_percent_needed_for_attributed_excess"][str(fraction)] = (
                    100 * float(cumulative_mass[index]) / mass_denominator
                )
            result["targets"][f"ep{ep}"] = target
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
