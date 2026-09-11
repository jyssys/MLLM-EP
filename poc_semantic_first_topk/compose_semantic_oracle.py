#!/usr/bin/env python3
"""Compose isolated group perturbations into globally budgeted K schedules.

Each measured row is a *state* for one spatial group, not an incremental step.
The response to decreasing K is empirically non-monotonic, so a greedy K=8 ->
K=7 -> ... path is invalid.  We solve a multiple-choice knapsack over the
directly measured K states instead.  This also guarantees that the full 1..8
grid cannot be worse than the coarse grid for an exactly shared drop budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _pareto_dp(
    by_group: dict[int, dict[int, dict]], allowed: set[int]
) -> tuple[int, dict[int, tuple[float, tuple[int, ...]]]]:
    groups = sorted(by_group)
    total = sum(next(iter(by_group[group].values()))["group_size"] * 8 for group in groups)
    # dropped assignments -> (additive isolated KL, selected K tuple)
    dp: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for group in groups:
        values = by_group[group]
        group_size = int(next(iter(values.values()))["group_size"])
        states = [(8, 0.0)]
        states.extend(
            (candidate, max(float(values[candidate]["kl_per_answer_token"]), 0.0))
            for candidate in sorted(allowed)
            if candidate < 8 and candidate in values
        )
        next_dp: dict[int, tuple[float, tuple[int, ...]]] = {}
        for prior_drop, (prior_risk, prior_k) in dp.items():
            for candidate, state_risk in states:
                dropped = prior_drop + (8 - candidate) * group_size
                proposal = (prior_risk + state_risk, prior_k + (candidate,))
                incumbent = next_dp.get(dropped)
                if incumbent is None or proposal[0] < incumbent[0]:
                    next_dp[dropped] = proposal
        dp = next_dp
    return total, dp


def allocation(
    entries: list[dict], target_fraction: float, allowed: set[int],
    exact_drop: int | None = None,
) -> dict:
    by_group: dict[int, dict[int, dict]] = {}
    for row in entries:
        if row["group"] >= 0 and row["k"] in allowed:
            by_group.setdefault(row["group"], {})[row["k"]] = row
    total, dp = _pareto_dp(by_group, allowed)
    target = round(total * target_fraction) if exact_drop is None else exact_drop
    feasible = [drop for drop in dp if drop <= target]
    # Use the largest non-overshooting budget.  Risk minimization at a fixed
    # budget happens inside the DP; choosing a smaller budget solely because it
    # has lower error would not answer the same-compute comparison.
    dropped = max(feasible) if feasible else 0
    risk, group_k = dp[dropped]
    return {"group_k": list(group_k), "dropped": dropped,
            "total": total, "drop_fraction": dropped / max(total, 1),
            "additive_kl_oracle": risk, "solver": "multiple_choice_dp",
            "target_dropped": target}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fractions", type=float, nargs="+", default=[.1, .2, .3, .4, .5])
    args = parser.parse_args()
    rows = []
    for root in args.input:
        for path in sorted(root.glob("sensitivity_*.jsonl")):
            rows.extend(json.loads(line) for line in path.read_text().splitlines())
    request_ids = sorted({row["request_id"] for row in rows})
    result = {"source": [str(path.resolve()) for path in args.input],
              "requests": {}, "global_policies": {}}
    for request in request_ids:
        selected = [row for row in rows if row["request_id"] == request]
        dataset = selected[0]["dataset"]
        policies = {}
        for fraction in args.fractions:
            coarse = allocation(selected, fraction, {1, 2, 4, 6, 8})
            full = allocation(selected, fraction, set(range(1, 9)),
                              exact_drop=coarse["dropped"])
            policies[f"semantic_full_{fraction:g}"] = full
            policies[f"semantic_coarse_{fraction:g}"] = coarse
        result["requests"][request] = {"dataset": dataset, "policies": policies}
    # A deployable spatial prior learned from the calibration cohort: unlike
    # the request oracle above, it does not inspect the evaluation request or
    # its answer.  Equal unit group sizes express assignment fraction on a
    # canonical 4x4 grid; the evaluator reports the realized fraction.
    aggregate = []
    valid = [row for row in rows if row.get("group", -1) >= 0]
    for group in sorted({row["group"] for row in valid}):
        for k in sorted({row["k"] for row in valid if row["group"] == group}):
            values = [float(row["kl_per_answer_token"]) for row in valid
                      if row["group"] == group and row["k"] == k]
            aggregate.append({"group": group, "group_size": 1, "k": k,
                              "kl_per_answer_token": float(np.median(values))})
    for fraction in args.fractions:
        coarse = allocation(aggregate, fraction, {1, 2, 4, 6, 8})
        full = allocation(aggregate, fraction, set(range(1, 9)),
                          exact_drop=coarse["dropped"])
        result["global_policies"][f"semantic_full_{fraction:g}"] = full
        result["global_policies"][f"semantic_coarse_{fraction:g}"] = coarse
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
