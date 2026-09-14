#!/usr/bin/env python3
"""Merge causal trajectory quality with would-be physical deferred work."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


PREFIX = "[LLADA_DENOISE]"


def accept_times(log: Path):
    by_policy = defaultdict(dict)
    with log.open(errors="replace") as handle:
        for line in handle:
            where = line.find(PREFIX)
            if where < 0:
                continue
            record = json.loads(line[where + len(PREFIX):])
            policy = record.get("policy", "baseline")
            if str(record.get("request_id", "")).startswith("warmup"):
                continue
            for seq, block, iteration, accepted in zip(
                record["sequence_ids"], record["block_starts"],
                record["block_iterations"], record["accepted_mask"]
            ):
                for position, selected in enumerate(accepted):
                    if selected:
                        by_policy[policy][(int(seq), int(block), position)] = int(iteration)
    return by_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    args = parser.parse_args()
    oracles = pd.read_csv(args.task_root / "DORMANT_ORACLES.csv")
    rows = []
    for task in ("gsm8k", "humaneval"):
        raw = args.campaign / "raw" / "causal" / task / "r1"
        quality = pd.read_csv(args.campaign / "analysis" / f"{task}_policy_quality.csv")
        trajectory = accept_times(
            args.campaign / "logs" / f"causal_{task}_ep4_b32_mini32_r1_g32.log"
        )
        physical = defaultdict(lambda: defaultdict(float))
        for path in raw.glob("dormant_rank*.jsonl"):
            with path.open() as handle:
                for line in handle:
                    record = json.loads(line)
                    policy = record["policy"]
                    for field in (
                        "physical_rows", "masked_rows", "dormant_rows", "deferred_rows",
                        "route_changed_rows", "destination_changed_rows",
                        "deferred_remote_assignments", "deferred_remote_destinations",
                    ):
                        physical[policy][field] += float(record[field])
        baseline_accept = trajectory["baseline"]
        for quality_row in quality.to_dict("records"):
            policy = quality_row["policy"]
            metrics = physical.get(policy, {})
            current = trajectory.get(policy, {})
            common = set(baseline_accept) & set(current)
            acceptance_exact = sum(
                baseline_accept[key] == current[key] for key in common
            )
            acceptance_abs_drift = sum(
                abs(baseline_accept[key] - current[key]) for key in common
            ) / max(len(common), 1)
            policy_oracle = None
            if policy.startswith("h1_route"):
                policy_oracle = "route_trigger_k8"
                horizon = 1
            elif policy.startswith("h1_k"):
                policy_oracle = f"periodic_k{policy.rsplit('k', 1)[1]}"
                horizon = 1
            elif policy == "h1_stale":
                policy_oracle = "perfect_remove"
                horizon = 1
            elif policy.startswith("h2_k"):
                policy_oracle = f"periodic_k{policy.rsplit('k', 1)[1]}"
                horizon = 2
            else:
                horizon = 0
            oracle = None
            if policy_oracle:
                match = oracles[
                    (oracles.task == task)
                    & (oracles.horizon == horizon)
                    & (oracles.policy == policy_oracle)
                ]
                if not match.empty:
                    oracle = match.iloc[0]
            rows.append({
                "task": task,
                "subset": "bounded32",
                "policy": policy,
                "horizon": horizon,
                "period": int(quality_row.get("nfe", 0)) if policy == "baseline" else (
                    999 if policy == "h1_stale" else int(policy.rsplit("k", 1)[1])
                ),
                "trigger": "route+destination" if "route" in policy else "periodic",
                "router_fresh": True,
                "shared_fresh": True,
                "routed_rows_deferred_pct": 100 * metrics.get("deferred_rows", 0) / max(metrics.get("physical_rows", 0), 1),
                "remote_payload_deferred_pct": None if oracle is None else float(oracle.remote_payload_saved_pct),
                "final_sequence_exact": int(quality_row["final_sequence_exact"]),
                "final_sequence_total": int(quality_row["final_sequence_total"]),
                "nfe": int(quality_row["nfe"]),
                "nfe_equal": int(quality_row["nfe"]) == int(quality[quality.policy == "baseline"].iloc[0].nfe),
                "acceptance_iteration_exact_pct": 100 * acceptance_exact / max(len(common), 1),
                "acceptance_iteration_abs_drift": acceptance_abs_drift,
                "benchmark_score": int(quality_row["benchmark_correct"]),
                "baseline_score": int(quality[quality.policy == "baseline"].iloc[0].benchmark_correct),
                "quality_delta_pp": 100 * float(quality_row["benchmark_delta_vs_baseline"]) / int(quality_row["benchmark_total"]),
                "per_sample_score_flips": int(quality_row["per_sample_score_flips"]),
                "feasible_e2e_oracle_pct": None if oracle is None else float(oracle.feasible_e2e_pct),
                "evidence_boundary": "causal output replacement; no measured runtime saving",
            })
    fields = list(rows[0])
    with (args.task_root / "REFRESH_POLICY_RESULTS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
