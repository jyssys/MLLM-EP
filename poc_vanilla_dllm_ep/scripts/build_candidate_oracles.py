#!/usr/bin/env python3
"""Build request-level upper bounds for Candidates C/E/F/G/H/I.

All latency savings are explicitly upper bounds.  The script never equates an
assignment reduction with a measured latency reduction: assignment-based
coverage is applied only to the measured dispatch/combine span and is labeled
as a proportionality oracle.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep4-rank0", type=Path, required=True)
    parser.add_argument("--p2p-json", type=Path)
    parser.add_argument(
        "--clean-request-ms", type=float,
        help="Matched clean total wall for the captured requests; observer-heavy wall is fallback only.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank0 = json.loads(args.ep4_rank0.read_text())
    rank_paths = sorted(args.ep4_rank0.parent.glob(
        args.ep4_rank0.name.replace("_rank0.json", "_rank*.json")
    ))
    payloads = [json.loads(path.read_text()) for path in rank_paths]
    instrumented_request_ms = float(rank0["request_wall_s"]) * 1000
    request_ms = args.clean_request_ms or instrumented_request_ms
    # Tensor capture has a large observer tax.  Stage *shares* are measured on
    # that internally consistent timeline; when a clean denominator is given,
    # scale all event spans by clean/observed wall instead of mixing raw traced
    # numerators with clean wall time (which can otherwise exceed 100%).
    clean_scale = request_ms / instrumented_request_ms

    by_call = defaultdict(list)
    for payload in payloads:
        for row in payload["stage_rows"]:
            by_call[int(row["call_id"])].append(row)
    critical = {}
    for call_id, rows in by_call.items():
        critical[call_id] = {
            stage: max(float(row[f"{stage}_ms"]) for row in rows) * clean_scale
            for stage in ("router", "prepare", "dispatch", "expert", "combine", "moe")
        }

    route_rows = [
        row for row in rank0["stage_rows"]
        if row.get("forward_phase") == "refine" and "selected_experts" in row
    ]
    total_comm_ms = sum(
        critical[int(row["call_id"])]["dispatch"]
        + critical[int(row["call_id"])]["combine"]
        for row in route_rows
    )
    total_moe_ms = sum(critical[int(row["call_id"])]["moe"] for row in route_rows)
    total_router_prepare_ms = sum(
        critical[int(row["call_id"])]["router"]
        + critical[int(row["call_id"])]["prepare"]
        for row in route_rows
    )

    p2p_copy_ms = None
    if args.p2p_json and args.p2p_json.exists():
        p2p = json.loads(args.p2p_json.read_text())
        p2p_copy_ms = next(
            float(row["copy_p50_ms"] if "copy_p50_ms" in row else row["copy_median_ms"])
            for row in p2p["results"] if int(row["expert_count"]) == 1
        )

    # C: perfect-future block-hot replicas placed on source rank 0.  Expert
    # compute remains and critical-rank load penalties are optimistically zero.
    # The saved time is therefore already an optimistic communication-only cap.
    block_rows = defaultdict(list)
    for row in route_rows:
        block_rows[(str(row["request_id"]), int(row["block_id"]))].append(row)
    replica_oracles = {}
    expert_bytes = 3 * 2048 * 768 * 2
    for budget in (1, 2, 4, 8, 16, 32, 64, 128):
        saved_comm_ms = 0.0
        added_critical_expert_ms = 0.0
        copies = 0
        for rows in block_rows.values():
            demand = Counter()
            for row in rows:
                for experts in row["selected_experts"]:
                    demand.update(expert for expert in experts if expert >= 32)
            cached = {expert for expert, comments in demand.most_common(budget)}
            copies += len(cached)
            for row in rows:
                selected = row["selected_experts"]
                cached_assignments = sum(
                    expert in cached for experts in selected for expert in experts
                )
                remote = max(int(row["remote_assignments_from_source"]), 1)
                coverage = cached_assignments / remote
                span = critical[int(row["call_id"])]
                saved_comm_ms += coverage * (span["dispatch"] + span["combine"])
                original_rank_load = list(row["rank_assignment_counts"])
                new_rank_load = list(original_rank_load)
                for experts in selected:
                    for expert in experts:
                        if expert in cached:
                            owner = expert // 32
                            new_rank_load[owner] -= 1
                            new_rank_load[0] += 1
                load_inflation = max(new_rank_load) / max(max(original_rank_load), 1) - 1
                added_critical_expert_ms += max(load_inflation, 0.0) * span["expert"]
        copy_ms = 0.0 if p2p_copy_ms is None else copies * p2p_copy_ms
        net = max(saved_comm_ms - copy_ms - added_critical_expert_ms, 0.0)
        replica_oracles[str(budget)] = {
            "perfect_future_saved_comm_ms": saved_comm_ms,
            "expert_copies": copies,
            "copy_cost_ms": None if p2p_copy_ms is None else copy_ms,
            "proportional_critical_expert_penalty_ms": added_critical_expert_ms,
            "peak_hbm_budget_bytes": budget * expert_bytes,
            "optimistic_net_request_gain_fraction": net / request_ms,
        }

    # E: how much selected router mass is physically rank-local or two-rank.
    scope = defaultdict(list)
    for row in route_rows:
        for experts, weights in zip(row["selected_experts"], row["router_weights"]):
            rank_mass = [0.0] * 4
            for expert, weight in zip(experts, weights):
                rank_mass[int(expert) // 32] += float(weight)
            rank_mass.sort(reverse=True)
            scope["top1_rank_mass"].append(rank_mass[0])
            scope["top2_rank_mass"].append(rank_mass[0] + rank_mass[1])
    conditional_scope = {
        "top1_rank_mass_mean": sum(scope["top1_rank_mass"]) / len(scope["top1_rank_mass"]),
        "top2_rank_mass_mean": sum(scope["top2_rank_mass"]) / len(scope["top2_rank_mass"]),
        "top1_rank_mass_ge_0.9_fraction": sum(x >= 0.9 for x in scope["top1_rank_mass"]) / len(scope["top1_rank_mass"]),
        "top2_rank_mass_ge_0.9_fraction": sum(x >= 0.9 for x in scope["top2_rank_mass"]) / len(scope["top2_rank_mass"]),
        "absolute_free_global_scope_request_cap_fraction": total_moe_ms / request_ms,
        "warning": "Router mass is not a correctness or verifier-acceptance result.",
    }

    # I: perfect rank-coherent top-8 choice from the captured top-16 list at
    # bounded router-mass sacrifice.  This remains an assignment oracle.
    near_tie = {}
    for loss_limit in (0.0, 0.001, 0.005, 0.01):
        baseline_fanout = 0
        alternative_fanout = 0
        baseline_remote = 0
        alternative_remote = 0
        tokens = 0
        for row in route_rows:
            for experts16, weights16 in zip(
                row["router_top16_experts"], row["router_top16_weights"]
            ):
                baseline_experts = experts16[:8]
                baseline_mass = sum(float(x) for x in weights16[:8])
                best = (len({expert // 32 for expert in baseline_experts}), baseline_mass, baseline_experts)
                for subset_size in range(1, 5):
                    for ranks in itertools.combinations(range(4), subset_size):
                        allowed = [
                            (expert, float(weight))
                            for expert, weight in zip(experts16, weights16)
                            if expert // 32 in ranks
                        ][:8]
                        if len(allowed) < 8:
                            continue
                        mass = sum(weight for _, weight in allowed)
                        if mass + 1e-12 < baseline_mass * (1 - loss_limit):
                            continue
                        experts = [expert for expert, _ in allowed]
                        candidate = (len(set(expert // 32 for expert in experts)), mass, experts)
                        if candidate[0] < best[0] or (
                            candidate[0] == best[0] and candidate[1] > best[1]
                        ):
                            best = candidate
                baseline_fanout += len({expert // 32 for expert in baseline_experts})
                alternative_fanout += best[0]
                baseline_remote += sum(expert >= 32 for expert in baseline_experts)
                alternative_remote += sum(expert >= 32 for expert in best[2])
                tokens += 1
        remote_reduction = 1 - alternative_remote / max(baseline_remote, 1)
        near_tie[str(loss_limit)] = {
            "tokens": tokens,
            "baseline_fanout_mean": baseline_fanout / tokens,
            "alternative_fanout_mean": alternative_fanout / tokens,
            "remote_assignment_reduction_fraction": remote_reduction,
            "proportional_comm_request_oracle_fraction": remote_reduction * total_comm_ms / request_ms,
        }

    exact_delta = {
        "exact_dense_delta_byte_reduction_fraction": 0.0,
        "exact_request_gain_fraction": 0.0,
        "fp8_delta_absolute_request_cap_fraction": 0.5 * total_comm_ms / request_ms,
        "warning": "An exact dense hidden delta has the same element count as the full hidden; FP8 is approximate and excludes codec overhead.",
    }
    layer_global = {
        "free_all_router_prepare_request_cap_fraction": total_router_prepare_ms / request_ms,
        "free_all_moe_request_cap_fraction": total_moe_ms / request_ms,
        "warning": "These are physically impossible all-layer upper bounds and collide with Epoch/DICE-style selective refresh.",
    }
    overlap = {
        "free_all_dispatch_combine_request_cap_fraction": total_comm_ms / request_ms,
        "batch1_independent_slack_observed": False,
        "warning": "Sequential vanilla refinement exposes no independent next-iteration work before the current logits/acceptance decision.",
    }
    result = {
        "request_wall_ms_instrumented": instrumented_request_ms,
        "request_wall_ms_denominator": request_ms,
        "observer_to_clean_stage_scale": clean_scale,
        "measured_critical_stage_sums_ms": {
            "refine_dispatch_combine": total_comm_ms,
            "refine_moe": total_moe_ms,
            "refine_router_prepare": total_router_prepare_ms,
        },
        "candidate_c_block_hot_replica": replica_oracles,
        "candidate_e_conditional_scope": conditional_scope,
        "candidate_f_temporal_delta": exact_delta,
        "candidate_g_layer_global": layer_global,
        "candidate_h_overlap": overlap,
        "candidate_i_near_tie_rank_coherence": near_tie,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
