#!/usr/bin/env python3
"""Build a request-weighted, zero-transition elastic-EP oracle.

The input is the observer-heavy EP1/2/4 run.  Same-device event durations are
first reduced to a critical-rank duration for every logical MoE invocation.
The oracle then chooses one EP degree for an entire denoising iteration; it
never mixes rank timestamps or claims that topology transition is free in a
deployable system.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instrumented_dir", type=Path)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    clean: dict[tuple[int, str], list[float]] = defaultdict(list)
    for path in args.clean_dir.glob("vanilla_ep*_restart*_rank0.json"):
        ep = int(re.search(r"ep(\d+)", path.name).group(1))
        payload = json.loads(path.read_text())
        for record in payload["records"]:
            clean[(ep, str(record["id"]))].append(float(record["elapsed_s"]) * 1000)

    # ep -> logical invocation -> critical-rank same-device CUDA duration
    calls: dict[int, dict[tuple, float]] = defaultdict(dict)
    request_ids: set[str] = set()
    for rank0_path in sorted(args.instrumented_dir.glob("*_rank0.json")):
        ep = int(re.search(r"ep(\d+)", rank0_path.name).group(1))
        rank_paths = sorted(args.instrumented_dir.glob(
            rank0_path.name.replace("_rank0.json", "_rank*.json")
        ))
        payloads = [json.loads(path.read_text()) for path in rank_paths]
        joined: dict[int, list[dict]] = defaultdict(list)
        for payload in payloads:
            for row in payload["stage_rows"]:
                joined[int(row["call_id"])].append(row)
        for rows in joined.values():
            exemplar = rows[0]
            request_id = str(exemplar["request_id"])
            request_ids.add(request_id)
            key = (
                request_id,
                str(exemplar.get("forward_phase")),
                int(exemplar.get("block_id", -1)),
                int(exemplar.get("denoising_iteration", -1)),
                int(exemplar["layer"]),
            )
            calls[ep][key] = max(float(row["moe_ms"]) for row in rows)

    common = set.intersection(*(set(calls[ep]) for ep in (1, 2, 4)))
    if not common:
        raise SystemExit("no logical MoE invocations common to EP1/2/4")
    by_step: dict[tuple, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for key in common:
        step = key[:4]
        for ep in (1, 2, 4):
            by_step[step][ep] += calls[ep][key]

    static_stage_ms = {
        ep: sum(per_ep[ep] for per_ep in by_step.values()) for ep in (1, 2, 4)
    }
    best_static = min(static_stage_ms, key=static_stage_ms.get)
    choices = {step: min(per_ep, key=per_ep.get) for step, per_ep in by_step.items()}
    oracle_stage_ms = sum(by_step[step][ep] for step, ep in choices.items())
    best_static_stage_ms = static_stage_ms[best_static]
    clean_request_ms = sum(
        statistics.median(clean[(best_static, request_id)])
        for request_id in sorted(request_ids)
    )
    saved_ms = max(best_static_stage_ms - oracle_stage_ms, 0.0)
    ordered_steps = sorted(by_step)
    topology_changes = sum(
        choices[ordered_steps[index]] != choices[ordered_steps[index - 1]]
        for index in range(1, len(ordered_steps))
    )
    result = {
        "common_logical_moe_invocations": len(common),
        "logical_steps": len(by_step),
        "request_ids": sorted(request_ids),
        "critical_moe_sum_ms_by_static_ep": static_stage_ms,
        "best_static_ep": best_static,
        "best_static_moe_sum_ms": best_static_stage_ms,
        "zero_transition_dynamic_moe_sum_ms": oracle_stage_ms,
        "zero_transition_saved_moe_ms": saved_ms,
        "matched_clean_request_denominator_ms": clean_request_ms,
        "zero_transition_direct_request_gain_fraction": saved_ms / clean_request_ms,
        "topology_changes": topology_changes,
        "winner_step_counts": {
            str(ep): sum(choice == ep for choice in choices.values())
            for ep in (1, 2, 4)
        },
        "transition_sensitivity": {
            str(cost_ms): max(saved_ms - topology_changes * cost_ms, 0.0)
            / clean_request_ms
            for cost_ms in (0.1, 1.0, 5.0, 10.0)
        },
        "feasibility_warning": (
            "The layouts contain different resident expert sets.  This is a perfect "
            "iteration-level oracle only: it excludes weight movement, redundant "
            "copies, communicator switching, graph recapture, and synchronization."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
