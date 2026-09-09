#!/usr/bin/env python3
"""Reduce clean request runs and observer traces without rank double-counting."""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
from pathlib import Path
import statistics


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo] * (hi - pos) + values[hi] * (pos - lo)


def parse_label(label: str) -> dict:
    result = {"label": label, "repeat": -1, "batch_size": -1, "policy": "unknown", "wave": -1}
    parts = label.split(":")
    if len(parts) >= 4 and parts[0].startswith("r") and parts[1].startswith("b"):
        result.update(repeat=int(parts[0][1:]), batch_size=int(parts[1][1:]), policy=parts[2],
                      wave=int(parts[3][1:]))
    elif label.startswith("profile:"):
        result.update(policy="single_request_profile", wave=int(parts[-1]))
    return result


def load_requests(run: Path) -> list[dict]:
    rows = []
    for path in sorted(run.glob("requests_dp*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    return rows


def reduce_clean(runs: list[Path], out: Path) -> None:
    wave_rows = []
    schedule_rows = []
    sct_rows = []
    token_sets: dict[str, set[tuple[int, ...]]] = collections.defaultdict(set)
    for restart, run in enumerate(runs):
        requests = [row for row in load_requests(run) if not row["warmup"]]
        grouped: dict[str, list[dict]] = collections.defaultdict(list)
        for row in requests:
            grouped[row["label"]].append(row)
            token_sets[row["source_request_id"]].add(tuple(row["output_ids"]))
            sct_rows.append({
                "restart": restart, "run": run.name, "label": row["label"],
                "source_request_id": row["source_request_id"], "family": row["family"],
                "prompt_tokens": row["prompt_tokens"], "ttft_s": row["ttft_s"],
                "e2e_s": row["e2e_s"], "processor_s": row["processor_s"],
            })
        policy_totals: dict[tuple[int, str, int], float] = collections.defaultdict(float)
        for label, rows in grouped.items():
            meta = parse_label(label)
            gpu_start = min(row.get("gpu_batch_start_s", row["engine_submit_s"]) for row in rows)
            bct = max(row["complete_s"] for row in rows) - gpu_start
            frontend_bct = max(row["complete_s"] for row in rows) - min(row["start_s"] for row in rows)
            wave_rows.append({
                "restart": restart, "run": run.name, **meta, "requests": len(rows),
                "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
                "image_count": sum(row["image_count"] for row in rows),
                "pixels": sum(row["total_pixels"] for row in rows),
                "gpu_bct_s": bct, "frontend_bct_s": frontend_bct,
                "sct_p50_s": percentile([row["e2e_s"] for row in rows], 0.5),
                "sct_p90_s": percentile([row["e2e_s"] for row in rows], 0.9),
                "sct_p99_s": percentile([row["e2e_s"] for row in rows], 0.99),
            })
            policy_totals[(meta["batch_size"], meta["policy"], meta["repeat"])] += bct
        for (batch_size, policy, repeat), bct in policy_totals.items():
            schedule_rows.append({"restart": restart, "run": run.name, "repeat": repeat,
                                  "batch_size": batch_size, "policy": policy, "gpu_bct_s": bct})

    for rows, name in [(wave_rows, "BCT_RESULTS.csv"), (schedule_rows, "SCHEDULE_BASELINES.csv"),
                       (sct_rows, "SCT_RESULTS.csv")]:
        with (out / name).open("w", newline="") as sink:
            writer = csv.DictWriter(sink, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    correctness = {
        "requests_checked": len(token_sets),
        "exact_greedy_agreement_requests": sum(len(values) == 1 for values in token_sets.values()),
        "all_agree": all(len(values) == 1 for values in token_sets.values()),
        "bad_request_ids": [key for key, values in token_sets.items() if len(values) != 1],
    }
    (out / "CORRECTNESS.json").write_text(json.dumps(correctness, indent=2) + "\n")


def reduce_observer(run: Path, out: Path) -> None:
    requests = load_requests(run)
    # EngineCore appends a collision-avoidance suffix to the frontend ID.
    # Scheduler/operation traces use that internal ID.
    request_map = {row["internal_request_id"]: row for row in requests}
    records = []
    proofs = []
    for path in sorted((run / "operations").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("kind") == "runtime_proof":
                proofs.append(row)
            elif row.get("kind") in {"operation", "moe"}:
                records.append(row)
    # One logical operation is represented on multiple ranks.  The critical
    # same-device duration is the maximum rank-local duration, never a sum.
    group: dict[tuple, list[dict]] = collections.defaultdict(list)
    for row in records:
        ids = tuple(row.get("request_ids", []))
        matched_requests = [request_map[rid] for rid in ids if rid in request_map]
        labels = tuple(sorted({request["label"] for request in matched_requests}))
        # Full-shape warmups deliberately carry labels such as
        # ``full_plan_warmup0``.  Test the recorded warmup bit rather than a
        # label prefix so these operations cannot leak into measured rows.
        if not labels or any(request["warmup"] for request in matched_requests):
            continue
        stage = row.get("stage", "moe" if row.get("kind") == "moe" else "unknown")
        group[(labels, row.get("step", -1), row.get("layer", -1), stage)].append(row)

    by_label: dict[str, dict] = collections.defaultdict(lambda: {
        "vision_ms": 0.0, "attention_ms": 0.0, "router_ms": 0.0,
        "dispatch_ms": 0.0, "expert_ms": 0.0, "combine_ms": 0.0,
        "moe_ms": 0.0, "layer_ms": 0.0, "llm_forward_ms": 0.0,
        "active_experts_sum": 0.0, "route_hhi_sum": 0.0,
        "rank_max_mean_sum": 0.0, "route_layers": 0,
    })
    for (labels, step, layer, stage), rows in group.items():
        if len(labels) != 1:
            continue
        label = labels[0]
        target = by_label[label]
        if stage == "vision_encoder":
            target["vision_ms"] += max(row.get("vision_encoder_ms", 0.0) for row in rows)
        elif stage in {"attention", "router", "layer", "llm_forward"}:
            target[stage + "_ms"] += max(row.get(stage + "_ms", 0.0) for row in rows)
        elif stage == "moe":
            for key in ["dispatch_ms", "expert_ms", "combine_ms", "moe_ms"]:
                target[key] += max(row.get(key, 0.0) for row in rows)
            hist = []
            rank_loads = []
            for row in sorted(rows, key=lambda value: value.get("ep_rank", -1)):
                local = row.get("local_expert_histogram") or []
                hist.extend(local)
                rank_loads.append(sum(local))
            total = sum(hist)
            if total:
                active = sum(value > 0 for value in hist)
                hhi = sum((value / total) ** 2 for value in hist)
                mean_rank = statistics.mean(rank_loads) if rank_loads else 0.0
                target["active_experts_sum"] += active
                target["route_hhi_sum"] += hhi
                target["rank_max_mean_sum"] += max(rank_loads) / mean_rank if mean_rank else 1.0
                target["route_layers"] += 1

    atlas = []
    by_request_label: dict[str, list[dict]] = collections.defaultdict(list)
    for row in requests:
        if not row["warmup"]:
            by_request_label[row["label"]].append(row)
    for label, timing in sorted(by_label.items()):
        request_rows = by_request_label[label]
        route_layers = timing["route_layers"]
        active_experts = timing["active_experts_sum"] / route_layers if route_layers else math.nan
        route_hhi = timing["route_hhi_sum"] / route_layers if route_layers else math.nan
        rank_max_mean = timing["rank_max_mean_sum"] / route_layers if route_layers else math.nan
        component = {key: value for key, value in timing.items()
                     if key not in {"route_layers", "active_experts_sum", "route_hhi_sum", "rank_max_mean_sum"}}
        atlas.append({
            **parse_label(label),
            "requests": len(request_rows),
            "families": "+".join(sorted(row["family"] for row in request_rows)),
            "prompt_tokens": sum(row["prompt_tokens"] for row in request_rows),
            "image_count": sum(row["image_count"] for row in request_rows),
            "pixels": sum(row["total_pixels"] for row in request_rows),
            **component,
            "active_experts_mean": active_experts,
            "route_hhi_mean": route_hhi,
            "rank_max_mean": rank_max_mean,
            "gpu_bct_s": max(row["complete_s"] for row in request_rows) - min(row["gpu_batch_start_s"] for row in request_rows),
        })
    with (out / "MODULE_COST_ATLAS.csv").open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(atlas[0]))
        writer.writeheader(); writer.writerows(atlas)
    (out / "RUNTIME_PROOFS.json").write_text(json.dumps(proofs, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clean", type=Path, nargs="*")
    parser.add_argument("--observer", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.clean:
        reduce_clean(args.clean, args.out)
    if args.observer:
        reduce_observer(args.observer, args.out)


if __name__ == "__main__":
    main()
