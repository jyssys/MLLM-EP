#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from poc_mllm_rank_aware_active_expert_reduction.contract import atomic_json


def _read(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("stage.pid*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _bootstrap(values: list[float], seed: int = 3407) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(10000, len(array)))].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _pct(base: float, changed: float) -> float:
    return 100.0 * (base - changed) / base if base else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    raw = _read(args.run / "raw_stage")
    grouped: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        grouped[(row["condition"], int(row["replicate"]), int(row["decode_step"]), int(row["layer"]))].append(row)
    invocations = []
    for (condition, replicate, step, layer), rows in sorted(grouped.items()):
        by_rank = {int(row["ep_rank"]): row for row in rows}
        if set(by_rank) != {0, 1, 2, 3}:
            raise RuntimeError((condition, replicate, step, layer, sorted(by_rank)))
        rank_rows = [by_rank[index] for index in range(4)]
        invocations.append({
            "condition": condition, "replicate": replicate, "step": step, "layer": layer,
            "dispatch_ms": max(row["dispatch_ms"] for row in rank_rows),
            "expert_execution_ms": max(row["expert_execution_ms"] for row in rank_rows),
            "combine_ms": max(row["combine_ms"] for row in rank_rows),
            "decode_moe_ms": max(row["decode_moe_ms"] for row in rank_rows),
            "pairs_by_rank": [row["token_expert_pairs"] for row in rank_rows],
            "active_experts_by_rank": [row["distinct_active_experts"] for row in rank_rows],
            "total_pairs": sum(row["token_expert_pairs"] for row in rank_rows),
            "max_rank_pairs": max(row["token_expert_pairs"] for row in rank_rows),
            "max_rank_active_experts": max(row["distinct_active_experts"] for row in rank_rows),
            "total_active_experts": sum(row["distinct_active_experts"] for row in rank_rows),
        })
    conditions = sorted({row["condition"] for row in invocations})
    aggregate = {}
    for condition in conditions:
        rows = [row for row in invocations if row["condition"] == condition]
        aggregate[condition] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("dispatch_ms", "expert_execution_ms", "combine_ms", "decode_moe_ms",
                        "total_pairs", "max_rank_pairs", "max_rank_active_experts",
                        "total_active_experts")
        } | {"observations": len(rows)}

    keyed = {(row["condition"], row["replicate"], row["step"], row["layer"]): row for row in invocations}
    clusters: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    comparisons = {}
    for candidate in ("NATIVE_TAIL_K6", "RANK_AWARE_K6", "SAME_PAIR_COLLAPSE25"):
        paired_rows = []
        for row in invocations:
            if row["condition"] != candidate:
                continue
            base = keyed[("NATIVE_K8", row["replicate"], row["step"], row["layer"])]
            paired_rows.append((base, row))
        result: dict[str, Any] = {"pairs": len(paired_rows)}
        for metric in ("dispatch_ms", "expert_execution_ms", "combine_ms", "decode_moe_ms",
                       "max_rank_active_experts", "total_pairs", "max_rank_pairs"):
            deltas = []
            for base, changed in paired_rows:
                value = _pct(float(base[metric]), float(changed[metric]))
                clusters[(candidate + ":" + metric, changed["replicate"], changed["step"])].append(value)
            cluster_values = [float(np.mean(values)) for (name, _, _), values in clusters.items()
                              if name == candidate + ":" + metric]
            result[metric + "_reduction_pct"] = float(np.mean(cluster_values))
            result[metric + "_reduction_95ci"] = _bootstrap(cluster_values)
        comparisons[candidate] = result

    tail = comparisons["NATIVE_TAIL_K6"]
    rank = comparisons["RANK_AWARE_K6"]
    collapse = comparisons["SAME_PAIR_COLLAPSE25"]
    rank_vs_tail: dict[str, Any] = {}
    for metric in ("expert_execution_ms", "decode_moe_ms", "max_rank_active_experts",
                   "total_pairs", "max_rank_pairs"):
        cluster_values = []
        for replicate in sorted({row["replicate"] for row in invocations}):
            for step in sorted({row["step"] for row in invocations}):
                layer_values = []
                for layer in range(48):
                    tail_row = keyed.get(("NATIVE_TAIL_K6", replicate, step, layer))
                    rank_row = keyed.get(("RANK_AWARE_K6", replicate, step, layer))
                    if tail_row is not None and rank_row is not None:
                        layer_values.append(_pct(float(tail_row[metric]), float(rank_row[metric])))
                if layer_values:
                    cluster_values.append(float(np.mean(layer_values)))
        rank_vs_tail[metric + "_reduction_pct"] = float(np.mean(cluster_values))
        rank_vs_tail[metric + "_reduction_95ci"] = _bootstrap(cluster_values, seed=3410)
    rank_increment = (
        rank["max_rank_active_experts_reduction_pct"]
        - tail["max_rank_active_experts_reduction_pct"]
    )
    same_pair_exact = (
        abs(collapse["total_pairs_reduction_pct"]) < 1e-9
        and abs(collapse["max_rank_pairs_reduction_pct"]) < 1e-9
    )
    same_pair_near_matched = (
        abs(collapse["total_pairs_reduction_pct"]) < 1.0
        and abs(collapse["max_rank_pairs_reduction_pct"]) < 2.0
    )
    collapse_expert_ci = collapse["expert_execution_ms_reduction_95ci"]
    collapse_moe_ci = collapse["decode_moe_ms_reduction_95ci"]
    captured_steps = sorted({int(row["step"]) for row in invocations})
    captured_replicates = sorted({int(row["replicate"]) for row in invocations})
    expected_invocations = 4 * len(captured_replicates) * len(captured_steps) * 48
    complete_capture = (
        len(captured_steps) >= 4
        and len(captured_replicates) >= 3
        and len(invocations) >= expected_invocations * 0.99
    )
    if not same_pair_near_matched or not complete_capture:
        verdict = "RUNTIME_INCONCLUSIVE"
    elif collapse_moe_ci[0] > 0 and collapse_expert_ci[0] > 0 and rank_increment > 0:
        verdict = "PHYSICAL_RANK_COLLAPSE_SUPPORTED"
    elif collapse_expert_ci[0] > 0 and rank_increment > 0:
        verdict = "EXPERT_STAGE_ONLY_HEADROOM"
    else:
        verdict = "STRUCTURAL_ONLY_NO_RUNTIME_GAIN"
    payload = {
        "schema_version": "rank_bridge_analysis.v1",
        "evidence": "MEASURED_TRUE_EP4_FROZEN_INPUT_ROUTE_INTERVENTION",
        "raw_event_count": len(raw), "invocation_count": len(invocations),
        "aggregate": aggregate, "comparisons_vs_native_k8": comparisons,
        "rank_aware_vs_native_tail": rank_vs_tail,
        "rank_aware_increment_vs_native_tail_pp": rank_increment,
        "captured_decode_steps": captured_steps,
        "captured_replicates": captured_replicates,
        "complete_capture": complete_capture,
        "same_pair_count_and_rank_load_exact": same_pair_exact,
        "same_pair_count_and_rank_load_near_matched": same_pair_near_matched,
        "bootstrap_cluster": "replicate_x_decode_step; layer events averaged inside cluster",
        "verdict": verdict,
        "limitations": [
            "SAME_PAIR_COLLAPSE25 is a non-semantic controlled route replay, not a deployable policy.",
            "RANK_AWARE_K6 is a physical-headroom oracle and does not use functional safety labels.",
            "CUDA event durations are reduced by max over the four EP ranks per layer invocation.",
            "A fixed128 generation has 127 decode-model forwards because its first token is sampled from prefill; requested step 128 therefore has no MoE invocation. The captured physical bridge uses steps 16/32/64/96.",
        ],
    }
    atomic_json(args.output, payload)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# PoC-1 — TRUE EP4 rank-aware physical bridge\n\n"
        f"Evidence: **{payload['evidence']}**. Verdict: **{verdict}**.\n\n"
        "The controlled `SAME_PAIR_COLLAPSE25` lane transforms each condition's current "
        "native route, keeps every path on its original owner rank, and changes only the "
        "within-rank expert identity. Path count and per-rank destinations are exact within "
        "each invocation; cross-wave matching is measured explicitly below.\n\n"
        "## Primary paired results\n\n"
        "| Condition | max-rank active reduction | expert execution gain | decode MoE gain | dispatch gain | combine gain |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {name} | {row['max_rank_active_experts_reduction_pct']:.3f}% "
            f"| {row['expert_execution_ms_reduction_pct']:.3f}% "
            f"[{row['expert_execution_ms_reduction_95ci'][0]:.3f}, {row['expert_execution_ms_reduction_95ci'][1]:.3f}] "
            f"| {row['decode_moe_ms_reduction_pct']:.3f}% "
            f"[{row['decode_moe_ms_reduction_95ci'][0]:.3f}, {row['decode_moe_ms_reduction_95ci'][1]:.3f}] "
            f"| {row['dispatch_ms_reduction_pct']:.3f}% | {row['combine_ms_reduction_pct']:.3f}% |"
            for name, row in comparisons.items()
        )
        + "\n\n"
        f"Rank-aware increment over native tail in max-rank active-expert reduction: "
        f"**{rank_increment:.3f} pp**. Exact cross-run pair/rank-load match: **{same_pair_exact}**; "
        f"near-match gate: **{same_pair_near_matched}**. Each collapse transform itself preserves "
        "the current invocation's rank destinations and path count exactly.\n\n"
        f"Direct matched K6 rank-aware vs native-tail expert-stage gain: "
        f"**{rank_vs_tail['expert_execution_ms_reduction_pct']:.3f}%** "
        f"(95% CI [{rank_vs_tail['expert_execution_ms_reduction_95ci'][0]:.3f}, "
        f"{rank_vs_tail['expert_execution_ms_reduction_95ci'][1]:.3f}]); decode-MoE gain: "
        f"**{rank_vs_tail['decode_moe_ms_reduction_pct']:.3f}%** "
        f"(95% CI [{rank_vs_tail['decode_moe_ms_reduction_95ci'][0]:.3f}, "
        f"{rank_vs_tail['decode_moe_ms_reduction_95ci'][1]:.3f}]).\n\n"
        "No TPOT/E2E claim is made; this report isolates the physical MoE bridge.\n",
        encoding="utf-8",
    )
    print(json.dumps({"verdict": verdict, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
