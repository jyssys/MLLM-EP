"""Analyze paired clean/profiled actual EP4 first-token runs."""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from poc_mllm_ep_foundation.io import atomic_json


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _timings(root: Path) -> dict[tuple[int, tuple[str, ...]], float]:
    grouped: dict[tuple[int, tuple[str, ...]], list[float]] = defaultdict(list)
    by_wave: dict[int, set[str]] = defaultdict(set)
    rows = []
    for name in glob.glob(str(root / "timings.dp*.jsonl")):
        rows.extend(_jsonl(Path(name)))
    if not rows:
        raise RuntimeError(f"missing repeat-preserving timing ledger under {root}")
    for row in rows:
        if not bool(row.get("anchor_measured", True)):
            continue
        wave = int(row["batch_id"])
        by_wave[wave].add(str(row.get("logical_request_id", row["request_id"])))
        grouped[(wave, ())].append(float(row["wall_ms"]))
    return {
        (wave, tuple(sorted(by_wave[wave]))): max(values)
        for (wave, _), values in grouped.items()
    }


def _profile_rows(root: Path, pattern: str) -> list[dict[str, Any]]:
    output = []
    for name in sorted(glob.glob(str(root / f"profile/raw_live/{pattern}"))):
        rank = int(Path(name).stem.split("rank")[-1])
        for row in _jsonl(Path(name)):
            if bool(row.get("measured", True)):
                output.append({**row, "physical_ep_rank": rank})
    return output


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--ep-profile-root", type=Path,
                        help="optional separate low-overhead EP-subevent pass")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    clean = _timings(args.clean_root)
    profiled = _timings(args.profile_root)
    ep_profile_root = args.ep_profile_root or args.profile_root
    ep_profiled = _timings(ep_profile_root)
    clean_by_ids: dict[tuple[str, ...], list[float]] = defaultdict(list)
    profiled_by_ids: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for (_, ids), value in clean.items():
        clean_by_ids[ids].append(value)
    for (_, ids), value in profiled.items():
        profiled_by_ids[ids].append(value)
    shared = sorted(set(clean_by_ids) & set(profiled_by_ids))
    paired = []
    for ids in shared:
        left = sorted(clean_by_ids[ids])
        right = sorted(profiled_by_ids[ids])
        count = min(len(left), len(right))
        for clean_ms, profile_ms in zip(left[:count], right[:count]):
            paired.append({"request_ids": list(ids), "clean_ms": clean_ms,
                           "profile_ms": profile_ms,
                           "overhead_fraction": (profile_ms - clean_ms) / clean_ms})

    full = _profile_rows(args.profile_root, "full_rank*.jsonl")
    ep = _profile_rows(ep_profile_root, "rank*.jsonl")
    # Full and EP diagnostic passes deliberately use disjoint wave-id ranges
    # so artifacts cannot collide.  Join the two passes by the stable logical
    # request set plus measured-repeat index, never by the launcher-local wave
    # number.
    full_wave_ids: dict[int, set[str]] = defaultdict(set)
    for row in full:
        wave = int(row["wave"])
        full_wave_ids[wave].add(str(row["request_id"]))
    ep_wave_ids: dict[int, set[str]] = defaultdict(set)
    for row in ep:
        wave = int(row["wave"])
        ep_wave_ids[wave].add(str(row["request_id"]))

    Signature = tuple[int, tuple[str, ...]]

    def signature(row: dict[str, Any], ids: dict[int, set[str]]) -> Signature:
        wave = int(row["wave"])
        return int(row.get("iteration", 0)), tuple(sorted(ids[wave]))

    full_groups: dict[tuple[Signature, str, int], list[float]] = defaultdict(list)
    for row in full:
        key = signature(row, full_wave_ids)
        full_groups[(key, str(row["kind"]), int(row["layer"]))].append(float(row["ms"]))
    ep_groups: dict[tuple[Signature, str, int], list[float]] = defaultdict(list)
    for row in ep:
        key = signature(row, ep_wave_ids)
        for kind in ("dispatch", "expert", "combine"):
            value = row[kind].get("critical_ms", row[kind]["ms"])
            ep_groups[(key, kind, int(row["layer"]))].append(float(value))

    per_wave = []
    signatures = sorted({key[0] for key in full_groups})
    for item_signature in signatures:
        def total(
            kind: str,
            groups: dict[tuple[Signature, str, int], list[float]],
        ) -> float:
            return sum(
                max(values)
                for (group_signature, item_kind, _), values in groups.items()
                if group_signature == item_signature and item_kind == kind
            )

        decoder = total("decoder_layer", full_groups)
        attention = total("attention", full_groups)
        moe = total("moe_total", full_groups)
        vision = total("vision_encoder", full_groups)
        dispatch = total("dispatch", ep_groups)
        expert = total("expert", ep_groups)
        combine = total("combine", ep_groups)
        routed_subevent_sum = dispatch + expert + combine
        iteration, ids = item_signature
        clean_wall = _median(clean_by_ids[ids]) if ids in clean_by_ids else None
        per_wave.append({
            "wave": next(
                wave for wave, wave_ids in full_wave_ids.items()
                if tuple(sorted(wave_ids)) == ids
                and any(
                    int(row["wave"]) == wave
                    and int(row.get("iteration", 0)) == iteration
                    for row in full
                )
            ),
            "iteration": iteration,
            "request_ids": list(ids),
            "clean_wall_ms": clean_wall,
            "vision_encoder_ms": vision,
            "decoder_total_ms": decoder,
            "attention_ms": attention,
            "routed_moe_total_ms": moe,
            "decoder_other_ms": decoder - attention - moe,
            "dispatch_critical_ms": dispatch,
            "expert_critical_ms": expert,
            "combine_critical_ms": combine,
            "router_layout_or_overlap_residual_ms": moe - routed_subevent_sum,
            "subevent_sum_ms": routed_subevent_sum,
            "routed_moe_fraction_of_clean_wall": (
                moe / clean_wall if clean_wall else None
            ),
            "unattributed_clean_wall_ms": (
                max(0.0, clean_wall - vision - decoder) if clean_wall else None
            ),
        })

    component_keys = (
        "vision_encoder_ms", "decoder_total_ms", "attention_ms", "routed_moe_total_ms",
        "decoder_other_ms", "dispatch_critical_ms", "expert_critical_ms",
        "combine_critical_ms", "router_layout_or_overlap_residual_ms", "subevent_sum_ms",
    )
    overhead = [row["overhead_fraction"] for row in paired]
    ep_profiled_by_ids: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for (_, ids), value in ep_profiled.items():
        ep_profiled_by_ids[ids].append(value)
    ep_overhead = []
    for ids in sorted(set(clean_by_ids) & set(ep_profiled_by_ids)):
        left = sorted(clean_by_ids[ids])
        right = sorted(ep_profiled_by_ids[ids])
        ep_overhead.extend((profile_ms - clean_ms) / clean_ms
                           for clean_ms, profile_ms in zip(left, right))
    overhead_median = max(_median(overhead), _median(ep_overhead))
    overhead_p90 = max(float(np.quantile(overhead, 0.9)),
                       float(np.quantile(ep_overhead, 0.9)))
    aggregate = {key: _median([row[key] for row in per_wave]) for key in component_keys}
    aggregate["clean_wall_ms"] = _median([
        row["clean_wall_ms"] for row in per_wave if row["clean_wall_ms"] is not None
    ])
    aggregate["routed_moe_fraction_of_clean_wall"] = _median([
        row["routed_moe_fraction_of_clean_wall"] for row in per_wave
        if row["routed_moe_fraction_of_clean_wall"] is not None
    ])
    aggregate["unattributed_clean_wall_ms"] = _median([
        row["unattributed_clean_wall_ms"] for row in per_wave
        if row["unattributed_clean_wall_ms"] is not None
    ])
    result: dict[str, Any] = {
        "schema_version": "mllm_ep4_full_profile.v1",
        "scope": "ACTUAL_FULL_MODEL_EP4_INPUT_READY_FIRST_TOKEN",
        "parallelism": {"tp": 2, "dp": 2, "ep": 4, "sp": 1, "pp": 1},
        "paired_timing_repeats": len(paired),
        "profiling_overhead": {
            "median_fraction": overhead_median,
            "p90_fraction": overhead_p90,
            "full_pass_median_fraction": _median(overhead),
            "ep_pass_median_fraction": _median(ep_overhead),
            "separate_passes": args.ep_profile_root is not None,
            "status": ("PASS_LE_5_PERCENT" if overhead_median <= 0.05 else
                       "DIAGNOSTIC_OVER_10_PERCENT" if overhead_median > 0.10 else
                       "WARN_5_TO_10_PERCENT"),
        },
        "aggregate_median_ms": aggregate,
        "component_semantics": {
            "routed_moe_total_ms": "outer Qwen3MoeSparseMoeBlock CUDA span; includes router/layout/EP waits",
            "dispatch_expert_combine": "nested critical spans; diagnostic and not additively combined with outer MoE",
            "router_layout_or_overlap_residual_ms": "outer MoE minus sum of nested critical spans; may include overlap residual",
            "shared_expert": "ABSENT_IN_CHECKPOINT_CONFIG",
            "wall_boundary": "after processor/input preparation through one generated token",
        },
        "paired_wall": paired,
        "per_wave": per_wave,
    }
    atomic_json(args.output, result)
    print(json.dumps({"paired_timing_repeats": len(paired),
                      "profiling_overhead": result["profiling_overhead"],
                      "aggregate_median_ms": aggregate}, indent=2))


if __name__ == "__main__":
    main()
