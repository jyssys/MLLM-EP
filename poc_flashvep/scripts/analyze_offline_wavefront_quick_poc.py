"""Aggregate four EP-rank replays and publish the offline wavefront gate."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median": float(statistics.median(values)),
        "p10": float(_percentile(values, 0.1)),
        "p90": float(_percentile(values, 0.9)),
        "mean": float(statistics.fmean(values)),
        "stddev": float(statistics.stdev(values) if len(values) > 1 else 0.0),
    }


def _critical_samples(rows: list[dict[str, Any]], key: str) -> list[float]:
    rank_samples = [[float(value) for value in row[key]] for row in rows]
    count = min(len(values) for values in rank_samples)
    return [max(values[index] for values in rank_samples) for index in range(count)]


def _paired_speedup(baseline: list[float], candidate: list[float]) -> dict[str, float]:
    count = min(len(baseline), len(candidate))
    return _stats([baseline[index] / candidate[index] for index in range(count)])


def _load_rank_results(directory: Path) -> list[dict[str, Any]]:
    paths = [directory / f"rank{rank}.json" for rank in range(4)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing rank results: {missing}")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(row.get("status") not in {"ok", "correctness_failed"} for row in rows):
        raise RuntimeError(f"rank replay failed: {[row.get('status') for row in rows]}")
    return rows


def _aggregate_o1(rank_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for batch in (16, 32, 64, 128):
        rows = [
            next(row for row in rank["o1"] if row["batch_equivalent"] == batch)
            for rank in rank_results
        ]
        wall = _critical_samples(rows, "wall_ms")
        dispatch = _critical_samples(rows, "dispatch_ms")
        expert = _critical_samples(rows, "expert_ms")
        combine = _critical_samples(rows, "combine_ms")
        wall_stats = _stats(wall)
        dispatch_stats = _stats(dispatch)
        expert_stats = _stats(expert)
        combine_stats = _stats(combine)
        tokens = int(rows[0]["workload"]["real_tokens"])
        assignments = int(rows[0]["workload"]["total_routed_assignments"])
        critical_rank = max(
            range(4), key=lambda rank: statistics.median(rows[rank]["wall_ms"])
        )
        critical_row = rows[critical_rank]
        output.append(
            {
                "batch_equivalent": batch,
                "workload": rows[0]["workload"],
                "wall_ms": wall_stats,
                "dispatch_ms": dispatch_stats,
                "expert_ms": expert_stats,
                "combine_ms": combine_stats,
                "expert_fraction": expert_stats["median"] / wall_stats["median"],
                "communication_to_expert": (
                    dispatch_stats["median"] + combine_stats["median"]
                )
                / expert_stats["median"],
                "tokens_per_second": tokens * 1000.0 / wall_stats["median"],
                "assignments_per_second": assignments
                * 1000.0
                / wall_stats["median"],
                "critical_rank": critical_rank,
                "critical_rank_breakdown_ms": {
                    "dispatch": statistics.median(critical_row["dispatch_ms"]),
                    "expert": statistics.median(critical_row["expert_ms"]),
                    "combine": statistics.median(critical_row["combine_ms"]),
                    "wall": statistics.median(critical_row["wall_ms"]),
                },
                "route_identity_all_ranks": all(row["route_identity"] for row in rows),
                "token_order_restoration_all_ranks": all(
                    row["token_order_restoration"] for row in rows
                ),
                "peak_memory_allocated_bytes": max(
                    int(row["peak_memory_allocated_bytes"]) for row in rows
                ),
            }
        )
    return output


def _aggregate_o2(rank_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    configurations = sorted(
        {
            (int(row["batch_equivalent"]), int(row["microbatches"]))
            for row in rank_results[0]["o2"]
        }
    )
    output = []
    for batch, microbatches in configurations:
        rows = [
            next(
                row
                for row in rank["o2"]
                if row["batch_equivalent"] == batch
                and row["microbatches"] == microbatches
            )
            for rank in rank_results
        ]
        full_rows = [row["full_batch_serial"] for row in rows]
        micro_rows = [row["microbatch_serial"] for row in rows]
        wave_rows = [row["expert_centered_wavefront"] for row in rows]
        full_wall = _critical_samples(full_rows, "wall_ms")
        micro_wall = _critical_samples(micro_rows, "wall_ms")
        wave_wall = _critical_samples(wave_rows, "wall_ms")
        full_stats = _stats(full_wall)
        micro_stats = _stats(micro_wall)
        wave_stats = _stats(wave_wall)
        speedup_full = _paired_speedup(full_wall, wave_wall)
        speedup_micro = _paired_speedup(micro_wall, wave_wall)
        full_expert = statistics.median(_critical_samples(full_rows, "expert_ms"))
        micro_expert = statistics.median(_critical_samples(micro_rows, "expert_ms"))
        full_communication = statistics.median(
            [
                left + right
                for left, right in zip(
                    _critical_samples(full_rows, "dispatch_ms"),
                    _critical_samples(full_rows, "combine_ms"),
                )
            ]
        )
        micro_communication = statistics.median(
            [
                left + right
                for left, right in zip(
                    _critical_samples(micro_rows, "dispatch_ms"),
                    _critical_samples(micro_rows, "combine_ms"),
                )
            ]
        )
        rank_speedups = [
            statistics.median(row["full_batch_serial"]["wall_ms"])
            / statistics.median(row["expert_centered_wavefront"]["wall_ms"])
            for row in rows
        ]
        overlap_by_rank = [
            statistics.median(
                row["expert_centered_wavefront"]["actual_overlap_fraction"]
            )
            for row in rows
        ]
        tokens = int(rows[0]["workload"]["real_tokens"])
        assignments = int(rows[0]["workload"]["total_routed_assignments"])
        critical_rank = max(
            range(4),
            key=lambda rank: statistics.median(
                rows[rank]["expert_centered_wavefront"]["wall_ms"]
            ),
        )
        critical_breakdown = {}
        for variant in (
            "full_batch_serial",
            "microbatch_serial",
            "expert_centered_wavefront",
        ):
            variant_row = rows[critical_rank][variant]
            critical_breakdown[variant] = {
                stage: statistics.median(variant_row[f"{stage}_ms"])
                for stage in ("dispatch", "expert", "combine", "wall")
            }
        output.append(
            {
                "batch_equivalent": batch,
                "microbatches": microbatches,
                "microbatch_tokens_global": rows[0]["microbatch_tokens_global"],
                "workload": rows[0]["workload"],
                "full_batch_serial_ms": full_stats,
                "microbatch_serial_ms": micro_stats,
                "expert_centered_wavefront_ms": wave_stats,
                "speedup_vs_full_batch": speedup_full,
                "speedup_vs_microbatch_serial": speedup_micro,
                "throughput_tokens_per_second": tokens * 1000.0 / wave_stats["median"],
                "throughput_assignments_per_second": assignments
                * 1000.0
                / wave_stats["median"],
                "expert_fragmentation_penalty": micro_expert / full_expert - 1.0,
                "collective_repetition_penalty": micro_communication
                / full_communication
                - 1.0,
                "actual_overlap_fraction_by_rank": overlap_by_rank,
                "actual_overlap_fraction_min_rank": min(overlap_by_rank),
                "dispatch_expert_overlap_ms_by_rank": [
                    statistics.median(
                        row["expert_centered_wavefront"][
                            "dispatch_expert_overlap_ms"
                        ]
                    )
                    for row in rows
                ],
                "expert_combine_overlap_ms_by_rank": [
                    statistics.median(
                        row["expert_centered_wavefront"][
                            "expert_combine_overlap_ms"
                        ]
                    )
                    for row in rows
                ],
                "rank_speedups": rank_speedups,
                "critical_rank": critical_rank,
                "critical_rank_stage_breakdown_ms": critical_breakdown,
                "correctness_all_ranks": all(
                    row["microbatch_serial_correctness"]["passed"]
                    and row["wavefront_correctness"]["passed"]
                    and row["route_identity"]
                    and row["token_order_restoration"]
                    for row in rows
                ),
                "max_abs_error": max(
                    float(row["wavefront_correctness"]["max_abs_error"])
                    for row in rows
                ),
                "mean_abs_error_max_rank": max(
                    float(row["wavefront_correctness"]["mean_abs_error"])
                    for row in rows
                ),
                "cosine_similarity_min_rank": min(
                    float(row["wavefront_correctness"]["cosine_similarity"])
                    for row in rows
                ),
                "peak_memory_allocated_bytes": max(
                    int(row["peak_memory_allocated_bytes"]) for row in rows
                ),
                "full_batch_peak_memory_allocated_bytes": max(
                    int(row["full_batch_peak_memory_allocated_bytes"])
                    for row in rows
                ),
            }
        )
    return output


def _gate(o2: list[dict[str, Any]]) -> dict[str, Any]:
    correctness = bool(o2) and all(row["correctness_all_ranks"] for row in o2)
    best_speedup = max(
        (row["speedup_vs_full_batch"]["median"] for row in o2), default=0.0
    )
    representative = {}
    for row in o2:
        batch = int(row["batch_equivalent"])
        representative[batch] = max(
            representative.get(batch, 0.0),
            float(row["speedup_vs_full_batch"]["median"]),
        )
    speedup_gate = (
        any(value >= 1.15 for value in representative.values())
        and sum(value >= 1.10 for value in representative.values()) >= 2
    )
    uncertainty_clear = bool(o2) and all(
        row["speedup_vs_full_batch"]["p10"] > 1.0
        for row in o2
        if row["speedup_vs_full_batch"]["median"] >= 1.10
    )
    actual_overlap = bool(o2) and any(
        row["actual_overlap_fraction_min_rank"] > 0.01 for row in o2
    )
    fragmentation = bool(o2) and all(
        row["expert_fragmentation_penalty"] < 0.15 for row in o2
    )
    rank_balanced = bool(o2) and any(
        min(row["rank_speedups"]) > 1.0 for row in o2
    )
    memory_reasonable = bool(o2) and all(
        row["peak_memory_allocated_bytes"]
        <= 1.5 * row["full_batch_peak_memory_allocated_bytes"]
        for row in o2
    )
    if not correctness or not actual_overlap or best_speedup < 1.05:
        decision = "NO-GO"
    elif all(
        (
            speedup_gate,
            uncertainty_clear,
            fragmentation,
            rank_balanced,
            memory_reasonable,
        )
    ):
        decision = "GO"
    else:
        decision = "HOLD"
    return {
        "decision": decision,
        "best_core_speedup": best_speedup,
        "core_exceeds_1_10": best_speedup > 1.10,
        "attention_router_extension_executed": False,
        "attention_router_extension_stop_rule": (
            "not executed because core best speedup did not exceed 1.10x"
            if best_speedup <= 1.10
            else "eligible by speed only; not part of the completed core D/E/C run"
        ),
        "criteria": {
            "correctness": correctness,
            "representative_speedup_gate": speedup_gate,
            "gain_exceeds_uncertainty": uncertainty_clear,
            "actual_overlap": actual_overlap,
            "fragmentation_below_15_percent": fragmentation,
            "not_one_rank_accident": rank_balanced,
            "memory_reasonable": memory_reasonable,
        },
    }


def _render_report(
    result_dir: Path,
    capture: dict[str, Any],
    rank_results: list[dict[str, Any]],
    o1: list[dict[str, Any]],
    o2: list[dict[str, Any]],
    gate: dict[str, Any],
) -> str:
    lines = [
        "# FlashVEP offline expert-centered wavefront Quick PoC",
        "",
        f"- 결론: **{gate['decision']}**",
        f"- Core 최고 speedup: **{gate['best_core_speedup']:.3f}x**",
        f"- 결과 디렉터리: `{result_dir}`",
        "- 범위: layer 24 Core D/E/C만; vLLM request는 capture 1회에만 사용",
        "- Workload: `synthetic batch scaling from real captured request`",
        "- Backend: 실제 `TritonExperts`, 명시적 NCCL all-gather/reduce-scatter",
        "",
        "## Capture",
        "",
        f"- token: {capture['original_token_count']} (vision {capture['vision_token_count']})",
        f"- hidden/intermediate/top-k: {capture['hidden_size']} / {capture['expert_intermediate_size']} / {capture['top_k']}",
        f"- experts: {capture['global_num_experts']} total, {capture['local_experts_per_rank']} per EP rank",
        f"- runtime DPEP chunks: `{rank_results[0]['capture_runtime']['dp_chunk_sizes']}`",
        "",
        "## Routed workload",
        "",
        "| B_eq | real/vision tokens | assignments | rank assignments | critical assignments | max local-expert tokens | active local experts/rank |",
        "|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in o1:
        workload = row["workload"]
        lines.append(
            f"| {row['batch_equivalent']} | {workload['real_tokens']}/{workload['vision_tokens']} | "
            f"{workload['total_routed_assignments']} | `{workload['rank_routed_assignments']}` | "
            f"{workload['critical_rank_assignments']} | {workload['max_local_expert_token_count']} | "
            f"`{workload['active_local_experts_per_rank']}` |"
        )
    lines.extend(
        [
        "",
        "## O1 — full-batch serial D→E→C",
        "",
        "| B_eq | tokens | D@critical ms | E@critical ms | C@critical ms | E max ms | MoE critical ms | expert-max % | M assignments/s | critical rank |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in o1:
        lines.append(
            "| {batch_equivalent} | {tokens} | {dispatch:.3f} | {expert_critical:.3f} | "
            "{combine:.3f} | {expert_max:.3f} | {wall:.3f} | {fraction:.1f} | {throughput:.2f} | {rank} |".format(
                batch_equivalent=row["batch_equivalent"],
                tokens=row["workload"]["real_tokens"],
                dispatch=row["critical_rank_breakdown_ms"]["dispatch"],
                expert_critical=row["critical_rank_breakdown_ms"]["expert"],
                combine=row["critical_rank_breakdown_ms"]["combine"],
                expert_max=row["expert_ms"]["median"],
                wall=row["wall_ms"]["median"],
                fraction=100.0 * row["expert_fraction"],
                throughput=row["assignments_per_second"] / 1e6,
                rank=row["critical_rank"],
            )
        )
    lines.extend(
        [
            "",
            "`E max`는 rank별 expert duration의 최대값이고, `D/E/C@critical`은 같은 critical-wall rank의 coherent breakdown입니다.",
        ]
    )
    lines.extend(
        [
            "",
            "## O2 — expert-centered wavefront",
            "",
            "| B_eq | K | full ms | micro-serial ms | wavefront ms | vs full | vs micro | frag. | repeat coll. | min-rank overlap |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in o2:
        lines.append(
            "| {batch_equivalent} | {microbatches} | {full:.3f} | {micro:.3f} | "
            "{wave:.3f} | {speedup:.3f}x | {micro_speedup:.3f}x | {frag:.1f}% | "
            "{collective:.1f}% | {overlap:.1f}% |".format(
                batch_equivalent=row["batch_equivalent"],
                microbatches=row["microbatches"],
                full=row["full_batch_serial_ms"]["median"],
                micro=row["microbatch_serial_ms"]["median"],
                wave=row["expert_centered_wavefront_ms"]["median"],
                speedup=row["speedup_vs_full_batch"]["median"],
                micro_speedup=row["speedup_vs_microbatch_serial"]["median"],
                frag=100.0 * row["expert_fragmentation_penalty"],
                collective=100.0 * row["collective_repetition_penalty"],
                overlap=100.0 * row["actual_overlap_fraction_min_rank"],
            )
        )
    lines.extend(
        [
            "",
            "### Critical-rank stage breakdown",
            "",
            "| B_eq/K | variant | D sum ms | E sum ms | C sum ms | wall ms |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in o2:
        for variant, label in (
            ("full_batch_serial", "full serial"),
            ("microbatch_serial", "micro serial"),
            ("expert_centered_wavefront", "wavefront"),
        ):
            stage = row["critical_rank_stage_breakdown_ms"][variant]
            lines.append(
                f"| {row['batch_equivalent']}/{row['microbatches']} | {label} | "
                f"{stage['dispatch']:.3f} | {stage['expert']:.3f} | "
                f"{stage['combine']:.3f} | {stage['wall']:.3f} |"
            )
    worst_max = max((row["max_abs_error"] for row in o2), default=float("nan"))
    worst_mean = max(
        (row["mean_abs_error_max_rank"] for row in o2), default=float("nan")
    )
    min_cosine = min(
        (row["cosine_similarity_min_rank"] for row in o2), default=float("nan")
    )
    lines.extend(
        [
            "",
            "## Correctness / overlap / gate",
            "",
            f"- assert_close 전체 rank/config: `{gate['criteria']['correctness']}`",
            f"- max abs / max-rank mean abs / min cosine: `{worst_max:.6g}` / `{worst_mean:.6g}` / `{min_cosine:.9f}`",
            f"- route identity와 token/order restoration: `{'PASS' if gate['criteria']['correctness'] else 'FAIL'}`",
            f"- 실제 CUDA event overlap: `{gate['criteria']['actual_overlap']}`",
            f"- fragmentation <15%: `{gate['criteria']['fragmentation_below_15_percent']}`",
            f"- gain > uncertainty: `{gate['criteria']['gain_exceeds_uncertainty']}`",
            f"- rank 균형: `{gate['criteria']['not_one_rank_accident']}`",
            f"- memory 합리성: `{gate['criteria']['memory_reasonable']}`",
            "",
            f"Attention/Router 확장: **미수행** — {gate['attention_router_extension_stop_rule']}.",
            "",
            f"최종 판정은 **{gate['decision']}**입니다.",
            "",
            "다음 단 하나의 작업: 이 결과를 archive하고 이번 mechanism branch를 중단합니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.report, args.gate, args.result_dir / "analysis.json"):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    rank_results = _load_rank_results(args.result_dir)
    capture = rank_results[0]["capture_metadata"]
    o1 = _aggregate_o1(rank_results)
    o2 = _aggregate_o2(rank_results)
    gate = _gate(o2)
    analysis = {
        "result_dir": str(args.result_dir),
        "capture": capture,
        "o1": o1,
        "o2": o2,
        "gate": gate,
    }
    (args.result_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.gate.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        _render_report(args.result_dir, capture, rank_results, o1, o2, gate),
        encoding="utf-8",
    )
    args.gate.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
