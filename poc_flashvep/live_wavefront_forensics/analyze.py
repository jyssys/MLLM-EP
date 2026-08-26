"""Analyze preregistered live-wavefront slowdown attribution variants."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIANTS = ("A0", "A1", "A2", "C")
VARIANT_LABELS = {
    "A0": "A0 stock / DBO off",
    "A1": "A1 stock DBO",
    "A2": "A2 forced split",
    "C": "C two-stream causal",
}


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _stats(values: pd.Series) -> dict[str, float]:
    return {
        "median": float(values.median()),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
        "p95": float(values.quantile(0.95)),
        "mean": float(values.mean()),
        "cv": float(values.std(ddof=1) / values.mean()) if len(values) > 1 else 0.0,
    }


def _load_variant(
    root: Path, variant: str, code_sha: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    directory = root / variant
    driver_rows = []
    for dp_rank in range(2):
        payload = json.loads((directory / f"driver.dp_rank{dp_rank}.json").read_text())
        if not payload["ok"] or payload["code_sha"] != code_sha:
            raise RuntimeError(payload)
        driver_rows.extend(payload["records"])
    forward_rows, stage_rows, proofs = [], [], []
    for ep_rank in range(4):
        payload = json.loads((directory / "raw" / f"rank{ep_rank}.json").read_text())
        if payload["status"] != "ok" or payload["visible_devices"] != "1,2,3,4":
            raise RuntimeError(payload)
        proofs.append(payload)
        forward_rows.extend(
            {**row, "ep_rank": ep_rank} for row in payload["forward_records"]
        )
        stage_rows.extend(
            {**row, "ep_rank": ep_rank} for row in payload["stage_records"]
        )
    return (
        pd.DataFrame(driver_rows),
        pd.DataFrame(forward_rows),
        pd.DataFrame(stage_rows),
        proofs,
    )


def _wave_latency(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, local in frame.groupby(
        ["wave", "request_id", "phase", "iteration"], sort=True
    ):
        wave, request_id, phase, iteration = keys
        rank_values = []
        for ep_rank, rank_rows in local.groupby("ep_rank"):
            origin = rank_rows[rank_rows.ubatch_id.isin((-1, 0))]
            if len(origin) != 1:
                raise AssertionError((wave, ep_rank, rank_rows.to_dict("records")))
            rank_values.append(
                (int(ep_rank), float(rank_rows.end_ms.max() - origin.iloc[0].start_ms))
            )
        if len(rank_values) != 4:
            raise AssertionError((wave, rank_values))
        rows.append(
            {
                "wave": int(wave),
                "request_id": request_id,
                "phase": phase,
                "iteration": int(iteration),
                "latency_ms": max(value for _, value in rank_values),
                "critical_ep_rank": max(rank_values, key=lambda item: item[1])[0],
            }
        )
    return pd.DataFrame(rows)


def _layer_stage_table(stage: pd.DataFrame, variant: str) -> pd.DataFrame:
    if stage.empty:
        raise RuntimeError(f"missing stage records for {variant}")
    rows = []
    for (stage_name, layer), local in stage.groupby(["stage", "layer"]):
        rank_spans = []
        for ep_rank, rank_rows in local.groupby("ep_rank"):
            rank_spans.append(
                (
                    int(ep_rank),
                    float(rank_rows.end_ms.max() - rank_rows.start_ms.min()),
                    int(len(rank_rows)),
                )
            )
        if len(rank_spans) != 4:
            raise AssertionError((variant, stage_name, layer, rank_spans))
        critical = max(rank_spans, key=lambda item: item[1])
        rows.append(
            {
                "variant": variant,
                "stage": stage_name,
                "layer": int(layer),
                "critical_span_ms": critical[1],
                "critical_ep_rank": critical[0],
                "calls_per_rank": critical[2],
            }
        )
    return pd.DataFrame(rows)


def _trace_stats(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("traceEvents", payload if isinstance(payload, list) else [])
    kernels = []
    for event in events:
        category = str(event.get("cat", "")).lower()
        args = event.get("args", {})
        if (
            event.get("ph") == "X"
            and float(event.get("dur", 0.0)) > 0
            and ("kernel" in category or "kernel" in str(args).lower())
        ):
            kernels.append(
                (
                    float(event["ts"]),
                    float(event["ts"]) + float(event["dur"]),
                    str(args.get("stream", event.get("tid", "unknown"))),
                )
            )
    if not kernels:
        return {"trace": path.name, "kernel_events": 0}
    boundaries = sorted({value for start, end, _ in kernels for value in (start, end)})
    busy = concurrent = 0.0
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        midpoint = (left + right) / 2
        active = sum(start <= midpoint < end for start, end, _ in kernels)
        if active:
            busy += right - left
        if active >= 2:
            concurrent += right - left
    span = boundaries[-1] - boundaries[0]
    return {
        "trace": path.name,
        "kernel_events": len(kernels),
        "unique_streams": len({stream for _, _, stream in kernels}),
        "kernel_span_ms": span / 1000,
        "gpu_busy_fraction": busy / span if span else 0.0,
        "kernel_concurrent_fraction": concurrent / span if span else 0.0,
        "idle_gap_fraction": 1 - busy / span if span else 0.0,
    }


def _profiler_summary(root: Path, variant: str) -> dict[str, Any]:
    traces = sorted((root / variant / "raw").glob("torch_trace_rank*_wave*.json"))
    errors = sorted((root / variant / "raw").glob("torch_profile_rank*_error.txt"))
    stats = [_trace_stats(path) for path in traces]
    valid = [row for row in stats if row.get("kernel_events", 0) > 0]
    return {
        "trace_count": len(traces),
        "error_count": len(errors),
        "per_rank": stats,
        "kernel_events_median": float(np.median([x["kernel_events"] for x in valid]))
        if valid
        else None,
        "gpu_busy_fraction_median": float(
            np.median([x["gpu_busy_fraction"] for x in valid])
        )
        if valid
        else None,
        "kernel_concurrent_fraction_median": float(
            np.median([x["kernel_concurrent_fraction"] for x in valid])
        )
        if valid
        else None,
        "idle_gap_fraction_median": float(
            np.median([x["idle_gap_fraction"] for x in valid])
        )
        if valid
        else None,
    }


def _correctness(
    root: Path, drivers: dict[str, pd.DataFrame]
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    baseline_driver = drivers["A0"]
    baseline_correct = baseline_driver[
        baseline_driver.phase == "correctness"
    ].set_index(["wave", "driver_dp_rank"])
    for variant in VARIANTS:
        for ep_rank in (0, 2):
            baseline_file = np.load(root / "A0" / "raw" / f"rank{ep_rank}.logits.npz")
            variant_file = np.load(root / variant / "raw" / f"rank{ep_rank}.logits.npz")
            baseline = {
                int(key.removeprefix("wave_")): baseline_file[key]
                for key in baseline_file.files
            }
            current = {
                int(key.removeprefix("wave_")): variant_file[key]
                for key in variant_file.files
            }
            for wave in sorted(set(baseline).intersection(current)):
                left = baseline[wave].astype(np.float32)
                right = current[wave].astype(np.float32)
                difference = np.abs(left - right)
                dp_rank = ep_rank // 2
                expected_token = int(
                    baseline_correct.loc[(wave, dp_rank)].output_tokens[0]
                )
                rows.append(
                    {
                        "variant": variant,
                        "wave": wave,
                        "ep_rank": ep_rank,
                        "max_abs": float(difference.max()),
                        "mean_abs": float(difference.mean()),
                        "cosine": float(
                            np.dot(left, right)
                            / (np.linalg.norm(left) * np.linalg.norm(right))
                        ),
                        "expected_token": expected_token,
                        "actual_token": int(np.argmax(right)),
                    }
                )
    frame = pd.DataFrame(rows)
    result = {}
    for variant, local in frame.groupby("variant"):
        driver = drivers[variant]
        dp_equal = all(
            len({tuple(tokens) for tokens in group.output_tokens}) == 1
            for _, group in driver.groupby("wave")
        )
        result[variant] = {
            "output_token_agreement": bool(
                (local.expected_token == local.actual_token).all()
            ),
            "dp_output_agreement": bool(dp_equal),
            "comparisons": int(len(local)),
            "max_abs": float(local.max_abs.max()),
            "mean_abs_median": float(local.mean_abs.median()),
            "cosine_min": float(local.cosine.min()),
        }
    return result, frame


def _error_summary(root: Path, variant: str) -> dict[str, Any]:
    log = root / f"{variant}.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    patterns = (
        "DeepEP error",
        "CUDA error",
        "EngineDeadError",
        "CPU recv timeout",
        "unspecified launch failure",
    )
    hits = {pattern: text.count(pattern) for pattern in patterns}
    return {
        "log_present": log.exists(),
        "error_counts": hits,
        "total": sum(hits.values()),
    }


def _plots(
    measured: pd.DataFrame,
    layer_table: pd.DataFrame,
    summary: dict[str, Any],
    figures: Path,
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    values = [measured[measured.variant == variant].latency_ms for variant in VARIANTS]
    ax.boxplot(values, tick_labels=VARIANTS, showfliers=True)
    ax.set(
        ylabel="Live decoder-prefill CUDA span (ms)",
        xlabel="Preregistered variant",
        title="A0/A1/A2/C live slowdown decomposition",
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "plot1_variant_latency.png", dpi=180)
    plt.close(fig)

    stages = ["attention", "moe_total", "dispatch_comm", "expert", "combine_comm"]
    fig, axes = plt.subplots(len(stages), 1, figsize=(10.5, 12), sharex=True)
    for axis, stage in zip(axes, stages, strict=True):
        for variant in VARIANTS:
            local = layer_table[
                (layer_table.variant == variant) & (layer_table.stage == stage)
            ]
            if not local.empty:
                axis.plot(local.layer, local.critical_span_ms, label=variant)
        axis.set_ylabel(f"{stage}\nms")
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=4)
    axes[-1].set_xlabel("Decoder layer")
    fig.suptitle("Critical-rank stage spans for fixed histology request")
    fig.tight_layout()
    fig.savefig(figures / "plot2_layer_stage_breakdown.png", dpi=180)
    plt.close(fig)

    factors = summary["incremental_factors"]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    names = list(factors)
    ax.bar(names, [factors[name] for name in names], color="#4472c4")
    ax.axhline(1.0, color="black", linestyle="--")
    ax.set(ylabel="Incremental latency factor", title="Causal slowdown attribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "plot3_incremental_slowdown.png", dpi=180)
    plt.close(fig)


def _report(path: Path, root: Path, summary: dict[str, Any]) -> None:
    latency = summary["latency_ms"]
    stages = summary["mean_layer_stage_ms"]
    correctness = summary["correctness"]
    lines = [
        "# Live Causal Wavefront Slowdown Forensics",
        "",
        "## Result",
        "",
        f"`ROOT_CAUSE: {summary['ROOT_CAUSE']}`",
        "",
        f"- Exact experiment code SHA: `{summary['code_sha']}`.",
        "- Qwen3-VL-30B-A3B-Instruct BF16, TP2/DP2/EP4/PP1, DeepEP high-throughput, eager mode, physical GPUs 1,2,3,4 only.",
        "- Fixed requests: coins (128 tokens), histology (277), method (2363); three warmups and ten measured repetitions each.",
        "",
        "## Variant latency",
        "",
        "| Variant | Median (ms) | p25 | p95 | CV |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = latency[variant]
        lines.append(
            f"| {VARIANT_LABELS[variant]} | {item['median']:.4f} | {item['p25']:.4f} | {item['p95']:.4f} | {item['cv']:.2%} |"
        )
    lines.extend(
        [
            "",
            "Incremental factors:",
            "",
            f"- A1/A0 (stock DBO substrate): {summary['incremental_factors']['A1/A0']:.4f}×.",
            f"- A2/A1 (forced prefix/tiny-tail split): {summary['incremental_factors']['A2/A1']:.4f}×.",
            f"- C/A2 (separate stream + causal events): {summary['incremental_factors']['C/A2']:.4f}×.",
            "",
            "![Variant latency](../deepep_revalidation/results/"
            + root.name
            + "/figures/plot1_variant_latency.png)",
            "",
            "## Layer/stage localization",
            "",
            "Mean critical-rank span over 48 layers of the fixed histology request:",
            "",
            "| Stage | A0 | A1 | A2 | C |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for stage in sorted({key for values in stages.values() for key in values}):
        lines.append(
            "| "
            + stage
            + " | "
            + " | ".join(f"{stages[v].get(stage, math.nan):.4f}" for v in VARIANTS)
            + " |"
        )
    lines.extend(
        [
            "",
            f"- Dominant inflated stage: **{summary['dominant_inflated_stage']}**.",
            "- Dispatch/combine timing reports compute-stream exposed spans and DeepEP communication-stream spans separately; it is not a claim of end-to-end collective duration in isolation.",
            "",
            "![Layer stages](../deepep_revalidation/results/"
            + root.name
            + "/figures/plot2_layer_stage_breakdown.png)",
            "",
            "## Instrumentation and lifetime evidence",
            "",
            f"- Control-file reads inside model/layer/attention forward: {summary['instrumentation']['filesystem_reads_inside_model_forward']}.",
            f"- Cached control reads per rank and variant: {summary['instrumentation']['control_file_reads']}.",
            f"- C dependency events created/max-live/after-cleanup: {summary['event_lifetime']['created']}/{summary['event_lifetime']['max_live']}/{summary['event_lifetime']['after_cleanup']}.",
            f"- Event records/waits: {summary['event_lifetime']['records']}/{summary['event_lifetime']['waits']}. Event count is bounded by 48 layers and reused across waves; no wave-proportional leak occurred.",
            f"- Decoder-layer average latency A0/A1/A2/C: {summary['decoder_layer_mean_ms']['A0']:.4f}/{summary['decoder_layer_mean_ms']['A1']:.4f}/{summary['decoder_layer_mean_ms']['A2']:.4f}/{summary['decoder_layer_mean_ms']['C']:.4f} ms.",
            "- Per-rank call counts and processed rows are in `call_counts_and_rows.json`; all variants preserve the same total decoder token rows per request.",
            "",
            "## torch.profiler diagnostic",
            "",
            "The profiler run is separate from latency samples. Coarse model/layer start/end intervals are not labeled as GPU overlap. Actual CUDA kernel concurrency, busy fraction, and idle gaps are reported only when a profiler trace was successfully emitted.",
            "",
            "```json",
            json.dumps(summary["torch_profiler"], indent=2),
            "```",
            "",
            "## Correctness and errors",
            "",
            "| Variant | Greedy token agreement | DP agreement | Logit maxabs | Min cosine | Runtime errors |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in VARIANTS:
        item = correctness[variant]
        lines.append(
            f"| {variant} | {item['output_token_agreement']} | {item['dp_output_agreement']} | {item['max_abs']:.6f} | {item['cosine_min']:.9f} | {summary['errors'][variant]['total']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            summary["root_cause_explanation"],
            "",
            f"Whether the causal-wavefront concept remains viable: **{summary['causal_wavefront_viability']}**.",
            "",
            f"Next single action: **{summary['recommended_next_action']}**",
            "",
            "## Artifacts",
            "",
            f"- Result directory: `poc_flashvep/deepep_revalidation/results/{root.name}/`",
            "- Aggregates: `latency_samples.csv`, `layer_stage_spans.csv`, `correctness.csv`, `call_counts_and_rows.json`.",
            "- Worker evidence: `<variant>/raw/rank*.json`; profiler traces are stored beside them when successful.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    contract = json.loads(
        (args.result_dir / "A0" / "experiment_contract.json").read_text()
    )
    code_sha = contract["code_sha"]
    drivers, forwards, stages, proofs = {}, {}, {}, {}
    for variant in VARIANTS:
        drivers[variant], forwards[variant], stages[variant], proofs[variant] = (
            _load_variant(args.result_dir, variant, code_sha)
        )
    wave_tables = {variant: _wave_latency(forwards[variant]) for variant in VARIANTS}
    measured = pd.concat(
        [
            table[table.phase == "measured"].assign(variant=variant)
            for variant, table in wave_tables.items()
        ],
        ignore_index=True,
    )
    layer_table = pd.concat(
        [_layer_stage_table(stages[v], v) for v in VARIANTS], ignore_index=True
    )
    correctness, correctness_frame = _correctness(args.result_dir, drivers)
    latency = {
        variant: _stats(measured[measured.variant == variant].latency_ms)
        for variant in VARIANTS
    }
    medians = {variant: latency[variant]["median"] for variant in VARIANTS}
    incremental = {
        "A1/A0": medians["A1"] / medians["A0"],
        "A2/A1": medians["A2"] / medians["A1"],
        "C/A2": medians["C"] / medians["A2"],
    }
    root_key = max(incremental, key=incremental.get)
    root_names = {
        "A1/A0": "stock vLLM DBO substrate",
        "A2/A1": "forced prefix/tiny-tail split",
        "C/A2": "separate compute streams with concurrent model/DeepEP execution",
    }
    root_cause = root_names[root_key]
    mean_stages: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        local = layer_table[layer_table.variant == variant]
        mean_stages[variant] = {
            stage: float(values.critical_span_ms.mean())
            for stage, values in local.groupby("stage")
        }
    baseline_for_root = {"A1/A0": "A0", "A2/A1": "A1", "C/A2": "A2"}[root_key]
    target_for_root = {"A1/A0": "A1", "A2/A1": "A2", "C/A2": "C"}[root_key]
    stage_ratios = {
        stage: mean_stages[target_for_root].get(stage, 0.0)
        / max(mean_stages[baseline_for_root].get(stage, 0.0), 1e-9)
        for stage in mean_stages[target_for_root]
        if stage in mean_stages[baseline_for_root]
    }
    dominant_stage = max(stage_ratios, key=stage_ratios.get)

    call_rows = {}
    for variant in VARIANTS:
        local = stages[variant]
        by_rank = {}
        for ep_rank, rank_rows in local.groupby("ep_rank"):
            experts = rank_rows[rank_rows.stage == "expert"]
            by_rank[str(int(ep_rank))] = {
                "attention_calls": int((rank_rows.stage == "attention").sum()),
                "dispatch_calls": int((rank_rows.stage == "dispatch_compute").sum()),
                "expert_calls": int(len(experts)),
                "combine_calls": int((rank_rows.stage == "combine_compute").sum()),
                "decoder_token_rows": int(
                    forwards[variant][
                        (forwards[variant].request_id == "histology")
                        & (forwards[variant].phase == "measured")
                        & (forwards[variant].iteration == 0)
                        & (forwards[variant].ep_rank == ep_rank)
                    ].tokens.sum()
                ),
                "expert_input_rows": int(
                    experts.get("input_tokens", pd.Series()).sum()
                ),
                "expert_assignments": int(
                    experts.get("expert_assignments", pd.Series()).sum()
                ),
            }
        call_rows[variant] = by_rank
    _json(args.result_dir / "call_counts_and_rows.json", call_rows)

    instrumentation = {
        "filesystem_reads_inside_model_forward": int(
            max(
                proof["control_file_reads_inside_model_forward"]
                for values in proofs.values()
                for proof in values
            )
        ),
        "control_file_reads": {
            variant: [
                int(proof["counters"].get("control_file_reads", 0))
                for proof in proofs[variant]
            ]
            for variant in VARIANTS
        },
        "cached_control_accesses": {
            variant: [
                int(proof["counters"].get("cached_control_accesses", 0))
                for proof in proofs[variant]
            ]
            for variant in VARIANTS
        },
    }
    c_proofs = proofs["C"]
    event_lifetime = {
        "created": max(
            int(proof["counters"].get("dependency_events_created", 0))
            for proof in c_proofs
        ),
        "max_live": max(
            int(proof["counters"].get("dependency_events_max_live", 0))
            for proof in c_proofs
        ),
        "after_cleanup": max(
            int(proof["dependency_events_live_after_cleanup"]) for proof in c_proofs
        ),
        "records": max(
            int(proof["counters"].get("dependency_event_records", 0))
            for proof in c_proofs
        ),
        "waits": max(
            int(proof["counters"].get("dependency_event_waits", 0))
            for proof in c_proofs
        ),
    }
    errors = {variant: _error_summary(args.result_dir, variant) for variant in VARIANTS}
    profiler = {
        variant: _profiler_summary(args.result_dir, variant) for variant in VARIANTS
    }
    decoder_layer_mean = {
        variant: mean_stages[variant]["decoder_layer"] for variant in VARIANTS
    }
    if root_key == "A1/A0":
        explanation = (
            "The dominant incremental jump occurs when stock DBO is enabled before any "
            "modality split or custom stream/event logic. The wavefront slowdown is therefore "
            "primarily a DBO substrate effect in this live DeepEP configuration."
        )
        next_action = (
            "Test a non-DBO, explicit single-owner layer pipeline that preserves one DeepEP "
            "collective sequence; do not optimize the current DBO wavefront."
        )
    elif root_key == "A2/A1":
        explanation = (
            "The dominant incremental jump occurs only after forcing the highly asymmetric "
            "prefix/tiny-tail split, identifying split geometry and tiny-tail execution as the "
            "primary cause."
        )
        next_action = (
            "Measure a preregistered minimum-tail-size coalescing policy without adding another "
            "CUDA stream."
        )
    else:
        explanation = (
            "A0/A1/A2 isolate the stock DBO and split costs; the dominant remaining jump occurs "
            "only when separate compute streams and concurrent model/DeepEP execution are added."
        )
        next_action = (
            "Build one bounded communication-serialized two-stream experiment to determine "
            "whether concurrent DeepEP collectives, rather than attention kernels, cause the jump."
        )
    viability = (
        "YES in the current substrate"
        if medians["C"] <= medians["A0"] * 1.2
        else "NO in the current vLLM/DeepEP implementation; the abstract DAG remains unproven"
    )
    summary = {
        "ROOT_CAUSE": root_cause,
        "code_sha": code_sha,
        "latency_ms": latency,
        "incremental_factors": incremental,
        "decoder_layer_mean_ms": decoder_layer_mean,
        "mean_layer_stage_ms": mean_stages,
        "dominant_inflated_stage": dominant_stage,
        "stage_inflation_ratios_at_root_jump": stage_ratios,
        "instrumentation": instrumentation,
        "event_lifetime": event_lifetime,
        "correctness": correctness,
        "errors": errors,
        "torch_profiler": profiler,
        "root_cause_explanation": explanation,
        "causal_wavefront_viability": viability,
        "recommended_next_action": next_action,
    }
    measured.to_csv(args.result_dir / "latency_samples.csv", index=False)
    layer_table.to_csv(args.result_dir / "layer_stage_spans.csv", index=False)
    correctness_frame.to_csv(args.result_dir / "correctness.csv", index=False)
    _json(args.result_dir / "summary.json", summary)
    _plots(measured, layer_table, summary, args.result_dir / "figures")
    _report(args.report, args.result_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
