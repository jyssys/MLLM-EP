from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from poc_mllm_ep_foundation.io import atomic_json


RESOLUTIONS = ("R0", "R05", "R025")
KS = (8, 6, 4)


def _load(path: Path | None) -> dict[str, Any] | None:
    return None if path is None else json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    split = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([head, split, *body])


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _config_table(configs: dict[str, dict[str, Any]]) -> str:
    rows = []
    for resolution in RESOLUTIONS:
        for k in KS:
            name = f"{resolution}_K{k}"
            value = configs[name]
            rows.append([
                resolution, _fmt(value["mean_visual_tokens"], 1), k,
                json.dumps({task: round(100 * score, 2)
                            for task, score in value["task_accuracy"].items()},
                           sort_keys=True),
                _fmt(value["mean_logical_visual_pairs"], 0),
                _fmt(value.get("measured_physical_padded_rows"), 0),
                _fmt(value.get("measured_max_mean"), 3),
                _fmt(value.get("measured_ep_critical_ms"), 3),
                _fmt(value.get("measured_ttft_ms"), 3),
                _fmt(value.get("measured_natural_e2e_ms"), 3),
                _fmt(value.get("mean_output_tokens"), 1),
                _fmt(100 * value.get("normal_termination_fraction", 0.0), 2),
                value.get("generation_timeout_count", 0),
                "ACTUAL_TRUE_EP4",
            ])
    return _table(
        ["R", "actual V", "K_v", "score % by task", "visual pairs", "physical rows",
         "max/mean", "EP critical ms", "TTFT ms", "natural E2E ms",
         "mean output tokens", "normal stop %", "generation timeout", "evidence"],
        rows,
    )


def _controls_table(analysis: dict[str, Any]) -> str:
    configs = analysis["configs"]
    rows = []
    for control, name in analysis["controls"].items():
        value = configs.get(name) if isinstance(name, str) else None
        rows.append([
            control, name,
            "N/A" if value is None else _fmt(100 * value["accuracy"], 2),
            "N/A" if value is None else _fmt(value.get("measured_ep_critical_ms"), 3),
            "N/A" if value is None else _fmt(value.get("measured_ttft_ms"), 3),
        ])
    return _table(["control", "config", "discovery score %", "EP critical ms", "TTFT ms"], rows)


def _plots(
    analysis: dict[str, Any],
    output: Path,
    virtual: dict[str, Any] | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    configs = analysis["configs"]
    strata = sorted(next(iter(configs.values()))["stratum_accuracy"])
    fig, axes = plt.subplots(1, len(strata), figsize=(5 * len(strata), 4), squeeze=False)
    for axis, stratum in zip(axes[0], strata):
        matrix = np.asarray([
            [100 * configs[f"{resolution}_K{k}"]["stratum_accuracy"][stratum]
             for k in KS]
            for resolution in RESOLUTIONS
        ])
        image = axis.imshow(matrix, cmap="viridis", aspect="auto")
        axis.set_xticks(range(len(KS)), [str(k) for k in KS])
        axis.set_yticks(range(len(RESOLUTIONS)), RESOLUTIONS)
        axis.set_xlabel("visual K")
        axis.set_title(stratum)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, f"{matrix[row, column]:.1f}",
                          ha="center", va="center", color="white")
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output / "task_quality_heatmap.png", dpi=180)
    plt.close(fig)

    names = [f"{r}_K{k}" for r in RESOLUTIONS for k in KS]
    if all(configs[name].get("measured_ep_critical_ms") is not None for name in names):
        fig, axis = plt.subplots(figsize=(7, 5))
        for name in names:
            value = configs[name]
            axis.scatter(value["mean_logical_visual_pairs"],
                         value["measured_ep_critical_ms"], label=name)
            axis.annotate(name, (value["mean_logical_visual_pairs"],
                                 value["measured_ep_critical_ms"]), fontsize=7)
        axis.set_xlabel("mean logical visual expert pairs / request")
        axis.set_ylabel("actual TRUE EP4 routed critical ms")
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / "visual_pairs_vs_ep_critical.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for name in names:
            value = configs[name]
            score = 100 * value["accuracy"]
            axes[0].scatter(value["measured_ttft_ms"], score)
            axes[0].annotate(name, (value["measured_ttft_ms"], score), fontsize=7)
            natural = value.get("measured_natural_e2e_ms")
            if natural is not None:
                axes[1].scatter(natural, score)
                axes[1].annotate(name, (natural, score), fontsize=7)
        axes[0].set(xlabel="actual TTFT ms", ylabel="discovery score %",
                    title="Accuracy–TTFT")
        axes[1].set(xlabel="actual natural E2E ms", ylabel="discovery score %",
                    title="Accuracy–natural E2E")
        for axis in axes:
            axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / "accuracy_ttft_e2e_frontier.png", dpi=180)
        plt.close(fig)

    controls = analysis["controls"]
    names = [name for name in controls.values() if name in configs]
    if names:
        fig, axis = plt.subplots(figsize=(9, 4))
        values = [configs[name].get("measured_ep_critical_ms", np.nan) for name in names]
        axis.bar(range(len(names)), values)
        axis.set_xticks(range(len(names)), names, rotation=35, ha="right")
        axis.set_ylabel("actual TRUE EP4 routed critical ms")
        fig.tight_layout()
        fig.savefig(output / "c3_c4_c5_comparison.png", dpi=180)
        plt.close(fig)

    virtual_configs = {} if virtual is None else virtual.get("configs", {})
    comparable = [
        name for name in configs
        if configs[name].get("measured_ep_critical_ms") is not None
        and virtual_configs.get(name, {}).get("EP4", {}).get(
            "mean_predicted_routed_ms"
        ) is not None
    ]
    if comparable:
        fig, axis = plt.subplots(figsize=(6, 6))
        for name in comparable:
            actual = float(configs[name]["measured_ep_critical_ms"])
            predicted = float(
                virtual_configs[name]["EP4"]["mean_predicted_routed_ms"]
            )
            all_in_domain = bool(
                virtual_configs[name]["EP4"].get("all_rows_in_domain", False)
            )
            axis.scatter(
                actual,
                predicted,
                marker="o" if all_in_domain else "x",
                color="tab:blue" if all_in_domain else "tab:red",
            )
            axis.annotate(name, (actual, predicted), fontsize=7)
        all_values = [
            value
            for name in comparable
            for value in (
                float(configs[name]["measured_ep_critical_ms"]),
                float(virtual_configs[name]["EP4"]["mean_predicted_routed_ms"]),
            )
        ]
        limits = [min(all_values), max(all_values)]
        pad = max(1.0, 0.05 * (limits[1] - limits[0]))
        axis.plot(
            [limits[0] - pad, limits[1] + pad],
            [limits[0] - pad, limits[1] + pad],
            linestyle="--",
            color="black",
            linewidth=1,
        )
        axis.set_xlabel("actual TRUE EP4 routed-stage ms")
        axis.set_ylabel("Part-1 simulator point estimate ms")
        axis.set_title("Actual vs simulator (red x = OOD rows present)")
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / "actual_vs_simulator_ep4.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--physical-costs", type=Path, required=True)
    parser.add_argument("--noop", type=Path, required=True)
    parser.add_argument("--noop-runtime", type=Path)
    parser.add_argument("--frozen-policy", type=Path, required=True)
    parser.add_argument("--virtual", type=Path)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()

    analysis = _load(args.analysis)
    heldout = _load(args.heldout)
    physical = _load(args.physical_costs)
    noop = _load(args.noop)
    noop_runtime = _load(args.noop_runtime)
    frozen = _load(args.frozen_policy)
    virtual = _load(args.virtual)
    batch4_runtime_limit = _load(
        args.artifacts / "analysis" / "batch4_runtime_limit.json"
    )
    assert analysis and heldout and physical and noop and frozen
    configs = analysis["configs"]
    controls = analysis["controls"]
    baseline = configs["R0_K8"]

    _plots(analysis, args.reports / "figures", virtual)
    _write(args.reports / "factorial_grid_quality_work_cost.md", f"""
# Factorial grid: quality, work, and physical cost

Evidence is the public-partial official generation lane on actual TRUE EP4.
Discovery generation timings are not used as clean performance evidence; the
table uses the separate paired representative timing cohort.

{_config_table(configs)}

The physical-row column remains `N/A` when the backend proof exposes only
logical compact routes and not kernel padding. It is never inferred from the
fixed-width tensor slots.
""")

    sensitivity_rows = []
    for name, value in sorted(configs.items()):
        sensitivity_rows.append([
            name, *[_fmt(100 * value["stratum_accuracy"][key], 2)
                    for key in sorted(value["stratum_accuracy"])],
        ])
    _write(args.reports / "task_resolution_vs_expert_sensitivity.md", f"""
# Task-stratified resolution and expert sensitivity

The MMStar `ocr_document_proxy` stratum is a text-sensitive MMStar subset, not
an OCR benchmark. Task labels are analysis strata only and are not available
to an online selector.

{_table(["config", *sorted(next(iter(configs.values()))["stratum_accuracy"])], sensitivity_rows)}

## Paired interaction

```json
{json.dumps(analysis['quality_interactions'], indent=2)}
```
""")

    _write(args.reports / "matched_pair_and_latency_frontiers.md", f"""
# Matched-pair and latency frontiers

Only pairs whose measured mean logical visual-pair counts differ by at most
5% are listed. Pair matching does not imply equal attention, encoder, padding,
communication, or wall time.

```json
{json.dumps(analysis['matched_pair_comparisons'], indent=2)}
```
""")

    _write(args.reports / "independent_vs_joint_controls.md", f"""
# Independent and joint controls

All controls use the same discovery candidates and the same provisional
paired quality constraint. Held-out data were not used to select them.

{_controls_table(analysis)}
""")

    c5 = controls["C5_joint_ep_critical_path"]
    c4 = controls["C4_joint_pair_only"]
    c3_names = [controls["C3_independent_resolution_then_k"],
                controls["C3_independent_k_then_resolution"]]
    comparison = [name for name in [c4, *c3_names] if name in configs]
    c5_gain = None
    control_gain = None
    increment = None
    if c5 in configs and configs[c5].get("measured_ep_critical_ms") is not None:
        base_cost = float(baseline["measured_ep_critical_ms"])
        c5_gain = 100 * (1 - float(configs[c5]["measured_ep_critical_ms"]) / base_cost)
        gains = [100 * (1 - float(configs[name]["measured_ep_critical_ms"]) / base_cost)
                 for name in comparison
                 if configs[name].get("measured_ep_critical_ms") is not None]
        control_gain = max(gains) if gains else None
        # Selecting C0 means the EP-aware objective rejected every Part-2
        # candidate.  It is not an EP-specific method gain over a regressive
        # cost-blind choice, so do not turn that selection regret into a
        # positive incremental contribution.
        increment = (
            0.0 if c5 == "R0_K8"
            else None if control_gain is None
            else c5_gain - control_gain
        )
    _write(args.reports / "ep_specific_increment.md", f"""
# EP-specific increment

- C5: `{c5}`
- C4: `{c4}`
- C5 routed-critical gain vs R0/K8: `{_fmt(c5_gain, 3)}%`
- strongest C3/C4 gain: `{_fmt(control_gain, 3)}%`
- EP-aware incremental gain: `{_fmt(increment, 3)} pp`

The increment is based on measured TRUE EP4 critical-path time. A positive
generic compression gain is not by itself an EP-specific contribution.
""")

    heldout_rows = []
    for name, value in heldout["results"].items():
        heldout_rows.append([
            name, value["n"], _fmt(100 * value["accuracy"], 2),
            _fmt(value["baseline_delta_pp"], 3),
            str([round(x, 3) for x in value["cluster_bootstrap_95ci_pp"]]),
            value["baseline_correct_to_wrong"], value["baseline_wrong_to_correct"],
            _fmt(value.get("mean_output_tokens"), 1),
            _fmt(value.get("p90_output_tokens"), 1),
            _fmt(value.get("mean_output_token_delta_vs_baseline"), 1),
            value["noninferiority_proven"],
        ])
    _write(args.reports / "heldout_quality.md", f"""
# Frozen held-out quality

The image-hash-disjoint heldout192 split was accessed once after freezing at
most two candidates. The -1pp margin is an operating target, not a theorem.

{_table(['config', 'n', 'score %', 'delta pp', '95% CI pp', 'C→W', 'W→C', 'mean out tok', 'P90 out tok', 'mean Δtok', 'NI proven'], heldout_rows)}
""")

    virtual_text = "NOT_RUN"
    if virtual is not None:
        virtual_text = json.dumps({
            key: value for key, value in virtual.items() if key != "rows"
        }, indent=2)
    _write(args.reports / "physical_ep4_or_virtual_projection.md", f"""
# Physical EP4 and virtual projections

Primary comparisons are measured on TRUE EP4 (TP2/DP2/EP4). EP8 is never
called measured or validated; it is structural replay plus sensitivity only.

```json
{json.dumps(physical, indent=2)}
```

## Virtual EP evidence

```json
{virtual_text}
```

## No-op adapter timing

```json
{json.dumps(noop_runtime, indent=2) if noop_runtime is not None else 'NOT_RUN'}
```

## Distinct-image global-batch4 runtime boundary

```json
{json.dumps(batch4_runtime_limit, indent=2) if batch4_runtime_limit is not None else 'NOT_RECORDED'}
```
""")

    heldout_candidates = [row["config"] for row in frozen["candidates"]]
    quality_states = []
    for name in heldout_candidates:
        value = heldout["results"][name]
        ci = value["cluster_bootstrap_95ci_pp"]
        quality_states.append(
            "LOCAL_NI_PROVEN" if ci[0] > -1.0
            else "MATERIAL_DROP" if ci[1] < -1.0
            else "QUALITY_UNRESOLVED"
        )
    quality_status = (
        "LOCAL_NI_PROVEN" if quality_states and all(x == "LOCAL_NI_PROVEN" for x in quality_states)
        else "QUALITY_FAILED" if "MATERIAL_DROP" in quality_states
        else "QUALITY_UNRESOLVED"
    )
    ep_status = (
        "EP_SPECIFIC_SUPPORTED"
        if c5 != "R0_K8"
        and c5_gain is not None
        and c5_gain > 0
        and increment is not None
        and increment >= 3.0
        else "GENERIC_COMPRESSION_ONLY"
    )
    best = c5 if c5 in configs else c4
    best_physical = physical["configs"].get(best, {})
    e2e_gain = None
    if best_physical.get("natural_e2e_ms") is not None:
        e2e_gain = 100 * (1 - float(best_physical["natural_e2e_ms"])
                          / float(physical["configs"]["R0_K8"]["natural_e2e_ms"]))
    if quality_status == "QUALITY_FAILED":
        verdict = "NO_GO_TESTED_FAMILY"
    elif ep_status != "EP_SPECIFIC_SUPPORTED":
        verdict = "GENERIC_ONLY"
    elif e2e_gain is not None and e2e_gain >= 15 and quality_status == "LOCAL_NI_PROVEN":
        verdict = "GO_BOUNDED"
    elif e2e_gain is not None and e2e_gain >= 5:
        verdict = "HOLD"
    else:
        verdict = "NO_GO_TESTED_FAMILY"

    summary = {
        "schema_version": "mllm_resolution_expert_tradeoff.summary.v1",
        "baseline": "PART1_FOUNDATION_READY_PUBLIC_PROTOCOL_PARTIAL",
        "controls": controls,
        "frozen_candidates": frozen["candidates"],
        "no_op_parity": noop["counts"],
        "no_op_runtime": noop_runtime,
        "ep_specific_increment_pp": increment,
        "ep_specific_status": ep_status,
        "quality_status": quality_status,
        "best_joint_ep": best,
        "best_natural_e2e_gain_pct": e2e_gain,
        "verdict": verdict,
        "public_protocol_status": "PUBLIC_PROTOCOL_PARTIAL",
        "ep8_status": "UNVALIDATED_EXTRAPOLATION",
        "batch4_secondary_status": batch4_runtime_limit,
        "do_not_implement_unbounded_production_runtime": True,
    }
    best_value = configs.get(best, {})
    baseline_physical = physical["configs"]["R0_K8"]
    best_physical = physical["configs"].get(best, {})
    ep_critical_gain = None
    if best_physical.get("ep_critical_ms") is not None:
        ep_critical_gain = 100 * (
            1
            - float(best_physical["ep_critical_ms"])
            / float(baseline_physical["ep_critical_ms"])
        )
    ttft_gain = None
    if best_physical.get("ttft_ms") is not None:
        ttft_gain = 100 * (
            1 - float(best_physical["ttft_ms"]) / float(baseline_physical["ttft_ms"])
        )
    fixed32_gain = None
    if best_physical.get("fixed32_ms") is not None and baseline_physical.get("fixed32_ms"):
        fixed32_gain = 100 * (
            1
            - float(best_physical["fixed32_ms"])
            / float(baseline_physical["fixed32_ms"])
        )
    fixed128_gain = None
    if best_physical.get("fixed128_ms") is not None and baseline_physical.get("fixed128_ms"):
        fixed128_gain = 100 * (
            1
            - float(best_physical["fixed128_ms"])
            / float(baseline_physical["fixed128_ms"])
        )
    pair_reduction = best_value.get("visual_pair_reduction_vs_baseline_pct")
    heldout_quality = {
        name: {
            "delta_pp": value["baseline_delta_pp"],
            "ci_pp": value["cluster_bootstrap_95ci_pp"],
            "correct_to_wrong": value["baseline_correct_to_wrong"],
            "wrong_to_correct": value["baseline_wrong_to_correct"],
            "mean_output_tokens": value.get("mean_output_tokens"),
            "p90_output_tokens": value.get("p90_output_tokens"),
            "mean_output_token_delta_vs_baseline": value.get(
                "mean_output_token_delta_vs_baseline"
            ),
            "finish_reason_counts": value.get("finish_reason_counts"),
        }
        for name, value in heldout["results"].items()
        if name != "R0_K8"
    }
    resolution_summary = {
        resolution: {
            "mean_visual_tokens": configs[f"{resolution}_K8"]["mean_visual_tokens"],
            "mean_actual_ratio": configs[f"{resolution}_K8"]["mean_actual_visual_ratio"],
            "duplicate_grids": configs[f"{resolution}_K8"]["duplicate_resolution_count"],
        }
        for resolution in RESOLUTIONS
    }
    interaction_values = [
        abs(float(value["interaction_pp"]))
        for value in analysis["quality_interactions"].values()
    ]
    stratum_ranges = []
    for stratum in next(iter(configs.values()))["stratum_accuracy"]:
        values = [float(value["stratum_accuracy"][stratum]) for value in configs.values()]
        stratum_ranges.append(100 * (max(values) - min(values)))
    tradeoff_status = (
        "TRADEOFF_SUPPORTED"
        if max(interaction_values or [0.0]) >= 3.0
        or max(stratum_ranges or [0.0]) >= 5.0
        else "TRADEOFF_WEAK"
    )
    summary.update({
        "tradeoff_status": tradeoff_status,
        "best_config": best,
        "actual_executed_visual_pair_reduction_pct": pair_reduction,
        "actual_ep_critical_path_gain_pct": ep_critical_gain,
        "actual_ttft_gain_pct": ttft_gain,
        "actual_fixed32_gain_pct": fixed32_gain,
        "actual_fixed128_gain_pct": fixed128_gain,
        "heldout_quality": heldout_quality,
        "resolution_grid": resolution_summary,
        "physical_padded_work": "N/A_BACKEND_DID_NOT_EXPOSE_PADDING",
        "online_selector": "NOT_IMPLEMENTED_UNIFORM_REQUEST_LEVEL_GRID_ONLY",
    })
    atomic_json(args.artifacts / "summary.json", summary)
    frozen_names = [row["config"] for row in frozen["candidates"]]
    virtual_label = (
        "NOT_RUN"
        if virtual is None
        else "EP4_STRUCTURAL_REPLAY; EP8_UNVALIDATED_EXTRAPOLATION"
    )
    final_required = f"""
FOUNDATION_BUNDLE_AND_BASELINE_STATUS: Part1 frozen bundle READY; public protocol PARTIAL; physical EP4 READY; Part1 simulator UNVALIDATED
MODEL_REVISION / GPU_SET / ACTUAL_TOPOLOGY: 9c4b90e1e4ba969fd3b5378b57d966d725f1b86c / 4,5,6,7 / TP2-DP2-SP1-EP4-PP1
PUBLIC_REPRODUCTION_STATUS: PUBLIC_PROTOCOL_PARTIAL; Part1 full MMStar and ChartQA reused only for the unchanged R0/K8 control
EVALUATION_LANE_AND_METRIC: OFFICIAL_REPRODUCTION_PUBLIC_PROTOCOL_PARTIAL / MMStar exact-option accuracy + ChartQA relaxed accuracy
SAMPLING_TOP_K / EXPERT_TOP_K_TEXT / EXPERT_TOP_K_VISION: 20 / 8 / {{8,6,4}}
DISCOVERY_COUNT / HELDOUT_COUNT: {analysis['base_requests']} image-question requests x 9 / {heldout['results']['R0_K8']['n']} image-hash-disjoint requests x baseline+{len(frozen_names)} frozen candidate(s)
RESOLUTION_GRID_AND_ACTUAL_VISUAL_TOKEN_COUNTS: {json.dumps(resolution_summary, sort_keys=True)}
PRIMARY_K_GRID: visual {{8,6,4}}; text=8; generated decode=8
NO_OP_PARITY: {json.dumps(noop['counts'], sort_keys=True)} over n={noop['n']}; runtime={json.dumps(noop_runtime, sort_keys=True) if noop_runtime is not None else 'NOT_RUN'}
BEST_RESOLUTION_ONLY: {controls['C1_resolution_only']}
BEST_EXPERT_ONLY: {controls['C2_expert_only']}
BEST_INDEPENDENT_TWO_AXIS: resolution-then-k={controls['C3_independent_resolution_then_k']}; k-then-resolution={controls['C3_independent_k_then_resolution']}
BEST_JOINT_PAIR_ONLY: {c4}
BEST_JOINT_EP_AWARE: {c5}
TASK_SENSITIVITY_AND_INTERACTION: {tradeoff_status}; {json.dumps(analysis['quality_interactions'], sort_keys=True)}
MATCHED_PAIR_COMPARISON: {json.dumps(analysis['matched_pair_comparisons'], sort_keys=True)}
ACTUAL_EXECUTED_VISION_PAIRS_REDUCTION: {_fmt(pair_reduction, 3)}%
PHYSICAL_PADDED_WORK_REDUCTION: N/A_BACKEND_DID_NOT_EXPOSE_PADDING; logical valid/invalid route slots reported separately
EP_CRITICAL_PATH_CHANGE: {_fmt(ep_critical_gain, 3)}% reduction for {best} vs R0_K8 on the paired representative TRUE-EP4 cohort
HELDOUT_QUALITY_DELTA_AND_CI: {json.dumps(heldout_quality, sort_keys=True)}
TRUE_EP4_TTFT_GAIN / NATURAL_E2E_GAIN / FIXED_DECODE_GAIN: {_fmt(ttft_gain, 3)}% / {_fmt(e2e_gain, 3)}% / fixed32={_fmt(fixed32_gain, 3)}%, fixed128={_fmt(fixed128_gain, 3)}%
SIM_EP4_GAIN_AND_UNCERTAINTY / SIM_EP8_LABEL: {virtual_label}
EP_SPECIFIC_INCREMENT_VS_STRONGEST_CONTROL: {_fmt(increment, 3)} pp; {ep_status}
ONLINE_SELECTOR_CAUSALITY_AND_COST: NOT_IMPLEMENTED; uniform request-level factorial only, no task label/answer/future routing used online
PRIOR_BASELINES_COMPLETED_OR_MISSING: FastMMoE visual-k component=C2 REIMPLEMENTED_COMPONENT_CONTROL; full FastMMoE/MoDES ports=NOT_RUN; MACS/ReaLB boundary audited
TRADEOFF_STATUS / EP_SPECIFIC_STATUS / QUALITY_STATUS: {tradeoff_status} / {ep_status} / {quality_status}
VERDICT: {verdict}
NOT_TESTED_AND_LIMITATIONS: public wrapper remains partial; physical padding not exposed; distinct-image global-batch4 K6/K4 unavailable after reproduced external-DP synchronization hang; Part1 EP2/EP4 simulator unvalidated; EP8 has no physical collective/E2E truth; no saliency/video/basis/training
NEXT_ACTION: {'Stop this EP-aware grid family as a main method.' if verdict in ('GENERIC_ONLY', 'NO_GO_TESTED_FAMILY') else 'Preserve the frozen candidate and validate only the scoped next bounded step.'}
DO_NOT_IMPLEMENT_UNBOUNDED_PRODUCTION_RUNTIME: true
""".strip()
    atomic_json(args.artifacts / "summary.json", summary)
    _write(args.reports / "final_summary.md", f"""
# Part 2 final summary

{final_required}

## Machine-readable summary

```json
{json.dumps(summary, indent=2)}
```

Part 1's reusable foundation remains valid regardless of this tested-family
verdict. No saliency model, training, video extension, or unbounded production
runtime was implemented.
""")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
