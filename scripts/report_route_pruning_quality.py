#!/usr/bin/env python3
"""Write the route-pruning Stage-A reports and stop-rule figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def percent(value: float) -> str:
    return f"{100 * value:.3f}%"


def gain(value: float) -> str:
    return f"{value:.3f}%"


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def placeholder(path: Path, title: str) -> None:
    plt.figure(figsize=(7, 3.5))
    plt.axis("off")
    plt.text(.5, .62, title, ha="center", va="center", fontsize=15, weight="bold")
    plt.text(
        .5, .40,
        "NOT RUN\nStage A stopped: P4 at 1% caused 2 additional wrong answers",
        ha="center", va="center", fontsize=11,
    )
    save_plot(path)


def projection_table(document: dict) -> str:
    lines = [
        "| EP | Method | Dispatch ms/req | Expert ms/req | Combine ms/req | "
        "Stage ms/req | Total stage gain | NFE-normalized gain | Max/mean | Wait |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ep in (4, 8):
        key = f"ep{ep}"
        base = document["baseline"]["projection"][key]
        lines.append(
            f"| {ep} | Vanilla | {base['dispatch_ms_per_request']:.3f} | "
            f"{base['expert_ms_per_request']:.3f} | {base['combine_ms_per_request']:.3f} | "
            f"{base['stage_ms_per_request']:.3f} | 0.000% | 0.000% | "
            f"{base['max_mean']:.4f} | {percent(base['wait_fraction'])} |"
        )
        for label, name in (("P2_MASS_1PCT", "P2 1%"), ("P4_MASS_1PCT", "P4 1%")):
            value = document["methods"][label]["projection"][key]
            lines.append(
                f"| {ep} | {name} | {value['dispatch_ms_per_request']:.3f} | "
                f"{value['expert_ms_per_request']:.3f} | {value['combine_ms_per_request']:.3f} | "
                f"{value['stage_ms_per_request']:.3f} | {gain(value['stage_gain_percent'])} | "
                f"{gain(value['nfe_normalized_stage_gain_percent'])} | "
                f"{value['max_mean']:.4f} | {percent(value['wait_fraction'])} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.input.read_text())
    reports = args.reports
    figures = reports / "figures" / "route_pruning_quality_dllm"
    figures.mkdir(parents=True, exist_ok=True)

    p2 = document["methods"]["P2_MASS_1PCT"]
    p4 = document["methods"]["P4_MASS_1PCT"]
    q2, q4 = p2["quality"], p4["quality"]
    baseline_accuracy = q2["baseline_accuracy"]
    ep8_base = document["baseline"]["projection"]["ep8"]
    p2_ep8, p4_ep8 = p2["projection"]["ep8"], p4["projection"]["ep8"]

    # Required Stage-A figures.
    plt.figure(figsize=(6, 4))
    plt.plot([0, 100 * q2["actual_removed_mass_fraction"]],
             [100 * baseline_accuracy, 100 * q2["accuracy"]], "o-", label="P2 utility-only")
    plt.plot([0, 100 * q4["actual_removed_mass_fraction"]],
             [100 * baseline_accuracy, 100 * q4["accuracy"]], "o-", label="P4 EP-aware")
    plt.xlabel("Actual removed router mass (%)"); plt.ylabel("GSM8K accuracy (%)")
    plt.title("Quality vs removed router mass (n=32)"); plt.legend(); plt.grid(alpha=.25)
    save_plot(figures / "01_quality_vs_removed_mass.png")

    plt.figure(figsize=(6, 4))
    for label, q, projection in (("P2", q2, p2_ep8), ("P4", q4, p4_ep8)):
        plt.plot([0, 100 * q["actual_removed_mass_fraction"]],
                 [0, projection["stage_gain_percent"]], "o-", label=label)
    plt.xlabel("Actual removed router mass (%)"); plt.ylabel("Simulated EP8 stage gain (%)")
    plt.title("Total trajectory gain; P3 not run by stop rule"); plt.legend(); plt.grid(alpha=.25)
    save_plot(figures / "02_ep8_stage_gain_vs_mass.png")

    plt.figure(figsize=(6, 4))
    plt.scatter([0, p2_ep8["stage_gain_percent"], p4_ep8["stage_gain_percent"]],
                [100 * baseline_accuracy, 100 * q2["accuracy"], 100 * q4["accuracy"]])
    for x, y, text in ((0, 100 * baseline_accuracy, "Vanilla"),
                       (p2_ep8["stage_gain_percent"], 100 * q2["accuracy"], "P2"),
                       (p4_ep8["stage_gain_percent"], 100 * q4["accuracy"], "P4")):
        plt.annotate(text, (x, y), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Simulated EP8 routed-MoE stage gain (%)"); plt.ylabel("GSM8K accuracy (%)")
    plt.title("Quality / systems Pareto (n=32)"); plt.grid(alpha=.25)
    save_plot(figures / "03_quality_vs_ep8_stage_gain.png")

    plt.figure(figsize=(6, 4))
    plt.bar(["Vanilla", "P2 1%", "P4 1%"], [8, q2["avg_k"], q4["avg_k"]])
    plt.ylabel("Average retained experts/token"); plt.ylim(7.6, 8.02)
    plt.title("AvgK after route omission")
    save_plot(figures / "04_avgk_vs_removed_mass.png")

    plt.figure(figsize=(6, 4))
    plt.bar(["Vanilla", "P2 1%", "P4 1%"],
            [ep8_base["max_mean"], p2_ep8["max_mean"], p4_ep8["max_mean"]])
    plt.ylabel("Mean invocation max/mean rank time"); plt.title("SIMULATED-EP8 imbalance")
    save_plot(figures / "05_ep8_max_mean.png")

    plt.figure(figsize=(6, 4))
    plt.bar(["Vanilla", "P2 1%", "P4 1%"],
            [100 * ep8_base["wait_fraction"], 100 * p2_ep8["wait_fraction"],
             100 * p4_ep8["wait_fraction"]])
    plt.ylabel("Synchronization wait fraction (%)"); plt.title("SIMULATED-EP8 wait")
    save_plot(figures / "06_ep8_wait_fraction.png")

    pressure_labels = ["0-.1", ".1-.25", ".25-.5", ".5-1", "1-2", ">=2"]
    plt.figure(figsize=(7, 4))
    x = np.arange(len(pressure_labels)); width = .36
    for offset, item, label in ((-.18, p2, "P2"), (.18, p4, "P4")):
        values = np.asarray(item["removed_pressure_hist"], dtype=float)
        values = values / max(values.sum(), 1)
        plt.bar(x + offset, 100 * values, width, label=label)
    plt.xticks(x, pressure_labels); plt.xlabel("Assignment-rank pressure bin")
    plt.ylabel("Pruned routes (%)"); plt.title("Pressure distribution of omitted routes")
    plt.legend(); save_plot(figures / "07_pruned_route_pressure.png")

    for number, title in (
        (8, "Adjacent low-mass persistence"),
        (9, "Adjacent high-pressure persistence"),
        (10, "Joint persistence"),
        (11, "Current-step vs temporal P4"),
        (12, "Current-step vs commit-aware P4"),
    ):
        placeholder(figures / f"{number:02d}_{title.lower().replace(' ', '_')}_NOT_RUN.png", title)

    quality_table = f"""| Method | Correct | Accuracy | Additional wrong | Parsed identity | Mean NFE | NFE delta | Actual mass | AvgK | Route reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 31/32 | 96.875% | 0 | 100% | {document['baseline']['mean_nfe']:.3f} | 0% | 0% | 8.000 | 0% |
| P2 utility-only 1% | {q2['correct']}/32 | {percent(q2['accuracy'])} | {q2['baseline_correct_to_wrong']} | {percent(q2['parsed_answer_identity'])} | {q2['mean_nfe']:.3f} | {q2['mean_nfe_change_percent']:.3f}% | {percent(q2['actual_removed_mass_fraction'])} | {q2['avg_k']:.4f} | {percent(q2['route_reduction_fraction'])} |
| P4 EP-aware 1% | {q4['correct']}/32 | {percent(q4['accuracy'])} | {q4['baseline_correct_to_wrong']} | {percent(q4['parsed_answer_identity'])} | {q4['mean_nfe']:.3f} | {q4['mean_nfe_change_percent']:.3f}% | {percent(q4['actual_removed_mass_fraction'])} | {q4['avg_k']:.4f} | {percent(q4['route_reduction_fraction'])} |"""

    gsm = f"""# Route-pruning quality: GSM8K Stage A

## Verdict

The 1% P4 run was completed for all 32 paired requests, but it failed the
pre-registered gate: accuracy changed from 31/32 to 29/32 and two
baseline-correct requests (IDs 15 and 21) became wrong. Request 21 also rose
from 44 to 108 refinement forwards (2.45x). All requests terminated by EOS and
had zero remaining masks, so this is a semantic/trajectory failure rather than
a malformed-generation failure.

P2 met the permissive 32-sample gate exactly (one additional wrong), but this
is only a provisional utility-only point: one sample equals 3.125 percentage
points and no GSM8K-128 confirmation was allowed after P4 failed.

{quality_table}

The paired exact p-values are {q2['paired_exact_pvalue']:.3f} (P2) and
{q4['paired_exact_pvalue']:.3f} (P4). They are not evidence of equivalence;
the explicit additional-wrong gate, not an underpowered p-value, controls
promotion. Exact generation identity was only {percent(q2['exact_generation_identity'])}
for P2 and {percent(q4['exact_generation_identity'])} for P4, showing that even
1% route mass omission perturbs most trajectories.

## Protocol and scope

- Actual BF16 LLaDA2.0-mini rollouts, pinned revision, threshold 0.95,
  block length 32, max 32 refinements/block, no weight renormalization, min-k 4.
- Vanilla reused the already-validated identical GSM8K IDs 0--31 rollout;
  a smoke parity run matched its token sequence exactly before pruning runs.
- P2/P4 were independently rolled out; baseline routes were not edited offline.
- Dense semantic emulation computes all branches and zeroes omitted branch
  coefficients. Its wall time is not a sparse-runtime speedup.

## Stop decision

The 2.5% and 5% budgets, GSM8K-128, and HumanEval were not run. This follows
the specification's stop rule for clear P4 regression already at 1%.
"""
    (reports / "route_pruning_quality_gsm8k.md").write_text(gsm)

    humaneval = """# Route-pruning quality: HumanEval

HumanEval was **not run**. GSM8K-32 P4 at the minimum nonzero 1% router-mass
budget caused two additional wrong answers, so Stage A did not permit the
GSM8K-128 promotion required before HumanEval. No HumanEval quality or systems
number is inferred.
"""
    (reports / "route_pruning_quality_humaneval.md").write_text(humaneval)

    ep_report = f"""# Route-pruning EP4/EP8 projection

These numbers are routed-MoE-stage projections, **not request E2E latency**.
EP4 and EP8 are labeled SIMULATED-EP4/8-EP2-CALIBRATED. Dispatch and combine
use the true-EP2-calibrated communication model; owner-local expert compute
uses actual GPU 0/1 grouped-mm replay, extended with 128 representative
route-pruned shapes per EP degree. The evaluated assignment ranges are fully
inside the resulting calibration envelope.

{projection_table(document)}

## P4 gain decomposition

P4's total projected routed-MoE-stage reductions are
{gain(p4['projection']['ep4']['stage_gain_percent'])} at EP4 and
{gain(p4_ep8['stage_gain_percent'])} at EP8. They cannot be called a
quality-safe latency gain. Mean NFE fell {abs(q4['mean_nfe_change_percent']):.3f}%,
which explains most of both numbers. After normalizing stage cost by NFE, the
remaining gains are only {gain(p4['projection']['ep4']['nfe_normalized_stage_gain_percent'])}
(EP4) and {gain(p4_ep8['nfe_normalized_stage_gain_percent'])} (EP8).

At EP8, P4 does improve physical shape relative to Vanilla: max/mean changes
from {ep8_base['max_mean']:.4f} to {p4_ep8['max_mean']:.4f}, and wait fraction
from {percent(ep8_base['wait_fraction'])} to {percent(p4_ep8['wait_fraction'])}.
But P2's total EP8 stage reduction is larger ({gain(p2_ep8['stage_gain_percent'])})
because its trajectory happened to use fewer NFEs; P4 therefore has no positive
total-gain increment over P2. On the NFE-normalized view P4 exceeds P2 by only
{p4_ep8['nfe_normalized_stage_gain_percent'] - p2_ep8['nfe_normalized_stage_gain_percent']:.3f}
percentage points, far below the 3-point EP-specific gate.

Remote-byte reductions include the changed trajectories and are not isolated
route-only savings. P4 EP8 remote logical bytes/request change from
{ep8_base['remote_logical_bytes_per_request'] / 2**30:.3f} to
{p4_ep8['remote_logical_bytes_per_request'] / 2**30:.3f} GiB.
"""
    (reports / "route_pruning_quality_ep_projection.md").write_text(ep_report)

    stopped = """# Route-pruning temporal persistence

Stage B was not run. The Stage-A practical EP-aware policy failed quality at
the minimum tested nonzero budget. Consequently temporal T1/T2 policies were
not searched, and no dLLM-specific persistence value is reported.
"""
    (reports / "route_pruning_temporal_persistence.md").write_text(stopped)
    (reports / "route_pruning_commit_aware.md").write_text(stopped.replace(
        "temporal persistence", "commit-aware audit").replace(
        "temporal T1/T2 policies", "commit-aware policies"
    ))

    final = {
        "quality_safe_point_found": True,
        "quality_safe_scope": "P2 utility-only 1% only; no safe P4 EP-aware point",
        "best_safe_utility_only": {
            "nominal_budget_percent": 1.0,
            "actual_budget_percent": 100 * q2["actual_removed_mass_fraction"],
            "correct": q2["correct"], "n": q2["n"],
            "ep4_stage_gain_percent": p2["projection"]["ep4"]["stage_gain_percent"],
            "ep8_stage_gain_percent": p2_ep8["stage_gain_percent"],
            "caveat": "provisional GSM8K-32 gate only",
        },
        "best_safe_ep_aware": None,
        "p4_completed": True,
        "p4_unsafe_result": {
            "actual_budget_percent": 100 * q4["actual_removed_mass_fraction"],
            "correct": q4["correct"], "n": q4["n"],
            "additional_wrong": q4["baseline_correct_to_wrong"],
            "ep4_stage_gain_percent": p4["projection"]["ep4"]["stage_gain_percent"],
            "ep8_stage_gain_percent": p4_ep8["stage_gain_percent"],
            "ep4_nfe_normalized_gain_percent": p4["projection"]["ep4"]["nfe_normalized_stage_gain_percent"],
            "ep8_nfe_normalized_gain_percent": p4_ep8["nfe_normalized_stage_gain_percent"],
        },
        "ep8_ep_specific_increment_at_safe_point_pp": None,
        "max_safe_removed_router_mass_percent": 100 * q2["actual_removed_mass_fraction"],
        "avg_k_at_best_safe_point": q2["avg_k"],
        "ep8_max_mean_before_after_safe_point": [ep8_base["max_mean"], p2_ep8["max_mean"]],
        "ep8_wait_before_after_safe_point": [ep8_base["wait_fraction"], p2_ep8["wait_fraction"]],
        "humaneval_run": False,
        "stage_b_run": False,
        "dllm_specific_signal": "FAIL",
        "practical_method_status": "NO-GO",
        "primary_blocker": "P4 at 1% caused two additional GSM8K-32 errors and did not clear the 3pp EP-specific gate.",
        "labels": ["SIMULATED-EP4-EP2-CALIBRATED", "SIMULATED-EP8-EP2-CALIBRATED"],
        "source_summary": str(args.input),
    }
    (reports / "route_pruning_quality_dllm_summary.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n"
    )

    summary = f"""# Quality-safe EP-aware route-pruning summary

## Outcome

**NO-GO for practical P4 and FAIL for dLLM-specificity.** P4 was completed for
all 32 requests. Its {gain(p4_ep8['stage_gain_percent'])} total simulated EP8
routed-MoE-stage reduction is dominated by a {abs(q4['mean_nfe_change_percent']):.3f}%
NFE reduction and comes with two additional wrong answers. The NFE-normalized
EP8 gain is {gain(p4_ep8['nfe_normalized_stage_gain_percent'])}, only
{p4_ep8['nfe_normalized_stage_gain_percent'] - p2_ep8['nfe_normalized_stage_gain_percent']:.3f}
points above P2, so the EP-specific 3-point gate is not met.

P2 1% is the only provisional gate-safe point (one additional wrong), but it
was not promoted to 128 because the matched P4 comparison failed. This is not
evidence for quality preservation at scale.

## Evidence boundary

- Quality and trajectories: actual GSM8K-32 model rollouts.
- EP4/EP8 times: EP2-calibrated simulation of routed-MoE stage only.
- Dense emulator wall time: deliberately not used as a latency result.
- HumanEval, temporal persistence, commit-aware policy, production runtime:
  not run by the pre-registered stop rule.

See `route_pruning_quality_gsm8k.md` and
`route_pruning_quality_ep_projection.md` for the complete tables.

QUALITY_SAFE_POINT_FOUND:
YES

BEST_SAFE_UTILITY_ONLY:
1% nominal / 0.998% actual, 30/32, EP4 16.496% total projected stage gain, EP8 16.078% total projected stage gain; provisional GSM8K-32 only

BEST_SAFE_EP_AWARE:
N/A; P4 1% failed with 29/32 and two additional wrong answers

EP8_EP_SPECIFIC_INCREMENT_AT_SAFE_POINT:
N/A

MAX_SAFE_REMOVED_ROUTER_MASS:
0.998% for provisional P2; 0% established for P4

AVG_K_AT_BEST_SAFE_POINT:
7.7529

EP8_MAX_MEAN_BEFORE_AFTER_SAFE_POINT:
1.2122 -> 1.2178 (P2 worsened)

EP8_WAIT_BEFORE_AFTER_SAFE_POINT:
17.017% -> 17.376% (P2 worsened)

HUMANEVAL_RUN:
NO

HUMANEVAL_QUALITY_DELTA:
N/A

TEMPORAL_LOW_MASS_PERSISTENCE:
N/A

TEMPORAL_HIGH_PRESSURE_PERSISTENCE:
N/A

TEMPORAL_JOINT_PERSISTENCE:
N/A

TEMPORAL_HEURISTIC_GAIN_OVER_CURRENT_P4:
N/A

COMMIT_AWARE_GAIN_OVER_CURRENT_P4:
N/A

DLLM_SPECIFIC_SIGNAL:
FAIL

PRACTICAL_METHOD_STATUS:
NO-GO

PRIMARY_BLOCKER:
P4 at 1% caused two additional GSM8K-32 errors and its NFE-normalized EP8 increment over P2 was only 0.578 percentage points.

NEXT_ACTION:
Retire the current no-renormalization P4 path; only revisit route pruning under a separately specified safety mechanism, otherwise prioritize exact balancing.

DO_NOT_IMPLEMENT_PRODUCTION_RUNTIME_AUTOMATICALLY:
true
"""
    (reports / "route_pruning_quality_dllm_summary.md").write_text(summary)


if __name__ == "__main__":
    main()
