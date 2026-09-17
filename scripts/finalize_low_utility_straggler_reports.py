#!/usr/bin/env python3
"""Merge exact GSM8K-128 and supplemental full-row evidence into reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def by(counter: dict, ep: int, policy: str, budget: float) -> dict:
    return next(
        row for row in counter["matched_mass"]
        if row["ep"] == ep and row["policy"] == policy and row["budget"] == budget
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--full-row-audit", type=Path, required=True)
    parser.add_argument("--full-row-counterfactual", type=Path, required=True)
    parser.add_argument("--calibration-extension", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    audit = json.loads(args.full_row_audit.read_text())
    counter = json.loads(args.full_row_counterfactual.read_text())
    calibration = json.loads(args.calibration_extension.read_text())
    summary["supplemental_full_row_audit"] = audit
    summary["supplemental_full_row_counterfactual"] = counter
    summary["calibration_extension"] = {
        "artifact": str(args.calibration_extension),
        **calibration,
        "combined_model": "artifacts/virtual_ep/20260918_low_utility_straggler/compute_model/ep8_grouped_mm_pruned_route_combined.csv",
        "note": "Only same-contract post-pruning route replays were merged; grouped-GEMM-only diagnostic rows were excluded.",
    }

    budget = .10
    ep4_generic, ep4_oracle = by(counter, 4, "P2_utility", budget), by(counter, 4, "P3_calibrated_oracle", budget)
    ep8_generic, ep8_oracle = by(counter, 8, "P2_utility", budget), by(counter, 8, "P3_calibrated_oracle", budget)
    ep8_practical = by(counter, 8, "P4_rank_pressure", budget)
    base8 = counter["baseline"]["8"]
    a4, a8 = audit["targets"]["ep4"], audit["targets"]["ep8"]
    ep8_mass_for_half_excess = a8["low_mass_first_curve"][
        "router_mass_percent_needed_for_attributed_excess"
    ]["0.5"]
    summary["decision"] = {
        "gates": {"g1": "FAIL", "g2": "PASS", "g3": "PASS", "g4": "PASS"},
        "verdict": "HOLD",
        "interpretation": "EP-specific systems oracle without critical-rank low-mass enrichment",
        "reasoning": (
            "Both the exact GSM8K-128 current-block population and the GSM8K-32 all-physical-row audit show low-mass fractions lower on critical than non-critical ranks, so the proposed enrichment mechanism is false. "
            "Low-mass routes are nevertheless common: the full-row audit attributes 39.41% of EP8 critical excess to <=10%-mass routes. On 2,048 exact full-row invocation replays, EP-aware selection beats utility-only by 4.60pp at a 5% mass budget and 7.69pp at 10%, while reducing max/mean and wait. This is HOLD rather than GO because the effect needs aggressive unvalidated approximation, the full-row policy is only a systems sample, and its opportunity is not shown to arise specifically from dLLM refinement."
        ),
    }
    required = f"""TRACE_SOURCE:
{summary['trace_source']} (exact current-block slots, GSM8K-128); {audit['source']} (all-physical-row audit, GSM8K-32); grouped-mm calibration extended with 105 actual post-pruning owner-rank replays

EP4_LOW_MASS_CRITICAL_ENRICHMENT:
{a4['bottom_25_global']['enrichment']:.4f}x / global bottom-25% normalized effective route mass, all-physical-row audit ({a4['bottom_25_global']['percentage_point_difference']:+.4f} pp; request-bootstrap 95% CI {a4['bottom_25_global']['request_cluster_bootstrap']['enrichment_95ci'][0]:.4f}-{a4['bottom_25_global']['request_cluster_bootstrap']['enrichment_95ci'][1]:.4f}x)

EP8_LOW_MASS_CRITICAL_ENRICHMENT:
{a8['bottom_25_global']['enrichment']:.4f}x / global bottom-25% normalized effective route mass, all-physical-row audit ({a8['bottom_25_global']['percentage_point_difference']:+.4f} pp; request-bootstrap 95% CI {a8['bottom_25_global']['request_cluster_bootstrap']['enrichment_95ci'][0]:.4f}-{a8['bottom_25_global']['request_cluster_bootstrap']['enrichment_95ci'][1]:.4f}x)

EP4_LOW_MASS_CRITICAL_EXCESS_SHARE:
{100*a4['mass_le_0.1']['sampled_attributed_excess_share']:.4f}% / normalized per-route mass <=10%, all-physical-row sampled ATTRIBUTED_EXCESS

EP8_LOW_MASS_CRITICAL_EXCESS_SHARE:
{100*a8['mass_le_0.1']['sampled_attributed_excess_share']:.4f}% / normalized per-route mass <=10%, all-physical-row sampled ATTRIBUTED_EXCESS

EP4_TAIL_7_8_CRITICAL_EXCESS_SHARE:
{100*a4['slots_7_8']['sampled_attributed_excess_share']:.4f}%

EP8_TAIL_7_8_CRITICAL_EXCESS_SHARE:
{100*a8['slots_7_8']['sampled_attributed_excess_share']:.4f}%

BEST_GENERIC_PRUNING_POLICY:
P2 utility-only / 10% removed router-mass budget / sampled full physical rows (EP8 stage gain {ep8_generic['stage_gain_percent']:.3f}%)

BEST_EP_AWARE_PRUNING_ORACLE:
P3 four-round calibrated-pressure ORACLE / 10% removed-mass budget (EP8 stage gain {ep8_oracle['stage_gain_percent']:.3f}%)

BEST_EP_AWARE_PRACTICAL_HEURISTIC:
P4 assignment-rank-pressure / 10% removed-mass budget (EP8 stage gain {ep8_practical['stage_gain_percent']:.3f}%)

EP4_GENERIC_STAGE_GAIN:
{ep4_generic['stage_gain_percent']:.3f}%

EP4_EP_AWARE_STAGE_GAIN:
{ep4_oracle['stage_gain_percent']:.3f}%

EP4_EP_SPECIFIC_INCREMENT:
{ep4_oracle['stage_gain_percent']-ep4_generic['stage_gain_percent']:+.3f} percentage points

EP8_GENERIC_STAGE_GAIN:
{ep8_generic['stage_gain_percent']:.3f}%

EP8_EP_AWARE_STAGE_GAIN:
{ep8_oracle['stage_gain_percent']:.3f}%

EP8_EP_SPECIFIC_INCREMENT:
{ep8_oracle['stage_gain_percent']-ep8_generic['stage_gain_percent']:+.3f} percentage points

EP8_MAX_MEAN_BEFORE_AFTER:
{base8['max_mean_mean']:.4f} -> {ep8_oracle['max_mean_mean']:.4f}

EP8_WAIT_FRACTION_BEFORE_AFTER:
{base8['wait_fraction_mean']:.4f} -> {ep8_oracle['wait_fraction_mean']:.4f}

EP8_ROUTER_MASS_NEEDED_FOR_50_PERCENT_EXCESS_REMOVAL:
{ep8_mass_for_half_excess:.4f}% / sampled all-physical-row low-mass-first ATTRIBUTED_EXCESS curve (not a measured removal or quality-safe point)

DLLM_PHASE_EFFECT:
Exact GSM8K-128 current-block analysis remains near/below 1x enrichment across early/middle/late; no growing dLLM-specific phase effect was found.

QUALITY_ROLLOUT_RUN:
NO

PRIMARY_INTERPRETATION:
EP-specific systems oracle, but the proposed critical-rank low-mass enrichment is absent

VERDICT:
HOLD

NEXT_ACTION:
Do not build a production runtime; if continued, run a separate conservative quality study against utility-only at min_k=4 before making any method claim."""
    summary["required_final_summary"] = required
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    marker = "\n## Supplemental all-physical-row audit\n"
    anatomy_path = args.report_dir / "low_utility_straggler_anatomy.md"
    anatomy_text = anatomy_path.read_text().split(marker)[0]
    anatomy_text += marker + f"""
The existing threshold-.95 GSM8K-32 heavy trace was audited over {audit['sampled_token_rows']:,} evenly sampled physical token rows from all {audit['invocations']:,} layer/refinement invocations. This includes prompt, prior blocks, and current block.

| target | definition | critical | non-critical | enrichment | difference | sampled ATTRIBUTED_EXCESS |
|---|---|---:|---:|---:|---:|---:|
| EP4 | bottom 25% | {100*a4['bottom_25_global']['critical_fraction']:.2f}% | {100*a4['bottom_25_global']['noncritical_fraction']:.2f}% | {a4['bottom_25_global']['enrichment']:.3f}x | {a4['bottom_25_global']['percentage_point_difference']:+.2f}pp | {100*a4['bottom_25_global']['sampled_attributed_excess_share']:.2f}% |
| EP4 | mass <=10% | {100*a4['mass_le_0.1']['critical_fraction']:.2f}% | {100*a4['mass_le_0.1']['noncritical_fraction']:.2f}% | {a4['mass_le_0.1']['enrichment']:.3f}x | {a4['mass_le_0.1']['percentage_point_difference']:+.2f}pp | {100*a4['mass_le_0.1']['sampled_attributed_excess_share']:.2f}% |
| EP8 | bottom 25% | {100*a8['bottom_25_global']['critical_fraction']:.2f}% | {100*a8['bottom_25_global']['noncritical_fraction']:.2f}% | {a8['bottom_25_global']['enrichment']:.3f}x | {a8['bottom_25_global']['percentage_point_difference']:+.2f}pp | {100*a8['bottom_25_global']['sampled_attributed_excess_share']:.2f}% |
| EP8 | mass <=10% | {100*a8['mass_le_0.1']['critical_fraction']:.2f}% | {100*a8['mass_le_0.1']['noncritical_fraction']:.2f}% | {a8['mass_le_0.1']['enrichment']:.3f}x | {a8['mass_le_0.1']['percentage_point_difference']:+.2f}pp | {100*a8['mass_le_0.1']['sampled_attributed_excess_share']:.2f}% |

This audit reverses no conclusion: low-mass work is common, but is depleted—not enriched—on the critical rank.

The sampled low-mass-first curve needs {ep8_mass_for_half_excess:.4f}% of total EP8 router mass to cover 50% of modeled `ATTRIBUTED_EXCESS`. This is an attribution curve, not proof that removing those routes realizes the full excess reduction.
"""
    anatomy_path.write_text(anatomy_text)

    oracle_path = args.report_dir / "low_utility_straggler_ep4_ep8_oracle.md"
    oracle_text = oracle_path.read_text().split("\n## Supplemental full-row counterfactual\n")[0]
    lines = []
    for row in counter["matched_mass"]:
        lines.append(
            f"| EP{row['ep']} | {row['policy']} | {100*row['budget']:.1f}% | "
            f"{row['stage_gain_percent']:.3f}% | {row['max_mean_mean']:.4f} | "
            f"{100*row['wait_fraction_mean']:.2f}% | {row['remote_bytes_reduction_percent']:.2f}% |"
        )
    oracle_text += "\n## Supplemental full-row counterfactual\n\n"
    oracle_text += f"The systems screen uses {counter['sample']['invocations']:,} evenly spaced full physical invocations ({counter['sample']['expert_slots']:,} expert slots) from the existing GSM8K-32 heavy trace. It is a sampled systems oracle, not a request-level quality result.\n\n"
    oracle_text += "| target | policy | mass budget | stage gain | mean max/mean | mean wait | remote-byte reduction |\n|---|---|---:|---:|---:|---:|---:|\n" + "\n".join(lines) + "\n"
    oracle_text += "\nAt 5% mass, P3 exceeds P2 by 3.23pp EP4 and 4.60pp EP8; P4 exceeds P2 by 1.96pp EP4 and 3.28pp EP8. At 10%, P3's EP8 increment is 7.69pp. These gains reduce max/mean and wait, but the approximation budget is not quality-validated.\n"
    oracle_path.write_text(oracle_text)

    summary_path = args.report_dir / "low_utility_straggler_summary.md"
    summary_path.write_text(f"""# Low-Utility EP Straggler — Summary

**Verdict: HOLD.** The proposed enrichment mechanism fails, while a separate EP-pressure pruning oracle survives only at aggressive, quality-unvalidated mass budgets.

## Gate result

- G1 low-utility enrichment: **FAIL**. EP8 bottom-25 enrichment is {a8['bottom_25_global']['enrichment']:.3f}x (request-bootstrap 95% CI {a8['bottom_25_global']['request_cluster_bootstrap']['enrichment_95ci'][0]:.3f}-{a8['bottom_25_global']['request_cluster_bootstrap']['enrichment_95ci'][1]:.3f}x) and the difference is {a8['bottom_25_global']['percentage_point_difference']:+.2f}pp on the all-row audit.
- G2 excess share: **PASS structurally**. `mass<=10%` routes carry {100*a8['mass_le_0.1']['sampled_attributed_excess_share']:.2f}% of sampled EP8 `ATTRIBUTED_EXCESS`, but they are common everywhere.
- G3 EP-specific advantage: **PASS only in the sampled systems oracle**. At 5% mass, EP8 P2/P3/P4 stage gains are {by(counter,8,'P2_utility',.05)['stage_gain_percent']:.2f}% / {by(counter,8,'P3_calibrated_oracle',.05)['stage_gain_percent']:.2f}% / {by(counter,8,'P4_rank_pressure',.05)['stage_gain_percent']:.2f}%.
- G4 scaling: **PASS**. The P3 increment at 5% is 3.23pp EP4 versus 4.60pp EP8.

## Interpretation

{summary['decision']['reasoning']}

No generation, quality rollout, weight renormalization, or production runtime was executed. EP4/EP8 remain calibrated simulations. The 128-request analysis is exact for current-block slots; the full-row anatomy and systems counterfactual use existing GSM8K-32 heavy-trace samples.

## Required final summary

```text
{required}
```
""")

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    for ep in (4, 8):
        plt.figure(figsize=(7, 4))
        for policy, label in (("P2_utility", "utility-only"),
                              ("P3_calibrated_oracle", "EP-aware oracle"),
                              ("P4_rank_pressure", "EP-aware heuristic")):
            rows = [row for row in counter["matched_mass"] if row["ep"] == ep and row["policy"] == policy]
            plt.plot([100 * row["budget"] for row in rows], [row["stage_gain_percent"] for row in rows], "o-", label=label)
        plt.xlabel("removed router mass budget (%)"); plt.ylabel("routed-MoE stage gain (%)")
        plt.title(f"EP{ep} full-row sampled counterfactual"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
        plt.savefig(args.figure_dir / f"ep{ep}_stage_gain_vs_router_mass.png", dpi=170); plt.close()
    for metric, filename, ylabel in (("max_mean_mean", "ep8_max_mean_vs_router_mass.png", "mean max/mean"),
                                     ("wait_fraction_mean", "ep8_wait_vs_router_mass.png", "mean wait fraction")):
        plt.figure(figsize=(7, 4))
        for policy, label in (("P2_utility", "utility-only"), ("P3_calibrated_oracle", "EP-aware oracle"), ("P4_rank_pressure", "EP-aware heuristic")):
            rows = [row for row in counter["matched_mass"] if row["ep"] == 8 and row["policy"] == policy]
            plt.plot([100 * row["budget"] for row in rows], [row[metric] for row in rows], "o-", label=label)
        plt.xlabel("removed router mass budget (%)"); plt.ylabel(ylabel); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
        plt.savefig(args.figure_dir / filename, dpi=170); plt.close()
    plt.figure(figsize=(7, 4))
    for ep in (4, 8):
        values = counter["ep_specific_increment_pp"][f"ep{ep}"]
        budgets = [str(value) for value in (.005, .01, .025, .05, .1)]
        plt.plot([100 * float(value) for value in budgets], [values[value] for value in budgets], "o-", label=f"EP{ep}")
    plt.axhline(3, color="red", ls="--", label="G3=3pp"); plt.xlabel("removed router mass budget (%)"); plt.ylabel("P3-P2 stage gain (pp)")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(args.figure_dir / "ep4_vs_ep8_increment.png", dpi=170); plt.close()

    plt.figure(figsize=(7, 4))
    for ep, target in ((4, a4), (8, a8)):
        curve = target["low_mass_first_curve"]["router_mass_percent_needed_for_attributed_excess"]
        fractions = (.25, .50, .75, .90)
        plt.plot([curve[str(value)] for value in fractions], [100 * value for value in fractions], "o-", label=f"EP{ep}")
    plt.xlabel("sampled router mass removed (%)")
    plt.ylabel("sampled ATTRIBUTED_EXCESS covered (%)")
    plt.title("All-physical-row low-mass-first attribution curve")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(args.figure_dir / "router_mass_vs_critical_excess.png", dpi=170); plt.close()

    print(required)


if __name__ == "__main__":
    main()
