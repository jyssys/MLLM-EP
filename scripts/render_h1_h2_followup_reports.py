#!/usr/bin/env python3
"""Render concise tracked reports from the H1/H2 deep-dive artifact."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil


def pct(value):
    return f"{100 * value:.2f}%"


def f3(value):
    return f"{value:.3f}"


def strip_large_fields(value):
    if isinstance(value, dict):
        return {key: strip_large_fields(item) for key, item in value.items()
                if key != "block_stage_totals"}
    if isinstance(value, list):
        return [strip_large_fields(item) for item in value]
    return value


def severity_table(h1):
    rows = []
    for ep in (2, 4, 8):
        row = h1[f"ep{ep}"]["time_severity"]
        def six(metric, *, percent=False):
            values = [metric[key] for key in ("p50", "p75", "p90", "p95", "p99", "max")]
            formatter = pct if percent else f3
            return " / ".join(formatter(value) for value in values)
        rows.append(
            f"| EP{ep} | {six(row['max_mean'])} | {six(row['max_second'])} "
            f"| {six(row['cv'])} | {six(row['synchronization_wait_fraction'], percent=True)} |"
        )
    return "\n".join(rows)


def critical_rank_distribution_table(h1):
    rows = []
    for ep in (2, 4, 8):
        counts = h1[f"ep{ep}"]["time_severity"]["critical_rank_distribution"]
        total = sum(counts)
        distribution = " / ".join(
            f"r{rank} {count:,} ({100 * count / total:.1f}%)"
            for rank, count in enumerate(counts)
        )
        rows.append(f"| EP{ep} | {distribution} |")
    return "\n".join(rows)


def persistence_table(h1):
    rows = []
    for ep in (4, 8):
        layer = h1[f"ep{ep}"]["persistence"]["layer_local"]
        forward = h1[f"ep{ep}"]["persistence"]["forward_aggregate"]
        run = layer["run_length"]
        rows.append(
            f"| EP{ep} | {pct(layer['adjacent_critical_rank_match'])} | {pct(layer['global_majority_accuracy'])} "
            f"| {pct(layer['same_request_cross_block_match'])} | {pct(layer['matched_random_critical_rank_match'])} "
            f"| +{layer['within_block_increment_over_global_pp']:.2f} pp | {run['p50']:.0f}/{run['p90']:.0f}/{run['p99']:.0f}/{run['max']:.0f} "
            f"| {pct(forward['adjacent_critical_rank_match'])} / {pct(forward['global_majority_accuracy'])} |"
        )
    return "\n".join(rows)


def persistence_vector_table(h1):
    rows = []
    for ep in (2, 4, 8):
        layer = h1[f"ep{ep}"]["persistence"]["layer_local"]
        forward = h1[f"ep{ep}"]["persistence"]["forward_aggregate"]
        rows.append(
            f"| EP{ep} | {layer['adjacent_rank_time_cosine']:.6f} "
            f"| {layer['matched_random_rank_time_cosine']:.6f} "
            f"| {layer['adjacent_rank_time_spearman']:.6f} "
            f"| {forward['adjacent_rank_time_cosine']:.6f} "
            f"| {forward['adjacent_rank_time_spearman']:.6f} |"
        )
    return "\n".join(rows)


def concentration_table(h1):
    rows = []
    for ep in (4, 8):
        concentration = h1[f"ep{ep}"]["concentration"]
        top = concentration["top_k_excess_mass_fraction"]
        needed = concentration["minimum_experts_for_excess_fraction"]
        rows.append(
            f"| EP{ep} | {pct(top['top_1']['p50'])} | {pct(top['top_2']['p50'])} | {pct(top['top_4']['p50'])} "
            f"| {pct(top['top_8']['p50'])} | {needed['25']['p50']:.0f} / {needed['50']['p50']:.0f} / {needed['75']['p50']:.0f} / {needed['90']['p50']:.0f} "
            f"| {pct(concentration['high_imbalance_top4_ge_50_fraction'])} |"
        )
    return "\n".join(rows)


def prediction_table(h1):
    rows = []
    for ep in (4, 8):
        prediction = h1[f"ep{ep}"]["predictability"]
        for k in (1, 2, 4, 8):
            early = prediction["early_1_2"][f"k{k}"]
            global_row = prediction["global_hot"][f"k{k}"]
            delta = prediction["dllm_increment_early2_minus_global_excess_mass_pp"][f"k{k}"]
            rows.append(
                f"| EP{ep} | {k} | {pct(early['recall_mean'])} | {pct(early['precision_mean'])} "
                f"| {pct(early['excess_mass_recall_mean'])} | {pct(global_row['excess_mass_recall_mean'])} | {delta:+.2f} pp |"
            )
    return "\n".join(rows)


def replica_table(h1, ep):
    rows = []
    policies = (
        ("global_static", "global static"),
        ("full_future_block", "full-future block"),
        ("full_future_oracle_envelope", "full-future envelope"),
        ("early_1", "early refinement 1"),
        ("early_1_2", "early refinements 1–2"),
    )
    replication = h1[f"ep{ep}"]["replication"]
    for key, label in policies:
        for budget in (1, 2, 4, 8):
            row = replication[key][f"budget_{budget}"]
            if key == "full_future_oracle_envelope":
                rows.append(
                    f"| {label} | {budget} | — | — | {pct(row['aggregate_stage_reduction'])} | — | — |"
                )
                continue
            expert = ((row["component_ms"]["baseline_expert"] - row["component_ms"]["candidate_expert"])
                      / row["component_ms"]["baseline_expert"])
            setup = row.get("block_triggered_replication", {}).get(
                "setup_charged_feasible_block_fraction"
            )
            rows.append(
                f"| {label} | {budget} | {pct(expert)} | {pct(row['aggregate_compute_only_stage_reduction'])} "
                f"| {pct(row['aggregate_stage_reduction'])} | {f3(row['candidate']['max_mean']['p50'])} "
                f"| {'—' if setup is None else pct(setup)} |"
            )
    return "\n".join(rows)


def h1_report(document):
    h1 = document["h1"]
    ep4_perfect = h1["ep4"]["perfect_balance"]
    ep8_perfect = h1["ep8"]["perfect_balance"]
    ep4_global = h1["ep4"]["replication"]["global_static"]["budget_8"]
    ep4_early = max(
        (h1["ep4"]["replication"][policy][f"budget_{budget}"]["aggregate_stage_reduction"], policy, budget)
        for policy in ("early_1", "early_1_2") for budget in (1, 2, 4, 8)
    )
    ep8_early = max(
        (h1["ep8"]["replication"][policy][f"budget_{budget}"]["aggregate_stage_reduction"], policy, budget)
        for policy in ("early_1", "early_1_2") for budget in (1, 2, 4, 8)
    )
    memory = ep4_global["memory"]
    return f"""# H1 Persistent EP Straggler Deep Dive

## Verdict

**HOLD.** The persistent-straggler signal is real and the divisible-work upper bound is meaningful: EP4 has {pct(ep4_perfect['aggregate_routed_moe_stage_reduction'])} aggregate routed-MoE headroom and EP8 has {pct(ep8_perfect['aggregate_routed_moe_stage_reduction'])}. Exact-semantics replica oracles recover only {pct(ep4_early[0])} in the strongest early-persistent EP4 configuration and {pct(ep8_early[0])} in EP8. More importantly, global-static replication is stronger than early block adaptation, so the incremental dLLM persistence value is not method-level.

## Evidence boundary and invocation semantics

- Trace: 128 GSM8K requests at threshold 0.95, requests 0–63 discovery and 64–127 held out; accuracy 118/128 = 92.19%.
- `277,134` means **19 routed-MoE layers × 14,586 refinement forwards**, not independent generations.
- Rank expert time is predicted by target-EP single-H100 BF16 grouped-MLP replays; the complete EP2 routed-MoE predictor previously validated at 3.87% median and 9.40% P90 APE.
- EP4 and EP8 are `SIMULATED-EP4/8-EP2-CALIBRATED`, not physical runs.
- Replica routing preserves the selected expert ID and weight semantics. The aggregate trace lacks source×expert identity, so post-replica A/U traffic is a measured-dedup-rate estimate. Both compute-only and full-stage estimates are reported.
- The per-expert excess attribution allocates the calibrated critical-rank excess by exact expert-row share. It is not a per-expert GPU timestamp.
- No new model rollout was run. No true-EP2 replica replay was run because primary EP4 replication stayed below the prescribed 5% stage gate and EP2's perfect routed-stage upper bound itself is only {pct(h1['ep2']['perfect_balance']['aggregate_routed_moe_stage_reduction'])}.

## Time-based straggler severity

Every cell below is ordered `P50 / P75 / P90 / P95 / P99 / max`.

| Target | Time max/mean | Time max/second | CV | Synchronization wait fraction |
|---|---:|---:|---:|---:|
{severity_table(h1)}

| Target | Critical-rank invocation distribution |
|---|---|
{critical_rank_distribution_table(h1)}

Assignment imbalance is much larger (EP4 P50/P95 max/mean 1.285/1.624; EP8 1.721/2.668), but the calibrated grouped kernel compresses assignment imbalance. The time-based result is the latency-relevant one: EP4 still loses 6.91% rank-time at P50 and EP8 15.63%.

## Critical-rank persistence

| Target | Adjacent layer-local | Layer-global baseline | Cross-block | Matched random | Adjacent − global | Layer run P50/P90/P99/max | Forward adjacent/global |
|---|---:|---:|---:|---:|---:|---:|---:|
{persistence_table(h1)}

The global bias is explicitly retained. At forward aggregate, EP4's 97.82% adjacent match is only +3.68 pp above the 94.14% global-majority baseline; EP8 is 100% for both, so EP8 forward-level identity persistence has no incremental within-block signal. Layer-local persistence has more incremental information (+15.16 pp EP4, +18.12 pp EP8), but cross-block persistence is already 95–96%.

| Target | Layer adjacent cosine | Matched-random cosine | Layer adjacent Spearman | Forward adjacent cosine | Forward adjacent Spearman |
|---|---:|---:|---:|---:|---:|
{persistence_vector_table(h1)}

Cosine is near one even for matched random controls because a large shared/global load component dominates vector magnitude. Critical-rank identity and Spearman rank ordering provide the more discriminative persistence evidence.

## Critical excess concentration

| Target | Top1 P50 | Top2 | Top4 | Top8 | Experts for 25/50/75/90% P50 | High-imbalance states where top4 ≥50% |
|---|---:|---:|---:|---:|---:|---:|
{concentration_table(h1)}

EP4 is moderately distributed: eight experts are needed for 50% of excess at P50. EP8 is more concentrated: top four explain 57.37% at P50, making small replica budgets structurally more plausible.

## Early hot-expert prediction versus global popularity

| Target | K | Recall | Precision | Early excess-mass recall | Global excess-mass recall | Early increment |
|---|---:|---:|---:|---:|---:|---:|
{prediction_table(h1)}

Early refinements add 2.3–7.3 pp EP4 and 2.7–10.0 pp EP8 excess-mass recall over global popularity. This is real dLLM-specific information, but it does not translate into a better stage oracle than pre-resident global replicas.

## Perfect-balance oracle

| Target | Expert reduction aggregate | Routed-stage reduction aggregate | Routed-stage P50/P95/P99 |
|---|---:|---:|---:|
| EP4 | {pct(ep4_perfect['aggregate_expert_stage_reduction'])} | **{pct(ep4_perfect['aggregate_routed_moe_stage_reduction'])}** | {pct(ep4_perfect['routed_moe_stage_reduction']['p50'])} / {pct(ep4_perfect['routed_moe_stage_reduction']['p95'])} / {pct(ep4_perfect['routed_moe_stage_reduction']['p99'])} |
| EP8 | {pct(ep8_perfect['aggregate_expert_stage_reduction'])} | **{pct(ep8_perfect['aggregate_routed_moe_stage_reduction'])}** | {pct(ep8_perfect['routed_moe_stage_reduction']['p50'])} / {pct(ep8_perfect['routed_moe_stage_reduction']['p95'])} / {pct(ep8_perfect['routed_moe_stage_reduction']['p99'])} |

This is a divisible-work upper bound, not an implementable placement policy. It establishes that the earlier ~0.3% one-owner placement result did not measure total straggler headroom.

## Exact-semantics replica oracles — EP4

| Policy | Replicas/layer | Expert reduction | Compute-only stage | Full stage estimate | Candidate max/mean P50 | Setup-amortizing blocks |
|---|---:|---:|---:|---:|---:|---:|
{replica_table(h1, 4)}

## Exact-semantics replica oracles — EP8

| Policy | Replicas/layer | Expert reduction | Compute-only stage | Full stage estimate | Candidate max/mean P50 | Setup-amortizing blocks |
|---|---:|---:|---:|---:|---:|---:|
{replica_table(h1, 8)}

The strongest global-static results are {pct(ep4_global['aggregate_stage_reduction'])} EP4 and {pct(h1['ep8']['replication']['global_static']['budget_8']['aggregate_stage_reduction'])} EP8. The best restricted full-future envelopes reach {pct(h1['ep4']['replication']['full_future_oracle_envelope']['budget_8']['aggregate_stage_reduction'])} and {pct(h1['ep8']['replication']['full_future_oracle_envelope']['budget_8']['aggregate_stage_reduction'])}; even future knowledge does not expose a large hidden replica opportunity.

The complete per-policy P50/P75/P90/P95/P99/max distributions for max-rank time, second-max time, mean, max/mean, max/second, CV, wait fraction, and dispatch/expert/combine/stage component totals are retained in `reports/h1_replication_oracle_summary.json`.

## Replica cost

- Exact checkpoint shapes: gate `[512,2048]`, up `[512,2048]`, down `[2048,512]`, BF16.
- One expert instance: {memory['expert_bytes']:,} bytes = 6.00 MiB.
- Eight replicas per each of 19 routed layers: {memory['total_pool_bytes_all_19_layers']:,} bytes = {memory['total_pool_bytes_all_19_layers']/2**30:.3f} GiB total pool, before allocator/alignment overhead.
- If replica destinations are balanced, that budget averages {memory['total_pool_bytes_all_19_layers']/4/2**20:.1f} MiB/rank at EP4 or {memory['total_pool_bytes_all_19_layers']/8/2**20:.1f} MiB/rank at EP8; the conservative one-rank concentration bound is the full {memory['total_pool_bytes_all_19_layers']/2**20:.1f} MiB.
- Early-1–2, budget 4 setup-charged break-even P50: {h1['ep4']['replication']['early_1_2']['budget_4']['block_triggered_replication']['setup_charged_break_even_refinements']['p50']:.2f} future refinements EP4 and {h1['ep8']['replication']['early_1_2']['budget_4']['block_triggered_replication']['setup_charged_break_even_refinements']['p50']:.2f} EP8.
- The setup charge is an evidence-derived sensitivity: endpoint payload time at measured 85.51 GB/s plus one measured EP2 dispatch-startup charge per affected layer. It is not a direct expert-weight-copy measurement.

## Decision

- H1A severity: **PASS**.
- H1B persistence: **PASS** for EP4/EP8.
- H1C concentration: **PASS**, especially EP8.
- H1D perfect-balance headroom: **PASS**.
- H1E practical EP4 replica headroom: **WEAK** (<5%).

The correct conclusion is **HOLD / STRUCTURAL_HEADROOM_ONLY**. There is genuine imbalance and a nontrivial upper bound, but small-budget exact replication captures less than 4% EP4 stage time, and global static popularity captures more than the dLLM-specific early-persistence policy. No production method is justified yet.
"""


def h2_report(document):
    h2 = document["h2"]
    state_rows = []
    for state in ("CURRENT_BLOCK_MASKED", "CURRENT_BLOCK_NEWLY_ACCEPTED", "CURRENT_BLOCK_DECODED"):
        row = h2["state_only"]["ep8"][state]
        state_rows.append(
            f"| {state} | {f3(row['max_mean']['p50'])} / {f3(row['max_mean']['p95'])} "
            f"| {f3(row['cv']['p50'])} / {f3(row['cv']['p95'])} "
            f"| {row['remote_bytes']['p50']/1024:.1f} / {row['remote_bytes']['p95']/1024:.1f} KiB "
            f"| {f3(row['fanout']['p50'])} |"
        )
    effect_rows = []
    for ep in (4, 8):
        effects = h2["full_workload_composition_effect"][f"ep{ep}"]
        effect_rows.append(
            f"| EP{ep} | {pct(effects['max_mean']['p50'])} / {pct(effects['max_mean']['p95'])} / {pct(effects['max_mean']['p99'])} "
            f"| {pct(effects['critical_expert_time']['p95'])} / {pct(effects['critical_expert_time']['p99'])} "
            f"| {pct(effects['remote_bytes']['p95'])} / {pct(effects['remote_bytes']['p99'])} "
            f"| {pct(effects['routed_moe_stage']['p50'])} / {pct(effects['routed_moe_stage']['p95'])} / {pct(effects['routed_moe_stage']['p99'])} / {pct(effects['routed_moe_stage']['max'])} |"
        )
    return f"""# H2 Mask-State Specialization: EP8 Re-analysis

## Verdict

**HOLD / NO METHOD.** EP8 makes expert-ID volatility more visible at the physical-rank level, but the full-workload systems consequence remains far below the 5% promotion gate.

## Same-position transition

| Metric | EP4 | EP8 |
|---|---:|---:|
| Destination-rank-set Jaccard P50 | {f3(h2['same_position']['ep4_destination_rank_set_jaccard']['p50'])} | **{f3(h2['same_position']['ep8_destination_rank_set_jaccard']['p50'])}** |
| Rank-set transition rate | {pct(h2['same_position']['ep4_rank_transition_rate'])} | **{pct(h2['same_position']['ep8_rank_transition_rate'])}** |
| Samples | {h2['same_position']['samples']:,} | {h2['same_position']['samples']:,} |

The expert-set Jaccard remains the prior 0.231 P50. Splitting 256 experts into 32-expert EP8 buckets lowers destination Jaccard from 0.75 to 0.60 and raises transition incidence from 66.7% to 87.1%.

## EP8 state-only geometry

| State | Max/mean P50/P95 | CV P50/P95 | Remote bytes P50/P95 | Fanout P50 |
|---|---:|---:|---:|---:|
{chr(10).join(state_rows)}

State-only imbalance is large, but these 1–32 current-block rows are embedded in 1,472–3,616 physical rows. State-only predicted times also fall outside parts of the measured EP8 compute envelope and are diagnostic only.

## Full physical workload composition effect

| Target | Max/mean P50/P95/P99 | Critical expert P95/P99 | Remote bytes P95/P99 | Routed stage P50/P95/P99/max |
|---|---:|---:|---:|---:|
{chr(10).join(effect_rows)}

The counterfactual replaces current MASKED route sets with a deterministic cycle of same-invocation DECODED route sets, while retaining the full prefix/prior-block workload. EP8 raises the P95 stage effect from 0.082% to **0.211%**, still two orders of magnitude below the 5% gate. P99 is 2.03%, and only isolated maxima reach 11.31%.

No H2 method should be created. EP8 amplifies rank identity transitions, not a reproducible full-stage bottleneck.
"""


def summary_report(document):
    h1, h2 = document["h1"], document["h2"]
    ep4_perfect = h1["ep4"]["perfect_balance"]["aggregate_routed_moe_stage_reduction"]
    ep8_perfect = h1["ep8"]["perfect_balance"]["aggregate_routed_moe_stage_reduction"]
    ep4_global = max(h1["ep4"]["replication"]["global_static"][f"budget_{b}"]["aggregate_stage_reduction"] for b in (1,2,4,8))
    ep4_early = max(h1["ep4"]["replication"][p][f"budget_{b}"]["aggregate_stage_reduction"] for p in ("early_1","early_1_2") for b in (1,2,4,8))
    ep8_early = max(h1["ep8"]["replication"][p][f"budget_{b}"]["aggregate_stage_reduction"] for p in ("early_1","early_1_2") for b in (1,2,4,8))
    return f"""# H1/H2 Follow-up Summary

The deep dive changes the interpretation of the earlier H3 result: one-owner placement had only ~0.3% headroom, but the actual divisible straggler upper bound is {pct(ep4_perfect)} EP4 and {pct(ep8_perfect)} EP8 routed-MoE stage time. Small-budget exact replication captures only {pct(ep4_global)} EP4 under the stronger global-static policy; early dLLM persistence reaches {pct(ep4_early)} EP4 and {pct(ep8_early)} EP8 and is not better than global popularity. H2 becomes more rank-volatile at EP8 but remains physically diluted ({pct(h2['full_workload_composition_effect']['ep8']['routed_moe_stage']['p95'])} P95 full-stage effect).

```text
H1_STRAGGLER_SEVERITY:
PASS

H1_CRITICAL_RANK_PERSISTENCE_EP4:
PASS

H1_CRITICAL_RANK_PERSISTENCE_EP8:
PASS

H1_TOP_EXPERT_EXCESS_CONCENTRATION:
PASS

H1_PERFECT_BALANCE_HEADROOM_EP4:
{pct(ep4_perfect)} aggregate routed-MoE stage ({pct(h1['ep4']['perfect_balance']['routed_moe_stage_reduction']['p50'])} P50)

H1_PERFECT_BALANCE_HEADROOM_EP8:
{pct(ep8_perfect)} aggregate routed-MoE stage ({pct(h1['ep8']['perfect_balance']['routed_moe_stage_reduction']['p50'])} P50)

H1_GLOBAL_STATIC_REPLICATION_HEADROOM_EP4:
{pct(ep4_global)}

H1_EARLY_PERSISTENCE_REPLICATION_HEADROOM_EP4:
{pct(ep4_early)}

H1_EARLY_PERSISTENCE_REPLICATION_HEADROOM_EP8:
{pct(ep8_early)}

H1_TRUE_EP2_REPLICA_DIRECTION:
NOT_RUN

H1_VERDICT:
HOLD

H2_EP4_DESTINATION_JACCARD_P50:
{h2['same_position']['ep4_destination_rank_set_jaccard']['p50']:.3f}

H2_EP8_DESTINATION_JACCARD_P50:
{h2['same_position']['ep8_destination_rank_set_jaccard']['p50']:.3f}

H2_EP8_FULL_WORKLOAD_P95_EFFECT:
{pct(h2['full_workload_composition_effect']['ep8']['routed_moe_stage']['p95'])} routed-MoE stage

H2_VERDICT:
HOLD

MAIN_INTERPRETATION:
Persistent EP stragglers have real upper-bound headroom, especially at EP8, but the implementable exact-replication headroom is modest and mostly explained by globally hot experts rather than additional same-block refinement information. H2 rank transitions increase at EP8 without a material full-workload cost.

DO_NOT_IMPLEMENT_PRODUCTION_METHOD_AUTOMATICALLY:
true
```
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--figure-source", type=Path, required=True)
    parser.add_argument("--figure-destination", type=Path,
                        default=Path("reports/figures/h1_straggler_deepdive"))
    args = parser.parse_args()
    document = json.loads(args.summary.read_text())
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "h1_persistent_straggler_deepdive.md").write_text(h1_report(document))
    (args.report_dir / "h2_ep8_mask_state_reanalysis.md").write_text(h2_report(document))
    (args.report_dir / "h1_h2_followup_summary.md").write_text(summary_report(document))
    concise_h1 = strip_large_fields(copy.deepcopy(document["h1"]))
    for ep in (4, 8):
        for policy, policy_rows in concise_h1[f"ep{ep}"]["replication"].items():
            if policy == "full_future_oracle_envelope":
                continue
            for budget in (1, 2, 4, 8):
                memory = policy_rows[f"budget_{budget}"]["memory"]
                memory["mean_pool_bytes_per_rank"] = memory["total_pool_bytes_all_19_layers"] / ep
                memory["worst_case_pool_bytes_one_rank"] = memory["total_pool_bytes_all_19_layers"]
    h1_summary = {
        "evidence": document["evidence"], "h1": concise_h1,
        "true_ep2_replica_direction": "NOT_RUN",
        "true_ep2_skip_reason": "primary EP4 exact-replication stage headroom stayed below 5%; EP2 perfect stage upper bound is only 2.21%",
        "verdict": "HOLD",
    }
    h2_summary = {"evidence": document["evidence"], "h2": document["h2"], "verdict": "HOLD"}
    (args.report_dir / "h1_replication_oracle_summary.json").write_text(
        json.dumps(h1_summary, indent=2) + "\n"
    )
    (args.report_dir / "h2_ep8_mask_state_reanalysis_summary.json").write_text(
        json.dumps(h2_summary, indent=2) + "\n"
    )
    args.figure_destination.mkdir(parents=True, exist_ok=True)
    for source in args.figure_source.glob("*.png"):
        shutil.copy2(source, args.figure_destination / source.name)


if __name__ == "__main__":
    main()
