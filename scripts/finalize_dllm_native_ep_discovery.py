#!/usr/bin/env python3
"""Render the final two-track discovery verdict and remaining required figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structural", type=Path, default=Path("artifacts/dllm_native_ep_discovery/structural_summary.json"))
    parser.add_argument("--branches", type=Path, default=Path("artifacts/dllm_native_ep_discovery/branch_stability_summary.json"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()
    structural = json.loads(args.structural.read_text())
    branches = json.loads(args.branches.read_text())
    persistence = structural["track_a"]["persistence"]["global"]["heldout"]
    controls = structural["track_a"]["controls"]
    systems = structural["track_a"]["systems"]
    stability = branches["stability"]
    branch_systems = branches["systems"]
    ownership = structural["track_b"]
    ep4 = ownership["ep4"]
    ep8 = ownership["ep8"]
    safe = stability["oracle_thresholds"]["0.05"]["raw_output_reusable_fraction"]
    practical = stability["practical_gate"]["heldout"]["E3"]["coverage_of_stay"]
    ep4_gain = branch_systems["ep4"]["policies"]["E3_practical"]["stage_gain_percent"]
    ep8_gain = branch_systems["ep8"]["policies"]["E3_practical"]["stage_gain_percent"]

    combined = {
        "evidence_labels": {
            "route_identity": "MEASURED_MODEL_TRACE_GSM8K128",
            "branch_vectors": "MEASURED_DUPLICATE_COMPUTE_GSM8K16",
            "ep4": "SIMULATED-EP4-EP2-CALIBRATED",
            "ep8": "SIMULATED-EP8-EP2-CALIBRATED",
            "quality_reuse": "NOT_RUN_GATE_FAILED",
        },
        "track_a": {
            "stay_route_fraction_heldout": persistence["stay_route_fraction"],
            "stay_router_mass_fraction_heldout": persistence["stay_router_mass_fraction"],
            "adjacent_control_gap_pp": 100 * (
                controls["adjacent_live_mask"]["stay_route_fraction"]
                - controls["matched_random_nonadjacent_refinement"]["stay_route_fraction"]
            ),
            "stability_oracle_reusable_fraction_l2_5pct": safe,
            "practical_E3_heldout_stay_coverage": practical,
            "practical_E3_heldout_precision": stability["practical_gate"]["heldout"]["E3"]["precision_for_safe"],
            "practical_E3_ep4_stage_gain_percent": ep4_gain,
            "practical_E3_ep8_stage_gain_percent": ep8_gain,
            "impossible_E1_ep8_whole_stage_gain_percent": systems["ep8"]["source_side"]["stage_gain_percent"],
            "quality_run": False,
            "dllm_specific_control": "PASS",
            "verdict": "NO-GO",
        },
        "track_b": {
            "ep8_token_home_affinity": ep8["token_home_affinity"]["mean"],
            "ep8_early1_home_match_to_future": ep8["early_home_match_to_future"]["ref1"],
            "ep4_early1_current_block_remote_byte_reduction_percent": ep4["policies"]["O2_early_ref1"]["current_block_remote_byte_reduction_percent"],
            "ep8_early1_current_block_remote_byte_reduction_percent": ep8["policies"]["O2_early_ref1"]["current_block_remote_byte_reduction_percent"],
            "ep4_early1_whole_remote_byte_reduction_percent": ep4["policies"]["O2_early_ref1"]["remote_byte_reduction_percent"],
            "ep8_early1_whole_remote_byte_reduction_percent": ep8["policies"]["O2_early_ref1"]["remote_byte_reduction_percent"],
            "ep4_stage_gain_after_M1_migration_percent": ep4["migration"]["M1_hidden_row"]["repeated_net_stage_gain_percent"],
            "ep8_stage_gain_after_M1_migration_percent": ep8["migration"]["M1_hidden_row"]["repeated_net_stage_gain_percent"],
            "ar_one_shot_break_even_M1": ep8["migration"]["M1_hidden_row"]["ar_one_shot_break_even"],
            "dllm_repeated_break_even_M1": ep8["migration"]["M1_hidden_row"]["dllm_repeated_break_even"],
            "dllm_specific_control": "FAIL",
            "verdict": "NO-GO",
        },
        "stronger_track": "NEITHER",
        "primary_discovery": "Adjacent live-MASK expert edges are genuinely persistent, but branch outputs are usually not numerically stable and full-row EP dilution eliminates systems headroom.",
        "primary_blocker": "Repeated prefix/prior rows dominate physical EP work; token-home affinity and early predictability are too weak to amortize migration.",
        "next_action": "Do not implement either runtime; retain edge persistence as characterization and seek a mechanism that removes dominant full-row work without approximation.",
        "structural_detail": structural,
        "branch_detail": branches,
    }
    args.reports.mkdir(parents=True, exist_ok=True)
    summary_json = args.reports / "dllm_native_ep_discovery_summary.json"
    summary_json.write_text(json.dumps(combined, indent=2, sort_keys=True))

    figures = args.reports / "figures/dllm_native_ep_discovery"
    figures.mkdir(parents=True, exist_ok=True)
    # Required A4: weighted-contribution drift, separated from raw drift.
    trace_dir = Path("artifacts/dllm_native_ep_discovery/branch_trace16/traces")
    r0, r1, ages, raw = [], [], [], []
    for path in sorted(trace_dir.glob("request_*.npz")):
        with np.load(path, allow_pickle=False) as source:
            r0.append(source["edge_weighted_r0_relative_l2"])
            r1.append(source["edge_weighted_r1_relative_l2"])
            ages.append(source["edge_route_age"])
            raw.append(source["edge_raw_output_relative_l2"])
    r0, r1, ages, raw = map(np.concatenate, (r0, r1, ages, raw))
    plt.figure(figsize=(7, 4))
    plt.hist(r0, bins=100, range=(0, 2), density=True, histtype="step", label="R0 prior weighted")
    plt.hist(r1, bins=100, range=(0, 2), density=True, histtype="step", label="R1 current-weight reweight")
    plt.xlabel("weighted contribution relative L2"); plt.ylabel("density"); plt.legend(); plt.tight_layout()
    plt.savefig(figures / "12_weighted_contribution_drift.png", dpi=180); plt.close()

    # Required A7: held-out precision/recall.
    gate = stability["practical_gate"]["heldout"]
    plt.figure(figsize=(5, 4))
    labels = ["E3", "E4"]
    x = np.arange(2)
    plt.bar(x - .18, [gate[k]["precision_for_safe"] for k in labels], .36, label="precision")
    plt.bar(x + .18, [gate[k]["recall_of_safe"] for k in labels], .36, label="recall")
    plt.xticks(x, labels); plt.ylim(0, 1); plt.ylabel("held-out fraction"); plt.legend(); plt.tight_layout()
    plt.savefig(figures / "13_practical_gate_precision_recall.png", dpi=180); plt.close()

    # Required B10/B11 distributions from compact structural summaries.
    plt.figure(figsize=(6, 4))
    plt.bar(["EP4", "EP8"], [ep4["token_home_affinity"]["mean"], ep8["token_home_affinity"]["mean"]])
    plt.ylabel("mean most-affine-rank share"); plt.tight_layout()
    plt.savefig(figures / "14_token_destination_affinity.png", dpi=180); plt.close()
    plt.figure(figsize=(6, 4))
    plt.bar(["EP4 ref1", "EP4 ref1+2", "EP8 ref1", "EP8 ref1+2"], [
        ep4["early_home_match_to_future"]["ref1"], ep4["early_home_match_to_future"]["ref1_2"],
        ep8["early_home_match_to_future"]["ref1"], ep8["early_home_match_to_future"]["ref1_2"],
    ]); plt.ylabel("home match to future"); plt.xticks(rotation=20); plt.tight_layout()
    plt.savefig(figures / "15_early_home_match.png", dpi=180); plt.close()

    # Required B13 and B14/B15.
    plt.figure(figsize=(7, 4))
    names = ["O0", "O1", "O2", "O3", "O4"]
    for ep, cell in ((4, ep4), (8, ep8)):
        values = [cell["baseline"]["source_remote_peer_fanout_mean"]] + [
            cell["policies"][name]["source_remote_peer_fanout_mean"]
            for name in ("O1_full_future", "O2_early_ref1", "O3_early_ref1_2", "O4_static_global")
        ]
        plt.plot(names, values, marker="o", label=f"EP{ep}")
    plt.ylabel("mean remote peer fanout/source"); plt.legend(); plt.tight_layout()
    plt.savefig(figures / "16_ownership_fanout.png", dpi=180); plt.close()

    summary_md = f"""# dLLM-native Expert Parallelism discovery summary

## Outcome

Both tracks are **NO-GO** for implementation. Track A found a strong and held-out reproducible dLLM-native structural fact—live-MASK adjacent edges persist—but the systems/semantic chain breaks twice. Only {100*safe:.3f}% of measured STAY branches have <=5% raw output relative-L2, and the impossible all-STAY source-cache oracle saves only {systems['ep8']['source_side']['stage_gain_percent']:.4f}% of the whole simulated EP8 routed stage because current live-MASK work is diluted by prompt/prior/current-decoded full rows. The practical E3 EP8 stage gain is {ep8_gain:.7f}% (rounded to {ep8_gain:.4f}% below).

Track B is exact-semantics but weak. EP8 mean most-affine-rank share is {100*ep8['token_home_affinity']['mean']:.3f}%, early-ref1 predicts the future home only {100*ep8['early_home_match_to_future']['ref1']:.3f}% of the time, and O2 removes {ep8['policies']['O2_early_ref1']['current_block_remote_byte_reduction_percent']:.3f}% of current-block but only {ep8['policies']['O2_early_ref1']['remote_byte_reduction_percent']:.4f}% of whole remote bytes. After even the minimum 4,096-byte M1 migration, the repeated dLLM EP8 net stage change is {ep8['migration']['M1_hidden_row']['repeated_net_stage_gain_percent']:.4f}%, so neither dLLM nor AR one-shot breaks even.

## Evidence boundaries

- Route persistence: measured threshold-.95 GSM8K-128 model trace; discovery 0--63, held-out 64--127.
- Branch stability: measured duplicate-compute diagnostic on 16 fixed requests and layers (1, 5, 10, 14, 19); model outputs were unmodified.
- EP4/EP8 timing: **SIMULATED-EP4/EP8-EP2-CALIBRATED**, not physical EP4/EP8.
- Quality: no reuse policy rollout was run because the systems upper bound failed first.
- No production cache, migration runtime, kernel, threshold policy, freeze policy or pruning method was implemented.

## Required final summary

TRACK_A_LIVE_MASK_STAY_ROUTE_FRACTION:
{100*persistence['stay_route_fraction']:.3f}% (held-out)

TRACK_A_STAY_ROUTER_MASS_FRACTION:
{100*persistence['stay_router_mass_fraction']:.3f}% (held-out)

TRACK_A_STAY_EP_COST_FRACTION:
{systems['ep8']['live_mask_only_source_side']['stage_gain_percent']:.3f}% calibrated live-MASK stage removable; {systems['work']['live_mask_fresh_pair_reduction_percent']:.3f}% pair-count persistence

TRACK_A_STABILITY_ORACLE_REUSABLE_FRACTION:
{100*safe:.3f}% at raw branch relative-L2 <=5%

TRACK_A_PRACTICAL_REUSABLE_FRACTION:
{100*practical:.3f}% of held-out STAY branches selected by E3

TRACK_A_EP4_STAGE_GAIN:
{ep4_gain:.4f}% (practical E3, simulated)

TRACK_A_EP8_STAGE_GAIN:
{ep8_gain:.4f}% (practical E3, simulated)

TRACK_A_QUALITY_RUN:
NO

TRACK_A_QUALITY_DELTA:
N/A

TRACK_A_DLLM_SPECIFIC_CONTROL:
PASS

TRACK_A_VERDICT:
NO-GO

TRACK_B_TOKEN_HOME_AFFINITY:
{100*ep8['token_home_affinity']['mean']:.3f}% mean most-affine EP8-rank share

TRACK_B_EARLY1_HOME_MATCH_TO_FUTURE:
{100*ep8['early_home_match_to_future']['ref1']:.3f}%

TRACK_B_EP4_REMOTE_BYTE_REDUCTION:
{ep4['policies']['O2_early_ref1']['current_block_remote_byte_reduction_percent']:.3f}% current-block / {ep4['policies']['O2_early_ref1']['remote_byte_reduction_percent']:.4f}% whole

TRACK_B_EP8_REMOTE_BYTE_REDUCTION:
{ep8['policies']['O2_early_ref1']['current_block_remote_byte_reduction_percent']:.3f}% current-block / {ep8['policies']['O2_early_ref1']['remote_byte_reduction_percent']:.4f}% whole

TRACK_B_EP4_STAGE_GAIN_AFTER_MIGRATION:
{ep4['migration']['M1_hidden_row']['repeated_net_stage_gain_percent']:.4f}% (M1)

TRACK_B_EP8_STAGE_GAIN_AFTER_MIGRATION:
{ep8['migration']['M1_hidden_row']['repeated_net_stage_gain_percent']:.4f}% (M1)

TRACK_B_AR_ONE_SHOT_BREAK_EVEN:
NO

TRACK_B_DLLM_REPEATED_BREAK_EVEN:
NO

TRACK_B_DLLM_SPECIFIC_CONTROL:
FAIL

TRACK_B_VERDICT:
NO-GO

STRONGER_TRACK:
NEITHER

PRIMARY_DISCOVERY:
Adjacent live-MASK expert edges are genuinely persistent, but their branch outputs are usually not stable enough to reuse and they are a tiny share of full physical EP work.

PRIMARY_BLOCKER:
Full-row prompt/prior dilution kills Track A, while weak EP8 destination affinity and migration cost kill Track B.

NEXT_ACTION:
Do not implement either runtime; seek a dLLM-native mechanism that removes dominant full-row EP work without approximate branch reuse.

DO_NOT_IMPLEMENT_PRODUCTION_RUNTIME_AUTOMATICALLY:
true
"""
    (args.reports / "dllm_native_ep_discovery_summary.md").write_text(summary_md)
    print(summary_md.split("## Required final summary\n\n", 1)[1])


if __name__ == "__main__":
    main()
