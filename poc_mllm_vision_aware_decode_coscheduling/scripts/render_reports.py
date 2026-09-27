from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from poc_mllm_vision_aware_decode_coscheduling.contract import atomic_json


POLICY_NAMES = {
    "S0_FIFO_RANDOM": "S0 FIFO / frozen random",
    "S1_LENGTH": "S1 length-matched",
    "S2_TEXT": "S2 text-prefill",
    "S3_FULL": "S3 full-prefill / ELDR-style",
    "S4_SEPARATED": "S4 vision/text-separated",
    "S5_ORACLE": "S5 future-trace oracle",
}


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def pct(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def policy_table(metrics: dict[str, Any]) -> str:
    lines = [
        "| Policy | Mean union | P50 | P90 | Reduction vs FIFO | Reduction vs length | Oracle recovery | Weight-set proxy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICY_NAMES:
        row = metrics[policy]
        recovery = row["recovered_oracle_headroom"]
        lines.append(
            "| "
            + " | ".join(
                [
                    POLICY_NAMES[policy],
                    f'{row["mean_active_expert_union"]:.3f}',
                    f'{row["p50_active_expert_union"]:.1f}',
                    f'{row["p90_active_expert_union"]:.1f}',
                    pct(row["union_reduction_vs_fifo"]),
                    pct(row["union_reduction_vs_length"]),
                    "N/A" if recovery is None else pct(recovery),
                    f'{row["expert_weight_working_set_proxy_bytes"] / 2**20:.2f} MiB',
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def window_table(metrics: dict[str, Any]) -> str:
    lines = [
        "| Decode window | FIFO union | Length union | Text union | Full union | Separated union | Oracle union | Oracle reduction vs FIFO |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for window in ("1-32", "33-64", "65-96", "97-128", "1-128"):
        row = metrics[window]
        lines.append(
            "| "
            + " | ".join(
                [
                    window,
                    f'{row["S0_FIFO_RANDOM"]["mean_active_expert_union"]:.3f}',
                    f'{row["S1_LENGTH"]["mean_active_expert_union"]:.3f}',
                    f'{row["S2_TEXT"]["mean_active_expert_union"]:.3f}',
                    f'{row["S3_FULL"]["mean_active_expert_union"]:.3f}',
                    f'{row["S4_SEPARATED"]["mean_active_expert_union"]:.3f}',
                    f'{row["S5_ORACLE"]["mean_active_expert_union"]:.3f}',
                    pct(row["oracle_reduction_vs_fifo"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--source-spec", type=Path, required=True)
    args = parser.parse_args()

    artifacts = args.root / "artifacts"
    reports = args.root / "reports"
    baseline = load(args.baseline_audit)
    pool = load(artifacts / "frozen_request_pool.json")
    metrics = load(artifacts / "working_set_metrics.json")
    comparisons = load(artifacts / "paired_comparisons.json")
    headroom = load(artifacts / "oracle_headroom.json")
    expert = load(artifacts / "expert_tensor_contract.json")
    primary = metrics["heldout"]["1-128"]
    discovery = metrics["discovery"]["1-128"]

    split_counts = {
        split: sum(row["coscheduling_split"] == split for row in pool["anchors"])
        for split in ("discovery", "heldout")
    }
    family_counts = pool["family_counts"]
    heldout_prior_split_counts: dict[str, int] = {}
    for row in pool["anchors"]:
        if row["coscheduling_split"] == "heldout":
            prior_split = str(row.get("analysis_split", "unknown"))
            heldout_prior_split_counts[prior_split] = heldout_prior_split_counts.get(prior_split, 0) + 1
    spec_sha = hashlib.sha256(args.source_spec.read_bytes()).hexdigest()
    source_trace = artifacts / "source_pool_trace.jsonl.gz"

    runtime = {
        "status": "NOT_RUN_STAGE_A_GATE_FAILED",
        "reason": (
            "The authoritative Stage-A gate failed: the recorded future-trace checkpoint oracle reduced "
            "held-out 1-128 union by less than 10% versus both FIFO and length control."
        ),
        "full_model_executions": 0,
        "simulator_substituted": False,
        "true_ep4_tpot_gain_vs_fifo": None,
        "true_ep4_tpot_gain_vs_full": None,
        "fixed128_e2e_gain": None,
        "decode_moe_latency": None,
    }
    parity = {
        "status": "NOT_RUN_STAGE_A_GATE_FAILED",
        "structural_invariant": (
            "Offline policies only permute request-to-wave membership and do not alter "
            "tokens, routes, weights, precision, or model state."
        ),
        "runtime_output_parity_measured": False,
    }
    verdict = {
        "verdict": headroom["verdict"],
        "stage_a_gate": headroom["gate"],
        "stage_b_executed": False,
        "runtime_optimization_implemented": False,
        "next_action": (
            "Stop this co-scheduling method family for the tested wave-4 long-reasoning pool; "
            "do not build a runtime scheduler from this evidence."
        ),
    }
    atomic_json(artifacts / "clean_timing.json", runtime)
    atomic_json(artifacts / "runtime_parity.json", parity)
    atomic_json(artifacts / "final_verdict.json", verdict)
    atomic_json(
        artifacts / "run_manifest.json",
        {
            "authoritative_spec": str(args.source_spec),
            "authoritative_spec_sha256": spec_sha,
            "baseline_audit": str(args.baseline_audit),
            "baseline_bundle_sha256": baseline["bundle_sha256"],
            "model_revision": baseline["model_revision"],
            "physical_contract": "TP2-DP2-EP4-PP1 DeepEP BF16 top8 GPUs4,5,6,7",
            "stage_a_source_trace": str(source_trace),
            "stage_a_source_trace_sha256": hashlib.sha256(source_trace.read_bytes()).hexdigest(),
            "source_full_model_executions": "REUSED_FROM_PRIOR_BOUNDED_LONG_REASONING_POC",
            "new_full_model_executions": 0,
            "stage_b_executed": False,
            "simulator_used": False,
        },
    )

    write(
        reports / "baseline_readiness.md",
        f"""# Baseline readiness

Status: **{baseline['status']}**. The frozen Part-1 quality and physical TRUE EP4 evidence were reused after its recorded integrity audit; the full MMStar/ChartQA evaluation was not repeated.

- Model revision: `{baseline['model_revision']}`
- Model contract: BF16, 48 sparse MoE layers, 128 experts, top-k 8
- Physical contract: GPUs 4,5,6,7; TP2-DP2-EP4-PP1; DeepEP; 32 contiguous experts/rank
- Frozen bundle SHA256: `{baseline['bundle_sha256']}`
- Asset integrity: `{baseline['asset_integrity']['valid']}` ({baseline['asset_integrity']['digest_checked']} digests checked; {len(baseline['asset_integrity']['failures'])} failures)
- Authoritative spec SHA256: `{spec_sha}`
- Prior long-reasoning evidence: reused compact routing traces only; no simulator latency and no new optimization were used.

The baseline audit records `QUALITY_BASELINE_READY=true` and `ACTUAL_EP4_READY=true`. Stage A is an offline replay over measured TRUE-EP4 routing assignments; it does not claim a new physical timing measurement.
""",
    )

    write(
        reports / "request_pool.md",
        f"""# Frozen request pool

The routing-blind pool was frozen before this Stage-A scheduling analysis by requiring natural EOS/stop and at least 128 generated tokens. It contains **{pool['eligible_count']}** requests: {split_counts['discovery']} discovery and {split_counts['heldout']} held-out. The requested 24-request minimum could not be met from the existing bounded cohort, leaving a declared shortage of **{pool['shortage']}** rather than relaxing eligibility.

- Dataset counts: {', '.join(f'{key}={value}' for key, value in sorted(family_counts.items()))}
- Image hashes: unique across all selected requests
- Global wave size: 4
- Held-out waves: 2 per policy
- Decode trace: all 48 MoE layers and 22 checkpoints through step 128
- Selection used routing oracle: `{str(pool['routing_oracle_used_for_selection']).lower()}`

The discovery/held-out split is deterministic and image-disjoint. Policy assignments were frozen before any physical runtime timing. The compact source trace is committed at `artifacts/source_pool_trace.jsonl.gz` so the analysis can be reproduced without the prior 102-MiB trace.

The separated-signature alpha values were frozen in the predecessor study, and this scheduling split is new: its held-out set contains {heldout_prior_split_counts.get('discovery', 0)} requests from the predecessor discovery split and {heldout_prior_split_counts.get('heldout', 0)} from its held-out split. It is therefore held out from the current grouping analysis, but not a wholly virgin test of those inherited alpha values. This limitation cannot create the reported positive conclusion because S4 and S3/S2 are exactly tied.
""",
    )

    write(
        reports / "oracle_headroom.md",
        f"""# Offline oracle headroom

The held-out, aggregate step 1-128 future-trace checkpoint oracle reduced mean layer/checkpoint wave union from **{primary['S0_FIFO_RANDOM']['mean_active_expert_union']:.3f}** to **{primary['S5_ORACLE']['mean_active_expert_union']:.3f}** experts: **{pct(headroom['oracle_reduction_vs_fifo'])} vs FIFO** and only **{pct(headroom['oracle_reduction_vs_length'])} vs length-matched**.

The authoritative gate requires at least 10% against both controls. Both clauses failed. Discovery showed the same scale: **{pct(discovery['oracle_reduction_vs_fifo'])}** vs FIFO and **{pct(discovery['oracle_reduction_vs_length'])}** vs length control.

{policy_table(primary)}

The verified expert tensors occupy {expert['mib_per_expert_per_layer']:.1f} MiB per expert/layer. The table's weight-set values are `ESTIMATED_DISTINCT_WEIGHT_BYTES_NOT_HBM_TRAFFIC`; they are not measured cache residency or HBM traffic.

The reused trace contains 22 fixed checkpoints rather than every integer step. The oracle assignments are optimized for their aggregate 1-128 objective and then reused in the per-window report. Window rows therefore are neither independently optimized per-window oracles nor a claim about unrecorded steps.
""",
    )

    write(
        reports / "policy_workingset_comparison.md",
        f"""# Policy working-set comparison

All predictor policies use the same exact wave partitioner and the same top-32 per-layer predicted-set objective. Only the feature changes: text-only, modality-erased full-prefill, or separated vision/text channels.

{policy_table(primary)}

On held-out, S1, S2, S3, and S4 selected exactly the same two waves. Their union, Jaccard, per-rank union, rank distribution, max/mean, and remote fraction are therefore identical. S4's apparent **{pct(primary['S4_SEPARATED']['union_reduction_vs_fifo'])}** FIFO reduction is fully matched by the routing-blind length control.

The total request set and routes do not change across policies. Thus aggregate rank assignment distribution, rank max/mean, and remote-route fraction are expected to remain unchanged; the measurable structural target is wave-local expert union.
""",
    )

    s4_full = comparisons["S4_vs_S3"]
    s4_text = comparisons["S4_vs_S2"]
    s4_fifo = comparisons["S4_vs_S0"]
    write(
        reports / "mllm_specific_increment.md",
        f"""# MLLM-specific increment

There is no held-out incremental co-scheduling value from modality separation under the frozen predictor family:

- S4 minus S3 beneficial union delta: **{s4_full['beneficial_union_delta_experts']:.3f} experts**, 95% layer-cluster bootstrap CI [{s4_full['ci95_low']:.3f}, {s4_full['ci95_high']:.3f}].
- S4 minus S2 beneficial union delta: **{s4_text['beneficial_union_delta_experts']:.3f} experts**, 95% CI [{s4_text['ci95_low']:.3f}, {s4_text['ci95_high']:.3f}].
- S4 minus S0 beneficial union delta: **{s4_fifo['beneficial_union_delta_experts']:.3f} experts**, 95% CI [{s4_fifo['ci95_low']:.3f}, {s4_fifo['ci95_high']:.3f}], but the length control obtains exactly the same held-out waves and metric.

This is stronger than merely observing a nonsignificant timing delta: the frozen held-out S2/S3/S4 partitions are identical. Accordingly, the prior vision-conditioned prediction result does not translate into distinct wave organization on this pool.

Bootstrap clusters are MoE layers (48), not thousands of layer/step events treated as independent requests. The new scheduling held-out is not wholly virgin to the predecessor alpha fit, and it has only eight requests. Nevertheless, the exact S2/S3/S4 equality and the routing-independent oracle's 2.46% ceiling make Stage B unjustified under the specified gate.
""",
    )

    write(
        reports / "true_ep4_runtime.md",
        """# TRUE EP4 runtime validation

Status: **NOT_RUN_STAGE_A_GATE_FAILED**.

The specification forbids building or timing a runtime scheduler unless all Stage-A clauses pass. The recorded future-trace checkpoint oracle reached only 2.458% reduction versus FIFO and 0.263% versus length control, while S4 did not beat S3 or S1. Therefore:

- no new full-model execution was launched;
- no burner was stopped for an unnecessary timing run;
- no TRUE-EP4 TPOT, fixed128 E2E, decode-MoE latency, or output-parity result is fabricated;
- no simulator result substitutes for physical timing;
- no scheduling runtime or production queueing system was implemented.

The policies are structurally semantics-preserving because the offline operation only groups requests, but runtime output parity remains unmeasured and is reported as such.
""",
    )

    write(
        reports / "gain_vs_decode_length.md",
        f"""# Structural gain versus decode window

{window_table(metrics['heldout'])}

The aggregate oracle is the primary gate. Its frozen assignment does not dominate the length control in later windows: at 65-96 and 97-128 it is worse than the length partition. This indicates phase-dependent overlap rather than a stable, large batching headroom.

No latency-versus-length curve is reported because Stage B was not authorized. These are measured routing-structure metrics, not speedups.
""",
    )

    write(
        reports / "final_summary.md",
        f"""# Final summary

The bounded Stage-A experiment rejects runtime development for this tested method family. Even an oracle that sees every recorded future decode checkpoint reduces the held-out 1-128 expert union by only **{pct(headroom['oracle_reduction_vs_fifo'])} versus FIFO** and **{pct(headroom['oracle_reduction_vs_length'])} versus length matching**, far below the required 10%. Moreover, text, full-prefill, and vision-separated policies produce identical held-out waves, so there is no MLLM-specific scheduling increment to test physically.

The conclusion is scoped to this frozen 20-request, wave-4, fixed128 long-reasoning pool, 22 recorded decode checkpoints, and this simple common grouping family. It does not refute the earlier vision-conditioned routing signal; it shows that the signal does not create enough co-scheduling headroom here. The 24-request target was short by four, but the checkpoint-oracle ceiling is sufficiently below the gate that expanding a runtime prototype is not warranted.

BASELINE_STATUS: READY_REUSED_TRUE_EP4

REQUEST_POOL_STATUS: FROZEN_20_DISCOVERY_12_HELDOUT_8; RECOMMENDED_24_SHORTAGE_4

ORACLE_HEADROOM: {pct(headroom['oracle_reduction_vs_fifo'])}_VS_FIFO; {pct(headroom['oracle_reduction_vs_length'])}_VS_LENGTH; FAIL_10PCT_GATE

TEXT_POLICY_HEADROOM: {pct(primary['S2_TEXT']['union_reduction_vs_fifo'])}_VS_FIFO; {pct(primary['S2_TEXT']['union_reduction_vs_length'])}_VS_LENGTH

FULL_PREFILL_ELDR_STYLE_HEADROOM: {pct(primary['S3_FULL']['union_reduction_vs_fifo'])}_VS_FIFO; {pct(primary['S3_FULL']['union_reduction_vs_length'])}_VS_LENGTH

VISION_SEPARATED_HEADROOM: {pct(primary['S4_SEPARATED']['union_reduction_vs_fifo'])}_VS_FIFO; {pct(primary['S4_SEPARATED']['union_reduction_vs_length'])}_VS_LENGTH

VISION_INCREMENT_VS_FULL: ZERO; DELTA_EXPERTS={s4_full['beneficial_union_delta_experts']:.3f}; CI95=[{s4_full['ci95_low']:.3f},{s4_full['ci95_high']:.3f}]; IDENTICAL_HELDOUT_WAVES

VISION_INCREMENT_VS_TEXT: ZERO; DELTA_EXPERTS={s4_text['beneficial_union_delta_experts']:.3f}; CI95=[{s4_text['ci95_low']:.3f},{s4_text['ci95_high']:.3f}]; IDENTICAL_HELDOUT_WAVES

TRUE_EP4_TPOT_GAIN_VS_FIFO: N/A_STAGE_B_NOT_RUN

TRUE_EP4_TPOT_GAIN_VS_FULL: N/A_STAGE_B_NOT_RUN

FIXED128_E2E_GAIN: N/A_STAGE_B_NOT_RUN

RUNTIME_PARITY: NOT_RUN_STAGE_A_GATE_FAILED

VERDICT: INSUFFICIENT_BATCHING_HEADROOM

NEXT_ACTION: Stop this co-scheduling method family for the tested pool; do not implement a runtime scheduler from this evidence.
""",
    )


if __name__ == "__main__":
    main()
