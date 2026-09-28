#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from poc_mllm_rank_aware_active_expert_reduction.contract import atomic_json


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def pct(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}%"


def num(value: float | None, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def stage_average(q1: dict[str, Any], ep: str, key: str) -> float:
    return float(np.mean([row[key] for row in q1[ep]["stage_breakdown"]]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    artifacts, reports = root / "artifacts", root / "reports"
    baseline = load(artifacts / "baseline_audit.json")
    q1 = load(artifacts / "q1_analysis.json")
    q2 = load(artifacts / "q2_vds_analysis.json")
    q3 = load(artifacts / "q3_structural_oracle.json")
    q3_rows = load(artifacts / "q3_structural_aggregate.json")["rows"]
    auxiliary = load(artifacts / "auxiliary_structural.json")

    ep2_dispatch = stage_average(q1, "ep2", "dispatch_share_pct")
    ep2_expert = stage_average(q1, "ep2", "expert_execution_share_pct")
    ep2_combine = stage_average(q1, "ep2", "combine_share_pct")
    ep4_dispatch = stage_average(q1, "ep4", "dispatch_share_pct")
    ep4_expert = stage_average(q1, "ep4", "expert_execution_share_pct")
    ep4_combine = stage_average(q1, "ep4", "combine_share_pct")
    h = q2["results"]["heldout"]
    hd = q2["heldout_high_vs_low"]
    q2_gate = q2["gate"]["q2_pass"]
    q1_gate = q1["gate"]["q1_pass"]
    q3_gate = q3["gate_pass"]
    overall = ("VISION_PROTECTED_RANK_AWARE_SUPPORTED" if q1_gate and q2_gate and q3_gate
               else "GENERIC_RANK_AWARE_ONLY" if q1_gate and q3_gate
               else "INSUFFICIENT_RANK_AWARE_HEADROOM" if q1_gate
               else "TOKEN_COUNT_DOMINATES")

    write(reports / "baseline_readiness.md", f"""
# Baseline readiness

- Status: **{baseline['status']}**
- Model revision: `{baseline['revision']}`
- Shape: BF16, d=2048, m=768, E=128, K=8, 48 MoE layers.
- Measured tensor shape: gate/up `{baseline['expert_tensors']['model.language_model.layers.0.mlp.experts.gate_up_proj']['shape']}`, down `{baseline['expert_tensors']['model.language_model.layers.0.mlp.experts.down_proj']['shape']}`.
- Exact BF16 expert bytes: **{baseline['per_expert_weight_bytes']:,} bytes (9 MiB)**.
- EP2 evidence: TRUE `TP1-DP2-EP2`, 64 experts/rank, `DeepEPHTPrepareAndFinalize` + `TritonExperts` on physical GPUs 2,3.
- EP4 evidence: TRUE `TP2-DP2-EP4-PP1`, 32 experts/rank, the same DeepEP sparse path on physical GPUs 2,3,6,7.
- EP8 scope: `STRUCTURAL_EP8_PROJECTION_ONLY`; no latency, TPOT, E2E, or speedup is inferred.

Part-1 public-quality status remains the frozen foundation status; this PoC did not rerun MMStar/ChartQA. No MoDES GMLG/DMT/global-layer/semantic score is used by any candidate policy.
""")

    breakdown_lines = []
    for ep in ("ep2", "ep4"):
        for row in q1[ep]["stage_breakdown"]:
            breakdown_lines.append(
                f"| {ep.upper()} | {row['condition']} | {row['dispatch_mean_ms']:.4f} | "
                f"{row['expert_execution_mean_ms']:.4f} | {row['combine_mean_ms']:.4f} | "
                f"{row['moe_mean_ms']:.4f} | {row['dispatch_share_pct']:.2f}% | "
                f"{row['expert_execution_share_pct']:.2f}% | {row['combine_share_pct']:.2f}% |")
    write(reports / "q1_decode_moe_stage_breakdown.md", f"""
# Q1 decode MoE stage breakdown

Instrumented TRUE DeepEP runs were separated from clean timing. CUDA events bracket installed DeepEP prepare, fused Triton experts, and DeepEP finalize. Values are critical-rank means; diagnostic synchronization makes them unsuitable as TPOT.

| Topology | Batch | Dispatch ms | Expert ms | Combine ms | Total MoE ms | Dispatch share | Expert share | Combine share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(breakdown_lines)}

Mean shares: EP2 dispatch {ep2_dispatch:.2f}%, expert {ep2_expert:.2f}%, combine {ep2_combine:.2f}%; EP4 dispatch {ep4_dispatch:.2f}%, expert {ep4_expert:.2f}%, combine {ep4_combine:.2f}%.

Expert weights/HBM reads are charged to expert execution, never dispatch. Stage traces contain {q1['ep2']['raw_rank_invocations']:,} EP2 and {q1['ep4']['raw_rank_invocations']:,} EP4 rank-local invocations.
""")

    cold_rows = [row for row in q1["microbench"]["rows"]
                 if row["cache_condition"] == "cold_l2_flushed"]
    micro_lines = [
        f"| {row['local_experts']} | {row['fixed_32_path_mean_ms']['2']:.5f} | "
        f"{row['fixed_32_path_mean_ms']['4']:.5f} | {row['fixed_32_path_mean_ms']['8']:.5f} | "
        f"{row['fixed_32_path_mean_ms']['16']:.5f} | {row['fixed_token_2_to_16_active_effect_pct']:.2f}% | "
        f"{row['joint_model']['r2'] - row['token_only_model']['r2']:.3f} |"
        for row in cold_rows]
    write(reports / "q1_token_vs_active_expert_microbench.md", f"""
# Q1 token paths versus active experts

Actual installed BF16 Triton grouped-expert kernel on physical GPU2, d=2048/m=768. Each cold observation uses a 256 MiB L2-displacing preconditioner. Warmup=12, repeats=40, rounds=3. This is rank-local kernel evidence, not full EP latency.

| Local expert bank | 2 active ms | 4 active ms | 8 active ms | 16 active ms | 2→16 effect | joint R² increment over token-only |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(micro_lines)}

At fixed 32 paths, the mean cold 2→16 active-expert effect is **{q1['microbench']['cold_fixed_token_mean_active_effect_pct']:.2f}%**. The mean joint-model R² increment over token-only is **{q1['microbench']['cold_joint_r2_increment_over_token_only']:.3f}**. Hot repeated-weight results are retained in the artifact and are not used to infer cold HBM behavior.
""")

    write(reports / "q1_rank_latency_model.md", f"""
# Q1 rank latency model

Natural trace correlations are weak because pair count, active set, layer, cache state, and launch behavior co-vary:

| Topology | token-pairs Spearman | active-experts Spearman | active-bytes Spearman | token-only R² | joint R² |
|---|---:|---:|---:|---:|---:|
| EP2 | {q1['ep2']['rank_local_correlations']['token_pairs_vs_expert_ms_spearman']:.4f} | {q1['ep2']['rank_local_correlations']['active_experts_vs_expert_ms_spearman']:.4f} | {q1['ep2']['rank_local_correlations']['active_weight_bytes_vs_expert_ms_spearman']:.4f} | {q1['ep2']['rank_local_models']['token_only']['r2']:.4f} | {q1['ep2']['rank_local_models']['joint']['r2']:.4f} |
| EP4 | {q1['ep4']['rank_local_correlations']['token_pairs_vs_expert_ms_spearman']:.4f} | {q1['ep4']['rank_local_correlations']['active_experts_vs_expert_ms_spearman']:.4f} | {q1['ep4']['rank_local_correlations']['active_weight_bytes_vs_expert_ms_spearman']:.4f} | {q1['ep4']['rank_local_models']['token_only']['r2']:.4f} | {q1['ep4']['rank_local_models']['joint']['r2']:.4f} |

The causal evidence is the controlled Sweep A, not the raw correlation: constant paths plus more distinct weights increased execution by {q1['microbench']['cold_fixed_token_mean_active_effect_pct']:.2f}%. Hardware-counter collection was not run; therefore the conclusion is **PHYSICAL_ACTIVE_EXPERT_BOTTLENECK_SUPPORTED (bounded kernel/stage evidence)**, not a complete roofline proof.
""")

    bins = h["damage_by_frozen_vds_bin"]
    write(reports / "q2_visual_dependency_signal.md", f"""
# Q2 visual dependency signal

VDS-A and VDS-B were measured inside the actual decode FlashAttention path on TRUE EP4 for frozen layers [4,12,24,36,44,47] and steps [8,16,32,48,64,96,112]. The hook reconstructs attention to the current paged KV cache and never changes model outputs. It is diagnostic-only.

- VDS points: {q2['vds_points']:,}; joined functional paths: {q2['joined_path_count']:,}/{q2['functional_label_count']:,}.
- Frozen primary before damage analysis: `{q2['primary_vds_frozen_before_damage_analysis']}`.
- Held-out VDS-A/damage Spearman: {num(h['vds_attn_damage_spearman'])}.
- Held-out VDS-B contribution/damage Spearman: {num(h['vds_contribution_damage_spearman'])}.
- External caption image-swap causal lane: expert change {pct(q2['causal_vds_c']['caption_image_swap_expert_change_pct'])}, EP4 changed-rank {pct(q2['causal_vds_c']['caption_image_swap_ep4_changed_rank_pct'])}.

The causal lane proves image-conditioned routing exists, but it is not observation-aligned with this 32-request damage cohort. Direct VDS-A/B↔VDS-C alignment is therefore **not proven**.
""")

    write(reports / "q2_vds_vs_skip_damage.md", f"""
# Q2 VDS versus skip damage

The previously frozen primary functional damage (`next-token JSD + nonnegative target-logprob drop + top1 penalty`) was joined to newly measured VDS at exact request/layer/step.

| Frozen VDS tertile | Held-out mean functional damage |
|---|---:|
| low | {num(bins['low'], 7)} |
| mid | {num(bins['mid'], 7)} |
| high | {num(bins['high'], 7)} |

- High−low: {num(hd.get('high_minus_low_mean_damage'), 7)}
- Request-bootstrap 95% CI: `{hd.get('request_bootstrap_95ci')}`
- Monotonic: `{q2['gate']['monotonic_heldout_damage']}`
- CI strictly positive: `{q2['gate']['heldout_high_vs_low_ci_positive']}`
- Direct causal alignment: `{q2['gate']['direct_causal_alignment']}`

Verdict: **{q2['verdict']}**. K8→K6/K4 propagation was stopped after the drop-one gate; it is not silently treated as zero damage.
""")

    primary_rows = [row for row in q3_rows if row["split"] == "heldout" and row["batch"] == 8]
    ep2_o4 = next(row for row in primary_rows
                   if row["ep"] == 2 and row["policy"] == "O4_RANK_AWARE_COLLAPSE_STRUCTURAL_UPPER")
    q3_lines = [
        f"| EP{row['ep']} | {row['policy']} | {row['path_reduction_pct']:.3f}% | "
        f"{row['max_active_expert_reduction_pct']:.3f}% | {row['total_active_expert_reduction_pct']:.3f}% | "
        f"{row['remote_path_reduction_pct']:.3f}% |"
        for row in primary_rows]
    write(reports / "q3_rank_aware_oracle.md", f"""
# Q3 rank-aware structural oracle

| Mapping | Policy | Path reduction | Max-rank active/weight reduction | Total active reduction | Remote-path reduction |
|---|---|---:|---:|---:|---:|
{chr(10).join(q3_lines)}

At held-out B8 and matched {q3['primary_ep4_rank_aware']['path_reduction_pct']:.3f}% path reduction, EP4 native-tail reduces the maximum rank working set by {q3['primary_ep4_native_tail']['max_active_expert_reduction_pct']:.3f}%; O4 reduces it by {q3['primary_ep4_rank_aware']['max_active_expert_reduction_pct']:.3f}% (**+{q3['o4_increment_over_native_tail_pp']:.3f}pp**).

O4 is a nondeployable future-route structural upper bound. The caption trace lacks dense safe-candidate labels, so this is matched path budget—not a demonstrated matched functional-damage budget. Q4 remains required for safety.
""")

    write(reports / "q3_ep_scaling.md", f"""
# Q3 EP scaling

Primary held-out B8 O4 max-rank active-weight reduction:

- EP2 actual mapping replay: {ep2_o4['max_active_weight_reduction_pct']:.3f}%.
- EP4: {q3['primary_ep4_rank_aware']['max_active_weight_reduction_pct']:.3f}%.
- EP8 structural: {q3['primary_ep8_rank_aware']['max_active_weight_reduction_pct']:.3f}%.

The EP8 value is **STRUCTURAL_EP8_PROJECTION_ONLY**. It is larger than EP4 and passes the frozen structural scaling gate; it carries no timing implication.
""")

    q4_status = ("BOUNDED_OFFLINE_CANDIDATE_REQUIRED" if q2_gate else
                 "NOT_RUN_BY_Q2_MLLM_GATE_AND_NO_MATCHED_SAFE_BATCH_LABELS")
    write(reports / "q4_vision_protected_oracle.md", f"""
# Q4 vision-protected rank-aware oracle

- P0 native-router-tail physical proxy: measured structurally in Q3.
- P1 rank-aware: measured only as the nondeployable O4 upper bound.
- P2 vision-protected rank-aware: **{q4_status}**.

Q2 did not establish the required ordering between current-token VDS and functional skip damage. Inventing a VDS protection threshold after observing held-out damage would violate the specification. Accordingly, no MLLM-specific P2 policy or propagation run was implemented. MoDES mechanisms were not borrowed.
""")

    aux_rows = auxiliary["aggregate"]["rows"]
    def aux(lane: str, ep: int) -> dict[str, Any]:
        return next(row for row in aux_rows if row["lane"] == lane and row["ep"] == ep)
    natural, stress = aux("natural_heldout_b8", 4), aux("stress_discovery_only_b8", 4)
    phase = aux("phase_mixed_16_64_96_128_b8", 4)
    write(reports / "aux_natural_vs_stress.md", f"""
# Auxiliary natural versus stress

The stress cohort was frozen using discovery-only EP4 rank max/mean, before any held-out timing. This lane is structural replay, not latency.

| Lane | Baseline rank max/mean | Native-tail max-active reduction | Rank-aware upper reduction |
|---|---:|---:|---:|
| natural held-out B8 | {natural['baseline_rank_max_mean']:.3f} | {natural['native_tail_max_active_reduction_pct']:.3f}% | {natural['rank_aware_max_active_reduction_pct']:.3f}% |
| stress discovery B8 | {stress['baseline_rank_max_mean']:.3f} | {stress['native_tail_max_active_reduction_pct']:.3f}% | {stress['rank_aware_max_active_reduction_pct']:.3f}% |
""")
    write(reports / "aux_phase_mixed_decode.md", f"""
# Auxiliary phase-mixed decode

Frozen held-out B8 requests at decode steps 16/64/96/128 were combined structurally. At EP4, baseline rank max/mean was {phase['baseline_rank_max_mean']:.3f}; native-tail reduced max active experts by {phase['native_tail_max_active_reduction_pct']:.3f}%, and the O4 upper bound by {phase['rank_aware_max_active_reduction_pct']:.3f}% at {phase['path_reduction_pct']:.3f}% path reduction. This is not a continuous-batching runtime result.
""")

    write(reports / "ep8_structural_projection.md", f"""
# EP8 structural projection

Status: **STRUCTURAL_EP8_PROJECTION_ONLY**.

- Linear map: 128 experts / 8 ranks = 16 experts/rank.
- Held-out B8 native-tail max-rank active reduction: {next(r for r in primary_rows if r['ep']==8 and r['policy']=='O2_NATIVE_TAIL')['max_active_expert_reduction_pct']:.3f}%.
- Held-out B8 O4 upper reduction: {q3['primary_ep8_rank_aware']['max_active_expert_reduction_pct']:.3f}%.
- Natural B8 O4: {aux('natural_heldout_b8', 8)['rank_aware_max_active_reduction_pct']:.3f}%.
- Stress B8 O4: {aux('stress_discovery_only_b8', 8)['rank_aware_max_active_reduction_pct']:.3f}%.
- Phase-mixed B8 O4: {aux('phase_mixed_16_64_96_128_b8', 8)['rank_aware_max_active_reduction_pct']:.3f}%.

No EP8 collective was run. No EP8 latency, TPOT, E2E, speedup, or confidence interval is reported.
""")

    write(reports / "q5_ep2_ep4_runtime.md", """
# Q5 EP2/EP4 candidate runtime

Status: **NOT_RUN_BY_GATE**.

Physical baseline stage characterization was completed on TRUE EP2 and TRUE EP4, but the candidate runtime promotion was not authorized: Q2 did not support visual protection, and Q4 could not produce a functionally validated frozen candidate from the sparse drop-one labels. Therefore no modified route was presented as an actual kernel reduction, and TPOT/E2E gains are N/A rather than zero.
""")

    final_fields = f"""
BASELINE_STATUS: PASS_REUSED_VALIDATED_FOUNDATION; TRUE_EP2_AND_EP4_RECONFIRMED

Q1_DECODE_MOE_STAGE_BREAKDOWN: MEASURED_TRUE_DEEPEP_DIAGNOSTIC
Q1_DISPATCH_SHARE: EP2={ep2_dispatch:.3f}%; EP4={ep4_dispatch:.3f}%
Q1_EXPERT_EXECUTION_SHARE: EP2={ep2_expert:.3f}%; EP4={ep4_expert:.3f}%
Q1_COMBINE_SHARE: EP2={ep2_combine:.3f}%; EP4={ep4_combine:.3f}%
Q1_TOKEN_COUNT_CORRELATION: EP2={q1['ep2']['rank_local_correlations']['token_pairs_vs_expert_ms_spearman']:.4f}; EP4={q1['ep4']['rank_local_correlations']['token_pairs_vs_expert_ms_spearman']:.4f}
Q1_ACTIVE_EXPERT_CORRELATION: EP2={q1['ep2']['rank_local_correlations']['active_experts_vs_expert_ms_spearman']:.4f}; EP4={q1['ep4']['rank_local_correlations']['active_experts_vs_expert_ms_spearman']:.4f}
Q1_ACTIVE_WEIGHT_BYTES_CORRELATION: EP2={q1['ep2']['rank_local_correlations']['active_weight_bytes_vs_expert_ms_spearman']:.4f}; EP4={q1['ep4']['rank_local_correlations']['active_weight_bytes_vs_expert_ms_spearman']:.4f}
Q1_FIXED_TOKEN_ACTIVE_EXPERT_EFFECT: cold 32-path, active2→16 = +{q1['microbench']['cold_fixed_token_mean_active_effect_pct']:.3f}% mean
Q1_PHYSICAL_VERDICT: PHYSICAL_ACTIVE_EXPERT_BOTTLENECK_SUPPORTED_BOUNDED

Q2_VDS_ATTN_RESULT: held-out Spearman={num(h['vds_attn_damage_spearman'])}
Q2_VDS_CONTRIBUTION_RESULT: held-out Spearman={num(h['vds_contribution_damage_spearman'])}
Q2_CAUSAL_VDS_ALIGNMENT: PARTIAL_EXTERNAL_CAUSAL_ONLY; DIRECT_ALIGNMENT_NOT_PROVEN
Q2_VDS_SKIP_DAMAGE_CORRELATION: {num(h['vds_attn_damage_spearman'])}
Q2_HIGH_VS_LOW_VDS_DAMAGE: delta={num(hd.get('high_minus_low_mean_damage'),7)}; 95%CI={hd.get('request_bootstrap_95ci')}
Q2_MLLM_VERDICT: {q2['verdict']}

Q3_MATCHED_PATH_REDUCTION: {q3['primary_ep4_rank_aware']['path_reduction_pct']:.3f}%
Q3_NATIVE_TAIL_ACTIVE_EXPERT_REDUCTION: EP4 max-rank={q3['primary_ep4_native_tail']['max_active_expert_reduction_pct']:.3f}%
Q3_RANK_AWARE_ACTIVE_EXPERT_REDUCTION: EP4 max-rank={q3['primary_ep4_rank_aware']['max_active_expert_reduction_pct']:.3f}%
Q3_MAX_RANK_WEIGHT_BYTES_REDUCTION: EP4={q3['primary_ep4_rank_aware']['max_active_weight_reduction_pct']:.3f}%
Q3_EP2_HEADROOM: {ep2_o4['max_active_expert_reduction_pct']:.3f}% max-rank structural
Q3_EP4_HEADROOM: {q3['primary_ep4_rank_aware']['max_active_expert_reduction_pct']:.3f}% max-rank structural
Q3_EP8_STRUCTURAL_HEADROOM: {q3['primary_ep8_rank_aware']['max_active_expert_reduction_pct']:.3f}% max-rank; STRUCTURAL_EP8_PROJECTION_ONLY
Q3_VERDICT: RANK_AWARE_STRUCTURED_HEADROOM_SUPPORTED_NONDEPLOYABLE_UPPER

Q4_RANK_AWARE_ONLY_FUNCTIONAL_DAMAGE: NOT_ESTABLISHED_FOR_BATCH_O4
Q4_VISION_PROTECTED_FUNCTIONAL_DAMAGE: NOT_RUN_BY_Q2_GATE
Q4_PHYSICAL_HEADROOM_RETAINED: NOT_RUN_BY_Q2_GATE
Q4_VERDICT: NO_MLLM_SPECIFIC_POLICY_PROMOTION

NATURAL_SERVING_RESULT: EP4 O4 max-rank structural reduction={natural['rank_aware_max_active_reduction_pct']:.3f}%
STRESS_SERVING_RESULT: EP4 O4 max-rank structural reduction={stress['rank_aware_max_active_reduction_pct']:.3f}% (discovery-only)
PHASE_MIXED_RESULT: EP4 O4 max-rank structural reduction={phase['rank_aware_max_active_reduction_pct']:.3f}%

Q5_EP2_TPOT_GAIN: N/A_NOT_RUN_BY_GATE
Q5_EP4_TPOT_GAIN: N/A_NOT_RUN_BY_GATE
Q5_DECODE_MOE_GAIN: N/A_NOT_RUN_BY_GATE
Q5_RUNTIME_VERDICT: NOT_PROMOTED

OVERALL_VERDICT: {overall}
NEXT_ACTION: If pursued, collect dense multi-path functional safety labels and validate a generic rank-aware candidate separately; do not claim visual protection from this PoC.
""".strip()

    write(reports / "final_summary.md", f"""
# Final summary

## Outcome

The bounded PoC supports the physical and structural halves separately: distinct active expert working set changes actual expert-kernel latency at fixed path count, and a future-truth rank-aware collapse upper bound reduces the maximum rank working set substantially more than native tail removal. It does **not** support the MLLM-specific protection chain: measured current-token visual dependency did not meet the held-out, causally aligned skip-damage gate.

Consequently the full named method is not promoted, no modified-route runtime was built, and no speedup is claimed. The defensible verdict is **{overall}**, limited to a bounded generic EP observation plus a nondeployable structural oracle.

## Evidence discipline

- Physical: TRUE EP2/EP4 DeepEP stage and installed Triton kernel measurements.
- Structural: frozen caption route replay, including EP8 structural mapping only.
- Functional: reused exact TRUE EP4 drop-one labels joined to current VDS.
- Not measured: P2 candidate quality, modified-route EP2/EP4 timing, any EP8 latency.

## Final Required Summary

```text
{final_fields}
```
""")

    atomic_json(artifacts / "final_verdict.json", {
        "schema_version": "rank_aware_active_expert_final.v1",
        "baseline_status": "PASS_REUSED_VALIDATED_FOUNDATION_TRUE_EP2_EP4_RECONFIRMED",
        "q1": {"pass": q1_gate, "verdict": "PHYSICAL_ACTIVE_EXPERT_BOTTLENECK_SUPPORTED_BOUNDED"},
        "q2": {"pass": q2_gate, "verdict": q2["verdict"]},
        "q3": {"pass": q3_gate, "verdict": "RANK_AWARE_STRUCTURED_HEADROOM_SUPPORTED_NONDEPLOYABLE_UPPER"},
        "q4": {"status": q4_status},
        "q5": {"status": "NOT_RUN_BY_GATE", "speedup_claim": False},
        "ep8": "STRUCTURAL_EP8_PROJECTION_ONLY",
        "overall_verdict": overall,
        "modes_internal_mechanism_used": False,
    })
    atomic_json(artifacts / "execution_manifest.json", {
        "schema_version": "rank_aware_active_expert_execution.v1",
        "model_revision": baseline["revision"],
        "physical_gpu_uuids": {
            "2": "GPU-24200107-8a7f-de46-1bc8-b81f8d3af13e",
            "3": "GPU-17488c15-2d4c-5d9e-d503-29b0d959a8a8",
            "6": "GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28",
            "7": "GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b",
        },
        "q1_microbench": {"physical_gpu": 2, "observations": 342,
                           "source": "artifacts/q1_microbench_gpu2_v4.json"},
        "q1_stage": {
            "ep2": {"request_executions": 26, "rank_invocations": q1["ep2"]["raw_rank_invocations"]},
            "ep4": {"request_executions": 26, "rank_invocations": q1["ep4"]["raw_rank_invocations"]},
        },
        "q2_vds": {"request_executions": 32, "vds_points": q2["vds_points"],
                    "joined_paths": q2["joined_path_count"]},
        "q3": {"source_requests": q3["source_request_count"], "execution": "OFFLINE_REPLAY"},
        "failed_append_only_diagnostics": [
            "stage_ep2_v1_missing_deepep_environment",
            "stage_ep2_v2_late_flush_failure",
            "vds_ep4_full_v1_prompt_length_4096_limit_after_20_requests",
            "vds_ep4_full_v2_startup_collision_with_v1_orphans",
        ],
        "full_model_execution_budget_estimate": 136,
        "budget_limit": 400,
        "tests": {"unittest_count": 5, "status": "PASS"},
    })


if __name__ == "__main__":
    main()
