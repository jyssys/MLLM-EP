#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from poc_mllm_vision_conditioned_decode_damage_skipping.contract import atomic_json, read_json


def pct(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{100*value:.{digits}f}%"


def pp_ci(value: dict) -> str:
    return f"{value['point_pp']:+.2f} pp [{value['ci95_low_pp']:+.2f}, {value['ci95_high_pp']:+.2f}]"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root, artifacts = args.root, args.root / "artifacts"
    reports = root / "reports"
    local = read_json(artifacts / "local_damage_summary.json")
    functional = read_json(artifacts / "functional_damage_summary.json")
    predictor = read_json(artifacts / "damage_predictor_comparison.json")
    oracle = read_json(artifacts / "drop_one_oracle_headroom.json")
    rare = read_json(artifacts / "rare_but_critical_paths.json")
    ep8 = read_json(artifacts / "ep8_structural_projection.json")
    cohort = read_json(artifacts / "manifests/frozen_damage_cohort.json")
    gate = predictor["stage_b_gate"]
    verdict = predictor["verdict"]

    prior = Path("poc_mllm_mixed_domain_online_calibration")
    write(reports / "baseline_readiness.md", f"""
# Baseline readiness

Status: **PASS_REUSED_VALIDATED_FOUNDATION**

- Model: `Qwen/Qwen3-VL-30B-A3B-Instruct`
- revision: `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`
- dtype / shape: BF16, d=2048, m=768, 48 sparse MoE layers, 128 experts, top-k=8
- TRUE EP4 path: TP2-DP2-EP4-PP1, DeepEP high-throughput, linear 32 experts/rank
- TRUE EP2 path: TP1-DP2-EP2, DeepEP high-throughput, linear 64 experts/rank
- frozen source cohort: `{prior}/artifacts/manifests/frozen_domain_pool.json`

The Part-1 public quality baseline and the mixed-domain EP2/EP4 path proofs were
integrity-checked and reused. Full MMStar/ChartQA evaluation was not repeated.
This PoC collected new branch-output and functional intervention evidence only;
no simulator latency is used.
""")

    write(reports / "path_damage_capture.md", f"""
# Path-damage capture

Status: **MEASURED_TRUE_EP4_BOUNDED**

- Frozen cohort: {cohort['request_count']} requests, {cohort['calibration_count']} calibration / {cohort['heldout_count']} held-out, four balanced domains.
- Exact selected-branch local labels: {local['path_labels']:,} paths across {local['groups']:,} token/layer groups and {local['requests']} requests.
- Functional drop-one labels: {functional['path_count']} paths across {functional['request_count']} requests.
- Intervention contract: one original selected path removed, no substitution, remaining original weights renormalized to sum one.
- Functional execution: TRUE TP2-DP2-EP4, same frozen teacher-forced natural continuation, remainder of the current step executed after intervention.
- Intervention marker audit: {functional['marker_audit']['probe_count']} probes; every materialized target fingerprint matched and exactly one physical rank owned/applied each dropped expert: **{functional['marker_audit']['all_pass']}**. Marker-count distribution across source/receiver ranks: `{functional['marker_audit']['marker_count_distribution']}`.

Full hidden-state dumps were not taken. The raw diagnostic tensors/logits are
kept locally under ignored `artifacts/runs/`; compact labels are versioned.

The first functional run completed cases 00--28 and then correctly rejected
case29 because the initially selected structural step 96 exceeded its 73-token
natural continuation. That failed run is preserved. A versioned plan applies a
damage-blind natural-length fallback to predeclared step 64 for case29, and a
resume run completed cases 29--31. No completed case was rerun or cherry-picked.
""")

    table = [
        "| Oracle skip | actual | safe precision | critical recall | mean damage | top1 flip among skipped |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in oracle["frontier"].items():
        table.append(
            f"| {name} | {pct(row['actual_skip_rate'])} | {pct(row['safe_skip_precision'])} "
            f"| {pct(row['protected_critical_recall'])} | {row['mean_damage_skipped']:.6g} "
            f"| {pct(row['top1_flip_in_skipped'])} |"
        )
    write(reports / "drop_one_oracle_headroom.md", f"""
# Drop-one oracle headroom

Status: **MEASURED_HELDOUT_FUNCTIONAL_ORACLE**

Absolute-safe held-out path fraction: **{pct(oracle['absolute_safe_headroom'])}**.
The frozen absolute-safe rule is target-token log-prob drop <=0.01 nat, JSD
<=1e-4, and no top-1 flip. Oracle selection sees true damage and is not deployable.

{chr(10).join(table)}

This establishes whether low-damage paths exist; it does not establish that an
online estimator can identify them or that multi-path skipping preserves task quality.
""")

    score = functional["functional_damage_score"]
    write(reports / "damage_metric_validation.md", f"""
# Damage metric validation

Primary metric was frozen before functional results:

`max(0, target_logprob_drop_nats) + 2*JSD + 1[top1_flip]`.

It is based on actual next-token logits after the remaining forward, while local
mixture relative-L2/cosine remain broad-coverage secondary labels. Functional
score mean/P50/P80/P90/P99 were {score['mean']:.6g} / {score['p50']:.6g} /
{score['p80']:.6g} / {score['p90']:.6g} / {score['p99']:.6g}. Top-1 flip rate was
{pct(functional['top1_flip_fraction'])}. Local-mixture vs functional-damage
Spearman was {functional['local_mixture_vs_functional_spearman']:.4f}. Local mixture relative-L2 P50/P90/P99
were {local['local_mixture_relative_l2']['p50']:.4f} /
{local['local_mixture_relative_l2']['p90']:.4f} /
{local['local_mixture_relative_l2']['p99']:.4f}.

No metric was selected after predictor results, and no natural-generation task
quality claim is made from these local/one-step labels.
""")

    predictor_table = [
        "| Family | Spearman | AUROC | AUPRC | safe precision @20% | critical recall @20% | false-safe @20% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, row in predictor["results"].items():
        point = row["frontier"]["20pct"]
        predictor_table.append(
            f"| {family} | {row['continuous_damage_spearman']:.4f} | {row['critical_auroc']:.4f} "
            f"| {row['critical_auprc']:.4f} | {pct(point['safe_skip_precision'])} "
            f"| {pct(point['protected_critical_recall'])} | {pct(point['false_safe_rate'])} |"
        )
    comparisons = predictor["comparisons"]
    write(reports / "damage_predictor_comparison.md", f"""
# Held-out damage predictor comparison

Status: **MEASURED_NO_FUTURE_LEAKAGE**

All families use the same fixed-alpha standardized ridge predictor. Fit,
normalization, frequency priors, and critical thresholds use calibration rows
only. Held-out requests are image-disjoint.

{chr(10).join(predictor_table)}

At the primary 20% nominal skip point:

- F5−F1 critical recall: {pp_ci(comparisons['F5_MINUS_F1']['protected_critical_recall'])}
- F5−F1 safe precision: {pp_ci(comparisons['F5_MINUS_F1']['safe_skip_precision'])}
- F5−F2 critical recall: {pp_ci(comparisons['F5_MINUS_F2']['protected_critical_recall'])}
- F5−F2 safe precision: {pp_ci(comparisons['F5_MINUS_F2']['safe_skip_precision'])}
- F5−F3 critical recall: {pp_ci(comparisons['F5_MINUS_F3']['protected_critical_recall'])}
- F5−F3 safe precision: {pp_ci(comparisons['F5_MINUS_F3']['safe_skip_precision'])}

Stage-B gate: **{'PASS' if gate['pass'] else 'FAIL'}**. A point estimate is not
treated as vision-specific evidence when the request-cluster bootstrap CI includes zero.
""")

    write(reports / "rare_but_critical_paths.md", f"""
# Rare-but-critical paths

Frozen definition: {rare['definition']}.

- Held-out count/rate: {rare['path_count']} / **{pct(rare['path_rate'])}**
- Domain counts: `{dict(rare['domain_counts'])}`
- Layer counts: `{dict(rare['layer_counts'])}`
- Decode-step counts: `{dict(rare['decode_step_counts'])}`
- Mean vision-mass affinity: {rare['mean_vision_mass_affinity']}
- Protection at 20% skip: `{rare['protection_at_20pct_skip']}`

This test asks whether F5 protects low-router-weight / low-generic-score but
high-measured-damage paths better than the generic control. Its result is folded
into the Stage-B decision; rarity alone is not evidence of vision causality.
""")

    later_status = "PENDING_STAGE_B_PASS" if gate["pass"] else "NOT_RUN_BY_STAGE_B_GATE"
    write(reports / "adaptive_budget_policy.md", f"""
# Adaptive budget policy

Status: **{later_status}**

The Stage-B held-out gate was {'passed' if gate['pass'] else 'not passed'}.
Accordingly, {'a bounded C0--C6 quality frontier is required next' if gate['pass'] else 'no C3--C6 propagation/runtime policy was implemented'}.
The nominal K8/K6/K4 policies were not confused with actual path reduction.
BEST_FROZEN_SKIP_BUDGET: {'PENDING' if gate['pass'] else 'N/A'}.
""")
    write(reports / "quality_frontier.md", f"""
# Full-model quality frontier

Status: **{later_status}**

Natural-generation C0--C6 quality was not run{' yet' if gate['pass'] else ' because the authoritative Stage-B gate failed'}.
No local vector or next-token result is reported as final-answer accuracy.

GENERIC_MODES_QUALITY_AT_BUDGET: N/A

VISION_POLICY_QUALITY_AT_BUDGET: N/A
""")
    write(reports / "ep2_ep4_runtime.md", f"""
# TRUE EP2 / EP4 runtime

Status: **{later_status}**

The previously validated TRUE DeepEP EP2/EP4 baseline paths were reused for
readiness, but no candidate timing was authorized without the quality gate.
No simulator value is substituted.

EP2_TPOT_GAIN: N/A

EP4_TPOT_GAIN: N/A
""")

    ep8_rows = []
    for family, row in ep8["policies"].items():
        ep8_rows.append(
            f"| {family} | {pct(row['path_reduction'])} | {row['removed_remote_paths']} "
            f"| {row['logical_remote_dispatch_combine_bytes_removed']:,} | "
            f"{row['rank_max_mean_before']:.3f}→{row['rank_max_mean_after']:.3f} |"
        )
    write(reports / "ep8_structural_projection.md", f"""
# EP8 structural projection

Status: **STRUCTURAL_EP8_PROJECTION_ONLY**

The 128 experts are mapped linearly to 8 ranks (16/rank). This is a bounded
held-out functional-path sample, not a latency model and not a promoted runtime policy.

| Policy score used for 20% offline mask | path reduction | remote paths removed | logical bytes removed | rank max/mean |
|---|---:|---:|---:|---:|
{chr(10).join(ep8_rows)}

The byte field is a BF16 hidden+contribution logical payload proxy and excludes
DeepEP padding/metadata. EP8 latency, TPOT, E2E, and speedup are explicitly N/A.
""")

    f1 = predictor["results"]["F1_MODES_STYLE_GENERIC"]
    f2 = predictor["results"]["F2_TEXT_PREFILL"]
    f3 = predictor["results"]["F3_FULL_PREFILL"]
    f5 = predictor["results"]["F5_VISION_SEPARATED"]
    summary = f"""
# Final summary

This bounded PoC measured actual selected-branch outputs for 9,312 paths and
actual next-token functional damage for 256 drop-one interventions on TRUE EP4.
The oracle/predictor distinction is preserved, and no future-decode truth was
used as a feature. The Stage-B decision was `{verdict}`.

There is substantial *oracle* safe-skip headroom ({pct(oracle['absolute_safe_headroom'])}),
but the tested vision-separated estimator does not identify it better than the
text-prefill control with statistical support. At the frozen 20% point, F5 and
F2 make an indistinguishable safety decision, while F5's point gains over F1
have confidence intervals spanning zero. Rare-but-critical paths are also only
{pct(rare['path_rate'])} of this held-out sample and F5 did not rescue the one
observed case. Local mixture error is a poor proxy for functional damage
(Spearman {functional['local_mixture_vs_functional_spearman']:.4f}).

The recent negative frequency-prediction result is not repackaged here: this
experiment targets `vision signal → skip damage`. {'The frozen gate passed, so quality propagation remains required before runtime.' if gate['pass'] else 'The frozen vision-increment gate failed, so C0--C6 quality propagation and physical candidate timing were correctly stopped.'}

## Final Required Summary

BASELINE_STATUS: PASS_REUSED_VALIDATED_TRUE_EP2_EP4

DAMAGE_CAPTURE_STATUS: MEASURED_LOCAL_9312_AND_FUNCTIONAL_256_TRUE_EP4

DROP_ONE_ORACLE_SAFE_SKIP_HEADROOM: {pct(oracle['absolute_safe_headroom'])}

PRIMARY_DAMAGE_METRIC: max(0,target_logprob_drop_nats)+2*JSD+1[top1_flip]

GENERIC_ROUTER_DAMAGE_PREDICTION: Spearman {predictor['results']['F0_GENERIC_LOCAL_ROUTER']['continuous_damage_spearman']:.4f}; critical recall@20 {pct(predictor['results']['F0_GENERIC_LOCAL_ROUTER']['frontier']['20pct']['protected_critical_recall'])}

MODES_STYLE_DAMAGE_PREDICTION: Spearman {f1['continuous_damage_spearman']:.4f}; critical recall@20 {pct(f1['frontier']['20pct']['protected_critical_recall'])}

TEXT_PREFILL_DAMAGE_PREDICTION: Spearman {f2['continuous_damage_spearman']:.4f}; critical recall@20 {pct(f2['frontier']['20pct']['protected_critical_recall'])}

FULL_PREFILL_DAMAGE_PREDICTION: Spearman {f3['continuous_damage_spearman']:.4f}; critical recall@20 {pct(f3['frontier']['20pct']['protected_critical_recall'])}

VISION_SEPARATED_DAMAGE_PREDICTION: Spearman {f5['continuous_damage_spearman']:.4f}; critical recall@20 {pct(f5['frontier']['20pct']['protected_critical_recall'])}

VISION_INCREMENT_VS_MODES: recall {pp_ci(comparisons['F5_MINUS_F1']['protected_critical_recall'])}; safe precision {pp_ci(comparisons['F5_MINUS_F1']['safe_skip_precision'])}

VISION_INCREMENT_VS_TEXT: recall {pp_ci(comparisons['F5_MINUS_F2']['protected_critical_recall'])}; safe precision {pp_ci(comparisons['F5_MINUS_F2']['safe_skip_precision'])}

RARE_BUT_CRITICAL_PATH_RATE: {pct(rare['path_rate'])} ({rare['path_count']}/{functional['heldout_paths']})

BEST_FROZEN_SKIP_BUDGET: {'PENDING_QUALITY' if gate['pass'] else 'N/A_STAGE_B_GATE_FAILED'}

GENERIC_MODES_QUALITY_AT_BUDGET: N/A

VISION_POLICY_QUALITY_AT_BUDGET: N/A

ACTUAL_EXPERT_PATH_REDUCTION: N/A_NO_PROMOTED_RUNTIME_POLICY

EP2_TPOT_GAIN: N/A_NOT_RUN_BY_GATE

EP4_TPOT_GAIN: N/A_NOT_RUN_BY_GATE

EP8_STRUCTURAL_EFFECT: STRUCTURAL_EP8_PROJECTION_ONLY; bounded offline 20% masks in report; no latency claim

VERDICT: {verdict}

NEXT_ACTION: {'Run bounded C0–C6 natural-quality frontier before any physical timing.' if gate['pass'] else 'Stop this vision-conditioned branch at Stage B; any future generic/text-aware skipping study must be qualified independently.'}
"""
    write(reports / "final_summary.md", summary)
    final = {
        "baseline_status": "PASS_REUSED_VALIDATED_TRUE_EP2_EP4",
        "damage_capture_status": "MEASURED_LOCAL_9312_AND_FUNCTIONAL_256_TRUE_EP4",
        "drop_one_oracle_safe_skip_headroom": oracle["absolute_safe_headroom"],
        "stage_b_gate": gate,
        "verdict": verdict,
        "quality_status": later_status,
        "ep2_runtime": None, "ep4_runtime": None,
        "ep8_status": "STRUCTURAL_EP8_PROJECTION_ONLY",
        "production_runtime_implemented": False,
    }
    atomic_json(artifacts / "final_verdict.json", final)
    # Required machine-readable stage artifacts remain explicit even when the
    # frozen Stage-B gate correctly prevents propagation/runtime work.  This
    # avoids encoding NOT_RUN as a zero measurement.
    atomic_json(artifacts / "frozen_budget_policy.json", {
        "status": later_status,
        "policy": None,
        "reason": "Stage C is authorized only after the frozen Stage-B gate.",
        "actual_expert_path_reduction": None,
    })
    atomic_json(artifacts / "quality_results.json", {
        "status": later_status,
        "generic_modes_quality_at_budget": None,
        "vision_policy_quality_at_budget": None,
        "reason": "Local and one-step functional labels are not task quality.",
    })
    atomic_json(artifacts / "timing_ledger.json", {
        "status": later_status,
        "ep2_tpot_gain": None,
        "ep4_tpot_gain": None,
        "simulator_substitution": False,
    })
    atomic_json(artifacts / "actual_modified_routes.json", {
        "status": later_status,
        "routes": None,
        "reason": "No adaptive route is materialized before quality promotion.",
    })
    atomic_json(artifacts / "run_manifest.json", {
        "schema_version": 1,
        "authoritative_spec": "/home/esjung/MLLM-EP-github/poc_flashvep/reports/MLLM_Vision_Conditioned_Decode_Damage_Estimation_and_Adaptive_Skipping_PoC.md",
        "model_revision": "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c",
        "dtype": "bfloat16",
        "physical_evidence": "TRUE_TP2_DP2_EP4_DEEPEP",
        "functional_probe_count": functional["path_count"],
        "local_path_label_count": local["path_labels"],
        "stage_b_gate": gate,
        "verdict": verdict,
        "commands": [
            "bash poc_mllm_vision_conditioned_decode_damage_skipping/scripts/run_vector_capture.sh path_vector_capture_v1 0",
            "bash poc_mllm_vision_conditioned_decode_damage_skipping/scripts/run_vector_capture.sh path_vector_capture_v3_remaining 25",
            "python -m poc_mllm_vision_conditioned_decode_damage_skipping.scripts.build_local_damage --capture poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/runs/path_vector_capture_v1 --capture poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/runs/path_vector_capture_v3_remaining --manifest poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/manifests/frozen_damage_cohort.json --compact poc_mllm_mixed_domain_online_calibration/artifacts/ep4_compact.jsonl.gz --criticality poc_mllm_vision_conditioned_decode_reduction/artifacts/frozen_vision_criticality.json --labels poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/local_path_damage_labels_v2.jsonl.gz --summary poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/local_damage_summary_v2.json --functional-plan poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/manifests/frozen_functional_probe_plan_v2.json",
            "bash poc_mllm_vision_conditioned_decode_damage_skipping/scripts/run_functional_probe.sh functional_damage_v1",
            "bash poc_mllm_vision_conditioned_decode_damage_skipping/scripts/run_functional_probe.sh functional_damage_v2_remaining '' 29 frozen_functional_probe_plan_v2.json",
            "python -m poc_mllm_vision_conditioned_decode_damage_skipping.scripts.analyze_functional_damage --run poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/runs/functional_damage_v1 --run poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/runs/functional_damage_v2_remaining --plan poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/manifests/frozen_functional_probe_plan_v2.json --local-labels poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/local_path_damage_labels_v2.jsonl.gz --labels poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/functional_path_damage_labels.jsonl.gz --summary poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/functional_damage_summary.json",
            "python -m poc_mllm_vision_conditioned_decode_damage_skipping.scripts.analyze_predictors --labels poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/functional_path_damage_labels.jsonl.gz --contract poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/research_contract.json --artifact-dir poc_mllm_vision_conditioned_decode_damage_skipping/artifacts",
            "python -m poc_mllm_vision_conditioned_decode_damage_skipping.scripts.build_ep8_structural --labels poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/functional_path_damage_labels.jsonl.gz --predictions poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/predictor_outputs.jsonl.gz --output poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/ep8_structural_projection.json",
            "python -m poc_mllm_vision_conditioned_decode_damage_skipping.scripts.render_reports --root poc_mllm_vision_conditioned_decode_damage_skipping",
        ],
        "production_runtime_implemented": False,
    })
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
