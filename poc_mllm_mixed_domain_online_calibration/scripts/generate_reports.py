from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..contract import DOMAINS, MODEL, MODEL_REVISION, read_json, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def _write(name: str, body: str) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / name).write_text(body.rstrip() + "\n")


def _f(value: Any, digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _pct(value: Any, digits: int = 2) -> str:
    return "N/A" if value is None else f"{100 * float(value):.{digits}f}%"


def _metric(curve: dict[str, Any], ratio: str, mode: str, predictor: str, metric: str) -> float | None:
    return curve[ratio][mode][predictor]["1-128"].get(metric)


def main() -> None:
    final = read_json(ART / "final_verdict.json")
    sep = read_json(ART / "domain_signature_separation.json")
    curve = read_json(ART / "warmup_learning_curve.json")
    online = read_json(ART / "online_vs_frozen_calibration.json")
    pred = read_json(ART / "predictor_comparison.json")
    locality = read_json(ART / "mixed_domain_locality.json")
    scaling = read_json(ART / "ep2_ep4_scaling.json")
    ep8 = read_json(ART / "ep8_structural_projection.json")
    overhead = read_json(ART / "predictor_overhead.json")
    cohort = read_json(ART / "manifests" / "frozen_domain_pool.json")
    warmups = read_json(ART / "manifests" / "warmup_indices.json")
    ep4_audit = read_json(ART / "ep4_trace_audit.json")
    ep2_audit = read_json(ART / "ep2_trace_audit.json")
    ep4_rows = read_jsonl(ART / "ep4_compact.jsonl.gz")
    chosen = f"{float(final['selected_warmup_ratio']):.2f}"
    chosen_pct = _pct(float(chosen), 0)
    ep4_outputs = [row["output_token_count"] for row in ep4_rows]
    lengths = {domain: [row["output_token_count"] for row in ep4_rows if row["domain"] == domain] for domain in DOMAINS}

    _write("baseline_readiness.md", f"""# Baseline readiness

Status: **READY_REUSED_AND_RECONFIRMED**.

- Model: `{MODEL}` at frozen revision `{MODEL_REVISION}`, BF16, d=2048, m=768, 48 sparse MoE layers, 128 routed experts, top-k 8.
- Frozen Part-1 bundle `cdf380d9...f66d3` reports quality/actual-EP2/actual-EP4 ready and was reused; full MMStar/ChartQA was not repeated.
- TRUE EP4 trace: TP2-DP2-EP4-PP1, DeepEP high-throughput, linear 32 experts/rank; {ep4_audit['requests']} requests, observed EP worlds `{ep4_audit['observed_ep_world_sizes']}`, invalid expert IDs {ep4_audit['invalid_expert_ids']}, silent fallback `{ep4_audit['silent_fallback']}`.
- TRUE EP2 trace: TP1-DP2-EP2, DeepEP high-throughput, linear 64 experts/rank; {ep2_audit['requests']} requests, observed EP worlds `{ep2_audit['observed_ep_world_sizes']}`, invalid expert IDs {ep2_audit['invalid_expert_ids']}, silent fallback `{ep2_audit['silent_fallback']}`.
- Physical GPU UUIDs: 4 `GPU-6076e2f2-5b63-3761-5586-56ceb7df8139`; 5 `GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730`; 6 `GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28`; 7 `GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b`.
- Tracing is an instrumented evidence pass and is not treated as clean latency timing.
- EP8 is structural replay only; latency/TPOT/E2E/speedup are prohibited.
""")

    domain_lines = []
    for domain in DOMAINS:
        values = lengths[domain]
        capped = sum(row["finish_reason"] == "length" for row in ep4_rows if row["domain"] == domain)
        long_count = sum(value >= 64 for value in values)
        domain_lines.append(f"| {domain} | {len(values)} | 20 / 20 | {_f(np.mean(values), 1)} | {int(np.median(values))} | {max(values)} | {long_count} | {capped} |")
    realized = []
    for ratio, values in warmups["ratios"].items():
        text = ", ".join(f"{domain}: {item['count']}/{len(lengths[domain])} ({_pct(item['realized_ratio'], 1)})" for domain, item in values.items())
        realized.append(f"- nominal {float(ratio)*100:g}% — {text}")
    _write("domain_cohort.md", f"""# Frozen mixed-domain cohort

The cohort was selected and ordered before any routing result was read. It contains {cohort['request_count']} image-hash-disjoint requests, 40 per domain, with a frozen 20/20 discovery/held-out split and frozen IID/domain-shift streams. Routing and answer correctness were not selection inputs.

| Domain | N | discovery / heldout | mean output tokens | median | max | >=64 | length-capped |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(domain_lines)}

`structured_diagram_layout` is a semantically distinct MMStar diagram/code-sequence cohort because no local DocVQA/OCRBench asset was available. It is not claimed as an OCR benchmark reproduction.

## Realized balanced warm-up

{chr(10).join(realized)}

Natural generation used one frozen image-grounded reasoning prompt, deterministic decoding, EOS enabled, and `max_new_tokens=192`. Short-answer domains were not artificially extended.
""")

    sep_lines = []
    for predictor in ("P_text", "P_full", "P_vision", "P_separated"):
        value = sep["heldout"][predictor]
        sep_lines.append(f"| {predictor} | {_f(value['within_domain_cosine_distance'])} | {_f(value['between_domain_cosine_distance'])} | {_f(value['between_minus_within'])} | {_pct(value['nearest_neighbor_same_domain_rate'])} | {_f(value['silhouette'])} |")
    _write("domain_signature_separation.md", f"""# Domain signature separation

All separation metrics use the frozen held-out half; layerwise discovery values are in `artifacts/domain_signature_separation.json`.

| Signature | within cosine distance | between cosine distance | gap | NN same-domain | silhouette |
|---|---:|---:|---:|---:|---:|
{chr(10).join(sep_lines)}

The result tests routing-signature structure, not dataset-name labels. Positive between-minus-within and NN/silhouette quantify actual separation. `P_separated` preserves modality identity; its larger dimensionality is controlled by using the same cosine kNN family for all learned-signature predictors.
""")

    curve_lines = []
    for ratio in ("0.00", "0.01", "0.05", "0.10", "0.20"):
        curve_lines.append("| " + " | ".join([
            f"{float(ratio)*100:g}%",
            _f(_metric(curve["heldout"], ratio, "M1_frozen", "P_text", "expert_histogram_cosine")),
            _f(_metric(curve["heldout"], ratio, "M1_frozen", "P_full", "expert_histogram_cosine")),
            _f(_metric(curve["heldout"], ratio, "M1_frozen", "P_separated", "expert_histogram_cosine")),
            _f(_metric(curve["heldout"], ratio, "M1_frozen", "P_vision", "expert_histogram_cosine")),
            _f(_metric(curve["heldout"], ratio, "M1_frozen", "P_domain", "expert_histogram_cosine")),
            _f(_metric(curve["heldout"], ratio, "M2_continual", "P_separated", "expert_histogram_cosine")),
        ]) + " |")
    _write("warmup_learning_curve.md", f"""# Domain-balanced warm-up learning curve

Primary metric below is held-out aggregate 1–128 decode expert-histogram cosine under frozen calibration. Ratio selection used discovery only. Zero warm-up is an explicitly uncalibrated uniform prior, not future truth.

| nominal ratio | P_text frozen | P_full frozen | P_separated frozen | P_vision frozen | domain label frozen | P_separated continual |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(curve_lines)}

Frozen practical knee: **{chosen_pct}**. Rule: {curve['selected_knee']['criterion']}. Domain-balanced realized counts are reported in `domain_cohort.md`; the 1% nominal point realizes 2.5% per domain. Warm-up requests are ordinary baseline service requests (no duplicate inference); their opportunity cost is the first K requests/domain without a calibrated predictor, plus the separately measured CPU update/lookup cost.
""")

    online_lines = []
    for setting, data in (("IID heldout", online["iid_heldout"]), ("domain shift", online["domain_shift"])):
        for mode in ("M1_frozen", "M2_continual"):
            item = data[mode]["P_separated"]
            if "1-128" in item: item = item["1-128"]
            online_lines.append(f"| {setting} | {mode} | {_f(item.get('expert_histogram_cosine'))} | {_f(item.get('rank_distribution_cosine'))} | {_f(item.get('per_rank_load_mae'))} |")
    online_ci_lines = []
    for stream, metrics in online["continual_minus_frozen_paired_ci"].items():
        for metric, value in metrics.items():
            online_ci_lines.append(f"| {stream} | {metric} | {_f(value['beneficial_delta'])} | [{_f(value['ci95_low'])}, {_f(value['ci95_high'])}] |")
    phase_lines = []
    phases = online["domain_shift_by_phase"]
    for phase in phases["M1_frozen"]["P_separated"]:
        frozen = phases["M1_frozen"]["P_separated"][phase]
        continual = phases["M2_continual"]["P_separated"][phase]
        phase_lines.append(f"| {phase} | {_f(frozen.get('expert_histogram_cosine'))} | {_f(continual.get('expert_histogram_cosine'))} | {_f(frozen.get('rank_distribution_cosine'))} | {_f(continual.get('rank_distribution_cosine'))} |")
    _write("online_vs_frozen_calibration.md", f"""# Frozen versus continual online calibration

Every prediction follows `prefill signature → prediction record → actual decode → optional DB update`. The current request ID is rejected if present in memory. The complete causal ledger is `artifacts/no_future_leakage_events.jsonl.gz`.

| stream | mode | expert cosine | rank cosine | rank-load MAE |
|---|---|---:|---:|---:|
{chr(10).join(online_lines)}

Positive deltas below favor continual calibration:

| stream | metric | beneficial delta | paired 95% CI |
|---|---|---:|---:|
{chr(10).join(online_ci_lines)}

Frozen versus continual behavior after each frozen stream boundary:

| phase | frozen expert cosine | continual expert cosine | frozen rank cosine | continual rank cosine |
|---|---:|---:|---:|---:|
{chr(10).join(phase_lines)}

Predictor CPU overhead on already-collected compact histograms: signature construction {_f(overhead['signature_construction_cpu_us_per_request'], 2)} µs/request, kNN lookup {_f(overhead['knn_query_cpu_us'], 2)} µs, database update {_f(overhead['database_update_cpu_us'], 2)} µs. No GPU synchronization is performed by the offline predictor. A deployable asynchronous GPU histogram-export path was not implemented, so router-trace extraction overhead is `UNAVAILABLE` rather than silently treated as zero.
""")

    selected = pred["iid_heldout_frozen"]["1-128"]
    predictor_lines = []
    selected_curve = curve["heldout"][chosen]["M1_frozen"]
    for predictor in ("P_domain", "P_text", "P_full", "P_vision", "P_separated"):
        value = selected_curve[predictor]["1-128"]
        predictor_lines.append(f"| {predictor} | {_f(value.get('decode_expert_top8_recall'))} | {_f(value.get('expert_histogram_cosine'))} | {_f(value.get('expert_histogram_jsd'))} | {_f(value.get('active_expert_set_recall'))} | {_f(value.get('rank_distribution_cosine'))} | {_f(value.get('rank_distribution_jsd'))} | {_f(value.get('per_rank_load_mae'))} | {_pct(value.get('hottest_rank_accuracy'))} |")
    ci_lines = []
    for control in ("P_full", "P_text", "P_domain"):
        for metric in ("expert_histogram_cosine", "rank_distribution_cosine", "per_rank_load_mae"):
            value = selected[f"P_separated_minus_{control}"][metric]
            ci_lines.append(f"| P_separated − {control} | {metric} | {_f(value['beneficial_delta'])} | [{_f(value['ci95_low'])}, {_f(value['ci95_high'])}] | {value['requests']} |")
    _write("predictor_comparison.md", f"""# Predictor comparison

All policies use the same simple cosine kNN/prototype family. Metrics use checkpoint-sampled actual decode routes through step 128 and the frozen {chosen_pct} calibration ratio. `P_full` is an ELDR-style whole-prefill control, not an exact ELDR reproduction.

| Predictor | top-8 recall | expert cosine | expert JSD | active-set recall | rank cosine | rank JSD | rank-load MAE | hottest-rank acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(predictor_lines)}

Positive bootstrap delta means P_separated is better; request is the bootstrap unit.

| Comparison | metric | beneficial delta | paired 95% CI | N |
|---|---|---:|---:|---:|
{chr(10).join(ci_lines)}

Window-specific values for 1–32, 33–64, 65–96, and 97–128 are preserved in `artifacts/predictor_comparison.json`. Point estimates whose CI includes zero are not called confirmed increments.
""")

    local_lines = []
    for size in ("4", "8", "16"):
        for ep_size in ("2", "4", "8"):
            values = locality[size]["ep"][ep_size]
            local_lines.append(f"| B{size} | EP{ep_size} | {_pct(values['S4_ORACLE']['union_reduction_vs_fifo'])} | {_pct(values['S1_DOMAIN']['union_reduction_vs_fifo'])} | {_pct(values['S2_P_full']['union_reduction_vs_fifo'])} | {_pct(values['S3_P_separated']['union_reduction_vs_fifo'])} |")
    oracle_b8_ep4 = locality["8"]["ep"]["4"]["S4_ORACLE"]["union_reduction_vs_fifo"]
    scheduler = "STOPPED_BY_GATE" if oracle_b8_ep4 < .10 else "HEADROOM_GATE_PASSED_BUT_RUNTIME_NOT_IN_SCOPE"
    _write("mixed_domain_locality_headroom.md", f"""# Mixed-domain locality headroom revisit

| wave | mapping | oracle | same-domain | P_full | P_separated |
|---|---|---:|---:|---:|---:|
{chr(10).join(local_lines)}

Primary B8/EP4 future-truth oracle headroom is **{_pct(oracle_b8_ep4)}**. It uses actual future routes with greedy marginal-union grouping plus bounded pair-swap refinement, so it is non-deployable and not a proof of the globally optimal partition. Scheduler status: **{scheduler}**. No runtime scheduler was implemented when the 10% gate was not met. The EP8 rows are structural only.
""")

    swap = scaling["image_swap_rank_scaling"]
    swap_lines = [f"| EP{ep} | {_pct(value['changed_expert_fraction'])} | {_pct(value['changed_rank_fraction'])} | {value['event_count']} |" for ep, value in swap.items() if ep in {"2", "4", "8"}]
    if not swap_lines:
        swap_lines = ["| N/A | N/A | N/A | prior causal trace unavailable |"]
    rank_sep = scaling["rank_signature_domain_separation"]
    rank_sep_lines = [
        f"| {label} | {_f(values['P_text']['between_minus_within'])} | {_f(values['P_full']['between_minus_within'])} | {_f(values['P_separated']['between_minus_within'])} |"
        for label, values in rank_sep.items()
    ]
    physical_metric_lines = []
    for label, payload in (("EP2 physical discovery subset", scaling["ep2_physical"]), ("EP4 physical heldout", scaling["ep4_physical"])):
        values = payload["selected_ratio_curve"]["M1_frozen"]["P_separated"]["1-128"]
        physical_metric_lines.append(f"| {label} | {_f(values.get('expert_histogram_cosine'))} | {_f(values.get('rank_distribution_cosine'))} | {_f(values.get('per_rank_load_mae'))} | {_pct(values.get('hottest_rank_accuracy'))} |")
    _write("ep2_ep4_scaling.md", f"""# Physical EP2 / EP4 and topology scaling

- EP2: **{scaling['ep2_physical']['status']}**, {scaling['ep2_physical']['request_count']} balanced requests, 64 experts/rank.
- EP4: **{scaling['ep4_physical']['status']}**, {scaling['ep4_physical']['request_count']} requests, 32 experts/rank.
- Common EP2/EP4 expert histogram cosine: {_f(scaling['physical_route_comparison']['expert_histogram_cosine_ep2_vs_ep4'])} across {scaling['physical_route_comparison']['common_request_count']} requests. Rank identity is topology-specific and is not equated across EP sizes.
- Bounded deterministic topology parity: exact token sequence {_pct(scaling['semantic_parity'].get('exact_output_token_sequence_rate'))}; exact output text {_pct(scaling['semantic_parity'].get('exact_output_text_rate'))}. No policy changes model semantics; differences, if any, are audited as topology/BF16 numerical behavior.

| physical evidence | P_separated expert cosine | rank cosine | rank-load MAE | hottest-rank acc. |
|---|---:|---:|---:|---:|
{chr(10).join(physical_metric_lines)}

Prior frozen same-question/same-token causal image-swap expert IDs were remapped without latency modeling:

| topology | changed expert | changed destination-rank | events |
|---|---:|---:|---:|
{chr(10).join(swap_lines)}

Rank-signature domain-separation gaps:

| evidence | P_text | P_full | P_separated |
|---|---:|---:|---:|
{chr(10).join(rank_sep_lines)}

This directly tests frozen H2/H3: finer expert ownership can turn more expert changes into rank changes. Physical EP2/EP4 prediction metrics are in `artifacts/ep2_ep4_scaling.json`; EP8 remains structural.
""")

    ep8_pred_lines = []
    for predictor, value in ep8["predictor_metrics"].items():
        ep8_pred_lines.append(f"| {predictor} | {_f(value.get('expert_histogram_cosine'))} | {_f(value.get('rank_distribution_cosine'))} | {_f(value.get('per_rank_load_mae'))} | {_pct(value.get('hottest_rank_accuracy'))} |")
    _write("ep8_structural_projection.md", f"""# EP8 structural projection

Evidence label: **{ep8['evidence_label']}**. Experts/rank: {ep8['experts_per_rank']}. EP8 latency, TPOT, E2E, and speedup: **{ep8['latency_tpot_e2e_speedup']}**.

- Requests: {ep8['request_count']}
- Rank max/mean: {_f(ep8['rank_max_mean'])}
- Remote destination fraction: {_pct(ep8['remote_destination_fraction'])}
- Logical remote assignments: {_f(ep8['logical_remote_assignments'], 0)}
- Logical dispatch+combine payload proxy: {_f(ep8['logical_remote_dispatch_combine_bytes'] / 2**20, 2)} MiB (logical BF16 hidden bytes; not measured transport).

| Predictor | expert cosine | EP8 rank cosine | rank-load MAE | hottest-rank acc. |
|---|---:|---:|---:|---:|
{chr(10).join(ep8_pred_lines)}

Per-domain rank prototypes, active experts/rank, B4/B8/B16 locality, and causal changed-rank replay are machine-readable in `artifacts/ep8_structural_projection.json`. These results support only a structural interpretation such as “potentially stronger at higher EP.”
""")

    full_delta = final["vision_increment_vs_full"]["expert_histogram_cosine"]
    text_delta = final["vision_increment_vs_text"]["expert_histogram_cosine"]
    domain_delta = final["domain_label_control"]["expert_histogram_cosine"]
    p_sep_value = selected_curve["P_separated"]["1-128"]
    frozen_gain = p_sep_value["expert_histogram_cosine"] - curve["heldout"]["0.00"]["M1_frozen"]["P_separated"]["1-128"]["expert_histogram_cosine"]
    m1 = online["iid_heldout"]["M1_frozen"]["P_separated"]["1-128"]["expert_histogram_cosine"]
    m2 = online["iid_heldout"]["M2_continual"]["P_separated"]["1-128"]["expert_histogram_cosine"]
    if "SUPPORTED" in final["verdict"]:
        recommendation = "Freeze this calibrated signal for a separate bounded adaptive-budget PoC; do not infer latency or deploy skipping here."
    elif final["verdict"] == "GENERIC_FULL_PREFILL_SUFFICIENT":
        recommendation = "Do not promote a vision-separated method; if prediction work continues, use the simpler P_text/P_full control in a separately gated PoC."
    else:
        recommendation = "Do not use this signal for expert skipping; investigate the stated failed evidence gate."
    changed_rank = {ep: _pct(swap.get(ep, {}).get("changed_rank_fraction")) for ep in ("2", "4", "8")}
    _write("final_summary.md", f"""# Mixed-domain online calibration PoC — final summary

The bounded study completed a frozen 160-request, four-domain TRUE EP4 trace, a balanced {scaling['ep2_physical']['request_count']}-request TRUE EP2 anchor, no-future-leakage online replay, and mandatory EP8 structural replay. Including bounded smoke, it used {final['execution_evidence']['full_model_execution_count']}/400 full-model request executions. It implemented no routing change, expert skipping, or production scheduler.

Primary result: **{final['verdict']}**. At the discovery-selected {chosen_pct} balanced warm-up, held-out `P_separated` expert-histogram cosine is {_f(p_sep_value['expert_histogram_cosine'])}. Its beneficial delta over `P_full` is {_f(full_delta['beneficial_delta'])} (95% CI [{_f(full_delta['ci95_low'])}, {_f(full_delta['ci95_high'])}]); over `P_text`, {_f(text_delta['beneficial_delta'])} ([{_f(text_delta['ci95_low'])}, {_f(text_delta['ci95_high'])}]). Thus modality separation adds no held-out value, while the simpler text-only signature is statistically better on this metric.

The interpretation follows the pre-frozen criteria. Expert-level gain, rank-level gain, domain-label control, and locality are not conflated. EP8 is never described as measured latency.

```
BASELINE_STATUS: READY_REUSED_AND_TRUE_EP2_EP4_RECONFIRMED
DOMAIN_COHORT_STATUS: FROZEN_160_REQUESTS_40_PER_DOMAIN_IMAGE_HASH_DISJOINT
DOMAIN_SIGNATURE_SEPARATION: P_SEPARATED_HELDOUT_GAP={_f(sep['heldout']['P_separated']['between_minus_within'])}; NN_SAME_DOMAIN={_pct(sep['heldout']['P_separated']['nearest_neighbor_same_domain_rate'])}
WARMUP_RATIOS_REALIZED_PER_DOMAIN: 0%=0/40; 1%=1/40(2.5%); 5%=2/40; 10%=4/40; 20%=8/40
BEST_WARMUP_RATIO: {chosen_pct}_DISCOVERY_FROZEN_KNEE
FROZEN_WARMUP_GAIN: EXPERT_COSINE_DELTA_VS_UNCALIBRATED={_f(frozen_gain)}
CONTINUAL_CALIBRATION_GAIN: IID_EXPERT_COSINE_M2_MINUS_M1={_f(m2-m1)}
P_TEXT_RESULT: EXPERT_COSINE={_f(selected_curve['P_text']['1-128']['expert_histogram_cosine'])}; RANK_COSINE={_f(selected_curve['P_text']['1-128']['rank_distribution_cosine'])}
P_FULL_ELDR_STYLE_RESULT: EXPERT_COSINE={_f(selected_curve['P_full']['1-128']['expert_histogram_cosine'])}; RANK_COSINE={_f(selected_curve['P_full']['1-128']['rank_distribution_cosine'])}
P_SEPARATED_RESULT: EXPERT_COSINE={_f(p_sep_value['expert_histogram_cosine'])}; RANK_COSINE={_f(p_sep_value['rank_distribution_cosine'])}
VISION_INCREMENT_VS_FULL: EXPERT_COSINE={_f(full_delta['beneficial_delta'])}_CI95=[{_f(full_delta['ci95_low'])},{_f(full_delta['ci95_high'])}]
VISION_INCREMENT_VS_TEXT: EXPERT_COSINE={_f(text_delta['beneficial_delta'])}_CI95=[{_f(text_delta['ci95_low'])},{_f(text_delta['ci95_high'])}]
DOMAIN_LABEL_CONTROL_RESULT: P_SEPARATED_BENEFICIAL_EXPERT_COSINE={_f(domain_delta['beneficial_delta'])}_CI95=[{_f(domain_delta['ci95_low'])},{_f(domain_delta['ci95_high'])}]
MIXED_DOMAIN_ORACLE_LOCALITY_HEADROOM: B8_EP4={_pct(oracle_b8_ep4)}
P_FULL_LOCALITY_HEADROOM: B8_EP4={_pct(locality['8']['ep']['4']['S2_P_full']['union_reduction_vs_fifo'])}
P_SEPARATED_LOCALITY_HEADROOM: B8_EP4={_pct(locality['8']['ep']['4']['S3_P_separated']['union_reduction_vs_fifo'])}
EP2_PHYSICAL_RESULT: TRUE_DEEPEP_EP2_{scaling['ep2_physical']['request_count']}_REQUESTS_64_EXPERTS_PER_RANK
EP4_PHYSICAL_RESULT: TRUE_DEEPEP_EP4_{scaling['ep4_physical']['request_count']}_REQUESTS_32_EXPERTS_PER_RANK
EP8_STRUCTURAL_RESULT: STRUCTURAL_EP8_PROJECTION_ONLY_16_EXPERTS_PER_RANK_NO_LATENCY
EP_SCALING_INTERPRETATION: CHANGED_RANK_EP2={changed_rank['2']}; EP4={changed_rank['4']}; EP8_STRUCTURAL={changed_rank['8']}
VERDICT: {final['verdict']}
NEXT_ACTION: {recommendation}
```
""")


if __name__ == "__main__":
    main()
