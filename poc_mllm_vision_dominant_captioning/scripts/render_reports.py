from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from poc_mllm_vision_dominant_captioning.contract import DOMAINS, UNIVERSAL_PROMPT, atomic_json, read_json, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
REASONING_CAUSAL = ROOT.parent / "poc_mllm_long_reasoning_decode_vision_signal/artifacts/causal_comparison.json"


def _read(path: Path, fallback: Any = None) -> Any:
    return read_json(path) if path.exists() else fallback


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.3f}%"


def _f(value: float | None, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _write(name: str, text: str) -> None:
    path = ROOT / "reports" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def _causal_value(causal: dict[str, Any], kind: str, window: str, ep: int, key: str) -> float | None:
    return causal.get("aggregate", {}).get(kind, {}).get(window, {}).get(str(ep), {}).get(key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--natural-results", type=Path, default=ROOT / "runs/ep4_natural200_v2/results.jsonl")
    args = parser.parse_args()
    cohort = _read(args.artifacts / "manifests/frozen_image_cohort.json", {})
    lengths = _read(args.artifacts / "caption_output_lengths.json", {})
    quality = _read(args.artifacts / "caption_quality_sanity.json", {})
    separation = _read(args.artifacts / "domain_separation.json", {})
    predictor = _read(args.artifacts / "predictor_comparison.json", {})
    causal = _read(args.artifacts / "causal_effect.json", {})
    ep2_causal = _read(args.artifacts / "ep2_causal_effect.json", {})
    locality = _read(args.artifacts / "batch_locality.json", {})
    timing = _read(args.artifacts / "clean_b8_timing.json", {})
    ep4_audit = _read(args.artifacts / "ep4_trace_audit.json", {})
    ep2_audit = _read(args.artifacts / "ep2_trace_audit.json", {})
    ep2_predictor = _read(args.artifacts / "ep2_predictor_comparison.json", {})
    rank_profiles = _read(args.artifacts / "ep_rank_profiles.json", {})
    step_audit = _read(args.artifacts / "trace_step_audit.json", {})
    video = _read(args.artifacts / "video_preprocessing_manifest.json", {})
    reasoning = _read(REASONING_CAUSAL, {})
    reasoning_change = reasoning.get("bootstrap_vs_noise", {}).get("changed_expert_fraction", {}).get("mean_difference")
    results = read_jsonl(args.natural_results)

    counts = {domain: sum(row["domain"] == domain for row in cohort.get("requests", [])) for domain in DOMAINS}
    visual = {
        domain: [row["vision_token_count"] for row in results if row["domain"] == domain]
        for domain in DOMAINS
    }
    cohort_lines = []
    for domain in DOMAINS:
        values = visual[domain]
        output = lengths.get(domain, {})
        cohort_lines.append(
            f"| {domain} | {counts[domain]} | {len(values)} | "
            f"{_f(float(np.median(values)) if values else None, 1)} | "
            f"{_f(output.get('median'), 1)} | {output.get('ended_before_cap', 'N/A')} |"
        )

    baseline_status = (
        "READY_TRUE_EP2_EP4_RECONFIRMED"
        if not ep4_audit.get("silent_fallback", True) and not ep2_audit.get("silent_fallback", True)
        else "PARTIAL_EP_PATH_EVIDENCE"
    )
    _write("baseline_readiness.md", f"""
# Baseline readiness

Status: **{baseline_status}**.

- Qwen3-VL-30B-A3B-Instruct revision `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`, BF16, d=2048, m=768, 48 sparse-MoE layers, 128 experts, K8.
- EP4: TP2-DP2-EP4-PP1, DeepEP high-throughput, linear 32 experts/rank; observed EP worlds `{ep4_audit.get('observed_ep_world_sizes')}`, invalid IDs {ep4_audit.get('invalid_expert_ids')}.
- EP2: TP1-DP2-EP2, DeepEP high-throughput, linear 64 experts/rank; observed EP worlds `{ep2_audit.get('observed_ep_world_sizes')}`, invalid IDs {ep2_audit.get('invalid_expert_ids')}.
- Full MMStar/ChartQA was not repeated; the frozen foundation and later physical path proofs were reused.
- Instrumented routing and clean B8 timing were separate. No latency simulator is used.
- Decode-step audit: `{json.dumps(step_audit, sort_keys=True)}`. An initial 50-request partial trace that counted external-DP padding forwards was explicitly excluded; v2 keys steps by unique M-RoPE position.
""")
    _write("image_domain_cohort.md", f"""
# Frozen image-domain cohort

The manifest was frozen before route inspection or generated-output inspection. It contains **{cohort.get('request_count')}** unique image hashes and uses balanced domain counts.

| domain | frozen images | executed EP4 | median visual tokens | median output tokens | EOS before cap |
|---|---:|---:|---:|---:|---:|
{chr(10).join(cohort_lines)}

Source deviations are explicit: `{json.dumps(cohort.get('source_deviations', {}), sort_keys=True)}`.
The dense-relation and scene/layout lanes are MMStar public-image substitutes, not claims of exact Image-Paragraph-Captioning or ADE20K reproduction.

Generation sanity: malformed={quality.get('malformed')}, refusals={quality.get('refusals')}, prompt leakage={quality.get('prompt_leakage')}, mean repeated-4gram fraction={_f(quality.get('mean_repeated_4gram_fraction'))}. CIDEr/BLEU/SPICE are not presented as official scores because this universal prompt is not each dataset's official caption protocol.
""")
    prompt_hash = hashlib.sha256(UNIVERSAL_PROMPT.encode()).hexdigest()
    observed_hashes = sorted({row.get("universal_prompt_sha256") for row in results})
    _write("universal_prompt_audit.md", f"""
# Universal prompt audit

- Contract hash: `{prompt_hash}`
- Observed hashes: `{observed_hashes}`
- Requests checked: **{len(results)}**
- Byte-identical raw prompt: **{'PASS' if observed_hashes == [prompt_hash] else 'FAIL'}**

No dataset name, category, question, options, reference caption, final-answer marker, or reasoning instruction was inserted.
""")
    text_sep = separation.get("P_text", {})
    vision_sep = separation.get("P_vision", {})
    _write("text_signature_leakage_audit.md", f"""
# Text-signature leakage audit

The linguistic prompt bytes are identical, but the text-token router signature is not assumed identical: text-token hidden states occur after image-token context and their positions vary with image grid length.

- P_text within-domain cosine distance: {_f(text_sep.get('within_domain_cosine_distance'))}
- P_text between-domain distance: {_f(text_sep.get('between_domain_cosine_distance'))}
- P_text gap: {_f(text_sep.get('between_minus_within'))}
- P_text same-domain NN: {_pct(text_sep.get('nearest_neighbor_same_domain_rate'))}
- P_vision gap / same-domain NN: {_f(vision_sep.get('between_minus_within'))} / {_pct(vision_sep.get('nearest_neighbor_same_domain_rate'))}

Thus any P_text variation is labeled **image-conditioned contextual/position variation in routed text rows**, not request-specific lexical task information or placeholder misclassification. The hook derives vision rows from the runtime multimodal mask; visual rows are subtracted from all prefill rows before P_text construction.
""")

    causal_lines = []
    for window in ("1-32", "33-64", "65-96", "97-128", "129-192", "193-256", "1-256"):
        causal_lines.append(
            f"| {window} | {_pct(_causal_value(causal, 'repeat_noise', window, 4, 'expert_changed_fraction'))} | "
            f"{_pct(_causal_value(causal, 'same_domain', window, 4, 'expert_changed_fraction'))} | "
            f"{_pct(_causal_value(causal, 'cross_domain', window, 4, 'expert_changed_fraction'))} | "
            f"{_pct(_causal_value(causal, 'cross_domain', window, 4, 'changed_rank_fraction'))} |"
        )
    _write("image_swap_causal_effect.md", f"""
# Causal identical-prompt image swap

Every comparison fixes the prompt, decode token IDs, decode positions, and replay length. Only the image changes. Forced replay is off-distribution for the replacement image and is interpreted as causal route sensitivity, not natural-response quality.

`expert change` and `rank change` below use the same histogram/multiset-displacement definition as the earlier reasoning PoC, so the regime comparison is like-for-like. Ordered top-k-slot churn is preserved separately in `artifacts/causal_effect.json` and is not substituted for this metric.

| sampled decode window | repeat expert change | same-domain expert change | cross-domain expert change | EP4 cross-domain rank change |
|---|---:|---:|---:|---:|
{chr(10).join(causal_lines)}

Aggregate cross-domain bootstrap vs repeat noise: `{json.dumps(causal.get('bootstrap_vs_repeat_noise', {}).get('cross_domain', {}).get('1-256', {}).get('4', {}), sort_keys=True)}`.
""")
    caption_change = _causal_value(causal, "cross_domain", "1-256", 4, "expert_changed_fraction")
    amplification = None if caption_change is None or reasoning_change is None else caption_change - reasoning_change
    _write("caption_vs_reasoning_regime.md", f"""
# Caption versus previous reasoning regime

- Previous bounded long-reasoning image effect above repeat noise: **{_pct(reasoning_change)}**
- Identical-prompt caption cross-domain expert change: **{_pct(caption_change)}**
- Caption minus reasoning: **{_pct(amplification)}**

The comparison aligns the changed-expert statistic but uses different frozen request pools. It is evidence about regime amplification, not a paired image-level causal comparison.
""")

    pred_lines = []
    for window in ("1-32", "33-64", "65-96", "97-128", "129-192", "193-256", "1-256"):
        for name in ("P_text", "P_vision", "P_full", "P_separated"):
            row = predictor.get("aggregate", {}).get(name, {}).get(window, {})
            pred_lines.append(
                f"| {window} | {name} | {_f(row.get('expert_histogram_cosine'))} | "
                f"{_f(row.get('decode_expert_top8_recall'))} | {_f(row.get('rank_distribution_cosine'))} | "
                f"{_f(row.get('per_rank_load_mae'))} |"
            )
    ci = predictor.get("paired_bootstrap", {})
    _write("signature_predictor_comparison.md", f"""
# Frozen held-out signature predictor comparison

Discovery rows are the only predictor memory; held-out image hashes are never added. All feature families use the same 3-NN predictor and normalization.

| window | predictor | expert hist cosine | top8 recall | rank cosine | rank-load MAE |
|---|---|---:|---:|---:|---:|
{chr(10).join(pred_lines)}

Aggregate paired request-bootstrap deltas:

- P_vision vs P_text: `{json.dumps(ci.get('vision_vs_text', {}).get('1-256', {}), sort_keys=True)}`
- P_separated vs P_text: `{json.dumps(ci.get('separated_vs_text', {}).get('1-256', {}), sort_keys=True)}`
- P_separated vs P_full: `{json.dumps(ci.get('separated_vs_full', {}).get('1-256', {}), sort_keys=True)}`
""")
    sep_lines = []
    for name in ("P_text", "P_vision", "P_full", "P_separated"):
        row = separation.get(name, {})
        sep_lines.append(
            f"| {name} | {_f(row.get('within_domain_cosine_distance'))} | "
            f"{_f(row.get('between_domain_cosine_distance'))} | {_f(row.get('between_minus_within'))} | "
            f"{_pct(row.get('nearest_neighbor_same_domain_rate'))} | {_f(row.get('silhouette'))} |"
        )
    _write("domain_separation.md", f"""
# Image-driven domain separation

| signature | within distance | between distance | gap | same-domain NN | silhouette |
|---|---:|---:|---:|---:|---:|
{chr(10).join(sep_lines)}

Actual decode-truth expert/rank separation: `{json.dumps(separation.get('decode_truth', {}), sort_keys=True)}`.
Layer-wise values are preserved in `artifacts/domain_separation.json`.
""")
    batch_lines = []
    for size in ("1", "2", "4", "8", "16"):
        for ep in ("2", "4", "8"):
            for policy in ("S0_FIFO", "S1_DOMAIN", "S2_P_full", "S2b_P_vision", "S3_P_separated", "S4_ORACLE"):
                row = locality.get(size, {}).get("ep", {}).get(ep, {}).get(policy, {})
                batch_lines.append(
                    f"| B{size} | EP{ep} | {policy} | {_f(row.get('mean_active_expert_union_per_layer'), 2)} | "
                    f"{_pct(row.get('union_reduction_vs_fifo'))} | {_f(row.get('rank_max_mean'))} |"
                )
    _write("batch_sensitivity.md", f"""
# Structural batch sensitivity

| batch | mapping | policy | mean layer expert union | reduction vs FIFO | rank max/mean |
|---|---|---|---:|---:|---:|
{chr(10).join(batch_lines)}

These are structural replays over measured routes. They are not latency predictions.
""")
    b8 = locality.get("8", {}).get("ep", {}).get("4", {})
    oracle_headroom = b8.get("S4_ORACLE", {}).get("union_reduction_vs_fifo")
    _write("mixed_domain_locality_headroom.md", f"""
# Mixed-domain locality headroom

Primary B8/EP4 structural oracle headroom is **{_pct(oracle_headroom)}**.
P_full/P_vision/P_separated reductions are **{_pct(b8.get('S2_P_full', {}).get('union_reduction_vs_fifo'))} / {_pct(b8.get('S2b_P_vision', {}).get('union_reduction_vs_fifo'))} / {_pct(b8.get('S3_P_separated', {}).get('union_reduction_vs_fifo'))}**.

{'The 10% oracle gate is not met, so no scheduler was implemented.' if oracle_headroom is not None and oracle_headroom < 0.10 else 'Structural headroom is reported only; no scheduler was implemented in this evidence PoC.'}

Clean B8 physical characterization: `{json.dumps(timing, sort_keys=True)}`.
""")
    ep2_rank = _causal_value(ep2_causal, "cross_domain", "1-256", 2, "changed_rank_fraction")
    ep4_rank = _causal_value(causal, "cross_domain", "1-256", 4, "changed_rank_fraction")
    ep8_rank = _causal_value(causal, "cross_domain", "1-256", 8, "changed_rank_fraction")
    _write("ep2_ep4_scaling.md", f"""
# EP2 / EP4 physical scaling

- EP2 cross-domain changed-rank fraction (physical EP2 routes): **{_pct(ep2_rank)}**
- EP4 cross-domain changed-rank fraction (physical EP4 routes): **{_pct(ep4_rank)}**
- EP2 held-out aggregate predictor metrics: `{json.dumps({name: ep2_predictor.get('aggregate', {}).get(name, {}).get('1-256', {}) for name in ('P_text', 'P_vision', 'P_full', 'P_separated')}, sort_keys=True)}`
- EP2 trace proof: `{json.dumps(ep2_audit, sort_keys=True)}`
- EP4 trace proof: `{json.dumps(ep4_audit, sort_keys=True)}`

Router results are not assumed identical across topology. EP8 below remaps the physical EP4 expert IDs and is structural only.
""")
    ep8_payload = {
        "status": "STRUCTURAL_EP8_PROJECTION_ONLY",
        "experts_per_rank": 16,
        "source_mapping_contract": "EP4 measured source ranks 0..3 retained; EP8 ranks 4..7 have no synthetic requests. Global request count is not replicated.",
        "cross_domain_changed_rank_fraction": ep8_rank,
        "domain_rank_separation": separation.get("decode_truth", {}).get("rank"),
        "domain_rank_profiles": rank_profiles.get("8"),
        "predictors": {
            name: predictor.get("aggregate", {}).get(name, {}).get("1-256", {})
            for name in ("P_text", "P_vision", "P_full", "P_separated")
        },
        "b8": locality.get("8", {}).get("ep", {}).get("8", {}),
        "b16": locality.get("16", {}).get("ep", {}).get("8", {}),
        "timing_claim": "PROHIBITED",
    }
    atomic_json(args.artifacts / "ep8_structural_projection.json", ep8_payload)
    _write("ep8_structural_projection.md", f"""
# EP8 structural projection

**STRUCTURAL_EP8_PROJECTION_ONLY** — 16 experts/rank.

- Cross-domain changed-rank fraction: **{_pct(ep8_rank)}**
- B8 structural metrics and held-out rank predictors are stored in `artifacts/ep8_structural_projection.json`.
- No EP8 latency, TPOT, E2E, or speedup is estimated.
""")
    _write("video_extension.md", f"""
# Secondary video extension

Status: **{video.get('status')}**.

ActivityNet Captions metadata was discoverable but source videos were not local. The available YouCook2 public media archive was {video.get('youcook2', {}).get('available_public_archive_bytes')} bytes. The optional lane was not allowed to turn into a multi-GB acquisition project ahead of the primary physical image study. No synthetic video was substituted; video signal and EP8 video results are **NOT_RUN**.
""")

    vision_ci = ci.get("vision_vs_text", {}).get("1-256", {}).get("expert_histogram_cosine", {})
    sep_text_ci = ci.get("separated_vs_text", {}).get("1-256", {}).get("expert_histogram_cosine", {})
    sep_full_ci = ci.get("separated_vs_full", {}).get("1-256", {}).get("expert_histogram_cosine", {})
    causal_ci = causal.get("bootstrap_vs_repeat_noise", {}).get("cross_domain", {}).get("1-256", {}).get("4", {}).get("expert_changed_fraction", {})
    late_values = [
        _causal_value(causal, "cross_domain", window, 4, "expert_changed_fraction")
        for window in ("65-96", "97-128", "129-192", "193-256")
    ]
    late_supported = any(value is not None and value > 0 for value in late_values)
    vision_over_text = max(vision_ci.get("ci95_low", -1), sep_text_ci.get("ci95_low", -1)) > 0
    sep_over_full = sep_full_ci.get("ci95_low", -1) > 0
    causal_supported = causal_ci.get("ci95_low", -1) > 0
    amplified = amplification is not None and amplification > 0
    ep_scaling = ep4_rank is not None and ep8_rank is not None and ep8_rank >= ep4_rank
    if vision_over_text and late_supported and causal_supported and amplified and ep_scaling and sep_over_full:
        verdict = "VISION_DOMINANT_CAPTION_REGIME_SUPPORTED_STRONG"
    elif vision_over_text or (causal_supported and amplified):
        verdict = "VISION_DOMINANT_CAPTION_REGIME_SUPPORTED_PARTIAL"
    elif sep_full_ci.get("beneficial_delta", 0) <= 0 and predictor.get("aggregate", {}).get("P_full", {}).get("1-256", {}).get("expert_histogram_cosine", 0) >= predictor.get("aggregate", {}).get("P_separated", {}).get("1-256", {}).get("expert_histogram_cosine", 0):
        verdict = "GENERIC_FULL_PREFILL_STILL_SUFFICIENT"
    elif predictor.get("aggregate", {}).get("P_text", {}).get("1-256", {}).get("expert_histogram_cosine", 0) >= predictor.get("aggregate", {}).get("P_vision", {}).get("1-256", {}).get("expert_histogram_cosine", 0):
        verdict = "TEXT_SIGNAL_STILL_DOMINANT"
    else:
        verdict = "NO_REGIME_AMPLIFICATION"
    overall = lengths.get("overall", {})
    _write("final_summary.md", f"""
# Final summary

This bounded study used a frozen, balanced 200-image cohort and the exact same raw prompt for every request. It preserves K8 routing and implements no optimization method.

BASELINE_STATUS: {baseline_status}

IMAGE_COHORT_STATUS: FROZEN_{cohort.get('request_count')}_UNIQUE_IMAGES_5X40; EP4_EXECUTED={len(results)}

UNIVERSAL_PROMPT_STATUS: {'PASS_BYTE_IDENTICAL' if observed_hashes == [prompt_hash] else 'FAIL'}

TEXT_SIGNATURE_LEAKAGE_STATUS: CONTEXTUAL_IMAGE_POSITION_VARIATION_AUDITED; P_TEXT_GAP={_f(text_sep.get('between_minus_within'))}

DOCCI_RESULT: N={lengths.get('docci', {}).get('n')}; MEDIAN_OUTPUT={_f(lengths.get('docci', {}).get('median'), 1)}

TEXTCAPS_RESULT: N={lengths.get('textcaps', {}).get('n')}; MEDIAN_OUTPUT={_f(lengths.get('textcaps', {}).get('median'), 1)}

NOCAPS_RESULT: N={lengths.get('nocaps', {}).get('n')}; MEDIAN_OUTPUT={_f(lengths.get('nocaps', {}).get('median'), 1)}

DENSE_RELATION_RESULT: SUBSTITUTE_MMSTAR; N={lengths.get('dense_relation', {}).get('n')}; MEDIAN_OUTPUT={_f(lengths.get('dense_relation', {}).get('median'), 1)}

SCENE_LAYOUT_RESULT: SUBSTITUTE_MMSTAR; N={lengths.get('scene_layout', {}).get('n')}; MEDIAN_OUTPUT={_f(lengths.get('scene_layout', {}).get('median'), 1)}

CAPTION_OUTPUT_LENGTH: MEAN={_f(overall.get('mean'), 1)}; MEDIAN={_f(overall.get('median'), 1)}; P90={_f(overall.get('p90'), 1)}; GE64={overall.get('at_least_64')}; GE128={overall.get('at_least_128')}; TRUNCATED={overall.get('truncated')}

P_TEXT_RESULT: EXPERT_COSINE={_f(predictor.get('aggregate', {}).get('P_text', {}).get('1-256', {}).get('expert_histogram_cosine'))}

P_VISION_RESULT: EXPERT_COSINE={_f(predictor.get('aggregate', {}).get('P_vision', {}).get('1-256', {}).get('expert_histogram_cosine'))}

P_FULL_RESULT: EXPERT_COSINE={_f(predictor.get('aggregate', {}).get('P_full', {}).get('1-256', {}).get('expert_histogram_cosine'))}

P_SEPARATED_RESULT: EXPERT_COSINE={_f(predictor.get('aggregate', {}).get('P_separated', {}).get('1-256', {}).get('expert_histogram_cosine'))}

VISION_INCREMENT_VS_TEXT: P_VISION_CI={json.dumps(vision_ci, sort_keys=True)}; P_SEPARATED_CI={json.dumps(sep_text_ci, sort_keys=True)}

VISION_INCREMENT_VS_FULL: {json.dumps(sep_full_ci, sort_keys=True)}

IMAGE_SWAP_EXPERT_CHANGE: CROSS_DOMAIN_EP4={_pct(caption_change)}; SAME_DOMAIN_EP4={_pct(_causal_value(causal, 'same_domain', '1-256', 4, 'expert_changed_fraction'))}; REPEAT={_pct(_causal_value(causal, 'repeat_noise', '1-256', 4, 'expert_changed_fraction'))}

REASONING_REGIME_IMAGE_SWAP_CHANGE: {_pct(reasoning_change)}

LATE_DECODE_VISION_PERSISTENCE: {'SUPPORTED' if late_supported else 'NOT_CONFIRMED'}; WINDOWS={late_values}

B8_MIXED_DOMAIN_ORACLE_HEADROOM: EP4={_pct(oracle_headroom)}

EP2_CHANGED_RANK: {_pct(ep2_rank)}

EP4_CHANGED_RANK: {_pct(ep4_rank)}

EP8_STRUCTURAL_CHANGED_RANK: {_pct(ep8_rank)}; STRUCTURAL_EP8_PROJECTION_ONLY

EP_SCALING_INTERPRETATION: {'FINER_MAPPING_INCREASES_OR_PRESERVES_RANK_MANIFESTATION' if ep_scaling else 'NOT_MONOTONIC_OR_UNRESOLVED'}

VIDEO_STATUS: {video.get('status')}

VIDEO_VISION_INCREMENT: NOT_RUN

VIDEO_EP8_STRUCTURAL_RESULT: NOT_RUN

VERDICT: {verdict}

NEXT_ACTION: Stop at evidence discovery. Do not implement skipping, scheduling, placement, routing modification, replication, or production runtime in this PoC.
""")
    atomic_json(args.artifacts / "final_verdict.json", {
        "verdict": verdict,
        "baseline_status": baseline_status,
        "vision_over_text": vision_over_text,
        "separated_over_full": sep_over_full,
        "causal_above_noise": causal_supported,
        "caption_minus_reasoning_expert_change": amplification,
        "late_persistence": late_supported,
        "ep_scaling": ep_scaling,
        "video_status": video.get("status"),
        "optimization_implemented": False,
    })


if __name__ == "__main__":
    main()
