from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..contract import read_json, read_jsonl_gz


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def fmt(value: object, digits: int = 4) -> str:
    if value is None: return "N/A"
    if isinstance(value, float): return f"{value:.{digits}f}"
    return str(value)


def ci(result: dict) -> str:
    bounds = result["bootstrap"]["ci95"]
    return f"{fmt(result['spearman'])} [{fmt(bounds[0])}, {fmt(bounds[1])}]" if bounds else "N/A"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--verdict", type=Path, required=True)
    parser.add_argument("--distributional", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    baseline, manifest, gate, verdict, distributional = map(
        read_json, (args.baseline, args.manifest, args.gate, args.verdict, args.distributional)
    )
    signals, labels = list(read_jsonl_gz(args.signals)), list(read_jsonl_gz(args.labels))
    held = [row for row in labels if row["split"] == "heldout" and row["condition"] == "orig"]
    by_domain = defaultdict(list)
    for row in held: by_domain[row["domain"]].append(row)
    write(args.reports / "baseline_readiness.md", f"""
# Baseline readiness

Status: **{baseline['status']}**.

- Model: Qwen3-VL-30B-A3B-Instruct `{baseline['model_revision']}`, BF16, 48 layers, E128, K8, d2048/m768.
- Physical contract: TRUE TP2-DP2-EP4-PP1 / DeepEP on GPUs 2,3,6,7.
- Prior bounded evidence: Q1={baseline['prior_q1']['verdict']}; Q3={baseline['prior_q3']['verdict']}; Q2-2={baseline['prior_q22_verdict']}.
- This run is instrumented semantic evidence. TPOT, E2E, and physical saving were not measured.
""")
    write(args.reports / "cohort_and_contract.md", f"""
# Frozen cohort and intervention contract

- Calibration: {manifest['calibration_requests']} requests; fresh held-out: {manifest['fresh_heldout_requests']} requests.
- Domain counts: `{manifest['domain_counts']}`.
- The universal prompt is byte-identical (`{manifest['universal_prompt_sha256']}`).
- Q2/Q2-2 request and image identities were excluded before capture.
- Same-domain counterfactuals were matched within 10% visual-token count.
- One predeclared layer/step site per request; selection used no routing, grounding, or damage outcome.
- Diagnostic reduction is native K8→K6/K4 tail removal, retained-weight renormalization, and no substitution.
""")
    write(args.reports / "visual_subspace_signal.md", f"""
# Single-pass visual grounding signal

The frozen score is `log((E_vision+1e-6)/(E_text+1e-6))`, where energies project the current decode pre-MoE hidden onto rank-4 visual/text prefill subspaces from the same layer. At most 16 rows per modality and TP shard were retained.

- Captured condition-sites: {len(signals)}.
- Held-out visual-score mean/median: {fmt(float(np.mean([r['separated_visual_score'] for r in held])))} / {fmt(float(np.median([r['separated_visual_score'] for r in held])))}.
- Held-out feature/damage Spearman: `{gate['heldout_all_feature_spearman']}`.
- Strongest control was frozen using calibration only: `{gate['control_selected_on_calibration_only']}`.
""")
    domain_lines = "\n".join(
        f"- {domain}: n={len(rows)}, K6 damage={fmt(float(np.mean([r['k6_damage'] for r in rows])))}, K4 damage={fmt(float(np.mean([r['k4_damage'] for r in rows])))}"
        for domain, rows in sorted(by_domain.items())
    )
    write(args.reports / "structured_reduction_damage.md", f"""
# Structured K8→K6/K4 functional damage

- Exact marker audit: `{gate['marker_audit']}`.
- Held-out K6 damage mean/median: {fmt(float(np.mean([r['k6_damage'] for r in held])))} / {fmt(float(np.median([r['k6_damage'] for r in held])))}.
- Held-out K4 damage mean/median: {fmt(float(np.mean([r['k4_damage'] for r in held])))} / {fmt(float(np.median([r['k4_damage'] for r in held])))}.
- Held-out structured damage mean/median: {fmt(float(np.mean([r['structured_damage'] for r in held])))} / {fmt(float(np.median([r['structured_damage'] for r in held])))}.

{domain_lines}

These are exact downstream-logit effects of semantic zero-weight interventions. They are not evidence that the physical sparse kernel skipped those rows.
""")
    inc = gate["increment_vs_frozen_control"]
    inc_ci = inc["bootstrap"]["ci95"]
    write(args.reports / "causal_grounding_gate.md", f"""
# Held-out visual-grounding gate

All intervals use 10,000 request-bootstrap repeats.

1. Original visual score vs structured damage: {ci(gate['association'])}; pass={gate['association']['pass']}.
2. Image-swap score delta vs damage delta: {ci(gate['causal_delta'])}; pass={gate['causal_delta']['pass']}.
3. Increment over calibration-frozen `{gate['control_selected_on_calibration_only']}`: {fmt(inc['spearman_difference'])} [{fmt(inc_ci[0] if inc_ci else None)}, {fmt(inc_ci[1] if inc_ci else None)}]; pass={inc['pass']}.

Joint Phase-1 gate: **{'PASS' if gate['gate_pass'] else 'FAIL'}**. Phase 2 was {'authorized but deliberately not implemented in this bounded task' if gate['gate_pass'] else 'not promoted'}.
""")
    nonlinear = distributional["original_condition"]
    causal_nonlinear = distributional["absolute_causal_delta"]
    write(args.reports / "nonlinear_distributional_diagnostic.md", f"""
# Exploratory nonlinear/distributional diagnostic

This was requested after partial held-out correlation was observed. It is
therefore exploratory and cannot alter the frozen verdict. All damage
thresholds, feature tertiles, and three-bin risk maps were nevertheless fit on
the 8 calibration requests only and evaluated on the 24 held-out requests.

| Feature | Original AUROC | Original AP | Damage-distribution JSD | Absolute causal-delta AUROC | Absolute causal-delta AP |
|---|---:|---:|---:|---:|---:|
| visual/text separated | {fmt(nonlinear['separated_visual_score']['heldout_auroc'])} | {fmt(nonlinear['separated_visual_score']['heldout_average_precision'])} | {fmt(nonlinear['separated_visual_score']['heldout_low_vs_high_damage_histogram_jsd'])} | {fmt(causal_nonlinear['separated_visual_score']['heldout_auroc'])} | {fmt(causal_nonlinear['separated_visual_score']['heldout_average_precision'])} |
| text energy | {fmt(nonlinear['text_energy']['heldout_auroc'])} | {fmt(nonlinear['text_energy']['heldout_average_precision'])} | {fmt(nonlinear['text_energy']['heldout_low_vs_high_damage_histogram_jsd'])} | {fmt(causal_nonlinear['text_energy']['heldout_auroc'])} | {fmt(causal_nonlinear['text_energy']['heldout_average_precision'])} |
| full energy | {fmt(nonlinear['full_energy']['heldout_auroc'])} | {fmt(nonlinear['full_energy']['heldout_average_precision'])} | {fmt(nonlinear['full_energy']['heldout_low_vs_high_damage_histogram_jsd'])} | {fmt(causal_nonlinear['full_energy']['heldout_auroc'])} | {fmt(causal_nonlinear['full_energy']['heldout_average_precision'])} |
| router removed mass | {fmt(nonlinear['router_removed_mass_control']['heldout_auroc'])} | {fmt(nonlinear['router_removed_mass_control']['heldout_average_precision'])} | {fmt(nonlinear['router_removed_mass_control']['heldout_low_vs_high_damage_histogram_jsd'])} | {fmt(causal_nonlinear['router_removed_mass_control']['heldout_auroc'])} | {fmt(causal_nonlinear['router_removed_mass_control']['heldout_average_precision'])} |

The visual score did not dominate the controls: full-prefill achieved the
higher original-condition AUROC/AP and much larger low-vs-high damage JSD;
router removed mass achieved the higher causal-delta AUROC. Thus nonlinear
distributional inspection does not provide a credible rescue signal.
""")
    write(args.reports / "final_summary.md", f"""
# Final summary — visual-grounding budget-risk Phase 1

The bounded result is **{verdict['verdict']}**. The primary score was frozen before damage measurement, evaluated on 24 fresh image-hash-disjoint held-out requests, and required both observational and same-token image-swap causal-delta evidence plus incremental value over a calibration-frozen generic control. MoDES was not compared, and no expert-selection or latency runtime was implemented.

```text
BASELINE_STATUS: {baseline['status']}
COHORT_STATUS: FROZEN_8_CALIBRATION_24_FRESH_HELDOUT
HELDOUT_REQUESTS: {len(held)}
PRIMARY_VISUAL_SCORE: LOG_VISUAL_TO_TEXT_PREFILL_SUBSPACE_ENERGY_RATIO
K6_DAMAGE_MEAN: {fmt(float(np.mean([r['k6_damage'] for r in held])))}
K4_DAMAGE_MEAN: {fmt(float(np.mean([r['k4_damage'] for r in held])))}
VISUAL_SCORE_DAMAGE_SPEARMAN_CI: {ci(gate['association'])}
CAUSAL_DELTA_SPEARMAN_CI: {ci(gate['causal_delta'])}
BEST_FROZEN_CONTROL: {gate['control_selected_on_calibration_only']}
VISUAL_INCREMENT_CI: {fmt(inc['spearman_difference'])} [{fmt(inc_ci[0] if inc_ci else None)}, {fmt(inc_ci[1] if inc_ci else None)}]
PHASE1_GATE: {'PASS' if gate['gate_pass'] else 'FAIL'}
PHASE2_PROMOTION: {'AUTHORIZED_NOT_IMPLEMENTED' if gate['phase2_promoted'] else 'NO'}
MODES_COMPARISON: NOT_RUN_BY_USER_SCOPE
EXPLORATORY_NONLINEAR_RESULT: NO_VISUAL_INCREMENTAL_RESCUE
TPOT_E2E_CLAIM: NOT_RUN
VERDICT: {verdict['verdict']}
NEXT_ACTION: {verdict['next_action']}
```
""")


if __name__ == "__main__": main()
