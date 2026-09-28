from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..contract import read_json


def effect(value: dict[str, Any]) -> str:
    ci = value.get("ci95")
    return f"{value['mean']:.6f} (request-bootstrap 95% CI {ci[0]:.6f}, {ci[1]:.6f})"


def comparison(value: dict[str, Any]) -> str:
    return (
        f"candidate damage {value['candidate_damage']:.6f}; "
        f"router-tail {value['tail_damage']:.6f}; random {value['random_damage']:.6f}; "
        f"gain vs tail {effect(value['vs_router_tail'])}; "
        f"gain vs random {effect(value['vs_random'])}"
    )


def write(root: Path, name: str, text: str) -> None:
    path = root / "reports" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    audit = read_json(root / "artifacts/baseline_audit.json")
    cohort = read_json(root / "artifacts/manifests/frozen_campaign_cohort.json")
    discovery = read_json(root / "artifacts/discovery/discovery_matrix.json")
    functional = read_json(root / "artifacts/discovery/h1_functional_summary.json")
    frozen = read_json(root / "artifacts/manifests/frozen_primary_candidate.json")
    confirmation_matrix = read_json(root / "artifacts/confirmation/confirmation_matrix.json")
    final = read_json(root / "artifacts/final_verdict.json")

    write(root, "baseline_readiness.md", f"""
# Baseline readiness

- Status: **{audit['status']}**.
- Base commit: `{audit['base_commit']}`; model revision `{audit['model_revision']}`.
- Model contract: BF16, d={audit['shape']['d']}, m={audit['shape']['m']}, E={audit['shape']['experts']}, K={audit['shape']['top_k']}, {audit['shape']['layers']} layers.
- Physical evidence path: GPUs 2,3,6,7, TRUE TP2-DP2-EP4-PP1, DeepEP HT, 32 experts/rank.
- Rich traces and exact interventions use current checkpoint weights. Offline output-vector replay and structural rank accounting are not latency measurements.
""")

    write(root, "cohort_and_controls.md", f"""
# Frozen cohorts and controls

- Discovery: **{cohort['split_counts']['discovery']}** requests; confirmation: **{cohort['split_counts']['confirmation']}** fresh requests.
- Image hashes are unique and split-disjoint; prior Q2-2/Q2-3 hashes were excluded before any route/damage inspection.
- Discovery domains: `{cohort['domain_counts']['discovery']}`.
- Confirmation domains: `{cohort['domain_counts']['confirmation']}`.
- Conditions: original image, same-domain token-count-matched image swap, and semantic-neutral same-geometry image.
- Question and teacher-forced token IDs are fixed across conditions. Layers `{cohort['layers']}` and steps `{cohort['steps']}` were frozen in advance.
- Confirmation was frozen before primary selection: `{cohort['confirmation_frozen_before_primary_selection']}`.
""")

    h1 = discovery["h1"]
    write(root, "h1_redundancy.md", f"""
# H1 — visual-conditioned concurrent expert redundancy

Exploratory original-minus-neutral effects (positive means more redundancy in the original condition):

- lower effective rank: {effect(h1['effective_rank_lower_is_redundant'])}
- mean pairwise output cosine: {effect(h1['mean_output_cosine'])}
- larger 0.95-cosine cluster: {effect(h1['largest_cluster'])}
- lower leave-one-out reconstruction residual: {effect(h1['reconstruction_residual_lower_is_redundant'])}

The metrics are internally mixed: effective rank/cosine move slightly toward redundancy, but reconstruction residual moves in the opposite direction and no near-duplicate cluster grows. Exact four-step K6 functional probes show:

- redundancy: {comparison(functional['comparisons']['redundancy_k6'])}
- redundancy + rank closure: {comparison(functional['comparisons']['redundancy_rank_k6'])}

Thus H1 produced a small discovery structural effect but failed the skip-utility gate. It was frozen to distinguish reproducible-but-useless structure from a discovery fluctuation on untouched data.
""")

    h2 = discovery["h2"]
    write(root, "h2_persistence.md", f"""
# H2 — decode-step routing persistence

- top-k persistence original-minus-neutral: {effect(h2['topk_persistence'])}
- router-distribution persistence: {effect(h2['router_persistence'])}
- EP-rank-set persistence: {effect(h2['rank_persistence'])}
- persistence-selected K6 gain vs router-tail: {effect(h2['local_skip_k6']['vs_router_tail'])}
- persistence-selected K6 gain vs random: {effect(h2['local_skip_k6']['vs_random'])}

No request-bootstrap CI establishes a positive modality-conditioned persistence effect, and the diagnostic local skip choice is worse than router-tail.
""")

    h3 = discovery["h3"]
    write(root, "h3_circuits.md", f"""
# H3 — cross-layer expert collaboration / circuit structure

- transition-motif persistence original-minus-neutral: {effect(h3['motif_persistence'])}
- circuit-selected K6 gain vs router-tail: {effect(h3['local_skip_k6']['vs_router_tail'])}
- circuit-selected K6 gain vs random: {effect(h3['local_skip_k6']['vs_random'])}

A tiny motif-persistence shift is measurable in discovery, but the selected paths have substantially higher local damage than router-tail. This is not a usable skipping circuit.
""")

    h4 = discovery["h4"]
    write(root, "h4_rank_occupancy.md", f"""
# H4 — modality-conditioned EP-rank occupancy

- B8 singleton-expert effect: {effect(h4['singleton_experts'])}
- rank max/mean effect: {effect(h4['rank_imbalance'])}
- working-set HHI effect: {effect(h4['working_set_hhi'])}
- mean max-rank active-expert reduction at K6: rank-closure {h4['rank_closure_k6']['mean_max_rank_active_expert_reduction']:.3f}, router-tail {h4['tail_closure_k6']['mean_max_rank_active_expert_reduction']:.3f}, random {h4['random_closure_k6']['mean_max_rank_active_expert_reduction']:.3f} experts.
- exact K6 rank-closure probe: {comparison(functional['comparisons']['rank_closure_k6'])}

Rank-aware choice creates more structural closure, reproducing the generic Q1/Q3 premise, but its modality effect is not robust and K6 exact damage is not better than generic controls.
""")

    h5 = discovery["h5"]
    write(root, "h5_router_geometry.md", f"""
# H5 — activation-budget geometry

- router entropy original-minus-neutral: {effect(h5['router_entropy'])}
- effective-K effect: {effect(h5['effective_k'])}
- full-distribution top-8 mass effect: {effect(h5['topk_mass'])}

All CIs include zero or are numerically negligible. No modality-specific entropy/margin/effective-K budget rule was promoted.
""")

    score_rows = "\n".join(
        f"| {key} | {value:.6f} | {'yes' if value > 0 else 'no'} |"
        for key, value in discovery["candidate_conservative_scores"].items()
    )
    write(root, "discovery_matrix.md", f"""
# Discovery matrix

The score is the conservative minimum of a modality-effect lower bound and a diagnostic K6 skip-gain lower bound. Values from different mechanisms are only a gate/ranking aid, not a shared physical unit.

| Hypothesis | conservative score | both lower bounds positive |
|---|---:|---|
{score_rows}

The frozen candidate is `{frozen['primary_hypothesis']}` / `{frozen['policy']}`. Rationale: {frozen['selection_rationale']}.

H1–H5 were all executed even though H1 did not pass the functional gate. No derived H6/H7 was added because the trace exposed no strong, direct-to-intervention signal beyond the predeclared set.
""")

    f6 = final["functional"]["6"]
    write(root, "final_confirmation.md", f"""
# Untouched fresh confirmation

- Frozen primary: `{final['primary_hypothesis']}` / `{final['primary_policy']}`; confirmation requests: {final['structural_effect']['requests']}.
- G1 structural modality effect: **{final['gates']['G1_modality']}**, {effect(final['structural_effect'])}.
- G2 exact K6 skip utility: **{final['gates']['G2_skip_utility']}**; gain vs router-tail {effect(f6['vs_router_tail'])}; gain vs random {effect(f6['vs_random'])}.
- G3 domain robustness: **{final['gates']['G3_robustness']}**; positive domains {final['domain_stability']['positive_domains']}/{final['domain_stability']['domains']}; `{final['domain_stability']['domain_means']}`.
- K6 predicted closure events/request: primary {final['rank_closure']['candidate_k6']:.3f}, rank-aware variant {final['rank_closure']['rank_candidate_k6']:.3f}, router-tail {final['rank_closure']['router_tail_k6']:.3f}, random {final['rank_closure']['random_k6']:.3f}.
- Verdict: **{final['verdict']}**.

The functional damage contains target-logprob drop, Jensen–Shannon divergence, and top-1 flip over a four-token teacher-forced horizon. It is not a correlation-only result.
""")

    runtime = "NOT_RUN_CONFIRMATION_GATE_FAILED" if not final["true_ep4_runtime_promoted"] else "NOT_RUN_NO_PHYSICAL_RAGGED_SKIP_KERNEL"
    primary_ci = final["structural_effect"]["ci95"]
    summary = f"""
# MLLM decode modality discovery campaign — final summary

## Outcome

The campaign found small image-conditioned changes in concurrent expert-output geometry and cross-layer motif persistence, but none produced a safer expert choice than router-tail on exact functional interventions. The frozen H1 mechanism was tested on a fresh image-hash-disjoint cohort. Its modality and skip gates are reported independently; generic rank-aware closure is not relabeled as an MLLM-specific mechanism.

## Evidence boundaries

- `MEASURED`: TRUE EP4 route/output trace and exact drop-only semantic interventions.
- `DIAGNOSTIC`: current-weight expert-output vector replay and B8 structural rank accounting.
- `NOT_RUN`: physical ragged expert-work removal and latency; no TPOT/E2E speedup is claimed.
- Failed setup attempts (wrong Python environment and a rank-local renormalization bug) were retained locally, fixed, and superseded by the marker-audited v5 run. The final evidence uses only successful runs.

## Required summary

CAMPAIGN_GPU_SET: 2,3,6,7
DISCOVERY_REQUESTS: {cohort['split_counts']['discovery']}
CONFIRMATION_REQUESTS: {cohort['split_counts']['confirmation']}

H1_VISUAL_CONDITIONED_REDUNDANCY: discovery structural shift; exact skip utility failed; held-out G1={final['gates']['G1_modality']}
H2_TEMPORAL_LAYER_PERSISTENCE: no robust modality effect; K6 worse than router-tail
H3_EXPERT_CIRCUIT_STRUCTURE: tiny discovery motif shift; skip link failed
H4_MODALITY_RANK_OCCUPANCY: generic closure headroom present; modality-specific/functional advantage absent
H5_ROUTER_GEOMETRY: no robust entropy/effective-K/top-8-mass shift

PRIMARY_DISCOVERED_SIGNAL: {frozen['primary_hypothesis']} — {frozen['mechanism']}
PRIMARY_MODALITY_EFFECT: {effect(final['structural_effect'])}
PRIMARY_SKIP_GAIN: K6 vs router-tail {effect(f6['vs_router_tail'])}; vs random {effect(f6['vs_random'])}
PRIMARY_RANK_CLOSURE_GAIN: rank-aware K6 {final['rank_closure']['rank_candidate_k6']:.3f} vs primary {final['rank_closure']['candidate_k6']:.3f} predicted closure events/request
PRIMARY_HELDOUT_CI: [{primary_ci[0]:.6f}, {primary_ci[1]:.6f}]

TRUE_EP4_RUNTIME_STATUS: {runtime}
FINAL_VERDICT: {final['verdict']}
NEXT_ACTION: preserve the generic rank-aware Q1/Q3 result; do not continue static vision-score skipping. If MLLM specificity remains required, test discrete decoder events with direct counterfactual functional labels rather than another scalar visual proxy.
"""
    write(root, "campaign_summary.md", summary)
    write(root, "final_summary.md", summary)

    # A compact JSON-facing audit of the held-out H1 diagnostics is useful for
    # readers who want to compare discovery and confirmation without parsing
    # markdown.
    (root / "artifacts/confirmation/confirmation_matrix_excerpt.json").write_text(
        json.dumps({
            "h1": confirmation_matrix["h1"],
            "h2": confirmation_matrix["h2"],
            "h3": confirmation_matrix["h3"],
            "h4": confirmation_matrix["h4"],
            "h5": confirmation_matrix["h5"],
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
