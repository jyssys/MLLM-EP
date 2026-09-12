# Vanilla dLLM MoE EP latency deep PoC

This directory contains the independent-vanilla study described by
`../poc_flashvep/reports/vanilla_dllm_moe_ep_latency_deep_poc.md`.

## Evidence map

- `reports/01_*`: fair Single/EP2/EP4 fixed-work result and stage localization.
- `reports/02_*`: EP4 temporal route/state structure.
- `reports/03_*` through `10_*`: candidate oracles and kill decisions.
- `reports/11_prior_art_audit.md`: adversarial novelty boundaries and links.
- `reports/final_decision.md`: gate accounting.
- `results/stage0_clean/`: clean derived metrics and bounded quality.
- `results/analysis/`: temporal summaries and compressed row-level tables.
- `results/oracles/`: measured P2P, elastic, local-draft, and candidate oracles.
- `results/plots/`: the 15 pre-registered diagnostics.

Raw per-rank trace JSON is excluded from Git because it is observer-heavy and
approximately 3.4 GB. It remains in the local result tree. The compressed
row-level temporal tables and all aggregate data needed to reproduce the report
are versioned.

## Reproduction order

All launchers hard-code the authorized physical-GPU subsets and the runner
validates CUDA UUIDs before model execution.

1. `scripts/run_stage0_clean.sh`
2. `scripts/score_stage0.sh` and `scripts/analyze_stage0.py`
3. `scripts/run_stage1_instrumented.sh` and `scripts/analyze_instrumented.py`
4. `scripts/run_stage1_timing.sh`
5. `scripts/run_rank_local_draft.sh`
6. `scripts/run_ep_degree_microbench.sh`
7. `scripts/finalize_analysis.py`

The final label is `CHARACTERIZATION-ONLY`; no method prototype was allowed
past the oracle gate.
