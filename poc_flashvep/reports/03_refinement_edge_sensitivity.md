# Refinement edge sensitivity

Status: `NOT_RUN_AFTER_BASELINE_GATE`.
Runtime and model identities appear in `00_runtime_audit.md`.
The fresh full baseline evaluation remains in progress.
No approximate policy passed the quality gate yet.

## Frozen partitions

| Split | Source | N | Labels available to online policy? |
|---|---|---:|---|
| Calibration | GSM8K training | 256 | No |
| Validation | Different GSM8K training questions | 128 | No |
| Final test | GSM8K test | 1319 | No |
| Final test | HumanEval | 164 | No |

The split seed is 20261027.
The preparation script asserts calibration/validation disjointness.
It stores validation answers separately for offline scoring.
It freezes every prompt manifest with SHA-256.

## Required test

Record previous-iteration confidence only after that iteration completes.
Record its availability before the next MoE execution.
Use the fresh current router for every candidate edge.
Preserve at least one routed edge per token.
Leave attention, shared experts, residuals, and original aggregation weights unchanged.

Stratify sensitivity by layer, block iteration, current mask status, and confidence history.
Compare isolated logit perturbations with complete changed generation trajectories.
Report unmask changes, final answer changes, NFE, and EOS status.
Do not infer quality from local hidden similarity alone.

| Method | Validation quality | Edge reduction | Logit change | Unmask change | Full-trajectory quality |
|---|---|---|---|---|---|
| Uniform P1 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| Router P2 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| Refinement P3 proxy | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| EP-only P4 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| Joint P5 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |

Previous route-pruning failures remain contextual evidence only.
They used different data and dense semantic emulation.
They do not establish this campaign's outcome.
