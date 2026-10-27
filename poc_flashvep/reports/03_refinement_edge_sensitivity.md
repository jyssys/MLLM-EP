# Refinement edge sensitivity

Status: `BASELINE_TRACE_ANALYSIS_COMPLETE_OMISSION_SENSITIVITY_NOT_RUN`.
Runtime and model identities appear in `00_runtime_audit.md`.
The user-approved 128/32 baseline pilot is complete.
Two independent EP4 replicas collected 16 different non-test validation trajectories.
The observer records prior completed-forward confidence before each current dispatch.
It does not change router results or omit expert work.
Observed-versus-bypass output parity remains a required per-request assertion.
This collection does not establish an approximate policy's quality or speedup.
No approximate policy passed the quality gate yet.

## Completed retained-all observations

All 16 requests preserved output tokens and NFE across baseline, no-op, and observer execution.
Each stored confidence feature predates its current dispatch.
The analysis uses the first MoE layer once per forward for commitment labels.
It does not count the same label 19 times across physical layers.

Prior confidence predicts next-iteration commitment with macro AUROC 0.92619.
The question-bootstrap 95% interval is [0.91610, 0.93418], with N=16 requests.
This is `TRACE_ANALYSIS`, not an expert-omission result.
Commit predictability does not establish marginal expert importance.
It also does not establish that expert work can be skipped safely.

The median observer-to-baseline wall ratio is 1.16964 across these 16 requests.
This overhead is approximately 16.96% under this parallel collection setup.
Do not use the instrumented absolute times as clean serving latency.
The rank-local stage shares reside in `pilot/observer_breakdown.csv` under the result directory.
Those shares do not identify the clean global critical path.

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
