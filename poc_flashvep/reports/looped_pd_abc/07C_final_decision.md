# Track C: decision

Reviewed decision file present.

Timing correction: legacy foreground-I/O runs are not valid clean method comparisons. Read `timing_correction.md`. Accepted comparisons require the registered nonblocking protocol.

Active model scope: ouro, huginn. Dory is deferred by explicit user instruction. Existing Dory evidence remains historical, not a new verdict.

Campaign status: `FINALIZED_WITH_LIMITATIONS`. Evidence snapshot: 2026-10-29T11:01:16.077838+00:00.

Overall status: `NO_GO_CORRECTNESS`. Natural batched placement changes output sequences. The repeated service-time ablation also lacks a demonstrated incremental gain. No lossless scheduling speedup is accepted.

| Model | Status | Reason |
| --- | --- | --- |
| ouro | NO_GO_CORRECTNESS | Loop versus service-time placement matches 125/144 outputs and gives 0.981x p95 E2E. A one-start synthetic four-GPU signal does not repair the natural-output gate or prove loop-specific novelty. |
| huginn | NO_GO_CORRECTNESS | Loop versus service-time placement matches 139/144 outputs and gives 1.016x p95 E2E with a 0.977-1.093x conditional CI. No lossless incremental gain is established. |

| Model / regime | Baseline | Candidate | Starts / frozen pairs | Exact outputs / N | Baseline p95 E2E ms | Candidate p95 E2E ms | Speedup | Paired 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ouro / C / P0 C8 | C_upstream | C_loop | 3 / 3 | 127/144 | 4040.493 | 4086.386 | 0.989× | 0.744–1.867× |
| ouro / C / P0 C8 | C_queue | C_loop | 3 / 3 | 123/144 | 3999.266 | 4086.386 | 0.979× | 0.548–1.824× |
| ouro / C / P0 C8 | C_token | C_loop | 3 / 3 | 124/144 | 4064.368 | 4086.386 | 0.995× | 0.557–1.629× |
| ouro / C / P0 C8 | C_service | C_loop | 3 / 3 | 125/144 | 4009.910 | 4086.386 | 0.981× | 0.716–1.843× |
| ouro / C / P0 C8 | C_DP4 | C_loop | 3 / 3 | 117/144 | 4101.735 | 4086.386 | 1.004× | 0.461–1.953× |
| ouro / C / P0 C8 | C_DP4_Mixed | C_loop | 3 / 3 | 116/144 | 4255.653 | 4086.386 | 1.041× | 0.329–2.392× |
| ouro / C / P0 C8 | C_PD1 | C_loop | 3 / 3 | 119/144 | 5520.884 | 4086.386 | 1.351× | 1.082–2.417× |
| ouro / C / P0 C8 | C_PD3 | C_loop | 3 / 3 | 128/144 | 2316.379 | 4086.386 | 0.567× | 0.478–1.111× |
| huginn / C / P0 C8 | C_upstream | C_loop | 3 / 3 | 137/144 | 6420.779 | 6635.428 | 0.968× | 0.948–0.998× |
| huginn / C / P0 C8 | C_queue | C_loop | 3 / 3 | 141/144 | 6897.639 | 6635.428 | 1.040× | 0.979–1.111× |
| huginn / C / P0 C8 | C_token | C_loop | 3 / 3 | 135/144 | 6869.298 | 6635.428 | 1.035× | 1.007–1.083× |
| huginn / C / P0 C8 | C_service | C_loop | 3 / 3 | 139/144 | 6741.919 | 6635.428 | 1.016× | 0.977–1.093× |
| huginn / C / P0 C8 | C_DP4 | C_loop | 3 / 3 | 137/144 | 7197.541 | 6635.428 | 1.085× | 0.905–1.152× |
| huginn / C / P0 C8 | C_DP4_Mixed | C_loop | 3 / 3 | 137/144 | 7205.596 | 6635.428 | 1.086× | 0.984–1.128× |
| huginn / C / P0 C8 | C_PD1 | C_loop | 3 / 3 | 137/144 | 6297.944 | 6635.428 | 0.949× | 0.923–0.966× |
| huginn / C / P0 C8 | C_PD3 | C_loop | 3 / 3 | 134/144 | 8674.593 | 6635.428 | 1.307× | 1.195–1.364× |

## Limits

Official-reference sequence agreement is incomplete for both active models.
Full GSM8K test accuracy and HumanEval accuracy are `NOT_MEASURED`.
A closed-loop cohort does not establish sustainable open-loop capacity.
No global queue simulator or infinite-bandwidth serving oracle is validated.
GPU kernel times and overlapping transfer durations must not be added to E2E.
Short profiled runs are separate from clean serving runs.

## Final review: strongest controls and scope

| Model | Candidate | Input / concurrency / cap | Best measured E2E baseline | Baseline p95 E2E ms | Candidate p95 E2E ms | Speedup [95% CI] | Exact outputs / evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| huginn | C_loop | natural / 8 / 64 | C_PD1 | 6297.944 | 6635.428 | 0.949× [0.923–0.966×] | 137/144 |
| ouro | C_loop | natural / 8 / 64 | C_PD3 | 2316.379 | 4086.386 | 0.567× [0.478–1.111×] | 128/144 |

No original-model-equivalent serving GO is accepted. Read `09_final_summary.md` for the full limits.
One-start scale points do not supply an accepted confidence interval.
The full frozen SLO grid remains available. Favorable cells are not a new acceptance rule.
