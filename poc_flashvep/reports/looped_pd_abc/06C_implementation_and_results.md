# Track C: measured implementation results

Reviewed decision file present.

Timing correction: legacy foreground-I/O runs are not valid clean method comparisons. Read `timing_correction.md`. Accepted comparisons require the registered nonblocking protocol.

Active model scope: ouro, huginn. Dory is deferred by explicit user instruction. Existing Dory evidence remains historical, not a new verdict.

Campaign status: `FINALIZED_WITH_LIMITATIONS`. Evidence snapshot: 2026-10-29T11:01:16.077475+00:00.

These are actual GPU measurements from completed runs.
Unfinished and failed runs do not contribute latency values.
Natural generation permits EOS and preserves the declared output cap.
Cap termination is not proof of answer completeness.
A lower latency with different output IDs is not an accepted lossless speedup.

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

## Mechanism and controls

C changes initial placement only. A and B remain disabled.
The main feature ablation compares loop-state placement with calibrated service-time placement.
Both policies receive the same legal progress telemetry.
C2 migration and dynamic role reassignment are NOT_MEASURED.
A one-prefill/one-decode pool has no initial routing choice.

## Limits

Official-reference sequence agreement is incomplete for both active models.
Full GSM8K test accuracy and HumanEval accuracy are `NOT_MEASURED`.
A closed-loop cohort does not establish sustainable open-loop capacity.
No global queue simulator or infinite-bandwidth serving oracle is validated.
GPU kernel times and overlapping transfer durations must not be added to E2E.
Short profiled runs are separate from clean serving runs.
