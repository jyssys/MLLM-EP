# C: loop-state placement discovery

Status: C1 initial placement implemented. Repeated four-GPU comparison is pending.

## Scope

C changes initial request-to-worker placement at normal PD admission boundaries.
It uses conventional complete-prefill ownership and the upstream transfer policy.
A and B remain disabled.
It does not change the intra-worker loop quantum.
It does not migrate an active request or restart a prompt.

The code implements C1 only.
C2 safe-boundary migration, dynamic role changes, and adaptive model depth are not implemented.
Their results must remain `NOT_MEASURED`.

## Observable information

The connector records each submitted frontier with its CUDA event.
It sends a progress message only after that event reports completion.
The coordinator can therefore use completed loop progress, not speculative submission progress.

The cost model uses prompt length, the fixed configured depth, generated-token count, and the declared output cap.
It never uses a future EOS position or future generated token.
Calibration uses separate warmed GPU runs.
The calibration includes observed batch-size effects on decode time.
The final comparison also includes observed prefill batching effects and uncovered handoff cost.
Both service and loop controls use these same calibrated terms.
The resulting completion scores are `MODEL_PREDICTION`, not measured future service times.

The added loop feature currently covers prefill frontiers only.
Decode-worker placement uses the same emitted-token progress and service model in both controls.
It does not use a decode token's in-flight recurrent position.
The policy also scales total prefill cost by completed recurrent rows.
It does not separately weight prelude, core, and coda time in that progress fraction.
These limits matter particularly for boundary-heavy Dory.
A negative result rejects this C1 variant, not every possible loop-state estimator.

## Strong controls

| Policy | Initial placement signal | Loop progress |
|---|---|---|
| Upstream | Reserved cache blocks, then transfer slots | No |
| Queue | Number of assigned unfinished requests | No |
| Token | Known prompt tokens or remaining output cap | No |
| Service | Calibrated service cost and elapsed worker time | No |
| Loop | Same calibrated costs plus GPU-completed frontiers | Yes |

Upstream is not round-robin placement.
Service and loop policies receive the same progress-message instrumentation.
Only the loop policy uses the frontier feature in its score.
This pair is the main loop-state feature ablation.

Service subtracts elapsed prefill work once per GPU queue.
It does not subtract elapsed service independently from every queued request.
The latter calculation would weaken the generic baseline.
Charging one independent weight reload per batched request would also weaken that baseline.
The final cost model instead uses separately measured prefill batch ratios.

The implementation also compares colocated DP4, colocated Mixed DP4, and static P1:D3/P3:D1.
The four-GPU main placement comparison uses P2:D2.
There is no placement choice in an isolated P1:D1 pool.
Its initial routing headroom is structurally zero.

## Heterogeneous prompt-length control

The original natural-question pilot contains relatively short prompts.
That workload alone cannot test broad prompt-service heterogeneity.
A separate one-start discovery matrix therefore adds mixed prompt lengths at global concurrency 16.
It compares eight policies on each of the three models.
These 24 GPU jobs remain queued until the main C phase.

Each model receives the same deterministic length sequence.
The 48 inputs contain 17 prompts of 128 tokens, one of 256, twelve of 512, and eighteen of 2048.
The 256-token exception preserves a question that does not fit its intended 128-token bucket.
The harness prepends synthetic context and retains the complete question and chat-generation suffix.
It uses real greedy generation with EOS enabled and output cap 64.
It does not freeze output tokens.

`workload_worker.py` changes only the input provider.
It records the manifest hash and its synthetic origin before it starts the frozen worker.
The raw request row retains the worker's original source label.
`workload_provenance.json` supplies the explicit source correction without rewriting that row.
The master table reports this corrected source as `effective_prompt_source`.

The matched queue, token, service, and loop policies retain staged G1 and conventional ownership.
The strong P1:D3/P3:D1 controls retain their normal atomic execution path.
Colocated Mixed DP4 remains a separate strong control.
No one-start result can establish an accepted gain or sustainable capacity.
Any promising point requires independent frozen confirmation.

## Bounded shadow capture before clean C timing

Three queued diagnostic jobs collect legal routing states before the clean C matrix.
Each model uses P2:D2, concurrency 16, 24 heterogeneous inputs, and output cap 16.
The diagnostics retain real generation and unchanged model execution.

At each feasible worker evaluation, `shadow_worker.py` records the current queue and completed prefill frontiers.
It evaluates queue, token, service, and loop scores on that same observed state.
It does not use future tokens, future arrivals, or future GPU completion times.
Alternative scores do not change the actual loop policy's score.
The extra CPU logging can change scheduling time and batch composition.
These runs therefore cannot supply clean latency or goodput comparisons.

`C_shadow_opportunities.csv` separates single-worker rounds from rounds with an actual placement choice.
It also records whether an observed frontier exists and whether service and loop scores select different workers.
A scoring round does not necessarily become an accepted admission.
The analysis excludes warmup requests.
Its one-start results do not establish population rates or a serving speedup.

## Measurement limits

Route logs record worker scores, pending counts, and selector CPU time.
The reported selector time excludes the subsequent synchronous log write.
Actual E2E includes that write.
The report must not describe selector CPU time as the complete scheduling overhead.
This statement describes the original recorder.
The registered timing correction moves durable writes to a background thread.
New selector counters still exclude subsequent serialization and enqueue costs.
`nonblocking_evidence_cost.csv` records those producer costs separately.
The final C calibration and measurements require the registered nonblocking protocol.
An early counter used a batch offset in place of its start timestamp.
The experiment fixed that counter before the repeated C comparison.
A CPU regression test checks that counter.
Earlier smoke-counter values cannot establish selector overhead.

The baseline cost model is affine and hardware-calibrated.
It is not a validated global queue simulator.
Prediction residuals and held-out incremental value remain unverified.
No oracle speedup is accepted from this model alone.

The experiment must show a live gain over the tuned service-time control.
A gain over queue-only placement does not establish loop-specific value.
The strongest equal-budget colocated and static-role baselines also remain required controls.
