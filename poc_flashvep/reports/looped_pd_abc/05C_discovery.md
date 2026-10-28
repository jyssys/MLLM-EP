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
The resulting completion scores are `MODEL_PREDICTION`, not measured future service times.

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

The implementation also compares colocated DP4, colocated Mixed DP4, and static P1:D3/P3:D1.
The four-GPU main placement comparison uses P2:D2.
There is no placement choice in an isolated P1:D1 pool.
Its initial routing headroom is structurally zero.

## Measurement limits

Route logs record worker scores, pending counts, and selector CPU time.
The reported selector time excludes the subsequent synchronous log write.
Actual E2E includes that write.
The report must not describe selector CPU time as the complete scheduling overhead.

The baseline cost model is affine and hardware-calibrated.
It is not a validated global queue simulator.
Prediction residuals and held-out incremental value remain unverified.
No oracle speedup is accepted from this model alone.

The experiment must show a live gain over the tuned service-time control.
A gain over queue-only placement does not establish loop-specific value.
The strongest equal-budget colocated and static-role baselines also remain required controls.
