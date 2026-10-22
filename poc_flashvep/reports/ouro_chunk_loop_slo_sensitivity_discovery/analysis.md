# Ouro C/G and SLO sensitivity analysis

Final verdict: **CONDITIONAL**.
The registered data capture and CPU analysis are complete.
This is a completed characterization study with explicit limits, not a demonstrated adaptive scheduling method.

| Independent decision | Result |
| --- | --- |
| CHARACTERIZATION | CONDITIONAL |
| ADAPTIVE_JOINT_HEADROOM | CONDITIONAL. No live adaptive comparison |
| MULTIGPU_PORTABILITY | NOT_RUN |

See `decision_gates.md` for the phase-specific reasons.

## Current interpretation

The study measures a C/G distributional trade-off in the pinned RLT runtime.
It does not establish online Joint C/G superiority or justify a new model-parallel engine.
The prior `CONDITIONAL_FIXED_QUANTUM_ONLY` decision remains unchanged.
The paused native-vLLM Ouro Loop PP project remains paused.

The experiment uses pretrained Base Ouro-1.4B BF16.
Every prompt and decode token requires full R4.
G changes prefill yield boundaries, not model depth.
Four physical GPUs4–7 execute independent single-GPU experiments.
They do not execute a single request in TP, PP, or shared DP serving.

## Main measured fixed-grid result

The new study contains 316 bounded continuity cohorts.
The full grid covers C128/256/512/1024, G1/2/4, stock Mixed, and a staged-G4 control.
P≈2K/8K and N4/8 are distinct factors.
Eight shortlisted policies each have ten cohorts at two primary shapes.
All comparisons use same-GPU process blocks with counterbalanced policy order.

At P≈8K/N8/C1024, Grouped G2 reduces mean cohort-p99 ITL by 51.212% relative to Atomic.
The 95% paired interval is [45.036%, 59.925%].
However, mean cohort-p95 rises from 38.845 ms to 130.377 ms with G2.
The mean drained-cohort time reduction is −2.159%, CI [−5.375%, 3.174%].
Thus the large tail reduction is not a measured E2E speedup.
Mixed C512 has a lower mean cohort-p99, 110.277 ms.
Its mean long TTFT is higher, at 1715.895 ms.
G2 C1024 gives 189.721 ms and 1027.939 ms respectively.
The policies favor different objectives.

## Why P99 improves while P95 worsens

Atomic has 253/8890 background gaps above 100 ms.
G2 has 480/8866.
Atomic's violating mass is below 5%, while G2's mass is above 5%.
Atomic has 196 gaps above 200 ms, versus 23 for G2.
G2 replaces some rare severe stalls with more frequent moderate stalls.
That measured distribution explains the opposing quantile movement.
The counts are descriptive, not independent token-level inference.

The source explains an important scheduling dependency.
The fairness counter resets on a recurrent batch, not a completed output token.
One static-R4 output interval can therefore include several prefill submissions.
Smaller C creates more interruptions and repeated core entry.
It does not guarantee a shorter complete output-token gap.

## Why G alone is not the complete cause

Staged G4 C1024 has mean cohort-p99 273.991 ms, versus Atomic's 388.869 ms.
Both use four-loop groups.
Thus the staged execution path itself changes the observed tail.
G2 improves another 30.756% relative to Staged G4, CI [21.428%, 43.868%].
Its drained-cohort benefit over Staged G4 is not resolved: 0.447%, CI [−3.082%, 6.902%].
The complete path, not only C×G, must appear in a cost model.
See `characterization.md`, `profiling.md`, and the separate heavy traces.

The graph setting also needs a precise interpretation.
Prompt-only recurrent calls use the native eager path in this RLT checkout.
Decode graph replay remains enabled.
Mixed can request graphs but exceed the configured query-count limit and fall back.
Thus equal graph settings do not imply equal graph execution.
This limits any claim about a more optimized prefill implementation.
The study does not change that engine behavior or resume native-vLLM development.

## SLO and serving interpretation

The registration freezes Strict/Moderate/Relaxed TBT at 135.946614/271.893227/679.733069 ms.
It also freezes absolute 75/100/125/150/200/250/500 ms and TTFT 2/5/10 s grids.
These are Ouro calibration values, not universal product requirements.
The reference uses warmed solo GPU4, P2033/N4, at 27.189323 ms.
It is not Sarathi's 4K/B32 reference.

The serving queue uses actual natural generation, common arrivals, and all-offered denominators.
It separates every-token request compliance from pooled p99.
It separates completed-in-window goodput, declared-drain goodput, and drained throughput.
It reports configured Poisson rate and realized offered count/window.
Queue growth disqualifies a sustainable-rate claim.
An unfinished in-service request can also fail the finite-window screen without proving divergence.

Short runs below 1,000 requests per arm/rate remain exploratory.
Their point estimates do not establish 90%-attainment production capacity.
Natural output content and length can differ between policies.
Their measured outcomes are not equal-work scheduling speedups.

## Correctness and limits

All 48 isolated component comparisons passed exactly for tokens, depth, prompt hidden/logits, and 96 KV namespaces.
These compare eight generated tokens, not complete reasoning answers.
New P≈8K/N8 bounded G2-versus-Atomic token matches are 72/80.
All 80 depth records and output lengths match.
The historical 112/120 Grouped-versus-Split result is a different comparison and stays unchanged.

The targeted ninth-token diagnostic shows a changed hidden state and a BF16 logit tie.
It does not identify every earlier source of the hidden difference.
It does not exclude graph, lifecycle, or stream-related alternatives.
Full GSM8K/MATH accuracy is NOT_RUN.
The study cannot claim quality-preserving method superiority.

## Research decision boundary

The fixed response surface can motivate further controlled characterization.
It does not establish that a two-dimensional online policy beats tuned C-only and stock Mixed.
The serving shortlist is representative rather than exhaustive per-SLO tuning.
Adaptive C-only and G-only controllers are NOT_RUN.
Their absence limits the joint-selection claim.

Do not implement Adaptive Joint or new TP/PP serving based only on this p99 result.
First resolve batched-output reproducibility with matched hidden and graph-state probes.
Then test any surviving SLO region against stronger fixed and one-dimensional controls.
Increase requests near a plausible capacity boundary before a capacity claim.

## Completed natural-serving evidence

The study executed 128 Poisson policy runs and 48 burst policy runs.
They contain 6,880 offered request instances across the eight policy arms.
These are not 6,880 independent source families.
The runs emitted 1,195,403 output tokens in total.
There are 6,171 complete natural responses, 707 capped responses, and two timeouts.
No admission-error outcome occurred.

The primary moderate SLO requires TTFT ≤5 s and every observed TBT ≤271.893227 ms.
Natural completion is an additional joint requirement.
At configured Poisson 0.25 req/s, Atomic C1024 passes 65/94 offered requests.
Grouped C1024 passes 74/94. Mixed C512 passes 85/94.
The paired G2-minus-Atomic difference is +9.574 percentage points, with 95% CI [1.481, 21.876].
G2-minus-Mixed is −11.702 percentage points, CI [−28.571, 0.000].
At 0.5 req/s, G2 gains +28.272 points over Atomic, CI [11.236, 50.289].
It does not establish an improvement over Mixed: −7.330 points, CI [−20.000, 8.000].

All policies show queued-backlog growth in every 1.0 req/s Poisson block.
Other finite-window failures can reflect unfinished in-service tails instead of queued growth.
No arm/rate qualifies sustainable 90%-attainment capacity.
The registered request-count and confidence requirements are not met.
The per-arm/rate Poisson sample counts are 52/94/191/379 at 0.125/0.25/0.5/1.0 req/s respectively.
The burst counts are 48/96 per arm at 0.25/0.5 req/s.
The full grids and paired intervals are in `serving_tables.md` and the SLO CSVs.

## Measurement controls and evidence preservation

All 48 isolated numerical comparisons passed exactly.
All 12 observer ON/OFF cohorts preserved tokens and depth records.
The observer time ratio is 1.002225, with 95% CI [0.997524, 1.007551].
This control covers Atomic C1024 at P≈2K/N4, not every workload.
Four Nsight captures exported successfully in the separate heavy lane.
Kernel service sums and device-activity coverage are not clean E2E shares.
The final quiet GPU4 check preserves the main tail ordering without four concurrent experiment workers.
It does not supply a universal shared-host correction.

The completeness audit passes all 71 distinct jobs required by the main manifests.
It also checks complete-request full-R4 work for 6,878 completed serving instances.
The 1,834 lossless raw/log archives passed SHA256 verification.
The compressed evidence occupies 164,532,627 bytes.
Local originals remain intact.
The resource wrapper restored recognized burners on physical GPUs 4–7 after data capture.

## Direct answers to the research questions

1. **Different interference mechanisms:** C changes chunk count and recurrent-core invocation count. G changes prompt yield boundaries within each chunk.
2. **Trade-off:** Smaller C can multiply eager launches and scheduling transitions. Smaller G can replace rare severe stalls with more frequent moderate stalls.
3. **Opposing quantiles:** G2 moves more than 5% of observed gaps above 100 ms, while reducing gaps above 200 ms.
4. **Grid dependence:** The response depends on C, G, staging, prompt shape, and batch packing. C×G alone does not predict client latency.
5. **Joint operating region:** G2 C1024 lies on the descriptive TTFT–p99 frontier. Statistical superiority over every tuned one-dimensional baseline is unproved.
6. **Strong baselines:** Mixed can deliver a lower decode tail. Split C1024 is a relevant fixed-grid control missing from the open-loop menu.
7. **Adaptive next step:** No live adaptive comparison establishes additional capacity. Resolve output fidelity and strengthen missing fixed controls before another scheduler.
