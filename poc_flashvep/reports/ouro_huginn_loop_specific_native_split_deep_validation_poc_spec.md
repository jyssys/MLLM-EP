# Deep validation of native Token, Loop, and 2D splits

## 1. Contract and budget

Start: 2026-11-05 12:38:46 Asia/Seoul.
Overall deadline: 2026-11-05 23:38:46.
GPU stop: 2026-11-05 22:23:46.
New long-job cutoff: 2026-11-05 22:03:46.
The 11-hour budget includes analysis, reports, ordinary push, remote verification, and resource restoration.
No interruption resets these deadlines.
The source parent is `a1c9845ba4f7a56b4e990b16e4f25f96311311de`.
The publication branch is `flashvep/loop-specific-native-split-deep-validation`.
All new code and evidence use their own namespaces.
The active checkout, staged index, external runtime, old reports, and old raw data remain unchanged.

This study does not implement a new online dynamic scheduler.
It tests whether exact native static redistribution and a validated offline oracle justify that next step.
An implementation failure is not a structural performance NO-GO.

## 2. Models and fixed limits

| Model | Revision | Work | GPU pair | Low / medium / heavy nominal QPS |
|---|---|---|---|---|
| ByteDance/Ouro-1.4B | 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 | BF16, R4, 4096/32 | 0/1 | 1.23460875 / 3.4569045 / 6.356 |
| tomg-group-umd/huginn-0125 | bb6621b65e90b6a4b9b29ef88dc83866d450470c | BF16, R32, 2048/16 | 4/5 | 0.37903355455 / 1.06129395274 / 1.971 |

Keep the qualified numerical kernels, original weights, Attention semantics, and full recurrent work.
Keep Huginn Prelude, Coda, injection state, and the 4096-token context limit.
No approximation, quantization, early exit, or parameter changes are permitted.
All methods use two P slots and eight configured D slots.
Actual active concurrency can be lower because of KV credit.
Ouro P/D token budgets remain 8192/4096. Huginn budgets remain 4096/2048.
The primary prompt chunk is the complete input.
Token split boundaries are independent of that chunk budget.
Prefill and Decode remain separate recurrent-core batches.
Native Mixed is not a contribution or a main comparison arm.

Freeze the previous full-precision `frozen_SLO.json` without recalibration.
Ouro limits in ms: TTFT 1112.9925646819174, TPOT 53.70702898374411, ITL p99 65.08287642151119.
Huginn limits in ms: TTFT 1963.9920117333531, TPOT 186.5715059141318, ITL p99 192.8212538175284.
The additional-work screen requires TPOT p95 ≤90% of its limit and ITL p99 ≤its limit.
Joint goodput uses request TTFT and request-average TPOT. The token-tail screen remains separate.
Timeouts and failures remain in offered-request denominators.
Finite overloaded cohorts do not establish sustainable capacity.

## 3. Diagnoses and predeclared interventions

### Scheduler conflict

First reproduce the actual inherited probe-hook mutation with its existing CPU regression.
Disable `D_local_probe_output_pacing` for every arm because this study has no independent local probes.
Record requested and effective P/D scheduler settings after all hooks construct each stage.
Require actual non-Wavefront scheduling for grouped native continuation.
Do not remove the native one-depth Wavefront progress guard.
Test the original Huginn Token1984 warm-up in a new GPU namespace.
Separate this configuration correction from later graph and performance interventions.

### Graph cache coverage

The hypothesis is startup-dependent admission into a bounded exact-shape cache.
First replay the cache admission order in a CPU regression.
Use the same primary-shape reservation policy and serial full-prompt warm-up for every method.
Prefer B1 and B2 primary prompt geometries before short startup geometries.
Do not give one method a larger cache.
Start with four entries per packed-prefill graph owner.
If four entries do not cover reachable primary shapes, compare eight entries in bounded controls.
Record entries, complete graph keys, hits, misses, eager fallbacks, capture time, warm-up time, and HBM.
Do not call counters that include startup measured-only counters.
Warm-up requests do not enter latency, throughput, or SLO statistics.
Capture still uses the native validity checks, stream ordering, operands, and numerical operations.
Do not infer a counterfactual optimized speedup by subtracting capture time.

## 4. Qualification and cut budget

Run Native reference and candidate N8 before N32 overlapping simultaneous submissions.
N32 means 32 offered requests, not 32 active GPU requests.
Keep state hashing separate from clean latency.
Each candidate must pass output sequence, valid prompt KV, recorded hidden state, transferred-byte identity, full work, and native lifetime checks.
The hidden-state gate covers its recorded boundary, not every tensor after every operation.
Original-HF fidelity and downstream accuracy remain separate, unmeasured items unless explicitly executed.
Preserve source buffers until native ACK. Require final coordinator credits and native KV allocations to retire.
Test mixed accepted/full-PD cuts where the existing connector supports them.
Do not present a forced cut list as a complete online admission policy.

The discovery budget is three partial candidates per axis and model:

| Model | Token prefix | Loop cut | 2D frontier (c,k) |
|---|---|---|---|
| Ouro | 3840, 3072, 4032 | 3, 2, 1 | (3,3072), (2,2048), (3,3840) |
| Huginn | 1984, 2016, 1792 | 31, 28, 24 | (31,1024), (28,1536), (31,1792) |

Native Atomic PD and native-path zero-offload are common controls.
Extra cuts require a logged hypothesis and an equal additional search budget for competing axes.
Do not expand a correctness failure into serving.
Stop an unsafe access or OOM on its affected configuration.
Keep every failed configuration and log.

The diagnostic may gather one KV layer before selecting pages to avoid copying unrelated layers.
Native fingerprints must preserve their original block-K/block-V hash order and every valid byte.
Partial fingerprints must preserve their original plane-K/plane-V hash order and every valid byte.
Compare old and new fingerprint functions in a CPU regression before GPU use.
This change applies only to heavy qualification. It is not a clean-serving runtime optimization.

## 5. Data partitions and clean serving

Use the previous ShareGPT/LongBench controlled prompt pool for discovery and regression.
That pool has only 184 sources, including 128 previous primary prompts.
It cannot provide 128 wholly new held-out prompts after previous exclusions.
Extend the pool from the same pinned public datasets for confirmation.
Exclude previous source IDs, source hashes, and model-specific first-128-token prefixes.
Freeze 128 unique held-out prompt IDs before cut selection.
Freeze separate calibration prompts with no held-out or discovery overlap.
Preserve actual tokenizer IDs and explicit length truncation.
Record source revisions, source hashes, source IDs, unique counts, and overlap checks.
This is controlled ignore-EOS generation, not natural response lengths or Azure traffic.

All paired methods share inputs, generation seeds, open-loop Poisson offsets, and output caps.
Use new arrival seeds for held-out starts.
Use randomized or counterbalanced method order across independent starts.
Prefix caching remains disabled, with zero reused-prefix tokens required.
N128 is the primary comparison size.
N512 is conditional on a promising qualified result and available budget.
Any cyclic N512 run must state its 128 unique-text denominator.

Compare low, medium, and heavy load after N32 qualification.
Use medium discovery to select the best Decode-safe partial cut per axis.
Selection ranks joint goodput first and TTFT p95 second.
Record all attempted candidates, including Decode-unsafe candidates.
Confirm selected cuts and the predeclared Huginn Loop31 on untouched held-out inputs.
Require at least three independent starts. Prefer five Huginn starts if time permits.
Keep selected fixed cuts unchanged across confirmation starts.

## 6. Measurement and uncertainty

Report request TTFT, Service TTFT, queue/admission latency, TPOT, ITL, maximum ITL, E2E, rates, goodput, and actual migration.
Request TTFT includes scheduled arrival to first output.
Service TTFT includes first actual model-stream Prefill entry through handoff, continuation, Coda, and first output.
Report the clock-mapping bracket uncertainty and its nonlinear-drift limitation.
Do not replace TTFT with Service TTFT.
Calculate per-request intervals before percentiles. Never add component p95 values.
Keep host durations, CUDA event enclosures, and exclusive kernel union separate.

Repeat marker-free confirmation as the primary clean-latency result.
Use matched marker ON/OFF controls for mechanism timing.
Keep the ≤5% absolute global-wall interference screen.
Also report local-tail changes because a global pass does not bound them.
Do not use a rejected CUPTI trace as clean-serving GPU activity.
Use bounded diagnostic captures only after the main correctness and latency gates.
Missing kernel/SM/bandwidth counters remain NOT_MEASURED.

Use paired independent-start ratios and 95% intervals.
The start, not each request or token, is the replication unit.
Report small-sample assumptions and cut-selection uncertainty.
A TTFT improvement does not imply improved goodput.
Attribute queue relief, P/D work, graph behavior, transfer, and admission separately.
Recurrent token rows and causal-pair counts are structural accounting, not measured FLOPs.

## 7. Validated offline oracle

Implement a dependency-aware event-driven model only from separate measured calibration.
Model P/D queues, resource contention, batching, Decode iterations, KV reservations, transfer, ACK/source retention, and graph shape cost.
Use the same causal and state-validity contract for Token, Loop, and 2D actions.
Do not give arbitrary tiles free execution or free transfers.
Use measured D interference and frozen Decode gates.
An oracle may use future arrivals and output caps only with an explicit offline label.
It is not an online achievable policy.

Predict untouched static-cut serving runs before comparing modeled headroom.
Report TTFT, Service TTFT, TPOT, and ITL prediction errors with start-level uncertainty.
The initial validation screen is median absolute relative error ≤15% and p95 error ≤25% over the validation cells.
Also require no predicted Decode-safe action that violates the measured frozen Decode screen.
These model-error thresholds do not replace or weaken actual SLOs.
If validation fails, report the model and errors but do not label its result a Feasible Oracle.
Do not tune the model on held-out confirmation results and then call those same results validation.
Native, Token-only, Loop-only, and 2D share all measured costs and constraints.
An optional compute-only relaxation is not a rigorous upper bound without a bound argument.
Missing costs remain NOT_MEASURED. Missing headroom remains INCONCLUSIVE.

## 8. Novelty, figures, and closeout

An identical generic logical-layer boundary with identical state, placement, and costs is EQUIVALENT_MECHANISM.
The 2D action space includes Token and Loop endpoints. More candidates alone do not establish novelty.
Both PD workers already hold full model weights. No weight-residency saving follows from the Loop name.
Separate a measured Token-versus-Loop trade-off from an exclusive Loop-specific claim.

Create the five requested figure families with measured, structural, and modeled labels.
Show deterministic-cache before/after controls, held-out Loop31 intervals, actual payload, Decode gates, and oracle validation error.
Do not draw invented idle time or missing oracle bars.
Record per-model GO_LOOP_SPECIFIC, CONDITIONAL_GO, NO_GO_LOOP_SPECIFIC, or INCONCLUSIVE.
Require both models and structural evidence for a common-method GO.

Use the existing exclusive quiet lane for clean 2GPU measurements.
Stop only verified approved burners. Leave unrelated user processes untouched.
Keep all devices quiet during a clean run if shared-host burner interference is possible.
Restore approved burners on physical GPUs 0–7 when no experiment owns the lane and after final experiments.
Verify ownership, active GPU processes, the real index signature, external runtime diff, and remote SHA.
Use an alternate publication index and ordinary push. Never force push.
Reserve at least 75 minutes after the GPU stop for figures, reports, publication, and resource verification.

## 9. Pre-serving amendment after the cache4 GPU qualification

Decision time: 2026-11-05, approximately 14:12 Asia/Seoul.
No clean discovery, cut selection, or held-out result existed at this decision.
The original four-entry qualification remains in its raw namespace and a separate ledger.
Ouro 2D N32 exposed a D batch of three requests.
The original power-of-two reservation policy did not admit that primary geometry.
The same run also captured a four-request D geometry after measurement began.
This is direct evidence that P-only primary warm-up does not stabilize the entire path.

Use a common eight-entry primary-shape budget for the main comparison.
Permit every legal integer D batch geometry, not only powers of two.
Repeat N8 and N32 qualification in distinct `reserved8` namespaces.
Heavy exactness qualification does not use the additional D warm-up cohort.
Its recorded shape coverage states which graph geometries the state comparison actually exercised.

Before non-heavy serving, warm each reachable D primary geometry with exact full-prompt requests and one output token.
Use the actual D KV pool to bound the startup cohort.
Only D scheduling waits until that cohort completes native import.
P continues calculation and ACK progress normally.
This barrier exists only during startup, not during measured serving.
Keep native ACK, fixed-depth work checks, and warm-up output records.
Record which additional warmed shapes lack a separate exact-state comparison.
Do not call those shapes universally state-qualified from a subset of observed geometries.

Keep legacy4, reserved4, and reserved8 controls for the primary Ouro 2D case.
Report HBM and startup costs separately.
The main baseline and every main split use the same eight-entry cap.

Remove marker-ON discovery uniformly across methods to preserve the unchanged wall budget.
Keep marker-free discovery, independent instrumented calibration, and paired ON/OFF held-out confirmation.
Do not remove an unfavorable method selectively.
The five-start Huginn confirmation remains preferred after the required three starts.
All original correctness gates, SLOs, isolation rules, and deadlines remain unchanged.

The inherited recovery state probe already selects the layer before the block gather.
An early candidate duplicated that function but did not replace the active recovery probe.
Remove the unused duplicate and preserve its archived source and CPU evidence.
Reuse `looped_pd_headroom_recovery.state_probe` without changes.
The CPU regression compares its bytes and hidden hash with the original legacy probe.
Native reserved8 N8/N32 records also compare against the retained reserved4 Native records.
The separate transfer-fingerprint intervention remains diagnostic-only.
It cannot improve clean-serving latency because clean serving disables state hashing.

## 10. Equal candidate-budget reduction before clean selection

Freeze two partial candidates per axis and model for the mandatory campaign.
Keep the first two candidates in each column of Section 4.
The third candidates are NOT_RUN_TIME_BUDGET, not failed methods.
No clean discovery or held-out result exists at this decision.
The heavy cache4 Huginn Native N8 dispatch took 287.774 seconds.
The corresponding N32 dispatch took approximately 614 seconds.
These are complete diagnostic-job durations, not clean-serving or pure model times.
Repeating all third candidates after cache8 qualification threatens the required paired confirmation in both models.
Use the same reduced search budget for Token, Loop, and 2D.
Complete three medium starts in both models before optional Huginn starts four and five.
Keep independent calibration before held-out prediction.
Do not change any correctness threshold, frozen SLO, or deadline.

## 11. Native transfer-region accounting

Apply the same read-only submission observer to every main method.
Count only successful native NIXL WRITE submissions.
Classify each actual local descriptor by its registered KV or hidden-storage bounds.
Require descriptor lengths to equal the native ticket size.
Keep startup and measured counters separate.
Compare measured accepted bytes and submission counts with native completed-transfer counters.
Record the observer's host cost separately from the original submission cost.
Record NIXL notification payload bytes separately.
PDEngine control messages, ACK serialization, and transport headers remain NOT_MEASURED.

The Huginn Token1984 N8 accounting smoke passed output, full-work, and lifetime gates.
Its measured KV payload was 44,248,596,480 bytes.
Its measured hidden payload was 5,575,680 bytes.
All 664 successful submissions matched the native completion counters.
The observer used 0.037779 host seconds in that finite measured cohort.
This host-cost accounting does not replace a paired wall-interference experiment.
The smoke does not select a cut or establish a serving speedup.

## 12. Geometry-aware correction to the two-cut Token budget

Apply this correction before any N128 discovery or held-out comparison.
Huginn Token candidates are 1984 and 1792, not 1984 and 2016.
The unchanged Down projection uses eight B1 tiles for 2048, 2016, and 1984 rows.
Prefix 1792 uses seven tiles.
Keeping only the two smallest suffixes would omit an aligned candidate with actual Down-work relief.
The new choice tests a stronger Token baseline.
It does not change the 256-row numerical fix.
All axes retain two mandatory cut candidates.
Huginn Token2016 becomes the third, unrun candidate.
Keep its original specification and the budget history visible.
The correction uses structural accounting, not favorable or unfavorable serving results.
Matched row counts remain different from matched FLOPs or actual device costs.

## 13. Full-prefix endpoint control before held-out confirmation

Native and feature-enabled zero-offload are valid full-prefix Token-only endpoints.
Keep both endpoint controls in every confirmation family.
Choose the safe Token reference from first-start discovery data only.
Use the same frozen Decode screen and graph-coverage gate.
Rank eligible endpoints and Token cuts by joint goodput, then TTFT p95.
Freeze this reference before the first held-out dispatch.
Do not replace an unsafe partial Token action with a claim of a universally optimal Token policy.
Keep the unsafe diagnostic partial and both endpoint comparisons in the report.
The modeled Token, Loop, and 2D families must all permit the same zero-offload endpoint.

## 14. Resource failure after state qualification

Huginn 2D(c28,k1536) passed N8/N32 state qualification but failed the larger D startup geometry.
The N128 dispatcher reported an out-of-memory error before measurement.
Retain this failure as BLOCKED_RESOURCE.
Do not expand the unchanged OOM action into calibration or an oracle.
State qualification does not prove workspace feasibility at every startup-warmed batch shape.
Continue other independently qualified methods.
Any memory repair requires separate qualification and matched allocation controls.

## 15. Terminal warm-up observer correction before confirmation

The D geometry warm-up hook rereads its completed marker on later scheduling callbacks.
The exact closure replay produced ten reads for ten callbacks after ALL_COMPLETE.
Reuse the existing recovery hook's terminal-state caching pattern.
Keep the active-import barrier and the native readiness predicate unchanged.
Retain historical c_ discovery with its original observer.
Use terminal-hook retirement for all new calibration and held-out methods.
Do not combine these observer versions into one clean paired result.

Qualify Huginn Token1984 at N8 and N32 with the complete D startup geometry.
Use exact output, transfer, KV, hidden, work, and lifetime gates.
This diagnostic adds 21 full-prompt startup requests.
The prior Token1984 N8 and N32 dispatches took 294.283 and 608.888 seconds.
Allow 600 seconds for startup and 1100 seconds for the entire expanded state diagnostic.
These bounded diagnostic limits do not change the serving SLO or campaign deadline.

Compare retired-hook controls with the retained first-start discovery inputs and arrivals.
Measure Huginn Native, Loop31, Token1984, and Ouro Token3840.
Keep global wall changes and local tail changes separate.
These controls do not select new cuts.
Defer optional legacy-cache sensitivity before sacrificing the core three-start confirmation.

## 16. Frozen cost-model correction and prospective validation

Version 1 froze before confirmation dispatches.
Its preparation enclosure used a projected CUDA event after host submission.
That interval can include an earlier queued GPU batch.
The serialized model can therefore count that device work twice.
Keep the original frozen model and every Version 1 prediction unchanged.
Do not use Version 1 to claim feasible headroom.

Version 2 uses host entry into ModelRunner._execute after input preparation.
This host enclosure can still contain synchronization.
It is not exclusive CPU time.
Version 2 uses independent calibration inputs only.
It froze at epoch 1793875757.023604, after four Huginn confirmation dispatches began.
This is a protocol deviation from the original global pre-confirmation freeze.
Exclude those four dispatches from Version 2 validation.
Do not create retrospective predictions for them.

Prepare each remaining Version 2 prediction before its own GPU dispatch.
Check raw-directory absence before and after CPU prediction.
Retain configuration hashes, model hashes, and prediction timestamps.
Validate those hashes and timestamps against the estimated subprocess start.
Static serving choices do not use either cost model.
This correction does not change measured model execution or cut selection.

Keep the original 15% median-error and 25% p95-error limits.
Require three prospective medium starts per model and method family.
Keep the accepted service-timing coverage requirement and false-safe rejection rule.
Do not retune costs after validation outcomes.
Prepared supplementary starts remain separate from primary serving estimates.
Do not run them solely to repair coverage when prediction-error screens already fail.
An invalid model cannot establish feasible-oracle speedup or absence of structural headroom.
Keep every SLO and the campaign deadline unchanged.
