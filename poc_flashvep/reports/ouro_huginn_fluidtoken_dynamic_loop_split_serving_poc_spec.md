# Ouro and Huginn: full Dynamic Loop-Split serving evaluation

## 1. Contract and budget

This is a new campaign based on commit `a4c9e4a69386b0cd9b50d8b3e5be3af05aa0b751`.
The campaign starts at 2026-11-04 14:05:34 KST.
The total deadline is 2026-11-05 01:05:34 KST.
The GPU stop is 2026-11-05 00:05:34 KST.
The new-job cutoff is 2026-11-04 23:35:34 KST.
The final hour reserves time for analysis, publication, and resource restoration.
The total budget includes pauses. A capacity interruption does not reset it.

The publication branch is `flashvep/full-dynamic-loop-split-serving`.
The source namespace is `poc_flashvep/dynamic_loop_split_serving`.
The evidence namespace is `poc_flashvep/reports/ouro_huginn_fluidtoken_dynamic_loop_split_serving`.
Earlier data, reports, staged entries, and external runtime modifications remain unchanged.
Publication uses a temporary Git index and ordinary pushes. Force push is prohibited.

## 2. Models and execution invariants

| Model | Checkpoint revision | Fixed depth | Primary input/output |
|---|---|---:|---:|
| ByteDance/Ouro-1.4B | 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 | R4 | 4096/32 |
| tomg-group-umd/huginn-0125 | bb6621b65e90b6a4b9b29ef88dc83866d450470c | R32 | 2048/16 |

Retain original weights, BF16, the qualified numerical fixes, and native attention semantics.
No early exit, quantization, approximation, training, or recurrent-work omission is allowed.
The qualified external runtime remains `/home/esjung/external/looped-pd-abc-rlt`.
Both workers retain full model residency. This is not layer-sharded pipeline execution.
Main split comparisons use separate Prefill and Decode phases, without shared-core co-batching.
The read-only `ref_loopserve` engine is a reference, not a runtime replacement.

## 3. Evidence labels

- MEASURED: actual GPU execution or host request observations, with explicit scope.
- TRACE_ANALYSIS: analysis of named recorded events.
- STRUCTURAL_ACCOUNTING: rows, bytes, or logical boundaries. These are not measured speedups.
- ANALYTICAL_ORACLE: an assumption-dependent bound, never an online result.
- NOT_MEASURED, NOT_RUN_TIME_BUDGET, or BLOCKED_CORRECTNESS: no accepted measurement exists.

Kernel union is not SM occupancy or physical FLOP saturation.
Finite overloaded throughput is not sustainable serving capacity.
Repeated prompts do not create independent texts.
No failed or perturbed trace can become accepted evidence through relabeling.

## 4. Phase A: Token suffix recovery

First reproduce both previous warm-up failures in new namespaces with the candidate disabled.
The expected exception is `wavefront completed prompt chunks out of order`.
Retain the CPU replay of the installed task builder and native progress method.
Then explicitly enable `token_suffix_origin_candidate` in new configurations.
The candidate uses the imported origin while retaining absolute hidden-state rows.
Do not disable the native progress assertion or bypass KV validity checks.

Test aligned and unaligned prefix boundaries with full-prompt and smaller execution budgets.
Use the primary failed boundaries first: Ouro prefix 3072 and Huginn prefix 1984.
Additional unaligned boundaries include Ouro 3071 and Huginn 1983.
Test at least two chunk budgets where the runtime supports them.
Every configuration must preserve output sequences, valid prompt KV, hidden state, and full recurrent work.
Transfer fingerprints must match before the receiver consumes the imported state.

Run N8 before N32 or larger concurrent qualification.
Qualify Static Loop Split at the same N32 level.
Check rejection, P-side continuation, source retention, commit, ACK, and final release separately.
An expected error reproduction is not a correctness failure of the new candidate.
A removed exception alone is not a successful recovery.
Any unexplained output or state difference blocks that configuration's performance expansion.
Models have independent ledgers. One model's success cannot certify the other.

## 5. Methods and transport controls

1. Static PD: native full-Prefill handoff and full-depth Decode.
2. Static Token Split: full-depth prefix KV and exact suffix-state migration.
3. Static Loop Split: valid depth-indexed KV and exact recurrent hidden-state migration.
4. Dynamic Token Split: an observed-state FluidToken-style policy.
5. Dynamic Loop-Split: an observed-state policy over qualified recurrent cuts.

Include staged native full-Prefill and full-depth snapshot controls.
Native streaming and snapshot transport are not interchangeable mechanisms.
Their difference must not be credited to the split axis.
Use common model, memory budget, graph policy, warm-up policy, and arrival trace within each comparison.

Initial Ouro loop cuts are 3, 2, and 1, with full-PD cut 4 as control.
Matched Token prefixes are 3072, 2048, and 1024.
Initial Huginn loop cuts are 31, 28, 24, and 16, with full-PD cut 32 as control.
Matched Token prefixes are 1984, 1792, 1536, and 1024.
Both axes receive the same number of tuning candidates per model.
Equal recurrent token rows do not imply equal Attention FLOPs or equal padded projection work.
The Huginn suffix-64 comparison can incur 256-row projection padding.
The suffix-256/Loop28 control separates this runtime artifact from a recurrent-structure claim.
Qualify additional cuts before using them in clean serving or online selection.

## 6. Datasets, cache, and load

Reuse pinned ShareGPT/LongBench controlled prefixes and model-specific token IDs.
Use fixed output caps with ignore-EOS. Do not describe these as natural output distributions.
Use identical prompt IDs, order, generation seeds, and Poisson arrivals across methods.
The clean primary matrix uses N128 distinct preserved records per start.
N512 persistence explicitly cycles the same 128-record pool.
Record native prefix-cache settings, lookup/hit counts, and actual reused token counts.
Prefer a common cold/no-prefix-reuse control if the native cache cannot provide reliable reuse accounting.
Do not silently enable caching for only one method.

Use three loads per model.
Low load starts near 0.25 times the retained finite P service reference.
Medium load starts near 0.70 times that reference.
Prefill-heavy load starts at nominal QPS 6.356 for Ouro and 1.971 for Huginn.
Check low and medium conditions with clean baseline queue observations before held-out comparisons.
Any calibration-driven rate adjustment must precede candidate evaluation and remain recorded.
Record nominal, realized arrival, and completed QPS separately.
The heavy condition can violate TTFT through finite queue growth. Do not call it stable serving.

## 7. Request timestamps and latency definitions

For each request, record arrival, actual first Prefill execution start, first token, and final token.
TTFT equals arrival to first token and always includes queueing.
Queue/Admission Latency equals arrival to actual first Prefill execution start.
Service TTFT equals that execution start to first token.
Service TTFT includes handoff, D continuation, Coda, sampling, and output delivery.
Do not substitute a coordinator admission receipt for actual Prefill execution.
Audit whether a native start timestamp records submission or device execution.
If only host submission is available, label the metric accordingly and retain the missing device distinction.

Compute per-request durations before p50/p95/p99 aggregation.
Do not add component percentiles.
Do not sum overlapping kernel, transfer, and host observations into exclusive E2E shares.
Report disjoint host component percentages with a summed-request-latency denominator.
Exclusive GPU components remain NOT_MEASURED when overlap prevents attribution.

## 8. Metrics and fixed SLO

Reuse the unchanged frozen TTFT, request-average TPOT, and ITL limits from the prior artifacts.
Keep the stricter 90% TPOT-margin diagnostic separate from joint TTFT-plus-TPOT goodput.
Report maximum ITL separately. A passing p99 gate is not an all-token deadline guarantee.

For every accepted run, report TTFT, Service TTFT, Queue/Admission, TPOT, ITL, and E2E percentiles.
Also report Prefill execution, handoff, D continuation, and their measurement scopes.
Record offered/completed/failed counts, input/output tokens/s, request throughput, and SLO attainment.
Failures and timeouts remain in the offered denominator.
Report actual offloaded requests, token rows, loops, payload bytes, fragments, commits, ACKs, and source lifetime.
Report P-ready queues, P/D device activity, KV blocks, admission refusals, and peak HBM.
Model submissions are not GPU kernel invocation counts.

## 9. Dynamic policy gates

Use measured calibration from records separate from the clean tuning cohort.
Both axes use comparable PPI/DPI, current Decode load, KV credit, deadline, and latency-slack checks.
Only qualified cuts can enter the online action set.
Recheck D admission at the actual frontier. Safely continue P work on rejection.
Never use future arrivals, actual future output length, or future SLO results.
Record every initial decision, frontier decision, rejection reason, and actual accepted continuation.
No D-local probe can count as a migrated serving request.
No large dynamic campaign starts after only N8 qualification.

[FluidPD](https://arxiv.org/html/2610.06917v1) provides the Token policy reference.
The current runtime and discrete cut selector differ from its SGLang implementation.
Use the label FluidToken-style reproduction unless the official implementation is actually executed.
FluidRole and worker role switching are out of scope.

## 10. Profiling and isolation

Use an isolated H100 pair for clean timing.
The existing resource guard must protect unrelated processes and stop only approved burner sessions.
Do not burn other GPUs during clean timing when shared-host or fabric interference is possible.
Correctness traces and instrumented calibration do not become clean latency evidence.
Pair selected activity captures with unprofiled runs and require at most 5% wall perturbation.
Retain failed profiler captures. Avoid replaying a full heavy Nsight campaign merely for visualization.
Use actual device timestamps and clearly mark missing counters.

## 11. Expansion and stopping rules

Prioritize both models' recovery and N32 qualification, then exact Static comparisons.
Expand promising online configurations to three independent starts and then N512 persistence.
Report paired start-level uncertainty. Tokens are not independent replications.
Do not weaken correctness or SLO thresholds to fill the matrix.
If time is short, retain the exact Static matrix and mark unfinished Dynamic cells NOT_RUN_TIME_BUDGET.
If a configuration fails correctness, mark its dependent cells BLOCKED_CORRECTNESS.
Do not classify an unmeasured method as a performance NO-GO.

A generic logical-layer split includes each recurrent boundary.
The same boundary, state, and execution cost imply EQUIVALENT_MECHANISM.
Positive Token-versus-Loop measurements alone cannot prove exclusive Loop-specific novelty.
Record both positive and negative conditions with equal visibility.

## 12. Deliverables and closeout

Write the final English report to `ouro_huginn_fluidtoken_dynamic_loop_split_serving_final_summary.md`.
Preserve configurations, source/runtime hashes, per-request IDs, decision logs, profiles, and failure logs.
Create six figure groups: motivation, TTFT decomposition, Static splits, Dynamic methods, matched-work advantage, and SLO-goodput/tails.
Use measured values only. Mark unavailable figures and metrics NOT_MEASURED.
Commit only this campaign's code and evidence. Preserve the user's staged index and external runtime.
Check ordinary remote push and SHA equality.
Restore approved burners on GPUs 0 through 7 and check ownership and device state.
The report must answer all eight user research questions without assuming Loop Split superiority.

## 13. Prospective budget priority at 17:15 KST

Additional cuts retain the same N8/N32 qualification and heavy-load tuning budget for both axes.
Low and medium loads prioritize native PD, both primary splits, and the three transport controls.
Other cut-by-normal-load cells receive `NOT_RUN_PREDECLARED_BUDGET_PRIORITY`.
This rule applies to both models and does not depend on a candidate's measured advantage.
The original new-job cutoff and total deadline do not change.

Dynamic calibration can start after both models finish cut qualification and the primary N128 screen.
It need not wait for every additional static serving cell.
The primary screen uses the same immutable run configurations as the full static controller.
All controllers retain the same exclusive quiet-lane lock.
This scheduling change does not permit concurrent clean GPU measurements or weaker correctness gates.

## 14. Fine-Token robustness control at 18:23 KST

The coarse matched-row map does not establish the best Token boundary.
Add Ouro Prefix3840/Suffix256 and Huginn Prefix2016/Suffix32 as strong-baseline controls.
Both configurations retain the original runtime, memory policy, graph policy, precision, and frozen SLO.
Each requires N8 followed by N32 exact-state qualification before serving or dynamic use.
The online Token selector can include the measured qualified fine cut.

This extension gives Token an additional candidate beyond the core matched-row map.
Ouro already enumerates every nonzero internal whole-loop cut.
Report the extra Token tuning cost explicitly.
Do not claim equal aggregate candidate counts or an exhaustive Token oracle.
Do not restrict the Token baseline to the minimum whole-loop offload fraction.
This control can favor Token. A negative Loop result remains valid evidence.

## 15. Transport-control budget priority at 18:32 KST

The same-cut transport controls run at medium load for both models.
They separate staged scheduling and Snapshot handoff from the split mechanism.
Their low-load and heavy-load duplicates receive `NOT_RUN_PREDECLARED_BUDGET_PRIORITY`.
Native PD and the primary Token/Loop comparisons retain all three loads.
Every additional matched-row cut retains heavy-load tuning and its full correctness gate.
This rule frees time for the online methods and matched instrumentation controls.
It applies to both models before those duplicate transport cells start.
Do not extrapolate a measured medium-load transport delta to another QPS as a causal result.

## 16. Dynamic startup control at 20:32 KST

No Dynamic branch GPU test started before this decision.
Remove the optional Dynamic-only cycle through short-prompt split geometries.
Those startup queries have 512 tokens, not the measured 4096 or 2048 tokens.
Cycling their shapes can consume the bounded native graph cache without warming the measured shapes.
This is a code-path risk, not a measured Dynamic slowdown.

Retain the ordinary startup request budget and Full-PD startup path.
Retain the same native graph-cache budget for both split axes.
Do not subtract first-use capture work from calibration or Serving latency.
The isolated finite predictors retain their recorded first-use work.
Dynamic forced-branch, Online N32, and Serving runs require their own correctness gates.
Record actual measured captures, replays, and fallbacks before attributing a performance difference to graph behavior.

## 17. Marker-free controls

An absolute ON/OFF wall difference above 5% fails the event-marker screen.
A negative difference does not exempt the pair.
Retain every failed pair.
Analyze completed marker-OFF controls as separate request-latency measurements.
Those controls cannot supply actual-execution Service TTFT.
Do not import event timestamps from their matched ON execution.
Event-marked figures must retain the applicable control status or missing-control label.

## 18. Concurrent forced-branch qualification at 20:55 KST

Serial N8 acceptance and rejection precede concurrent N32 branch tests.
Both Huginn Token N8 GPU executions completed before the first N32 branch configuration froze.
The N32 tests use a new `_concurrent_state32` namespace.
They offer 32 requests together and retain native slot, KV, transfer, ACK, and release limits.
Thirty-two offered requests do not imply 32 simultaneous GPU-active requests.

The diagnostic forces its selected frontier decision only in the explicit heavy-correctness phase.
Forced acceptance must produce 32 actual migrated requests and matching active D rows.
Forced rejection must produce 32 actual frontier rejections and zero D Prefill rows.
Both branches must preserve exact output, transported bytes, final prompt state, and complete recurrent work.
The independent unforced Online N32 gate remains mandatory before N128 Serving.
No real online admission or frontier SLO check is bypassed.

These traces can violate latency SLOs because hashing and forced decisions perturb scheduling.
They are not clean timing, safe online actions, or measured policy performance.
The change strengthens concurrent ownership coverage. It does not reduce the N32 population or any exactness check.
