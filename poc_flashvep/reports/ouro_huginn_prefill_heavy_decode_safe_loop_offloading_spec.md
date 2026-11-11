# Prefill-heavy discovery and conservative Decode-safe Loop offloading

## Scope and clock

This campaign starts after the Loop-width and tuned-Token campaign closes and publishes.
Its publication parent is `30e1717b446e5097a0f523da0bcc890476af3ba1`.
It performs Phase 1 discovery and conditional Phase 2 migration only.
It does not develop a complex predictor, a full dynamic scheduler, or a new Token-superiority campaign.

Start: 2026-11-11 12:27:21 UTC, or 21:27:21 Asia/Seoul.
The total deadline is 2026-11-11 22:27:21 UTC.
GPU work stops by 21:27:21 UTC. Final analysis and publication use the last hour.
Each job must fit its startup, execution, and cleanup allowances before the GPU deadline.
The campaign can finish earlier after a supported result or an exhausted useful search.

Source namespace: `poc_flashvep/prefill_heavy_decode_safe/`.
Evidence namespace: `poc_flashvep/reports/ouro_huginn_prefill_heavy_decode_safe_loop_offloading_20261111/`.
Publication branch: `flashvep/prefill-heavy-decode-safe-20261111`.
The active checkout, staged index, previous evidence, and external runtime remain unchanged.

## Research questions

1. Can P maintain measured Prefill execution pressure while D retains useful physical kernel inactivity?
2. Does that inactivity overlap legal source cuts, available state, legal KV credit, and Decode deadlines?
3. Can a small amount of exact Loop migration reduce TTFT while preserving Native-relative Decode latency?

Independent capacity ratios guide discovery. They do not prove online feasibility.
Growing queues describe a controlled overloaded interval, not sustainable serving capacity.
Finite-cohort completion rates do not establish steady-state capacity.

## Prior evidence

The campaign reads the six user-specified reports at their pinned commits.
It retains the old Ouro 4096/32 signal as a starting condition, not a completed migration result.
The reported D-local probes are not P-to-D continuations.

Old Huginn Loop28 used forced static migration without an active Decode-safe budget.
Its overlapping operation enclosures do not identify exclusive kernels or SM/HBM contributions.
Old Output4 kernel inactivity failed the profiling screen. The Output16 capture failed with a CUDA error.
Neither result supplies qualified physical slack for this campaign.

The completed GEMM campaign found a workload-specific Huginn benefit at Output4.
This campaign requires Output at least 16. Output4 therefore cannot qualify Phase 1 or Phase 2.
The earlier benefit also does not establish exclusive Loop-specific novelty.

## Model and numerical contract

| Model | Revision | Recurrent depth | Precision |
| --- | --- | --- | --- |
| ByteDance/Ouro-1.4B | 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 | R4 | Original BF16 |
| tomg-group-umd/huginn-0125 | bb6621b65e90b6a4b9b29ef88dc83866d450470c | R32 | Original BF16 |

Preserve exact Attention, all recurrent work, valid KV planes, Hidden State, and Huginn Prelude/Coda/reinjection.
Do not use approximation, early exit, quantization, precision reduction, or duplicate recurrent work.
Huginn's actual input, output, and required special tokens must remain within 4096 tokens.
Do not increase a model's verified context limit to admit a candidate.

The external runtime remains at its preserved qualified revision and patch.
Native-path Zero is the strongest comparison baseline.
Original Native remains a reference arm when needed.
Qualified projection tuning can apply to both arms under the same frozen policy.
Any new geometry requires qualification before performance claims.

## Workload and resource contract

Use one isolated 1P1D pair for primary timing.
Start with Ouro GPUs 0/1 and Huginn GPUs 4/5.
Do not assume that concurrent pairs are independent.
Use parallel timing only after a matched interference check supports that choice.

Reuse controlled ShareGPT/LongBench material and preserve original text IDs and token IDs.
Use 128 distinct prompt IDs for N128. Keep prefix caching disabled.
Use fixed output caps, ignored EOS, and paired open-loop Poisson arrivals.
Record truncation or extension rules for every new length.
Repeated prompts in persistence controls do not become independent texts.

Initial Ouro candidates are 4096/32, 6144/32, 8192/16 or 32, and longer legal prompts if useful.
Initial Huginn candidates are 2048/16, 2560/16, 3072/16, 3500/16, and a context-safe upper candidate.
Input/output/QPS and legal slots, memory, graph, and KV settings can change during discovery.
Each selected configuration freezes before its confirmation runs.
Do not degrade Native deliberately or change the model equations to create pressure.

Stop only approved owned burners or this campaign's verified process groups.
Keep approved burners active during CPU work and other GPU-free periods.
Clean timing can stop all managed burners under the existing isolation policy.
Restore all GPUs 0 through 7 after GPU work. Check ownership, generation rounds, and actual utilization.

## Frozen SLO and Decode protection

| Metric, milliseconds | Ouro | Huginn |
| --- | --- | --- |
| TTFT | 1112.9925646819174 | 1963.9920117333531 |
| Request-average TPOT | 53.70702898374411 | 186.5715059141318 |
| Additional-work TPOT p95 | 48.3363260853697 | 167.91435532271862 |
| ITL p99 | 65.08287642151119 | 192.8212538175284 |

New lengths do not receive a retrospectively relaxed SLO.
Report when longer contexts fail the existing limits.
Phase 2 also targets ITL p99 and TPOT p95 at most 1.05 times matched Native-path Zero.
Report maximum ITL and every threshold exceedance separately.
A p99 pass does not imply that every token interval passes.
Use the frozen delivery-based ITL contract. Identify coalesced output events explicitly.

## Phase 1-A: Capacity and instrumentation qualification

Measure P-only Prefill and prepared-D Decode with the same model, legal memory geometry, and qualified runtime.
Retain B1 and compatible larger-batch controls where available.
Use the strongest valid matched P reference for the 85% service screen.
Independent references remain separate from online same-window results.

First reproduce the previous profiling failure at a bounded diagnostic seam.
Do not repeat an unchanged failed collector indefinitely.
Test warmed activity collection, bounded captures, deferred export, and other justified low-overhead alternatives.
Record collector versions, callback modes, dropped records, invalid records, and activation/termination costs.
Do not replace physical kernel intervals with model-submission enclosures.

Compare instrumented and marker-free controls under identical inputs and arrivals.
Measure response wall time, TTFT, TPOT, ITL p99, and maximum ITL.
Use Native A/A repeats to identify variation. Report point screens and independent-start uncertainty separately.
An interference screen fails when a required latency ratio differs by more than 5%.
Short-window local effects also matter. A small whole-run wall difference cannot hide a large local tail effect.
If a collector fails the screen, its trace remains diagnostic-only.

## Phase 1-B: Adaptive discovery

Choose initial QPS near measured P capacity, then adjust it from observed bottlenecks.
Use short bounded pilots before N128 confirmation.
Classify each cell as P compute, D Decode, D KV admission, transfer/runtime, mixed, or qualified joint pressure.
Keep the measured evidence and failed gates for every classification.

If P busy is low, inspect graph coverage, service gaps, and queue ownership before increasing input or QPS.
If D Decode dominates, reduce output only to 16, or increase legal input and recalibrate QPS.
If KV admission dominates, inspect full-request reservations and legal memory budgets.
Free blocks for a small probe do not establish credit for a full continuation.
Do not treat coordinator waiting requests as P compute-ready requests.

## Phase 1-C: Same-window acceptance

Condition P requires all of the following within one server start and one common interval:

1. P kernel-interval union is at least 80% of that interval.
2. Audited compute-ready P pressure increases between its first and second halves.
3. Executable P work repeatedly exists. Submitted unretired work has explicit ownership.
4. P service is at least 85% of the strongest matched finite P reference.
5. D-credit or transfer waiting does not explain the apparent compute-ready queue.
6. Prefill execution persists throughout a nontrivial observation interval.

Condition D requires measured kernel inactivity, legal full-request credit, frozen Decode SLO, and useful contiguous windows.
Condition Joint requires Conditions P and D in the same interval.
Source-cut readiness and handoff must permit actual continuation within at least some windows.
Do not combine a P profile with D observations from another run.

Measure interval unions across streams, not summed durations.
Separate compute kernels, communication kernels, memory copies, and graph/API observations where the collector supports them.
Report busy/idle percentages, idle count, mean/p50/p95/p99/max, and every gap's start/end.
Report counts and accumulated durations above 1/2/5/10/20/30/50/100 ms.
Threshold rows overlap and must not be added.

Distinguish physical inactivity, compute-only packing, and request-feasible Decode-safe capacity.
Offline packing with future information is an Offline Oracle, not an online implementation.
Do not invent source-cut times that an Atomic trace does not expose.
Missing dependencies or unqualified timing leave request-feasible capacity NOT_MEASURED or UNVALIDATED.

Confirm promising configurations with at least three independent N128 server starts.
Use separate held-out arrival seeds after selection.
Add N384 through N512 persistence when a useful candidate survives.
Report actual distinct prompt counts and any cyclic reuse.

## Phase 2: Conservative migration, conditional on Phase 1

Do not enter Phase 2 for an unqualified model/workload.
Start with one remaining Loop: Ouro Loop3 or Huginn Loop31.
Only then consider Ouro Loop2 or Huginn Loop30/Loop28.
Qualify every new length/cut at N8, then N32, before serving expansion.

Use a small reproducible candidate request fraction and a minimal current-state admission guard.
Do not force every request into a static split.
Reject before commit when Decode execution, KV credit, state readiness, or the conservative completion allowance is insufficient.
Do not use future arrivals in an online guard.
Keep offline replay and online admission labels separate.

Preserve Native ACK, source lifetime, legal ownership, and exactly-once loop execution.
After commit, D owns the remaining state. Do not pretend that an unsupported reverse migration exists.
Measure progress and starvation. Stop expansion if accepted continuations wait excessively or Decode tails regress.
Do not discard a committed request to improve reported latency.

Measure actual P/D recurrent rows and completed Loops, not merely approved migrations.
Record descriptor preparation, payload bytes, observed transfer/ACK boundaries, D start/end, and first output.
Report wire completion only when the transport exposes a valid measurement.
Do not call a host receipt an exclusive DMA duration.

Compare TTFT, E2E, TPOT, ITL, maximum ITL, goodput, and finite-cohort rates against matched Native-path Zero.
Keep Service TTFT and exact queue delay NOT_MEASURED when execution timestamps are insufficient.
Compute request-level intervals before percentiles. Do not sum component percentiles.

Expand only after correctness, absolute SLO, and Native-relative Decode screens pass.
Use independent-start paired uncertainty for final claims.
Record the migration-fraction versus TTFT/Decode Pareto frontier, including failed points.

## Decisions and deliverables

Report Phase 1 and Phase 2 separately for each model.
Use Strong Success, Partial Success, Inconclusive, or No-go on Tested Envelope with explicit failed gates.
Do not infer structural impossibility outside the tested envelope.
Do not claim Phase 3/4 readiness without the requested Phase 1 and Phase 2 evidence.

Retain MEASURED, TRACE_ANALYSIS, OFFLINE_ORACLE, CORRECTNESS_ONLY, NOT_MEASURED, NOT_IDENTIFIED, NOT_RUN, and FAILED labels.
Create actual P/D timelines, queue/busy plots, idle distributions, source/window overlap, and migration trade-off figures.
Figures must not contain invented GPU activity or zero-valued unrun arms.

Maintain immutable configurations, hashes, GPU maps, model revisions, prompt IDs, commands, seeds, failures, and resume state.
Publish only related source, reports, figures, and lossless evidence on the new branch.
Verify Remote SHA, preserved staged-index signature, external-runtime diff, and all-eight-GPU restoration.
Final report: `ouro_huginn_prefill_heavy_decode_safe_loop_offloading_final_summary.md`.
