# Huginn adaptive Loop PD: autonomous net-benefit optimization

## 1. Objective and scope

Test whether exact adaptive Loop continuation can improve optimized Native PD.
Use Huginn-0125 with fixed R32 and original BF16 weights.
Do not change Attention math, recurrent work, or qualified numerical fixes.
Ouro is outside this primary campaign.

The publication parent is `6cadf1011a0b596efe39ccccefc9898dd8ef2925`.
The source directory is `poc_flashvep/huginn_net_benefit/`.
The evidence directory is `poc_flashvep/reports/huginn_autonomous_adaptive_loop_pd_net_benefit/`.
The publication branch is `flashvep/huginn-adaptive-loop-net-benefit`.
The active checkout and its staged index must remain unchanged.
External Runtime changes must remain unchanged.

## 2. Baselines and frozen contract

Native-path Zero is the strongest required baseline.
Original Native alone is not a sufficient reference.
Use the same packed Native wire format and Runtime revision across matched comparisons.
Retain full R32 computation, exact output tokens, valid KV, and hidden state.
Retain Native ACK, source retention, and destination ownership.
Do not execute a committed Loop on both GPUs.
Do not discard committed work after a deadline miss.

The anchor uses 2048 input tokens and 16 output tokens.
Its offered Poisson arrival rate is 1.971 requests/s.
Use one P GPU and one D GPU, with 2/8 logical slots.
Reuse 128 distinct controlled ShareGPT/LongBench prompt IDs.
Disable prefix caching and ignore EOS until the fixed output cap.
Record the model revision, seed, GPU map, command, source hash, and configuration hash.
N512 repeats of this pool are not 512 independent texts.

The frozen absolute limits are:

| Metric | Huginn limit |
| --- | ---: |
| TTFT | 1963.9920117333531 ms |
| Request-average TPOT | 186.5715059141318 ms |
| Additional-work TPOT p95 | 167.91435532271862 ms |
| ITL p99 | 192.8212538175284 ms |

The primary relative targets use the matched Native-path Zero reference.
The TTFT p95 target is a ratio of at most 0.95.
The E2E p95 target is no regression.
The ITL p99 and TPOT p95 targets are ratios of at most 1.05.
Report Maximum ITL and threshold exceedances separately.
Absolute SLO passage alone does not establish success.

## 3. Initial evidence and hypotheses

The prior recovery completed 142 offloaded requests among 384 requests.
It completed 853 offloaded Loops among 12,288 prompt recurrent Loops.
Loop/Zero geometric ratios were 1.3586 for TTFT p95 and 1.3328 for E2E p95.
The ITL p99 ratio was 1.3278.
These are frozen prior measurements, not new results.

Test four competing explanations before broad optimization.

1. Checkpoints increase P critical-path cost even without migration.
2. State preparation and handoff dominate the remaining compute saving.
3. The completion model understates D waiting and terminal Decode work.
4. Native-relative Decode protection leaves little useful execution capacity.

A frozen evidence replay must fail the requested performance targets before changes.
Fresh Native-path Zero A/A runs must establish a reproducible GPU baseline.
Use one controlled change per initial ablation.
Record Hypothesis, Evidence, Code Change, Cost, Result, and Decision for each candidate.

## 4. Execution plan

### Phase 0: preserve and inspect

Record Git status, staged-entry hash, external diff hash, GPU ownership, and burner state.
Inspect the prior final reports and their raw evidence.
Inspect local recurrent scheduling, grouped execution, and graph ownership.
Read FluidPD, DuetServe, Prefill Deflection, SGLang, and vLLM primary sources.
Check the Notion archive if its connected surface permits access.
An unavailable archive must not replace verification in GitHub originals.

### Phase 1: strong baseline and controls

Run fresh Native-path Zero A/A under the frozen anchor.
Compare Original Native, checkpoint-only without migration, selective checkpoint-only, and the previous Dynamic Loop.
Keep the Runtime revision, KV pool, graph budget, prompt pool, and arrival trace matched.
Separate no-migration checkpoint cost from actual transfer cost.
Do not infer CPU cost from submission counts.

### Phase 2: isolate costs

Measure source submission, boundary waits, graph selection, and destination quantum delays.
Use CUDA Events for explicitly labeled device-stream enclosures.
Use kernel activity only when its interference control passes.
Measure Native transfer and partial-cut transfer separately.
Do not call all transferred KV an incremental offload cost.
Measure descriptor preparation, transport submission, import, ACK, and release when observable.
Do not rename host intervals as DMA or kernel time.

### Phase 3: optimize and qualify

Prefer optimizations that address the largest measured critical-path cost.
Candidates include selective checkpoints, grouped graphs, predictive preparation, and asynchronous handoff.
Candidates also include descriptor reuse, delayed reservations, completion-aware admission, and compatible continuation batching.
These are candidates, not assumed improvements.
Retain a Native fast path behind an optional feature flag.
Revalidate D capacity at an actual GPU-complete boundary before ownership commit.
Reject stale epochs before commit.
Retain source buffers until Native ACK and transport drain.

Separate admission feasibility from immediate execution permission.
Price total completion against continued P execution from the same decision point.
Include uncertainty, transfer, known D work, and terminal execution.
The Online policy must not use future arrivals.
Already accepted work must receive progress within live Decode constraints.
Do not assume kernel preemption.
Record any completion miss and stop unsafe new admissions.

Qualify each changed path through CPU tests, N8 exact state, and N32 online exactness.
Mixed-cut qualification must include actual Online decisions.
Forced cuts are separate diagnostic controls.
Proceed to N32/N64 short serving and then N128 confirmation.
Use large Token chunks when efficient.
Do not weaken the Token baseline through asymmetric graph or chunk limits.

### Phase 4: confirmation

Use at least three paired N128 arrival seeds for a promising final method.
Add independent held-out arrival seeds when confidence intervals remain uncertain.
Balance execution order or use A/B/A controls.
Do not include only favorable selection trials.
Use independent starts as bootstrap units.
Report paired confidence intervals and zero-goodput cases without undefined ratios.
Extend to N512 only if the short comparison is promising and safe.
Finite-cohort drain rates are not sustainable capacity.

### Phase 5: decision and publication

Compare the final method with Zero and the strongest qualified Token method.
Provide actual P/D timelines and an optimization journal.
Distinguish implementation failure, measurement failure, and structural limits.
Publish only scoped changes through an isolated Git index.
Check the remote branch SHA.
Restore approved generation engines on GPUs 0–7.
Check engine ownership, generation progress, and utilization for every GPU.

## 5. Instrumentation and accounting

Report request TTFT from arrival to first output.
Report E2E from arrival to last output.
Compute percentiles from request-level timestamps.
Do not sum component percentiles.
Service TTFT requires an actual first Prefill execution timestamp.
If clean execution timestamps are unavailable, mark Service TTFT `NOT_MEASURED`.

Separate measured kernel unions, CUDA Event enclosures, host observations, and modeled predictions.
Separate physical idle, scheduler-visible idle, and Decode-safe execution budget.
The prior intrusive idle percentages are not clean Native capacity.
Check throughput, TTFT, TPOT, ITL p99, and Maximum ITL during instrumentation A/A controls.
Use a 5% interference target for primary tail measurements.
Failed captures remain diagnostic evidence.
Do not stop actual controlled Serving tests only because a collector fails.

Report full Loops only after every prompt token at that depth completes.
Report recurrent rows separately from FLOPs.
Account for Attention length, Prelude, Coda, and reinjection.
Report Native KV bytes, additional hidden bytes, descriptors, ACK time, and source lifetime separately.
No physical-capacity maximum follows from ready-free gap packing.

## 6. Resource and time policy

The campaign started at 2026-11-10 13:25:48 UTC.
Review progress after ten hours.
The user permits work beyond ten hours when evidence justifies it.
An initial sixteen-hour operational safety bound reserves ninety minutes for closure.
This bound is not a requirement to consume time.
An extension requires an explicit operational record before another GPU trial starts.
Every GPU trial has its own startup and execution timeout.
Stop only the failed owned trial after OOM, hang, or transport stall.

Run clean timing on one isolated GPU pair.
Stop approved burners on all eight GPUs during shared-host-sensitive timing.
Restore unused GPUs outside that isolated measurement lane.
Do not signal an unrelated process.
Retry only missing approved burners after restoration failure.
Explicitly select GPUs 0–7 because script defaults may select a subset.

## 7. Scientific verdict

`GO_LOOP_SPECIFIC_NET_BENEFIT` requires exactness, repeated Zero-relative improvement, protected Decode tails, and a benefit beyond strong Token or equivalent logical-layer execution.
`GO_GENERIC_PD_BENEFIT` requires a Zero-relative benefit without an established Loop-specific mechanism.
`CONDITIONAL_GO` identifies a reproducible benefit limited to explicit workloads or resource conditions.

Otherwise identify the supported failure category.
Use `NO_GO_CHECKPOINT_OVERHEAD`, `NO_GO_HANDOFF_OVERHEAD`, `NO_GO_DECODE_INTERFERENCE`, or `NO_GO_COMPLETION_ECONOMICS` for the corresponding measured limit.
Use `NO_GO_LOOP_SPECIFIC_NOVELTY` when generic alternatives explain the benefit.
Use `INCONCLUSIVE_MEASUREMENT` when measurement or reproducibility prevents a decision.
A failed implementation alone does not prove a structural impossibility.

## 8. Required final artifacts

Write `huginn_autonomous_adaptive_loop_pd_net_benefit_final_summary.md`.
Include the root-cause decomposition, optimization journal, exactness gates, and final implementation.
Include per-seed Native, Zero, checkpoint, previous Loop, final Loop, and Token comparisons.
Include P/D timelines, transfer accounting, graph behavior, and uncertainty.
Include AR execution only when environment, storage, and licenses permit a valid diagnostic.
Distinguish paper claims from actual AR measurements.
Preserve every failed trial and all raw provenance.
# Runtime extension addendum — 2026-11-10 UTC

The campaign can use the following optional extensions after exactness qualification.
Each extension has an independent feature flag.
An adopted common optimization must also apply to Native-path Zero and Token.

## Fixed-depth output delivery

`fixed_depth_submit` retains all model operations and their CUDA completion events.
It removes only an unused Host Gate result for a fixed-depth recurrent batch.
`decode_loop_grouping` uses the existing explicit multi-loop execution path.
Neither flag enables global Huginn Async scheduling.

`deferred_coda_collection` is a separate, fixed-depth extension.
It retains a CODA submission until its actual token result is ready.
The Native update function then performs delivery and termination.
The request cannot start its next token before that delivery.
This extension does not use speculative next-token execution.
Pending output storage and Request identity remain retained.

The implementation must bound the number of pending CODA batches.
It must prevent a first-output CODA wait for low-priority continuation work from blocking the ordinary Decode boundary stream.
It must preserve Native cancellation, CUDA event dependencies, KV ownership, and buffer lifetime.
N8 and N32 checks must precede serving selection.
A separate final Decode-state check must precede adoption of the best implementation.

## Independent execution lanes

`independent_prompt_stream` separates prompt submission from the Decode stream.
`independent_prompt_buffers` gives prompt execution its own bounded Native input-buffer lane.
`independent_coda_buffers` gives CODA its own bounded input-buffer lane.
These flags do not establish GPU overlap by themselves.
They do not establish a Decode safety bound.

`prompt_green_sms` can request a documented CUDA Green Context for prompt execution.
Record the requested and allocated SM counts.
Configure the prompt graph capture stream before capture.
Do not assume that changing a CUDA Graph replay stream changes captured node resources.
Do not use guessed offsets in CUDA's private stream structure.

The normal Decode stream retains its full-device resource access.
The prompt SM allocation is not a disjoint reservation against that stream.
This experiment tests spatial resource sharing.
It does not measure a physical kernel-idle window.
It does not establish a Loop-specific mechanism by itself.

## Transport selection

Whole-owned-block wire transfers can reduce descriptor count.
They also transfer invalid padding bytes.
Record semantic valid bytes and physical wire bytes separately.
Do not mark padding as valid state.

A bounded progress burst can repeat the existing Native progress function.
Producer event checks, in-flight byte credits, fragment counts, ACK, and source retention must remain enabled.
The Host-time budget is checked between calls.
It cannot interrupt a blocking Native transport call.

## Separate discovery workload

The primary workload remains Huginn R32, 2048 input tokens, and 16 output tokens.
A 4-output-token workload is a separate discovery cell.
It cannot replace a failed primary result.
It must retain every prompt token and all 32 recurrent Loops.
Its exactness check must compare every requested output token against the matching Native output prefix.
It must also compare all valid prompt KV planes and prompt hidden state.
Record the shorter output cap in every table and figure that uses it.

## Interpretation gates

Use the strongest qualified common Runtime for the final Zero baseline.
Calibrate the online policy on selection data only.
Keep confirmation Arrival Seeds separate from selection.
Evaluate both Native-relative ITL p99 and request TPOT p95.
Passing the frozen absolute SLO is not sufficient.
Report Maximum ITL and threshold exceedances.

Forced static-cut interventions are diagnostic executions.
They are not adaptive online decisions.
An exact forced cut that violates Decode protection is not a serving success.
If only one online cut is accepted, report that limitation.
Do not present it as demonstrated adaptive count diversity.

CUDA Event enclosures, kernel activity, Host intervals, structural work, and modeled costs remain separate evidence classes.
An invalid profiler trace cannot establish clean physical idle capacity.

### Effective quantum and reference selection

Check the submitted D prompt spans before scoring a quantum experiment.
A requested cap is not evidence of a bounded submission.
The Token Prelude can otherwise submit the complete recurrent suffix.
Apply the optional cap to staged Prelude and recurrent prompt phases.
Require observed coverage and cap passage in N8 and N32.
Keep Decode and the P atomic path unchanged.

Freeze the primary confirmation arms before reading any held-out result.
Use a verified full-device Native-path Zero reference.
Also retain each intervention's matched runtime control where it differs.
Do not replace a stronger Zero with a slower matched control for the scientific verdict.
Choose the strongest eligible actual adaptive Loop, fixed Loop diagnostic, and Token control from selection.
The fixed diagnostic does not satisfy the adaptive-success gate.
Prefer the smallest Decode regression among candidates that meet the selection TTFT target.
If no candidate meets that target, retain the best available performance and safety contrasts.
Record the reason and the configuration hash for each choice.
Use fresh Arrival seeds and a repeated Zero start for each seed.
Keep the short-output discovery and stock AR diagnosis separate.
An empirical serving failure can support a scoped implementation NO-GO without proving that every possible scheduler lacks physical headroom.
# Additional experiments after the first execution-path audit

## Process exit is a separate correctness condition

The coordinator can report a complete cohort before both GPU workers exit.
Every new qualification must record the actual P and D process exit codes.
A normal exit must have code zero for both workers.
Forced cleanup must retain the Native writer-before-receiver shutdown order.

Graph cleanup occurs after all Native work drains and before NIXL deregistration.
The cleanup resets prompt, Decode, and CODA graph executables in their owning contexts.
This cleanup does not change the measured model execution.

## Explicit Decode priority and hardware work queues

The campaign can compare strict ready-Decode submission with Native Refill ordering.
This comparison requires independent prompt buffers and the qualified delayed collector.
It must not starve an already committed continuation.
The current serial dynamic policy therefore cannot use this extension without a new progress design.

A separate candidate sets `CUDA_DEVICE_MAX_CONNECTIONS=32` before CUDA initialization.
All compared methods receive the same setting.
The campaign must measure its effect rather than assume stream concurrency.
Green contexts do not guarantee concurrent execution or forward progress.

## Owner-thread source progress pipeline

The optional source pipeline progresses completed transfers during another request's long graph submission.
It uses the existing Native producer events and transport path.
It does not change the current prepared batch.

The command reader accepts only capacity advertisements and negotiation replies for other requests.
The reader stops at the first unrelated command.
The normal worker later consumes that command in its original order.
Cancellation and current-request mutations must not cross the prepared-batch boundary.

The pipeline requires at least eight incomplete Native loop events from the current submission.
That condition shows queued device work, not physical GPU saturation or guaranteed overlap.
The same extension applies to Native Zero, Loop, and Token methods.
N8 and N32 qualification must precede serving comparisons.

## Predictive checkpoint preparation

The destination can advertise its remaining known Decode outputs and a drain-time forecast.
The forecast uses the requested output cap, delivered outputs, and past Decode cycle times.
It does not use future arrivals.

The source can use that forecast to expose a future checkpoint.
The forecast is not an admission or execution grant.
The source restores the actual advertisement immediately after checkpoint selection.
Every offer, commit, and destination quantum uses the unchanged current-state safety rules.
Token and Loop checkpoint preparation use the same forecast policy.

The separate 2048/4 discovery workload retains full R32 work for every requested token.
It does not replace the frozen 2048/16 Primary Anchor.
Its state checks compare the complete prompt state and every requested output prefix against the frozen Native reference.

## Monotone immutable KV protocol candidate

Early KV bytes do not transfer recurrent computation ownership.
The source must announce the immutable, full-prompt KV frontier through the coordinator.
The coordinator must check the current source owner before it forwards that announcement.

The destination can negotiate a later cut after immutable fragments arrive.
That cut must include every announced shallow plane.
The destination must remain in the receiving phase without a final fragment count or an activated request.
The announcement cannot include hidden state.

The optional fork changes exactly two policy predicates from the frozen parent protocol.
Each run preserves the parent hash, derived hash, changed line locations, and complete derived source.
The experiment does not change the Native transport implementation.
Native expected counts, producer events, import validity, ACK, and source retention remain required.

N8 and N32 qualification must include a real commit after at least one immutable fragment arrives.
An empty-receive negotiation does not qualify this new path.
The strongest comparison must include Native Zero with the same early-transfer opportunity.
A Loop-only streaming prototype cannot establish an advantage over a fully optimized Token streaming implementation.

## Disjoint SM intervention

The earlier prompt-only SM cap leaves Decode access to the full device.
The new diagnostic requests disjoint Decode and prompt groups from one CUDA resource split.
It records the actual resource counts and the common split indices.
Do not infer kernel occupancy from those counts.

The driver provides independent resource owners for the two execution paths.
The prompt owner captures prompt graphs.
The Decode owner captures Decode graphs and owns its boundary and readback streams.
Each graph teardown uses its owning context.

Compare an identically provisioned Zero control with the split method.
Also compare against the strongest unrestricted Native-path Zero.
A slower resource-restricted denominator cannot establish a Native-relative GO.
The diagnostic must retain exact state and the existing absolute SLO.

## Interruption recovery

Use a durable controller log for long blocks.
Do not restart a cell with a terminal GPU receipt and verified worker exits.
Recheck any missing CPU qualification from its preserved artifacts.
Do not replace the original launcher return code.
Resume unfinished cells only after the prior owner and its pending launcher stop.
