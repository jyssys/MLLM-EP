# Optimization journal

This file records selection experiments. It is not the final scientific verdict.
All durations below identify their measurement scope.

## H1 — Checkpoints create a cost before migration

Attribution correction: H14 identifies different wire credits and draining limits.
The initial comparison below does not isolate checkpoint cost.
Retain the measurements. Use matched-wire controls for the final attribution.

- Hypothesis: exposing every candidate boundary increases the source critical path.
- Evidence: the fresh Zero A/A runs reproduced the anchor configuration.
- Evidence: one matched N128 run used six checkpoints and performed no migration.
- Result: its TTFT p95 was 26571.8441 ms. The adjacent Zero run measured 17058.4852 ms.
- Limitation: this comparison uses one arrival seed. It does not isolate CUDA kernel time.
- Decision: implement selective checkpoint exposure. Retain the six-checkpoint control.
- Code: `huginn_net_benefit/planner.py` and `runtime.py`.
- Correctness: selective checkpoints passed N8 prompt state and output checks against the frozen Native reference.
- Limitation: the measured N8 offloads all used 16 remaining Loops. Adaptive count diversity is not established.

## H2 — Fixed-depth Decode need not return to Python after every Loop

- Hypothesis: explicit multi-loop batches can reduce control overhead without skipping recurrent work.
- Code: `huginn_net_benefit/fixed_decode.py` uses the existing multi-loop model execution path.
- Safety: it requires fixed minimum and maximum depth. It rejects stale and duplicate transitions.
- Evidence: N8 output, prompt KV, prompt hidden state, and full-work checks passed.
- Decision: perform N32 qualification before clean performance selection.
- Limitation: prompt state checks do not establish final Decode KV equality. A separate final-state check is prepared.
- Fairness: any adopted Decode optimization must also apply to the Zero and Token baselines.

## H3 — Immutable KV planes can move before the final cut

- Hypothesis: progressive transfer can overlap completed KV transfer with later source computation.
- Code: `huginn_net_benefit/progressive.py` and the optional connector extension in `runtime.py`.
- Safety: each completed KV plane is published once. The producer CUDA event protects its transfer.
- Safety: Native commit, hidden handoff, expected fragment counts, and ACK still control destination activation and source release.
- Failure: the first implementation passed N8 exact state checks but performed no measured migration.
- Root cause: the inherited ready-queue calculation excluded requests with queued early wire fragments.
- Correction: the extension counts source-owned, dependency-ready requests with immutable early transfers.
- Result: the second N8 implementation passed exact state checks. Two measured requests moved 32 Loops in total.
- Limitation: this is an instrumented qualification result. It is not a serving speedup or a Decode safety result.
- Decision: retain the failed evidence. Perform N32 qualification and a separate transfer control.

## H4 — Synchronous collection adds avoidable fixed-depth control work

- Hypothesis: the existing asynchronous execution path can reduce unnecessary collection at recurrent boundaries.
- Code: the first candidate requests the existing asynchronous Runtime option.
- Result: N8 and N32 output, prompt KV, prompt hidden state, and full-work checks passed.
- Audit: the resolved execution receipts show `effective_async_scheduling=false` on both workers.
- Root cause: the qualified Huginn configuration forces synchronous execution.
- Decision: do not run a performance campaign for this inactive option. Preserve its qualification evidence as a configuration audit.
- Limitation: these checks do not qualify asynchronous Huginn execution. Enabling it requires a separate Runtime design and exactness validation.

## Measurement rules

- CUDA Event intervals are execution enclosures. They are not a kernel union.
- Inclusive host counters overlap. Their sum is not a critical-path decomposition.
- NIXL completion observation includes polling delay. It is not isolated DMA duration.
- Nsight graph intervals include internal gaps. They cannot establish exact physical kernel idle.
- State qualification runs cannot supply clean latency evidence.
- Selection results cannot replace held-out confirmation.

## H5 — Fixed-depth logical submission can avoid an unused readback

- Hypothesis: synchronous Huginn can retain device dependencies without reading an unused Gate result after every recurrent submission.
- Code: `fixed_submit.py` uses Native `update_after_submission` only for exact fixed-depth Decode batches.
- Safety: the extension leaves model computation and graph replay unchanged.
- Safety: Native completion events and bounded submission buffers retain device dependencies and storage.
- Safety: CODA still collects actual generated tokens. No early exit or global Async mode is enabled.
- Result: separate N8 candidates passed output, prompt KV, prompt hidden, full-work, and lifetime checks.
- Result: one candidate retained single-loop submissions. The other grouped 32 explicit Loops.
- Decision: require N32 qualification and clean runtime selection.
- Limitation: these N8 qualification timings do not establish serving improvement.

## H6 — Existing graph events may support transfer without source checkpoints

- Evidence: Native packed recurrent graph execution already records a completion event after each replay.
- Hypothesis: an immutable prefix of KV planes can use that existing event for early transfer.
- Code: `early_tap.py` observes the selected completed graph depth inside the atomic Native submission.
- Safety: the source still executes all 32 Loops. The final Native handoff sends only unpublished planes and the required hidden state.
- Fairness: this is a candidate Zero optimization as well as a possible Loop optimization.
- Status: GPU qualification is prepared. No correctness or speedup claim applies yet.

## Diagnostic failure — final Decode state observer

- Failure: the first observer selected startup requests whose seeds overlapped the measured cohort.
- Evidence: its output-cap assertion failed during Warm-up, before the measured cohort.
- Correction: require the measurement marker before selecting final Decode states.
- Correction: require a valid fresh Native reference before executing a comparison observer.
- Decision: preserve the failed namespaces. Prepare new reference and candidate namespaces.
- Limitation: this observer failure does not invalidate earlier prompt-state qualification. It does not establish final Decode-state equality either.

## Diagnostic failure — collector lifecycle inside the measured cohort

- Evidence: `cudaProfilerStart` blocked for about 3.7 seconds. `cudaProfilerStop` blocked for about 6.3 seconds.
- Result: the short graph trace changed TTFT p95 and ITL p99 beyond the allowed interference target.
- Decision: retain this trace for diagnosis only. Move collector start before Warm-up and stop after cohort completion.
- Limitation: graph intervals cover internal gaps. Their union is not exact kernel busy time.

## H7 — Optimized Decode is a stronger common baseline

- Result: fixed-depth submission and grouping passed N8 and N32 prompt-state and output qualification.
- Result: a separate N8 observer also matched final valid Decode KV, final hidden state, and output tokens against a fresh Native reference.
- Limitation: the final-state observer synchronizes and hashes large states. Its latency is not serving evidence.
- Selection result: fixed submission plus grouping measured ITL p99 of 91.5166 ms on seed 2026110940.
- Selection controls: the original Zero measured 145.1623, 148.4301, and 123.0373 ms in three starts of that trace.
- Limitation: these controls also show material Native A/A tail variation. They are not three held-out confirmations of a Loop benefit.
- Decision: apply any selected Decode optimization to Zero, Loop, and Token. Recalibrate the common reserve before interpreting offload rejection.
- Limitation: no TTFT improvement has been established for the fixed-depth submission optimization.

## H8 — Source pressure omitted synchronous device work

- Evidence: the previous pending-source counter reads only the asynchronous in-flight list.
- Evidence: the resolved Huginn Runtime is synchronous. Previous admission records reported zero pending source device requests.
- Code: `source_events.py` also checks Native per-request producer CUDA events.
- Safety: the extension counts only live source-owned requests. It does not relax the distinct-request pressure threshold.
- Limitation: a pending CUDA event can include queued work. It does not prove physical kernel saturation.
- Status: CPU tests pass. GPU qualification is required before performance use.

## H9 — Native wire progress cadence may delay handoff

- Evidence: the Native connector submits at most one bounded fragment per request per progress call.
- Hypothesis: a bounded burst can submit more ready fragments before the next long model submission.
- Code: `wire_burst.py` repeats the unchanged Native progress call. It stops on no progress, a step limit, or a host-time budget.
- Safety: producer events, in-flight byte credits, fragment counts, Native ACK, and source retention remain unchanged.
- Limitation: the time budget is checked between calls. It cannot interrupt a blocking NIXL API call.
- Status: CPU tests pass. GPU qualification is prepared.

## H10 — Descriptor count and payload size form a trade-off

- Code: `block_wire.py` copies only physical pages owned by the exact semantic frontier.
- Cost: whole-page copies include invalid padding and unused layer bytes. They can increase wire bytes from about 5.71 GB to about 11.07 GB for this geometry.
- Result: N8 and N32 prompt KV, hidden state, output, full-work, and lifetime qualification passed.
- Limitation: copying extra invalid bytes does not make those bytes valid. Native import uses the original semantic frontier.
- Decision: compare clean performance before adoption. Fewer descriptors alone do not establish a benefit.

## H11 — Independent D streams need independent buffers

- Hypothesis: a low-priority continuation stream can overlap compatible work with a high-priority Decode stream.
- Result: the initial separate-stream candidate passed N8 and N32 exact-state qualification.
- Limitation: it still shares a bounded submission-buffer queue. Streams alone do not establish physical overlap or Decode safety.
- Code: an additional candidate gives prompt execution a separate Native submission queue and input buffers. Per-request dependencies and distinct graph pools remain required.
- Status: the separate-buffer variant is not GPU-qualified yet.

## Diagnostic failure — steady graph collection

- Failure: moving collector start before Warm-up did not remove the profiler-associated NIXL stall.
- Evidence: source thread stacks stopped inside `nixl_cu12._api.transfer`. The measured cohort was incomplete.
- Decision: stop only the verified namespace. Preserve the profiler output, stacks, and watchdog receipt.
- Limitation: the cause below the Python API remains inconclusive. The trace is not valid clean performance evidence.
- Decision: continue with actual GPU interventions and clean serving. Do not infer physical idle capacity from host gaps or sampled NVML utilization.

## H12 — Separate streams cannot remove a blocking output collector

- Evidence: synchronous `StageEngine._step` calls `Submission.collect` for CODA. Collection waits for the GPU event.
- Consequence: the Host cannot submit an independent continuation during that wait.
- Code: `deferred_coda.py` retains fixed-depth CODA submissions. It delivers only completed results through the unchanged Native update path.
- Safety: a pending request remains in CODA. It cannot start its next token before actual token delivery.
- Safety: first-output CODA for a continuation waits until its prompt producer event is ready. It cannot place a long prompt wait at the head of the Decode boundary stream.
- Code: optional separate CODA submission buffers prevent a pending boundary output from retaining the Decode core input-buffer lane.
- Limitation: this is a new scoped extension. It is not the unsupported global Huginn Async mode.
- Status: CPU checks pass. GPU exactness and serving validation are pending.

## H13 — A resource-bounded prompt stream may reduce Decode interference

- Hypothesis: a prompt stream with a bounded SM allocation can leave execution resources available for Decode.
- Design: use the installed PyTorch Green Context wrapper and the documented CUDA Driver API.
- Safety: do not guess internal CUDA stream-structure offsets for `libsmctrl` on this driver.
- Safety: configure the prompt graph capture stream before capture. A different graph launch stream does not change captured kernel-node resources.
- Limitation: a Green Context does not guarantee concurrent execution or a Native-relative Decode bound.
- Status: a 64-SM candidate is prepared. The actual allocation must be queried and recorded. GPU qualification is pending.
- Mechanism scope: this is spatial resource sharing. It is not proof of a physical kernel-idle window or a Loop-specific novelty.

## Selection results — transport and integer-validation candidates

- Scope: one paired selection arrival trace, seed 2026110940. These are not held-out confirmations.
- Control: the adjacent optimized Zero measured TTFT p95 of 9676.2237 ms, ITL p99 of 91.6437 ms, and TPOT p95 of 80.6362 ms.
- Whole-owned-block wire: TTFT p95 was 18546.7108 ms. ITL p99 was 142.2809 ms. The candidate did not provide a benefit.
- Native graph-event early KV publication: TTFT p95 was 9728.2826 milliseconds. ITL p99 was 125.6679 milliseconds. The original-runtime controls do not establish a clear TTFT benefit.
- Exact integer validation: TTFT p95 was 9969.8898 ms. ITL p99 was 126.3065 ms. No clear serving benefit is established against the original-runtime controls.
- Fairness: the transport and integer candidates above did not yet include the improved Decode flags. Compare their Decode tails to the matching original-runtime controls, not only to the stronger optimized Zero.
- Dynamic Token with the improved Decode flags: one request moved a 256-token suffix. TTFT p95 was 14415.2946 ms. ITL p99 was 104.3119 ms. TPOT p95 was 90.4763 ms.
- Decision: do not expand this Token policy as a successful method. Retain it for matched comparison and diagnose no-migration control cost.
- Limitation: this result does not establish a structural limit for every Token policy.

## Integration failure — missing selective checkpoint configuration

- Failure: the first combined fast-Decode/Loop N8 cell failed during engine construction. It did not reach the measured cohort.
- Root cause: the optional selector read a required `boundary_checkpoints` key. Its inherited configuration used the qualified Runtime default instead.
- Correction: use the same default in the selector. Prepare explicit common candidate frontiers in new `v2` namespaces.
- Preservation: do not overwrite the failed namespace or its dependent preflight failure receipts.
- Limitation: this integration failure is not a performance NO-GO or an exactness failure.

## H14 — A nested configuration replacement changed the comparison

- Evidence: optimized Zero specifies 8 GiB in flight and zero draining requests.
- Evidence: the recovery generator replaces `native_transport_limits` with a dictionary that contains only a shutdown timeout.
- Result: inherited recovery Loop and Token runs resolve to 256 MiB in flight and eight draining requests.
- Consequence: earlier bundled comparisons do not isolate checkpoint or policy cost.
- Correction: preserve those observations. Add explicit matched 8 GiB/zero-draining cells.
- Correction: add a resolved Native PDConfig receipt for all future arms, including Native.
- Limitation: changing both credit and draining policy does not identify their separate effects.
- Decision: require common resolved wire policy in final confirmation. Do not use the confounded controls to declare a checkpoint-specific NO-GO.

## H15 — Partial preparation builds a descriptor that it then discards

- Evidence: Native partial-path preparation first builds the ordinary fused-loop forward list. It then removes the CODA forward for a partial recurrent depth.
- Hypothesis: avoiding that unused CODA descriptor can reduce partial-source Host preparation cost.
- Code: `partial_prepare.py` returns no CODA descriptor only in the same partial fused-PRELUDE case.
- Safety: the qualified partial-path CODA filter remains. Full-depth CODA and Token-prefix CODA remain unchanged.
- Status: CPU checks pass. GPU qualification is pending.
- Limitation: no measured time saving is established yet.

## Matched-wire static intervention — first selection result

- Scope: one N128 selection start, Arrival Seed 2026110940. Native-relative confirmation is pending.
- Configuration: the static serving cell uses the same 8 GiB in-flight limit and zero draining requests as optimized Zero.
- Exactness: the static execution extension passed N8 and N32 valid state, output, work, and lifetime checks. Those state cells used the inherited qualification wire setting. Final adoption requires matched-wire qualification of the selected combination.
- Actual work: all 128 requests moved their final recurrent Loop. All requests completed.
- Result: static Loop31 TTFT p95 was 7234.2945 ms. Adjacent optimized Zero measured 9676.2237 ms.
- Result: completed finite-cohort QPS was 1.70970 versus 1.63198.
- Failure: ITL p99 was 123.1214 ms versus 91.6437 ms. This fails the Native-relative Decode target.
- Decision: do not declare GO. Test delayed output collection and resource-bounded concurrent execution.
- Structural accounting: a partial Huginn cut also moves valid CODA work from P to D. One of 32 recurrent Loops is not one of 32 equal total-runtime shares.

## H16 — Decode cohort retirement changes the asynchronous result path

- Failure: deferred CODA version 1 measured Zero ITL p99 of 149.8726 ms on the selection trace.
- Comparison: the preceding fixed-depth Zero measured 91.6437 ms. This is not an acceptable stronger baseline.
- Hypothesis: another Decode core group can enter before the previous cohort delivers its completed tokens.
- Code: version 2 retains Native Decode cohort retirement. Independent prompt preparation remains permitted.
- Result: matched-wire N8 and N32 output, valid state, work, and lifetime checks passed for Zero and static Loop31.
- Selection result: Zero TTFT p95 was 9584.1625 ms. ITL p99 was 88.4754 ms.
- Selection result: static Loop31 TTFT p95 was 7398.9598 ms. ITL p99 was 107.5914 ms.
- Actual work: static Loop31 completed the final full-prompt Loop for all 128 requests.
- Failure: Loop ITL remains above the Native-relative five-percent target.
- Limitation: these are one-start selection observations. They do not establish an independent confirmation or adaptive cut diversity.
- Decision: retain the Zero improvement. Test the already prepared resource-bounded prompt execution before final selection.

## H17 — A stopped worker receipt does not prove a clean process exit

- Evidence: six Green-context Loop namespaces contain a `cudaGraphExecDestroy` segmentation-fault signature after output completion.
- Scope: those cohorts completed output and intermediate-state comparisons. They did not establish safe Runtime teardown.
- Confirmed code issue: Native `StageClient.join` waits for process exit but does not reject a nonzero exit code.
- Hypothesis: graph executable destruction occurs too late in the context and transport lifetime.
- Limitation: the log alone does not establish the exact driver-level cause or the failing worker role.
- Code: `exit_integrity.py` records real worker exit codes. It rejects unsuccessful normal exits without interrupting Native forced-cleanup ordering.
- Code: `graph_lifetime.py` drains owned work and resets graph executables before Native NIXL deregistration.
- Safety: cleanup requires no pending transfer. It preserves Native ACK and source-retention ordering.
- Result: the first new Zero N8 qualification passes output, state, work, and lifetime checks. Both worker exit codes are zero.
- Status: real Green-context prompt execution still requires this new lifetime qualification.
- Decision: preserve the failed logs. Do not adopt a path solely because the coordinator reported a complete cohort.

## H18 — Refill ordering is not strict Decode-first submission

- Code evidence: Native Refill checks its Decode batch bound before recurrent Decode, but can select prompt work before that bound.
- Hypothesis: submitting already-ready Decode first can reduce the Host delay before its GPU launch.
- Code: `decode_priority.py` first selects ready Prelude, CODA, or recurrent Decode. Empty selections still permit the Native fallback policy.
- Code: the optional boundary-priority flag also prioritizes the Decode boundary and readback streams before graph capture.
- Safety: the candidate requires the qualified delayed collector and independent prompt buffers. It does not preempt a running kernel.
- Status: CPU tests pass. New N8/N32 and clean serving cells are prepared.
- Limitation: this candidate is not yet compatible with the serial dynamic budget. Do not apply it to committed dynamic work without a separate progress design.

## H19 — Streams can share finite hardware work queues

- Evidence: the observed worker environment leaves `CUDA_DEVICE_MAX_CONNECTIONS` unset.
- Reference: CUDA 12.9 documents a default of eight work queues and permits values from one through 32.
- Reference: NVIDIA warns that hardware connections can prevent concurrent Green-context progress, even with disjoint SM resources.
- Hypothesis: multiple Native, graph, transport, and prompt streams can incur false queue dependencies.
- Code: a per-namespace launch option sets 32 connections before worker CUDA initialization.
- Fairness: every compared Zero, Loop, and Token arm receives the same option.
- Safety: the option does not change weights, Attention, graph geometry, state ownership, or SLO thresholds.
- Status: qualification and serving cells are prepared. Hardware queue assignment is not measured.
- Decision: test this hypothesis directly before another broad SM-budget sweep.

References: [CUDA 12.9 environment variables](https://docs.nvidia.com/cuda/archive/12.9.0/pdf/CUDA_C_Programming_Guide.pdf), [Green contexts](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html).
# H20 — Progress completed transfers during another source graph submission

Hypothesis: A long source submission delays control messages and Native transfer progress for a different, completed request.

Evidence: `StageWorker.run` processes messages and `connector.progress` outside `StageEngine.step`.
`ModelRunner._execute_batch` submits all prepared recurrent forwards before it returns.
This code structure permits a delay. Its critical-path cost is not yet measured.

Code change: `source_pipeline.py` processes only safe control messages for other requests.
The reader stops at the first unrelated command and preserves FIFO order.
The same owner thread calls Native transfer progress after at least eight queued Native loop events remain incomplete.
The extension does not change the current batch, model math, producer event, or Native ACK rule.
Native Zero uses the same extension.

Cost: Additional event queries and owner-thread control work.
A Native progress call can exceed the requested polling interval.
Pending events do not prove GPU overlap or guarantee a compute horizon.

Validation: Four CPU FIFO and current-request isolation tests passed.
GPU state qualification and performance are NOT_RUN at this entry.

Decision: Test N8, then N32, before any serving claim.

## H21 — Prepare a checkpoint before known Decode work drains

- Hypothesis: a source request can lose a later empty Decode epoch when its initial decision selects one atomic R32 submission.
- Evidence: the source cannot negotiate an interior boundary after that atomic submission begins.
- Code: `predictive_prepare.py` advertises remaining known outputs and a forecast from past Decode cycle times.
- Code: both Token and Loop selection can use the forecast to prepare a checkpoint.
- Safety: preparation changes neither ownership nor actual destination admission.
- Safety: every offer, commit, and quantum still uses current destination conditions.
- Cost: failed forecasts can add source checkpoints without moving work.
- Validation: three CPU tests cover stale advertisements, unavailable destinations, immutable advertisements, and late drain forecasts.
- Status: GPU qualification is NOT_RUN at this entry.
- Decision: test the Primary Anchor and a separate 2048/4 discovery workload. Do not merge their conclusions.

## H22 — Separate immutable KV receipt from ownership commit

- Hypothesis: early transfer can reduce the post-decision handoff path without adding source scheduling checkpoints.
- Evidence: the previous policy rejects an offer after any destination fragment arrives.
- Evidence: a fragment from an immutable shallow KV plane does not contain the final continuation hidden state.
- Code: `monotone_early_protocol.py` derives an auditable optional fork from the frozen parent protocol.
- Code: the fork replaces exactly two empty-receive predicates with a monotone frontier predicate.
- Safety: the coordinator accepts frontier announcements only from the current source.
- Safety: the destination requires a full-prompt, KV-only frontier no deeper than the proposed cut.
- Safety: a cancelled, activated, committed, or non-receiving destination cannot use this exception.
- Safety: Native producer events, expected fragment counts, final hidden import, validity checks, ACK, and source retention remain unchanged.
- Safety: the destination removes the frontier record when Native transfer ownership ends.
- Cost: extra metadata messages, early descriptor work, and longer destination KV occupancy.
- Validation: six CPU tests passed. GPU qualification must include an actual commit after immutable fragments arrive.
- Limitation: this is a process-local research fork, not an upstream Runtime extension.
- Limitation: this candidate supports Loop frontiers. It does not establish superiority over an equally optimized Token streaming path.
- Status: GPU qualification is NOT_RUN at this entry.
- Decision: compare with an early-transfer Native Zero control. Do not weaken Native transfer validation.

## H23 — Allocate disjoint Decode and prompt SM resources

- Hypothesis: a prompt-only Green cap still competes with a Decode path that can use the full device.
- Evidence: reducing the prompt-only cap increased ITL p99 for the observed full-prompt Loop31 selections.
- Code: `green_pair.py` uses disjoint groups from one CUDA SM resource split.
- Code: Decode core, boundary, readback, and capture streams belong to the Decode resource owner.
- Code: the prompt graph uses the other resource owner.
- Safety: graph executables drain and reset in their owning contexts before Native transport cleanup.
- Cost: fewer available Decode SMs and shared HBM and hardware queues.
- Validation: four CPU resource-selection tests passed. Actual GPU allocation and state qualification are NOT_RUN at this entry.
- Decision: use an equally provisioned Zero diagnostic and the strongest full-device Zero denominator.
- Limitation: disjoint resource provisioning does not prove concurrent execution or a safety guarantee.
- GPU allocation: the documented split provides 112 SMs for Decode and eight SMs for prompt work. Twelve SMs remain unassigned.
- Validation: the provisioned Zero passes N8 and N32 exact-state checks and actual worker exit checks.
- Result: its one N128 selection measures TTFT p95 of 9789.06 milliseconds and ITL p99 of 96.53 milliseconds.
- Validation: Loop31 also passes N8/N32 exact state and actual worker exit checks.
- Result: the clean N128 Loop31 selection moves 128 full-prompt Loops. TTFT p95 is 8356.11 milliseconds. ITL p99 is 292.00 milliseconds.
- Decision: reject this provisioned candidate. Its clean ITL p99 exceeds the frozen 192.8213-millisecond limit. Do not extend it into repeated serving.
- Limitation: the measurements do not separate shared HBM interference, hardware queue effects, and the cost of the small prompt partition.

Reference: [CUDA 12.9 Green contexts](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html).

## H24 — Recover one terminal GPU cell after the API restart

- Evidence: the interrupted launcher records exit code 120. The two GPU workers record zero exit codes.
- Evidence: the terminal Native receipt and full state artifacts remain present.
- Code: the new serial controller writes to a durable file instead of the API output pipe.
- Validation: CPU analysis rechecks the completed N8 result against the frozen reference.
- Decision: retain the valid result. Resume only unfinished cells and the failed bootstrap variant in a new namespace.

## H25 — Check the checkpoint delay before calling it launch overhead

- Hypothesis: more P submissions cause the observed slowdown through CPU launch cost alone.
- Evidence: the initial checkpoint diagnostic increases the number of submissions from 128 to 896.
- Evidence: the measured inclusive `submit_prompt` owner CPU time decreases from 54.82 to 34.13 seconds.
- Evidence: the measured inclusive scheduler owner CPU time increases from 1.14 to 9.32 seconds.
- Limitation: the diagnostic transport configurations differ. The counters overlap and do not measure the critical path.
- Method: partition Host TTFT at the recorded milestones for all 128 paired requests in the wire-matched runs.
- Result: the all-checkpoint arm adds 7998.59 milliseconds of mean TTFT without migration.
- Result: arrival to admission adds 7752.06 milliseconds. P start to compute-release receipt adds 199.48 milliseconds.
- Result: the selective no-migration arm adds 71.01 milliseconds of mean TTFT.
- Decision: preserve the atomic path where possible. Do not attribute the slowdown to submission count or exclusive launch time.
- Evidence: `costs/controls.json` and `paired_host_attribution/`.

## H26 — Evaluate the hardware connection hypothesis

- Hypothesis: a higher CUDA connection limit lets the independent prompt stream make progress without a Decode queue dependency.
- Code change: set `CUDA_DEVICE_MAX_CONNECTIONS=32` before CUDA initialization for matched Zero and Loop31.
- State: the SM16 Zero and Loop31 N8 and N32 gates pass. Both workers exit with code zero.
- Cost: actual hardware work-queue assignment remains NOT_MEASURED.
- Result: one N128 start measures Zero ITL p99 of 88.2143 milliseconds.
- Result: Loop31 ITL p99 is 185.9927 milliseconds. TTFT p95 falls from 9737.6589 to 7609.8589 milliseconds.
- Decision: do not adopt this resource policy. The relative Decode gate still fails.
- Limitation: the selection does not isolate hardware queue assignment or HBM interference.

## H27 — Prepare a fixed-dose control

- Hypothesis: moving fewer requests can retain some P queue relief without the full-placement Decode penalty.
- Code change: `migration_dose_cells.py` selects one request per fixed request-index block.
- Safety: the existing Native mixed-cut protocol reserves each legal cut. It does not relax state or lifetime validation.
- Cost: N8 and N32 state qualification precede any clean N128 result.
- Status: the candidate is optional and NOT_RUN at this entry.
- Decision: use it only if the full-placement and quantum controls leave a material question about the migration fraction.
- Limitation: a forced assignment is not an adaptive policy. A new comparison must pair it with the same execution policy's Zero.

## H28 — Correct the effective Token execution quantum

- Hypothesis: smaller Token continuation groups reduce the longest uninterrupted submission.
- Evidence: `quantum6_token1984_1` requests a one-Loop cap.
- Evidence: all 128 measured D prompt spans actually submit 64 tokens through 32 Loops in the Prelude phase.
- Evidence: the SM32 Token selection also submits 32 Loops, despite its requested eight-Loop cap.
- Cause: the first cap only intercepts the recurrent prompt phase. The inherited Prelude can group all remaining Loops.
- Code change: `prefill_group.py` adds an optional cap on the staged prompt Prelude. Decode and the P atomic path retain their plans.
- Cost: the corrected route may add more Host scheduling and graph submissions.
- Validation: two CPU tests pass. The N8/N32 gate now checks the observed D span histogram as well as state and lifetime.
- Decision: retain the earlier runs as uncapped Token measurements. Test corrected 64-token and 256-token suffixes in new namespaces.
- Limitation: an observed submission bound does not measure kernel occupancy or preemption.
- Result: corrected Token64 passes N8, N32, all N128 outputs, and the observed cap gate.
- Result: it submits 512 D prompt spans. Every span uses 64 tokens and eight Loops.
- Result: TTFT p95 is 9704.70 milliseconds. ITL p99 is 148.41 milliseconds.
- Reference: the uncapped full-device Token64 has TTFT p95 of 9034.09 milliseconds and ITL p99 of 121.10 milliseconds.
- Limitation: these single starts do not isolate device interference from Decode cohort feedback.
- Decision: the smaller group does not protect Decode in this selection. Keep the larger suffix test separate.

## H29 — Test early immutable KV with an already selected cut

- Hypothesis: early publication may improve an exact fixed-cut handoff independently of any adaptive admission result.
- Evidence: the Native graph already exposes producer events for immutable R16 KV.
- Code change: reuse the early Native Zero configuration with cut R31 on P. Retain the same owner-thread progress path.
- Safety: reserve the cut before execution. Keep Native final hidden state, validation, and ACK.
- Status: N8/N32 and N128 cells are prepared. They are NOT_RUN at this entry.
- Decision: run only if the early-transfer Zero qualifies and the remaining GPU budget permits the matched test.

## H30 — Separate useful tiling from numerical padding

- Hypothesis: a smaller non-preemptive quantum protects Decode without losing the full-prompt Loop's benefit.
- Evidence: the 128-token Loop31 arm submits 2048 D prompt spans for 128 requests.
- Evidence: each span contains 128 tokens and one Loop. All 128 Loops complete.
- Validation: N8 and N32 pass. Every N128 output matches Zero.
- Result: TTFT p95 is 7469.2561 milliseconds. ITL p99 is 115.5143 milliseconds.
- Reference: untiled Loop31 on the same priority Runtime has TTFT p95 of 7238.1349 milliseconds.
- Reference: untiled ITL p99 is 104.6810 milliseconds.
- Cost: the observed D submission count rises from 128 to 2048.
- Cost: the structural padded dense estimate rises by 25.7% for D prompt work.
- Limitation: the structural estimate is not a hardware counter or an isolated timing attribution.
- Decision: test 256- and 512-token tiles, which match the qualified 256-row projection geometry.

## H31 — Check natural continuation batching opportunities

- Hypothesis: batching compatible imported requests can amortize short-quantum overhead.
- Evidence: the full-prompt, tiled Loop31, and full-device Token64 selections each move 128 requests.
- Evidence: each selection has zero intersecting D receive-to-first-output Host intervals.
- Evidence: median receive spacing is 568.57, 570.85, and 588.53 milliseconds, respectively.
- Evidence: median receive-to-output intervals are 82.99, 179.55, and 176.63 milliseconds, respectively.
- Cost: waiting for a second continuation can delay the current first output.
- Decision: do not add deliberate batching waits to this anchor without evidence of compatible ready work.
- Limitation: these are Host receipt intervals, not actual GPU availability or all possible higher-load behavior.
- Evidence: `runtime_observations/` for the three matched Seed 2026110940 selections.

## H32 — Retry only after a completion-miss epoch drains

- Hypothesis: the permanent completion-miss latch can suppress later safe opportunities.
- Evidence: the 2048/4 forecast selection commits nine four-Loop continuations before request index 16.
- Evidence: the last request exceeds a 350-millisecond reservation. Its first output arrives 466.28 milliseconds after commit.
- Evidence: 109 requests subsequently record only destination unavailability.
- Cause: the inherited empty-epoch recovery accepts a Decode tail reason, not a completion-reservation reason.
- Code change: `drained_recovery.py` permits one measured retry after two reservation periods.
- Safety: no active Decode, committed continuation, live offer lease, or pending quantum may remain.
- Safety: retry only half the failed suffix or less. Existing offer and commit checks remain in force.
- Safety: do not change the reservation deadline, Decode SLO, Native ACK, or committed owner.
- Cost: later Loop checkpoints may use a smaller suffix. Token still has its separate source frontier selection.
- Validation: CPU tests cover the cooldown, live work, reasons, smaller cap, and one-retry limit.
- Status: GPU N8, N32, and N128 cells are prepared. They are NOT_RUN at this entry.
- Limitation: the prototype cannot prove a future completion guarantee. It must measure new misses.
- Result: the diagnostic N128 moves four Loops on two requests, two Loops on eight requests, and one Loop on one request.
- Result: all 128 outputs match Zero. Observed full recurrent work also passes.
- Result: one drained-epoch retry occurs after a four-Loop reservation miss. It lowers the future ceiling to two Loops.
- Result: a later one-Loop reservation miss stops further retries.
- Result: TTFT p95 is 9929.42 milliseconds. ITL p99 is 82.47 milliseconds.
- Reference: matched Zero has TTFT p95 of 9662.43 milliseconds and ITL p99 of 77.42 milliseconds.
- Decision: reject this retry policy as a net-benefit method. The run demonstrates adaptation, not an improvement.
- Limitation: N32 passes exact state but does not exercise the retry trigger. N128 exercises it with output and full-work checks. No post-retry intermediate-state qualification is claimed.

## H33 — Separate early KV timing from transfer volume

- Hypothesis: early R16 publication reduces the final wire wait without increasing total payload.
- Evidence: the 2048/16 Zero path queues 2,854,748,160 early KV bytes per prompt.
- Validation: the exact N8 and N32 Zero gates pass. Both Native workers exit with code zero.
- Result: early and late publication both transfer 730,812,825,600 measured payload bytes over 128 requests.
- Result: both complete 11,008 measured transport submissions.
- Cost: early publication increases measured descriptor count from 1,028,030 to 1,092,480.
- Cost: both source paths have 4096 measured graph hits and zero measured captures or fallbacks.
- Result: TTFT p95 increases from 9762.66 to 12698.77 milliseconds.
- Result: the paired mean arrival-to-admit delay increases by 1943.08 milliseconds.
- Result: the paired mean source release-to-wire-commit interval increases by 120.30 milliseconds.
- Limitation: these Host receipts do not isolate CPU descriptor cost, DMA, or HBM interference.
- Decision: do not adopt the 2048/16 early Zero as a faster baseline. Retain the fixed-cut and proof-gated protocol as separate diagnostics.

## H34 — Recover a complete early-receipt test after analysis failure

- Evidence: the first early Loop N8 GPU cell completes with both Native worker exits at zero.
- Failure: the analysis reads a contract JSON file as an event file. It raises a missing-records error.
- Code change: the audit reads event records only. It excludes events before measurement begins.
- Code change: the serial controller records a per-cell observation failure. It does not qualify a failed audit.
- Validation: a CPU regression covers both the contract file and Warm-up exclusion.
- Result: reanalysis of the original GPU artifacts verifies one measured commit after an immutable fragment arrives.
- Result: the frozen state, output, lifetime, and real continuation gates pass for this N8 cell.
- Decision: retain the original GPU artifacts. Resume N32 without repeating N8.
- Evidence: `state/early_protocol_analysis_recovery.json`.

## H35 — Compare equal recurrent rows without assuming equal FLOPs

- Hypothesis: a full-prompt four-Loop frontier can execute more efficiently than a Token256 suffix with eight-Loop groups.
- Evidence: both methods move 1,048,576 recurrent rows over 128 requests. Both issue 512 D prompt submissions.
- Validation: N8 and N32 pass exact output, prompt KV, hidden state, full work, and Native lifetime checks.
- Result: every N128 output matches Zero for both methods.
- Result: Loop28 measures TTFT p95 of 3422.70 milliseconds and ITL p99 of 162.89 milliseconds.
- Reference: Token256 measures 4680.19 and 166.43 milliseconds, respectively.
- Cost: Loop28 moves more Coda work. Its padded dense plus ideal causal Attention estimate is about 6.4% greater on D.
- Cost: wire payload is 4,974,182,400 bytes for Loop28 and 4,998,533,760 bytes for Token256.
- Evidence: the Loop and Token D prompt paths record 512 and 4096 graph-cache selections, respectively. Every selection finds an existing entry.
- Evidence: neither prompt path records a measured capture or fallback. Equal submission counts do not imply equal internal graph-selection counts.
- Limitation: graph-cache selection counts are not CUDA kernel counts or GPU execution durations.
- Decision: include Loop28 in the frozen independent-Arrival confirmation before any held-out results exist.
- Limitation: neither arm passes the relative Decode gate in this selection. Equal rows do not isolate a Loop-specific benefit.
- Tiling result: Loop28 with 256-token tiles measures TTFT p95 of 3563.80 milliseconds and ITL p99 of 142.91 milliseconds.
- Validation: its N8/N32 state gates pass. All 128 serving outputs match Zero.
- Evidence: all 4096 observed D prompt spans use 256 tokens and one Loop. All four transferred Loops complete per request.
- Decision: retain the tiled arm as a diagnostic. It still fails the relative Decode target and is not a new held-out selection.

