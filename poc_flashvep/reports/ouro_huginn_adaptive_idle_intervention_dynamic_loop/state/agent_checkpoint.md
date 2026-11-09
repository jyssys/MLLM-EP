# Active campaign checkpoint

The start is 2026-11-08 18:07:15 KST.
The final deadline is 2026-11-09 08:07:15 KST.
The GPU deadline is 06:37:15 KST. Do not reset these clocks.

The publication parent is `ddbf49ae3ef610fefd78248d765a27641b21b98f`.
The branch ref is `flashvep/adaptive-idle-intervention-dynamic-loop-poc`.
Do not switch the actual checkout. Preserve its dirty files and staged index.
The private runtime remains unchanged on disk.

The Python executable is `/home/esjung/.venvs/looped-pd-abc/bin/python`.
The source namespace is `poc_flashvep.idle_intervention_dynamic_loop`.
The evidence root is this directory's parent.

The owner PID is 2659985. Its guard PID is 2660150.
The monitor PID is 2729083.
They use `state/queue.json`, `state/active.json`, and immutable job receipts.
They stop approved burners before each isolated experiment.
They restore approved missing burners when the queue becomes idle.
Check actual GPU engines during final restoration, not only worker PIDs.

## Completed evidence

- Both models passed forced mixed N32 output, prompt KV, hidden, full work, native byte checks, ACK, and final allocation checks.
- Both models passed actual boundary-negotiated N32 state tests with one nonzero cut each.
- Both models passed actual Token N32 state tests. Ouro's successful retry is `n32_ouro_actual_token_state32_v2`.
- An earlier marker-free Huginn N8 control selected 4 and 1 remaining loops in one run.
- Independent copy streams reduced an observed source snapshot serialization cost.
- The Ouro copy-stream N8 pilot improved TTFT. This is not confirmation or proof of adaptive diversity.
- Native A/A N24 runs completed. N128 Native and matched-wire zero controls also completed.
- The asynchronous probe reproduces Native prompt SHA256 records.
- N8 asynchronous wire checks preserve every valid KV byte and hidden byte.
- Detailed state probes are not performance or SLO evidence.

The Native N128 selection seed is 2026110900.
Ouro Native TTFT p95 is 2975.793 ms. Its ITL p99 is 43.230 ms.
Ouro matched-wire zero TTFT p95 is 2490.919 ms. Its ITL p99 is 41.031 ms.
This difference is a runtime confound, not a Loop benefit.
Huginn Native TTFT p95 is 9707.431 ms. Its ITL p99 is 124.841 ms.
Huginn matched-wire zero TTFT p95 is 9873.088 ms. Its ITL p99 is 127.220 ms.
These are selection controls, not held-out method confirmation.

## Current implementation

`runtime.py` implements GPU-completed boundary offers, D prepare, source commit, D revalidation, exact handoff, and Native fallback.
It separates admission horizons from immediate execution budgets.
It retains the original Native model and transport guards.
It counts ready P work and pending native device work separately.
Pending device work is not a kernel-busy measurement.
It supports bounded multiple quanta under the current output deadline.
Native service reserves exclude intervals that contain borrowed work.
Phase-aware costs distinguish terminal Coda work, nonterminal cores, and Token Prelude.
The Token comparator can use large suffixes and grouped recurrent execution.

`async_state.py` uses local read leases, event-ordered copies, and host SHA256 futures.
Its Source and D hidden checks occur at commit/import.
Its D KV check uses the immutable imported prefix in the completed prompt.
It hashes all original valid bytes. It does not weaken state comparison.
The connector attaches after its factory returns, not during StageEngine construction.
Earlier constructor failures remain in their original raw namespaces.
The newest probe uses eight hash threads and drains startup futures in startup statistics calls.

State-only v2 N32 jobs exempt observer-induced tail latching.
They retain live instantaneous budgets and Native validity guards.
They cannot establish Decode protection or serving performance.
Clean runs must retain the tail latch.

## Next work

1. Check active jobs and qualify both actual asynchronous N32 variants against the Native N32 references.
2. Keep actual diversity separate from a generic correctness PASS.
3. Let the queued intervention pilots and cost cells finish.
4. Run `costs` before `policy`. The policy generator requires completed full-suffix calibrations.
5. Review source graph captures and warm missing legal Token delta geometries when measured evidence requires them.
6. Test the calibrated policy at N8 and N32 before N128.
7. Confirm actual dynamic cuts on held-out Poisson arrival seeds, with all safety criteria unchanged.
8. Compare original Native, matched-wire zero, Token, full-prompt Loop, and tiled Loop.
9. Use at least three independent N128 starts for valid final candidates.
10. Measure Low/Medium controls and optional persistence only after safe qualification.
11. Use separate event controls for Service TTFT and execution timelines.
12. Produce the required seven figure families, final report, source manifests, and scoped Git publication.

The calibration queue includes efficient large Token chunks and recurrent groups up to R32.
Do not restrict the Token comparator to C256.
Do not use structural row savings as measured compute savings.
Do not infer physical idle gaps from host readiness or CUDA enclosures.
No validated physical kernel union exists yet in this campaign.
Primary Service TTFT remains NOT_MEASURED unless a separate valid device-entry control supplies it.

The current code compiles. Six CPU protocol tests pass.
The final report does not exist yet.
No final GO/NO-GO, commit, push, or final burner restoration is complete.

## Explicit resumption on 2026-11-09

The user requested completion after the original deadline expired.
Use `state/resumption_20261109.json`. Do not rewrite the original clock.
The resumption starts at 13:00:54 KST. Its GPU stop is 16:00:54 KST.
Its final deadline is 17:00:54 KST. Do not start long jobs after 15:40:54 KST.
Set `IDLE_LOOP_RESUMPTION` to the absolute resumption record path for dispatch commands.
Analysis commands do not need this variable.

The resumed owner PID is 3485842. Its guard PID is 3485991.
Unified execution sessions are owner 98860 and monitor 61149.
Check current state files before using these identifiers.
The existing owner loaded before the new dependency code.
Therefore, manually enforce Token N8 and N32 gates through `gated_enqueue.py`.
The dispatcher still preserves immutable raw namespaces and exclusive timing.

### Resumption findings

All four `r1_{model}_pinned_{static,dynamic}_state8` tests passed every state and lifetime check.
`r1_ouro_actual_mixed_state32` passed full output, state, handoff, work, and lifetime checks.
It migrated one request with three loops. It did not show adaptive diversity.
`r1_huginn_actual_mixed_state32` passed the same full checks.
It migrated three requests. Their remaining-loop counts were one, four, and four.
This is actual boundary negotiation, not preassigned mixed cuts.
Its heavy state-copy timings cannot establish Decode protection.

An older clean N32 Huginn run also selected one, two, and four loops.
Its name is `pred_huginn_tile_relative_clean32_workspace_v2`.
The forecast quantum budgets were 9.101, 17.316, and 26.869 ms for one, two, and four loops.
Its TTFT p95 was 5175.999 ms. The matched Native control was 3864.403 ms.
Its ITL p99 was 126.224 ms. The Native control was 126.939 ms.
Thus, actual variable-count selection exists, but this pilot did not improve TTFT.

### Observer repair

The four-slot CPU snapshot pool can block the Scheduler during state hashing.
A large-pool ablation, `r2_ouro_actual_mixed_state32`, timed out in startup after 307.299 seconds.
Do not interpret that failure as model or serving evidence.
Inspection showed dozens of full-byte hashes for startup shapes before measurement began.
The new `state_probe_measured_only` option skips startup hashing, not model warm-up.
It still hashes every valid byte for every measured request.
The pure `should_capture_state` rule has a regression test.
`r3_ouro_actual_mixed_state8` and `r3_ouro_actual_mixed_state32` passed all checks.
The N32 run still selected only three loops.
`r3_huginn_actual_mixed_state8` passed all checks.
The corresponding N32 run was active at this checkpoint.

### Strong Token recovery

`r3_ouro_actual_token_state8` and N32 failed during mixed Native/Token startup.
The exact error was `Invalid exact frontier from the coordinator`.
A zero-suffix startup request had k=T and entered the partial Staircase reservation guard.
The new process-local route calls the qualified Native reserve for k=T.
It does not relax the partial-frontier guard.
Nonpositive prefixes and prefixes above the actual prompt length still fail.
The CPU regression first reproduced this error and then passed after the route change.
Nine protocol tests and the evidence-bundle round-trip test pass.
GPU qualification of this reservation fix remains pending.

`r4_ouro_actual_token_state8` is queued before its N32 test.
`r3_huginn_actual_token_state8` will use the same corrected runtime.
The pending larger Token tests are deliberately absent from the live queue.
Run `gated_enqueue` after N8 completion. It adds N32 only after exact qualification.
Run it again after N32. It then adds the clean Token selection arm.
Keep all earlier failed configurations and raw evidence.

### Pending experiment and analysis sequence

1. Check the active job and qualify completed state tests.
2. Admit Token N32 and selection tests with `gated_enqueue`.
3. Complete `r1_{model}_{native,zero,rejection,loop,tile,token}_selection32`.
4. Inspect absolute Decode SLO, Native-relative ITL, actual cuts, and completion delays.
5. Run `confirmation` only after the controls and candidate state gates complete.
6. The confirmation module freezes three held-out arrival seeds: 2026110911, 2026110912, and 2026110913.
7. It recalibrates the shared estimator from the matched selection controls.
8. It does not expand Loop methods without actual mixed-cut N32 state qualification.
9. Stop repeat expansion after absolute Decode SLO failure. Retain every failed seed.
10. Add small Low/Medium and static controls only when time permits.

### New analysis and publication helpers

`campaign_report` derives serving, decisions, quanta, transfer, memory, and failure tables.
It separates marker-free, state-diagnostic, device-event, and instrumented evidence.
`paired_analysis` uses start-level paired bootstrap ratios, not independent token samples.
`figures` generates seven PNG/PDF families without invented kernel-idle intervals.
Physical kernel union and primary Service TTFT remain NOT_MEASURED.
`evidence_bundle` stores all raw bytes with SHA256 deduplication and bounded archive shards.
Its small-fixture round-trip and corruption test passes.
Do not build the real bundle until the scientific controllers stop and artifacts are final.

The latest runtime adds selection-time budgets to D quantum records.
Timing revision 2 starts before ModelRunner submission.
Older host completion records start after submission and do not enclose the whole quantum.
Neither record type is exclusive GPU execution time.

No final report, figures, commit, push, or final restoration exists yet.
Use a temporary Git index for publication. Do not stage the actual dirty checkout.
The original index entry signature still equals d0674d8fc531a451b003223020607e4b54dd264c5ca319a57ac387bd19a3e6b3.
The private runtime diff still equals 4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7.
Before final response, verify all eight real burner engines and Remote SHA equality.

## Checkpoint update at approximately 14:43 KST

The fixed resumption deadlines remain 16:00:54 GPU stop and 17:00:54 final closure.
Do not extend them. The owner and monitor still run in sessions 98860 and 61149.
Use `gated_enqueue` after each state cohort finishes.
Do not run `confirmation` until the current state and selection gates finish.
No held-out confirmation plan exists yet.

### New repairs and verified evidence

- Both strong Token paths passed N8 and N32 full output, KV, hidden, handoff, work, and lifetime checks.
- Ouro's successful strong Token N32 is `r4_ouro_actual_token_state32`.
- Huginn's successful strong Token N32 is `r4_huginn_actual_token_state32`.
- Huginn selected suffixes 32, 64, and 128 in that N32 state cohort.
- A readiness observation can expire before `_take`. The scheduler now tries another phase or returns to Native Decode policy.
- The chosen quantum cost is rechecked against the live budget. The Native progress guard stays enabled.
- The failed Huginn N32 did not capture the exact failed selection. A readiness race remains an inference, not a proven unique cause.
- Source pressure counts distinct ready and pending request IDs. It no longer sums possibly overlapping sets.
- Ouro clean `r1_ouro_tile_selection32` selected three and two loops in one run.
- It passed output parity, total work, and lifetime checks. TTFT p95 was 1800.344 ms versus Native 1162.309 ms.
- Its ITL p99 was 40.379 ms versus Native 38.540 ms. This is not a net-benefit success.

### Shared empty-D admission correction

The earlier forecast subtracted a Batch-1 reserve even when D had no active Decode.
The optional `empty_admission_uses_current_cap` policy uses the existing 200 ms cap in that state.
It applies to both Token and Loop methods. It does not change any SLO.
This cap is not a measured physical idle window or a guarantee about future arrivals.
Execution and cut commit still recheck current D state.
The new selection jobs are `r6_{model}_{token,loop,tile}_selection32`.
Final confirmation configurations enable the same flag for all methods.

### Grouped source boundary repair

An earlier Huginn cut exposed `Unexpected source checkpoint`.
Two requests at different depths can share a grouped native batch.
That batch can end below one request's planned target.
The scheduler now records each request's target.
An intermediate depth continues through the original Native callback.
Only the completed target can create an offer. Skipping the target remains an error.
All new actual offers require a non-null completed native CUDA event.
The CPU regression reproduces the old failure and passes after this repair.
Thirteen protocol tests and one bundle test pass.

### Full-byte observer and shutdown limits

`state_hash_defer_until_close` preserves every measured byte in immutable pinned buffers.
It postpones CPU SHA256 work until serving finishes.
Each worker uses count+2 slots for this diagnostic. It is not a clean timing configuration.
`r6_ouro_actual_mixed_state32` and `r5_ouro_actual_mixed_state32` saved all 32 state hashes but failed finalization.
Do not mark those runs complete or use them for performance.
The top-level `pd_shutdown_timeout=300` did not configure Native PDConfig.
Ouro still used the default 30-second close window. Deferred hashing consumed about 28 seconds.
`r8_ouro_actual_mixed_state32` sets `native_transport_limits.shutdown_timeout=300` explicitly.
Huginn already had that nested field. The runtime now records resolved PDConfig values.

### Current qualification strategy

`r7_huginn_actual_mixed_state32` passed every state check but migrated no measured request.
It is not an adaptive success. Most offers failed the remaining-cost admission gate.
The primary clean N32 runs show actual variable cuts, but observers change their schedule.
The final correctness stress uses simultaneous submissions, not forced cuts.
Its names are `r9_{model}_actual_mixed_state8` and `r9_{model}_actual_mixed_state32`.
These jobs retain actual fresh D decisions, exact state, native ACK, and the same SLO settings.
Their timing is diagnostic only. Primary performance remains Poisson.
The N32 stress enters the queue only after an N8 cohort completes an actual continuation.
At this checkpoint, Ouro N8 passed and Ouro N32 was admitted. Huginn N8 was pending or active.

### Pending final work

1. Check current jobs and run `gated_enqueue`.
2. Finish the R9 mixed-state gates and R6 clean selection jobs.
3. Keep unproven adaptation separate from generic correctness PASS.
4. Reload the owner after the current queue drains. Its current process predates safety-repeat code.
5. The new owner checks `repeat_safety_references` before later held-out starts.
6. Freeze `confirmation` only once. It uses seeds 2026110911, 2026110912, and 2026110913.
7. Original Native and matched zero controls always run. Token requires exact N32 qualification.
8. Tile requires an actual mixed-cut N32 state gate. Whole-prompt Loop also needs clean N32 variable cuts.
9. Retain failed and blocked starts in the final report. Do not replace them with favorable seeds.
10. Run small Low/Medium Native controls only if the remaining GPU budget permits them.
11. Run `campaign_report`, `work_accounting`, `completion_audit`, `paired_analysis`, then `figures`.
12. Review the eight generated figure families. They contain no invented GPU-idle bars.
13. Write the final English summary. Keep statistical, model, and instrumentation limits explicit.
14. Close scientific controllers, restore all eight actual burners, and verify their ownership.
15. Build and verify the lossless evidence bundle. Publish with a temporary Git index.
16. Push the new branch and verify the Remote SHA. Preserve the actual index and private runtime.

The completion model still uses guarded serial costs, not a validated online completion predictor.
Use `completion_audit` to show observed import-to-output waiting and this limitation.
Do not treat the entire KV payload as additional cost over Native, which also transfers KV.
Measured active-row saving is not measured P critical-path time or FLOPs.
Generic logical-layer execution at identical boundaries remains EQUIVALENT_MECHANISM.

## Scientific closure update, 2026-11-09 approximately 16:03 KST

All scientific work is closed. Do not launch another GPU experiment.
The final hard deadlines remain 16:00:54 GPU stop and 17:00:54 publication closure.
The original 15:40:54 start cutoff remains in the frozen resumption record.
A separate bounded-completion amendment permits only already declared jobs that fit their full startup, timeout, and cleanup margin.
That amendment does not extend either hard deadline or change any scientific configuration.
Its evidence is `state/bounded_completion_window.json`.
The completed scientific queue closes at 15:53:20 KST.
All owners, guards, and the monitor are stopped. No scientific worker remains active.

### Completed core evidence

Both models pass actual dynamic N8/N32 full output, valid KV, hidden, wire, recurrent work, and native lifetime checks.
Ouro's mixed N32 gate is `r10_ouro_actual_mixed_state32`.
It has 7/32 actual offloads and choices of one and three remaining Loops.
Huginn's mixed N32 gate is `r9_huginn_actual_mixed_state32`.
It has 2/32 offloads and choices of four and eight remaining Loops.
These large pinned-state observations are correctness diagnostics, not performance evidence.

All twenty frozen N128 confirmation starts complete.
There are three Native, three Token, three tiled Loop, and one zero control per model.
The primary total is 2560 requests. All primary output, work, lifetime, and absolute Decode checks pass.
The paired configuration hashes and loaded runtime/protocol/warm-up/bootstrap hashes remain consistent.
There are no analysis errors in the final collector.

Ouro Loop migrates only 3/384 primary requests and 20,480 recurrent Prefill rows.
Each primary run has one accepted request. Its choices across starts are two, one, and two Loops.
This is not within-run primary adaptive diversity.
Loop/Native TTFT ratio is 1.0086579, CI [0.9982686, 1.0275611].
Loop/Native ITL ratio is 1.0053039, CI [0.9555755, 1.0336659].
Loop/Native goodput ratio is 0.9650992, CI [0.8408878, 1.1065378].
Ouro Token migrates 5/384 requests. Rare ITL exceeds the frozen threshold five times, with maximum 146.858 ms.
The p99 absolute gate still passes. Do not hide the maximum.

Huginn Loop migrates 5/384 primary requests and 10,240 rows.
All accepted primary requests move only one Loop.
Loop/Native TTFT ratio is 1.4720148, CI [1.0953059, 1.9816691].
Loop/Native ITL ratio is 1.0034000, CI [0.9665255, 1.0396067].
Loop/Native goodput ratio is 0.3126368, CI [0.1237852, 0.5063133].
Huginn Token migrates 0, 0, and 4 requests across the three starts.
Its Native-relative ITL target is not established.
Huginn Loop has lower goodput than Token under the absolute SLO, but this is not an intrinsic granularity verdict.

The scientific verdict is NO_GO_NET_BENEFIT for both tested implementations and workloads.
There is no GO_ADAPTIVE_LOOP or GO_LOOP_OVER_TOKEN.
Do not claim that all possible schedulers fail or that physical slack is absent.
Physical kernel union, exclusive saved P time, and primary Service TTFT remain NOT_MEASURED.

### Important negative mechanism

One Huginn request accepts the last Loop on a 34.144 ms serial estimate.
Its import-to-first-output host delay is 86,524.743 ms.
Eight actual quantum cost bounds sum to 72.786 ms. That sum is not GPU occupancy.
P source release occurs about 563 ms after Cut commit, not 86 seconds later.
The downstream wait is therefore not an 86-second source-ACK stall.
The unfinished request holds 4128 KV blocks, or 10.3931 GiB.
The 23,552-block D pool admits only five such reservations despite eight configured slots.
The serial admission forecast ignores long Decode-priority or feedback-suspension waiting and the associated KV residence.
This is a controller and completion-cost limitation, not proof of an intrinsic lack of Loop headroom.

### Secondary and failed conditions

All four N16 Native Low/Medium controls complete.
All four secondary Static N32 serving cells are budget-blocked with retained receipts.
N512 is intentionally not expanded because no repeated Native benefit appears.
There are 293 configurations: 248 complete, 25 failed/incomplete, four budget-blocked, sixteen never enqueued.
Three old stopped receipts retain RUNNING with return code -15. They are not active processes.
The derived inventory keeps their original status field and normalizes the operational result.
The forced-rejection selection controls arm zero requests. Do not claim they measure armed-rejection cost.

### Resources and preservation

GPU 0 through 7 all run approved resident VLLM EngineCore processes.
All latest workers log actual generation and completed rounds.
The new final burner check requires both ownership and completed generation, not wrappers or startup activity alone.
The engine PIDs are 3728154, 3728161, 3728168, 3728180, 3728182, 3728597, 3728189, and 3728590.
The wrapper SIDs are 3726537 through 3726544.
Direct utilization reaches 100% on every GPU.
Leave those burners running after publication and the final response.

The index-entry SHA256 remains d0674d8fc531a451b003223020607e4b54dd264c5ca319a57ac387bd19a3e6b3.
The external diff SHA256 remains 4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7.
The active checkout still points to the original recovery branch and f4ecea59.
Use the new publication ref and a temporary index. Never stage the actual dirty checkout.

### Remaining closure work only

The final English report exists and contains the complete scientific result.
Ten figure families exist, including paired effect intervals and separate mixed-state diagnostics.
The final figure generator has one small readability update to regenerate before bundling.
Review the remaining figure layouts and run final prose checks.
Then build and verify the lossless `evidence_bundle`.
Perform a full extraction into a new temporary subdirectory and record its SHA verification.
The bundle freezes the active burner log once at closure. Do not regenerate raw scientific tables afterward.
Run `python -m poc_flashvep.idle_intervention_dynamic_loop.publish`.
It verifies the bundle, report, all eight generating burners, original index, and private runtime.
It publishes only new scoped source, reports, derived artifacts, and the lossless bundle.
Raw and expanded worker-source directories remain local and are recoverable from that bundle.
The helper creates an evidence commit and a metadata child, then verifies Remote SHA equality.
No commit or push exists at this checkpoint. The publication branch still equals parent ddbf49ae.
After push, query live GPU status again without changing frozen scientific evidence.
Deliver the concise Korean verdict, report/figure links, Branch, final SHA, remote equality, and restoration status.
