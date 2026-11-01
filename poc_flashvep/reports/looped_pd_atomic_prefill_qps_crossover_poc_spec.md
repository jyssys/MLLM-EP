# Atomic Prefill PD: Reproduction and Automatic Crossover Search

## 1. Contract and non-goals

This document implements the user's 2026-11-01 experiment contract.
The base commit is `f9bdad32855e4af620cace1111b366c366039c40`.
The branch is `flashvep/looped-pd-atomic-prefill-qps-crossover`.
Source code belongs in `poc_flashvep/looped_pd_atomic_prefill/`.
Evidence belongs in `poc_flashvep/reports/looped_pd_atomic_prefill_qps_crossover/`.
Previous code, staged work, checkpoints, source archives, and raw results remain unchanged.

The goal is a measured P-compute bottleneck with usable D compute, KV, and Decode-SLO reserve.
Do not implement Dynamic A, Loop Split, Token Split, or a new TP/PP engine.
Do not infer an architectural impossibility from a failed finite search.
Use MEASURED, TRACE_ANALYSIS, STRUCTURAL_ACCOUNTING, ANALYTICAL_ORACLE, and NOT_MEASURED labels explicitly.
No oracle or independent-rate ratio is a measured Serving speedup.

## 2. Models and immutable semantics

| Key | Model | Revision | Depth |
|---|---|---|---|
| qwen8 | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | Native AR |
| ouro | `ByteDance/Ouro-1.4B` | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` | Fixed R4 |
| huginn | `tomg-group-umd/huginn-0125` | `bb6621b65e90b6a4b9b29ef88dc83866d450470c` | Fixed R32 |

Use original BF16 weights and BF16 KV.
Keep all original recurrent work, attention semantics, tokenizer behavior, and greedy sampling.
Keep the qualified `down_norm` and `down_norm_adapter` numerical fixes.
No early exit, approximation, quantization, training, or weight changes are permitted.
Internal runtime parity is separate from original-HF fidelity and downstream accuracy.

Qwen uses the supported native vLLM/NIXL path.
Ouro and Huginn use the qualified RLT engine and native NIXL handoff.
Native AR and RLT have different kernels and scheduling paths.
Those differences prevent architecture-only attribution from absolute latency.

Verify actual context support, RoPE, allocation, and legal positions before each new length.
Huginn never exceeds 4,096 total tokens.
Ouro's checkpoint limit does not establish support in an unchanged local harness.

## 3. Workload matrix

| Model | Input/output candidates |
|---|---|
| Qwen3-8B | 8000/200, 8000/100, 3900/100, 3900/64, 3900/32 |
| Ouro R4 | 4096/100, 4096/64, 8192/200, 8192/100, 8192/64, 8192/32 |
| Ouro optional | 16384/100, after runtime and memory qualification |
| Huginn R32 | 3900/100, 3900/64, 3900/32, 3500/100, 3500/64, 3500/32 |

Qwen 8000/200 is the first AR reproduction target.
Ouro 8192/200 and 8192/32 are the first Looped long-context targets.
Huginn 3900/32 and 3900/64 are the first near-limit targets.
Preserve every requested cell in the status matrix, including blocked and unrun cells.

Use actual public ShareGPT conversation text where enough valid long sources exist.
Retain source revision, content hash, Prompt ID, original messages, and exact model-tokenizer IDs.
Explicit tokenizer-prefix length matching is a controlled workload, not a natural conversation-length distribution.
Never add undisclosed filler or call it an Azure text trace.
The initial availability audit found only 29 ShareGPT records meeting the common 8K support rule.
The final input pool therefore supplements ShareGPT with pinned public LongBench documents.
This choice precedes GPU measurements and does not depend on candidate performance.
Record the dataset/task for each Prompt and the exact supplemental revision.
If a long-source pool is insufficient, record the shortage instead of silently repeating or fabricating text.
An explicitly declared cyclic test pool may support larger offered cohorts.
Keep calibration and test text/prefixes disjoint.

Main generation uses `ignore_eos=True` and the declared exact output cap.
Do not call it Natural-EOS generation.
Prompt length plus output cap must fit the verified model and runtime limits.

## 4. Atomic and chunked execution

Full-Prefill PD means P owns all Prompt computation.
Atomic Full-Prompt means one logical Prompt span per request, with range `[0,L)`.
It does not require one CUDA kernel or one recurrent loop.
KV transfer may still contain several wire descriptors or fragments.

Compare C256, C1024, C2048, C4096, and Full-Prompt.
For Full-Prompt, set the per-request chunk limit to at least L.
The batch token budget must also fit each complete selected Prompt span.
Do not silently split a final request because another request consumed the batch budget.
Use fixed input lengths per cell and a compatible aggregate token budget.
Trace the actual selected spans and assert `[0,L)` in every Atomic request.

The first implementation uses native atomic-per-chunk Prefill for the entire chunk sweep where supported.
This holds the recurrent execution policy constant while changing C.
A full-span staged-G1 control can separate token chunking from recurrent scheduler boundaries.
If a native Atomic path is unsupported, report it before using another path.
Do not relabel staged execution as native Atomic.

Keep the native PD/NIXL protocol for all primary PD controls.
Do not replace it with Snapshot handoff or attribute a transport change to Atomic Prefill.
Keep actual P/D scheduling and transport settings in each run configuration.
C4096 and Full-Prompt can be equivalent at L3900.
Mark equivalent settings and do not claim their labels are distinct mechanisms.

Allow legal multi-request Prefill batching.
Start with a common aggregate token budget of 32,768 where memory permits.
Measure the actual batch shape and token budget, rather than assuming a full batch forms.
Equalize tuning opportunities across Atomic, chunked, and colocated controls.

The DuetServe motivation uses an 8,192-token aggregated budget.
Add a Qwen 8000/200 budget-8192 control before a reproduction claim.
Keep its results separate from the larger tuned aggregate budget.
The local vLLM implementation and numerical mode can differ from Dynamo in the paper.

## 5. Memory and slot search

Use the measured cache allocation geometry, not parameter count, for legal concurrency.
Check required reservation blocks before launching a batch.
Keep room for weights, runtime buffers, graph capture, and activation peaks.
Do not allocate both an old and a replacement KV pool accidentally.
OOM and watchdog failures remain separate immutable namespaces.

Search legal P slots in {1,2,4,8} and D slots in {2,4,8,16,32}.
Prune settings that cannot fit the complete fixed-depth state.
P and D may use different slot limits if the native engine supports them.
Different memory/slot settings are separate controls, not the same baseline.
Do not retain a slow P-slot setting merely because it produces P pressure.

Use B1/B2/B4/B8 finite P cohorts where memory permits.
For each C, select the smallest supported concurrency within 95% of its best finite P rate.
Use the strongest measured C rate as a tuning control, not only the Atomic rate.
Do not call an all-ready D cohort an attainable staggered-arrival D capacity.
Measure the online D trace before accepting reserve.

The initial Huginn 3900-token qualification failed with CUDA OOM.
Its large KV pool competed with whole-Prompt activation buffers.
The next compatibility control uses 24,576 KV blocks and a two-Prompt aggregate budget.
Apply that memory control equally to all subsequent Huginn methods.
Preserve the failed 26,624-block, 31,200-token-budget runs.

The initial Qwen 8K PD/DP2 qualification matched 7/8 output sequences.
Repeat the default PD before diagnosing the cause.
Test the installed native `VLLM_BATCH_INVARIANT=1` mode as a separate correctness control.
If it passes, use it for every qualified AR timing method.
Keep default and batch-invariant timing families separate.
This flag does not change BF16 weights or model depth.

Qwen's batch-invariant PD/DP2 comparison also matches 7/8 sequences.
The native flag alone does not resolve that difference.
Keep this result as an unresolved cross-mode fidelity limitation.
Use a separate, matching native-PD repeat as the Static-PD reference.
Every token-chunk candidate must still match that native-PD reference exactly.
This implements the requested Native-PD and Atomic-versus-Chunked checks.
It does not convert the failed colocated parity test into a pass.
No lossless PD-versus-DP speedup or original-model fidelity claim is allowed.

The optional Qwen batch-invariant family subsequently failed its native-PD repeat at 7/8.
Do not run its capacity or QPS campaign.
The original default native-PD family has two matching 8/8 starts.
Require a third default start to match both before allowing its chunk gates.
Use default mode only if that recheck and each candidate chunk gate pass.
Keep every failed optional-mode run and its first mismatch.
This does not establish a batch-invariant-mode fix or a cause for its failure.

Separate actual D Decode occupancy from KV already reserved for unfinished P work.
A reservation for a waiting P request is not an active Decode request.
Report both quantities and the refusal reasons.
Also report whether an existing reservation can support that same request's continuation without extra KV.
This observation is diagnostic only. It does not authorize a new migration method.

## 6. Correctness gates

Run short input gates before full-length gates.
Use multiple batch sizes and orders where legal.
Compare native local, native chunked PD, and Atomic PD on matching input histories and seeds.
Check exact output IDs, stop reason, and fixed recurrent-depth work.
For Looped transport gates, check valid Prompt KV, final hidden state, and Huginn injection/Coda state.
Verify source-buffer lifetime and transfer acknowledgment.
Record the first mismatch and its inputs.
Block performance expansion only for the affected invalid model/setting.

Initial gates use 2–8 requests.
Long-context qualification then uses at least eight full-length requests where memory permits.
Use an independent 32-request qualification before a long final candidate campaign.
Do not weaken a failed equality test because BF16 differences seem plausible.
Original-HF fidelity remains a separate unmeasured item unless an actual comparison runs.

## 7. Independent finite service measurements

Measure P without D reservation backpressure.
Measure D only after actual valid Prompt KV and all necessary state are ready.
Exclude preparation from the D-only timer, but retain its separate cost.
Use the actual local or native-PD path and identify the boundary explicitly.
Measure legal batch sizes 1/2/4/8, plus larger batches where useful and supported.
Use at least three repeated cohorts for finite service estimates.
Do not treat within-start repetitions as independent server starts.

Record P service, first sampling, and post-first-token D service separately.
Record effective rows, submissions, memory, and CPU overhead.
Capacity ratios are search diagnostics, not sustainable online capacities.
Prepared simultaneous D cohorts can overestimate online D efficiency.
Confirm any apparent D reserve with actual open-loop readiness.

## 8. Calibration and SLO freeze

Use a disjoint calibration set before held-out QPS measurements.
Use serial/no-queue native PD after correctness and basic tuning.
Freeze thresholds per model and input-length cell.
Use sixteen calibration requests where practical.
If only eight complete, record that reduced calibration size before test evaluation.

Primary request SLO combines TTFT and request-average TPOT.
Set each threshold to twice the corresponding calibration request p95.
Set the secondary token threshold to twice the calibration population ITL p99.
Keep the calibration output cap and all denominators in the frozen file.
Use the same thresholds for all policies at the same evaluated cell.
Do not change thresholds after observing candidate Serving results.
Also report fixed absolute ITL limits and 1.5/2/4 calibration-multiplier sensitivity.

## 9. Open-loop search procedure

Use seeded Poisson arrivals independent of completion.
Generate new actual arrival offsets whenever the target rate changes.
The default arrival seed family starts at 20261101.
Matched policies use identical Prompt IDs, output caps, arrival offsets, and seed.
Report target QPS, realized QPS, admitted rate, and completed throughput separately.
Include all offered failures and timeouts in the SLO denominator.

For each qualified cell, derive a reference from measured finite P/D service rates.
Start with factors {0.25,0.50,0.75,1.00,1.25,1.50,2.00} times the smaller measured finite rate.
Also inspect rates near P capacity when measured D reserve supports that range.
Do not label the independent reference as an online capacity estimate.

The automatic search advances through these stages:

1. Check context, memory, loader, and correctness.
2. Measure Atomic and chunked finite costs over legal batching settings.
3. Retain the declared primary workloads and all viable P-limited candidates.
4. Run the seven-rate smoke grid with at least 24 offered requests per setting.
5. Compare Atomic with the strongest qualified chunked control at identical arrivals.
6. Select finalists using only predeclared P/D pressure and reserve criteria.
7. Confirm finalists with at least 128 offered requests per QPS when feasible.
8. Repeat critical points with at least three independent server starts and arrival seeds.
9. Analyze all outcomes and restore approved resource policy.

Twenty-four-request results are smoke results only.
Use a sufficiently long arrival window for any sustained-pressure claim.
Target at least 30 seconds of central-window evidence and 128 offered requests for a final classification.
Increase request count for fast conditions rather than claiming saturation from a few seconds.
Record finite queue growth and drain. Do not claim infinite-horizon stability.

The controller must generate the next stage or a terminal blocked status automatically.
It must not silently stop because a preliminary static queue drained.
Completed namespaces are resume-safe and never run twice.
Failed attempts require new retry names.

## 10. Preregistered diagnostic thresholds

These thresholds classify the evidence. They are not proven safety guarantees.
Evaluate the central 20–90% arrival window and preserve the complete trace.
Require at least 30 seconds and 128 requests for a final TRUE classification.

P pressure requires increasing ready Prefill backlog and material queue waiting.
Require a positive backlog slope of at least 0.05 times realized offered requests/s.
Also require second-half mean ready backlog at least one request above the first half.
Require P service within 15% of its strongest matched finite P reference.
Actual P kernel-busy evidence must support at least 80% activity in the central interval.
NVML utilization alone cannot establish this condition.

D reserve requires no growing Decode-ready backlog and primary Decode-SLO margin.
Require request TPOT p95 at most 90% of its frozen limit.
Report ITL p99 and request-maximum violations separately.
Require at least 20% measured service reserve against the observed completed P rate.
Require usable D KV credit, with sampled admission refusal at most 1% for the strict TRUE label.
For this threshold, a binding D refusal means that D blocks a request while P has a usable slot and KV credit.
Also report unconditional D refusals, including simultaneous P/D-full samples.
Do not call a simultaneous full reservation a D-caused stall when P is already compute-busy.
Confirm the binding cause with actual P idle/service and queue traces.
Keep already-reserved P state separate from this strict new-admission criterion.
A condition with high reservation refusals can be diagnostically interesting but cannot silently pass the strict gate.

Do not require D kernel activity below 80%.
Kernel activity is not SM occupancy or maximum batching capacity.
Report D service at the observed batch and at larger legal batches separately.
A larger all-ready D rate is an optimistic screen, not sufficient proof of usable online reserve.
Before a strict TRUE label, validate staggered D readiness at 1.2 times the observed P completion rate where the native prepared-KV path permits it.
Report the finite cohort size and its KV limit.
This diagnostic is not Dynamic A and does not prove that arbitrary Prefill work can meet D SLOs.

Classification precedence is as follows:

- `TRUE_PREFILL_HEAVY`: all P compute/queue and D compute/KV/SLO requirements pass, with repeated confirmation.
- `KV_ADMISSION_BOUND`: reservation waiting materially blocks admission, without an independently qualified compute-only explanation.
- `DECODE_BOUND`: D service or Decode SLO is limiting and P has spare capacity.
- `GENERIC_TUNING_ARTIFACT`: apparent P pressure disappears with a stronger legal chunk/batching setting at identical arrivals.
- `MIXED_OR_INCONCLUSIVE`: multiple resources limit progress or necessary causal evidence is absent.

If low-overhead profiling cannot qualify clean P kernel activity, retain an unconfirmed candidate rather than forcing TRUE.
No threshold changes are permitted after held-out results.

## 11. Instrumentation and isolation

Record arrival, admission attempts, P start/end, valid frontier, transfer, D activation, first token, and completion.
Record native queue state, credits, active Decode rows, token events, memory, and actual chunk ranges.
Record actual NIXL bytes, descriptors/fragments, completion observations, and exposed handoff separately.
Host transfer duration is not pure DMA time.
Host model-forward spans are not exclusive kernel time.

Use one clean GPU pair at a time unless an actual contention calibration qualifies parallel timing.
Keep all owned burners off during a clean timing lane if they can interfere.
Restore them during safe idle periods and after all measurements.
The burner script's current defaults are not assumed to include all eight devices.
Pass the intended GPU list explicitly and check actual worker ownership.
Never signal an unrelated user's process.

Use separate GPU-only tracing where available.
Measure tracing overhead against unprofiled controls before using busy fractions as clean-runtime evidence.
If tracing changes service by more than 5%, do not directly transfer its utilization denominator to clean Serving.
SM occupancy remains NOT_MEASURED if no valid counter is available.
Do not call NVML GPU utilization SM occupancy.

## 12. Colocated controls and figures

Compare 1P1D with two independent colocated replicas at the same two-GPU budget.
Give DP2 a valid tuned native scheduler.
The old Dynamic-A wrapper forbids Mixed co-batching.
Do not inherit that restriction as an intentional weakness of the new colocated control.
For Ouro/Huginn, qualify a separate native-Mixed worker that applies only the same numerical fixes.
It uses the existing RLT Mixed policy and original recurrent core.
It does not use a new scheduler or recurrence migration.
Test C256 and C1024 with exact output and full-depth gates before performance expansion.
Keep separate-phase DP2 as another explicit control.
Native AR DP2 has an unresolved repeat mismatch and is not automatically qualified by a later matching run.
Where supported, include native Mixed/chunked colocated execution as a strong generic control.
Record any scheduler difference rather than forcing a deliberately weak DP configuration.
Native Mixed is existing functionality, not a new contribution.

### Diagnostic extensions after the first smoke results

The primary SLO file was frozen before held-out QPS results.
No later diagnostic changes that file or the classification thresholds.
Qwen 8000/200 native-PD outputs differ in 17/24 requests between the first two arrival rates.
The exact-output gate therefore stops that synchronous family's performance expansion.
A separate official asynchronous-scheduling repeat tests compatibility without changing BF16 or model weights.
It is a qualification diagnostic, not an accepted performance result.

The first Ouro 8192/32 smoke runs show large host-observed exposed handoff intervals.
The existing connector also checks global in-flight wire credit before scheduling Prefill work.
These facts motivate supported `PDConfig.max_inflight_bytes` controls at 256 MiB, 1 GiB, and 8 GiB.
The 64-MiB fragment limit and native NIXL protocol remain unchanged.
Each control requires output, full-depth, and valid-state byte checks before clean timing.
Wire-credit changes are separate transport controls. They are not evidence of a Loop Split benefit.
An isolated model-free NIXL benchmark separates transport capability from Serving polling and scheduling.
Its measured host completion is not pure GPU DMA or model E2E speedup.

### Bounded memory and baseline controls before final confirmation

The inherited harness assigns the P token budget to the Decode-only worker.
The native runner allocates row-dependent work buffers from that budget.
Test the supported D budget of 128 as a separate generic memory control.
Keep the fixed recurrent graph geometry, numerical fixes, BF16, and full depth unchanged.
Qualify larger legal D pools before timing them.
Use 25,088 blocks for the Ouro control and 28,288 blocks for Huginn at 3500/32.
The latter supports a fourth reservation but leaves little activation memory.
Record actual memory and admission credit. Do not assume that four active requests fit every execution geometry.

The optional Ouro 16K cell requires its own D-workspace output and valid-state byte gate.
Its calibration precedes its held-out rate search.
Keep an original-workspace control at the same length.
Do not change existing primary SLO thresholds.

At Huginn 3500/32 and Ouro 16384/100, qualify colocated controls before selecting a final baseline.
Compare separate-phase Full, separate-phase C1024, native Mixed C256, and compact native Mixed C256.
The compact control uses existing non-wavefront scheduling and a smaller legal token budget.
This is bounded generic tuning, not a new scheduling method.
All variants require exact output and full-depth checks.
Native Mixed must retain actual batch-composition evidence.
Failed memory or output gates exclude only the affected setting.
For Huginn 3500/32, also qualify a 1-GiB native wire-credit control for Full and C2048.
The fragment limit remains 64 MiB.
For Ouro 8192/32, combine the independently tested compact D pool with 1-GiB and 8-GiB wire credit.
Test Full and C4096 with a separate byte/output gate for every combination.
This prevents an untreated interaction between two generic controls from becoming a false architectural conclusion.
Independent final confirmation uses new arrival seeds that did not select these tuning settings.

Produce QPS versus TTFT p95, TPOT/ITL, total/output throughput, and P/D utilization plots.
Use the same workload within a rate curve.
Plot different input/output cells as separate series.
Produce an actual request/worker timeline with observed phase boundaries.
Separate the host request timeline from device kernel/transfer activity.
Label idle only where the selected device trace actually supports it.
Do not draw conceptual blank time as measured GPU idle.

## 13. Closeout and decision

Preserve configuration/status matrices, raw request IDs and tokens, failures, source hashes, model revisions, and all analysis code.
Keep all unmeasured values explicit.
Reports use English and self-identifying snake_case names.
The final user summary uses Korean.

Report AR, Ouro, and Huginn separately for each research question.
If no candidate qualifies, narrow context, output, KV, batching, and host-runtime explanations using controlled evidence.
Use INCONCLUSIVE when engine/model causality remains mixed.
Do not convert an unexplored condition into a negative result.
Do not begin Dynamic A even if an eligible regime appears.
Recommend a later static Loop/Token offloading gate instead.

Each job has a bounded watchdog and a durable terminal record.
The new session starts at 2026-11-01 04:03:08 UTC, or 13:03:08 KST.
An unattended controller uses a 24-hour operational checkpoint from that new start.
This is not the expired 12-hour limit of the previous study.
At that checkpoint, stop new GPU jobs and separate unfinished cells from measured results.
Do not claim completion because a checkpoint or model gate stopped the queue.
The progress guard records elapsed time, terminal runs, live processes, and incomplete namespaces.
It may resume a stopped controller only when all previous namespaces are terminal.
It never overwrites an incomplete or failed run.
Pilot costs determine the later matrix and runtime estimate.
No old campaign start time or expired deadline carries into this new study.
Record any operational time cap before it truncates the search.
Commit only owned changes and push normally.
Check the remote SHA and actual restored burners before claiming completion.
