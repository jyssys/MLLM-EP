# dLLM multi-GPU E2E atlas

## Decision

**Verdict: GENERIC_ONLY.** A large physical cost exists in this native runtime.
It does not establish a new parallel architecture.

An exact, same-TP4 storage control reduced held-out clean E2E by **39.54%**
[37.33%, 41.54%], using 24 requests and 2,000 paired request-bootstrap resamples.
Mean native E2E was 881.08 ms. Mean compact E2E was 518.49 ms.
We average the paired percentage reductions across requests.
We do not divide the two mean times to obtain that percentage.
Mean paired speedup was 1.666× [1.608×, 1.719×].

The cause is concrete. An 8 MiB layer KV view shares a 320 MiB backing allocation.
Native `slice_scatter` clones that entire allocation. Forty such copies occur
per cached forward on each rank. The exact control copies only the logical view.
It does not modify the original cache in place.

All tested output, acceptance, NFE, KV-value, and logit comparisons passed.
The candidate leaves model weights, precision, routing, masks, sampling, and parallel layout unchanged.
Official benchmark accuracy is NOT_RUN. The generation cap remains 64 tokens.

## 1. Model and environment

- Model: non-preview `inclusionAI/LLaDA2.0-mini`.
- Checkpoint revision: `dad945cac317da394b390f82c7b40691d8a881ed`.
- Runtime: existing dInfer `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f`, with its pinned pre-existing patch.
- PyTorch: 2.8.0+cu128. SGLang: 0.5.3.post1. CUDA: 12.8. NCCL: 2.27.3.
- DeepEP 1.2.1+73b6ea4 is installed but inactive in this EP1 path.
- Physical GPUs: H100 80GB HBM3, IDs 4, 5, 6, and 7. Selected pairs report NV18 connectivity.
- Logical ranks 0, 1, 2, and 3 map to physical GPUs 4, 5, 6, and 7, respectively.
- Actual native layout: TP4/EP1/DP1/PP1/CP1. Each rank owns four Q heads and one KV head.

Full UUIDs, topology, source hashes, original dirty-state hashes, and launch commands are recorded.
Other users remained active on GPUs outside the permitted mask. We did not change their processes.
No global clock, profiling permission, or server setting changed.

The model has 20 full-attention layers, 19 sparse layers, hidden size 2,048,
256 routed experts, and top-k 8. The trained context limit is 32,768 positions.
`use_sliding_window=false` disables the nominal sliding-window field.
Prefix KV is cached. Canvas Q/K/V remain fresh during refinement.
The native padded, unmasked regular-refinement behavior remains unchanged.

## 2. Measurement contract and validity

Three accounts remain separate:

1. Clean CPU-monotonic request wall, with one final GPU completion wait.
2. GPU work/service time per rank, including overlapping work.
3. Non-overlapped rank activity and partial dependency bounds.

Clean runs have no Atlas hooks. Coarse runs record every actual iteration,
input staging, graph replay, sampler, and refresh occurrence.
No last graph replay is multiplied by an iteration count.
Events resolve after the request, not inside model operators.
One cumulative rank supplies a request's component intervals.
Unrelated component maxima from different ranks are never added.

The request boundary excludes model loading, graph capture, warmup, tokenization,
server queueing, and final CPU-list serialization. It includes native prefix prefill,
block generation, refresh, native output filtering, and final GPU completion.
This is loaded-model cold-prefix latency, not server cold-start or HTTP latency.
Native generation does not stream token delivery. Serving TTFT is NOT_MEASURED.

The existing metadata freeze supplies eight calibration requests and 24 held-out requests.
Calibration has three coding, three math, and two document requests.
All inputs use real frozen source text. Selection does not inspect performance.
The held-out backgrounds are disjoint token ranges, not guaranteed disjoint articles or source files.
Domain task templates repeat. This is a performance cohort, not a new accuracy benchmark.

Primary settings remain B1, block32, threshold 0.9, temperature 0, EOS, and output cap64.
Each primary context has three warmups, three clean repeats, and three coarse repeats.
A separate parity-only run checks the native update schedule.
Technical repeats, layers, and iterations are not independent statistical samples.

| P | Mean coarse overhead | 95% CI | Outer timing status |
|---:|---:|---|---|
| 512 | 9.76% | [6.49%, 14.04%] | UNTRUSTED for quantitative component attribution |
| 4,096 | 4.16% | [2.80%, 5.63%] | VALIDATED_OUTER_ONLY |
| 8,192 | 0.98% | [0.86%, 1.10%] | VALIDATED_OUTER_ONLY |
| 16,384 | 1.99% | [0.57%, 4.80%] | VALIDATED_OUTER_ONLY |

Clean P512 E2E remains valid. Its component percentages are not promoted.
One P16K coarse refresh contains a 381.218 ms sampling-scope outlier.
Other refresh sampling scopes are about 0.15 ms. The outlier remains in means and raw evidence.
Its cause is UNRESOLVED. It is not evidence for a recurring removable sampler cost.

Nsight Systems 2024.6 collected one short node trace, one long graph trace,
and one long source-correlated node trace with an extra eager diagnostic forward.
The extra forward is excluded from native service tables. Its total wall is not a timing baseline.
Graph-level profiling added about 1.3% on its representative request.
Short node tracing added about 34%. Node traces classify operations, not headline latency.
OS runtime APIs were captured. CPU sampling was unavailable under the server's perf policy.

A separate low-overhead Torch trace recorded all CUDA occurrences.
Its overhead was 3.86–5.18% across ranks. The paired control traces added 3.22–5.32%.
An int16 marker bounds the CUPTI-to-event clock offset.
Native offset brackets were 108–126 microseconds, exceeding the 100-microsecond point-attribution gate.
Compact brackets were 66–72 microseconds. Fine native charts therefore remain BOUNDED/PARTIAL.
We did not relax that gate or claim a complete cross-rank critical DAG.

## 3. New native E2E and phase atlas

Historical absolute timings are not pooled with this campaign.
Only one native graph set exists during Stage A. Full graph prefill fits P16K here.
The older experiment retained both clean and marked graph sets and encountered OOM.
All four new primary points use graph prefill and graph refinement.

| P | Clean E2E ms, 95% CI | Coarse prefill envelope ms | Coarse warm block envelope ms | NFE | TPF | Clean output TPS |
|---:|---|---:|---:|---:|---:|---:|
| 512 | 247.17 [171.29, 319.53] | 21.83, UNTRUSTED | 246.24, UNTRUSTED | 38.00 | 1.821 | 279.11 |
| 4,096 | 373.48 [269.97, 469.96] | 41.68 | 344.67 | 40.00 | 1.738 | 178.83 |
| 8,192 | 533.84 [397.20, 661.79] | 74.86 | 463.10 | 42.25 | 1.612 | 120.11 |
| 16,384 | 863.97 [673.54, 1,049.10] | 193.37 | 690.15 | 42.75 | 1.482 | 67.83 |

These are means of request-level measurements. Different contexts change content,
output length, and NFE. The curve is not a pure token-length causal effect.
Coarse phases and clean E2E are separate lanes. Their values must not be added
to construct an exact clean-wall partition.

At P16K, mean coarse wall is 885.07 ms. Observed phase envelopes are approximately
193.37 ms prefill, 624.77 ms regular refinement, and 54.78 ms refresh.
The corresponding ratios to mean coarse wall are 21.85%, 70.59%, and 6.19%.
The remainder includes control, cache management, finalization, and unresolved intervals.
Refinement and refresh already contain their sampler work. Sampling is not added again.
The refresh mean includes the isolated outlier described above.
The disjoint host-scope partition closes numerically, but host scopes are not GPU compute categories.

### Average regular refinement: actual outer intervals

P16K contains 981 observed regular-refinement intervals across eight requests and three repeats.
They are repeated measurements, not 981 independent requests.

| Outer interval | Mean ms | Fraction of the 15.285 ms refinement envelope |
|---|---:|---:|
| Native graph execution envelope | 14.749 | 96.49% |
| Graph input staging envelope | 0.294 | 1.92% |
| Sampling/acceptance envelope | 0.145 | 0.95% |
| Other iteration control/unresolved | 0.098 | 0.64% |

The graph envelope contains compute, copies, collectives, internal gaps, and dependencies.
It is not synonymous with GPU compute. It is also not a measured attention fraction.

The paired representative trace gives these rank-0 regular-iteration activity values:

| Category | Native ms | Compact ms | Interpretation |
|---|---:|---:|---|
| Source-confirmed KV copy | 8.808 | 0.234 | Exclusive rank activity. Removability independently tested by clean E2E |
| Attention kernels | 0.726 | 0.682 | Exclusive rank activity, not full attention critical time |
| Routed expert kernels | 0.914 | 0.966 | Shared expert and some dense work are separate/unresolved |
| TP all-reduce | 0.656 | 0.772 | Payload movement versus rank wait remains unresolved |
| Dense GEMM, norm, router, elementwise, overlap, gaps | Recorded separately | Recorded separately | No forced fine critical-path partition |

This is one calibration request, not a population estimate.
Native clock uncertainty and graph dependency limits remain visible.
Do not add service times across streams or ranks and call that E2E.

## 4. Physical cause and repeated costs

The storage evidence has three independent checks:

1. Source-labelled eager execution identifies `LLaDA2Attention.forward -> Tensor.slice_scatter`.
   All four ranks reproduce exact native logits for that same-input diagnostic.
2. CUPTI reports real 335,544,320-byte Device-to-Device copies at that operation.
   The layer view contains only 8,388,608 bytes. Each fresh canvas has 8,192 bytes.
3. A CPU tensor regression reproduces the 40× backing-storage clone and checks exact compact-copy values.

The [PyTorch 2.8 implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/aten/src/ATen/native/TensorShape.cpp)
intentionally preserves size, stride, and storage offset through a full-storage clone.
The generic operator is correct. The runtime's monolithic-view use is expensive.
This is not inferred from the name `memcpy128` alone.

In the source-correlated native request, 14 cached forwards issue 560 full-storage copies per rank.
Their requested D2D volume is 175 GiB per rank. This is not measured DRAM read traffic.
These are same-device copies, not NVLink/P2P transfers or EP communication bytes.
No HBM hardware counter or true bandwidth claim is made.
The corresponding low-overhead profile records about 123 ms of exclusive copy activity per rank.
That activity is a causal-trial target, not an automatic E2E speedup.

The same-process paired trace has 12 native forwards and 440 copy occurrences.
Copy service falls from 96.887 ms to 2.577 ms with the compact control.
The occurrence count stays unchanged. Non-copy work remains similar.
The per-operation storage footprint falls from 320 MiB to 8 MiB.
Candidate physical D2D bytes were not independently recounted with a second CUPTI byte lane.
Its storage footprint is checked during graph capture. Its copy time and E2E are measured.

Actual native collectives include BF16 SUM all-reduce and token-state broadcast.
The cached forward has 40 all-reduce kernels: two per layer.
A 32-row hidden reduction has a 128 KiB logical input payload per call.
This is a payload proxy, not ring wire bytes.
No EP dispatch/combine A2A exists. No all-gather or reduce-scatter was identified in these request traces.
Initialization barriers are outside headline request timing.

CPU synchronization often waits for necessary GPU work.
Its API duration is not an independent removable delay.
The common-clock graph trace shows about 9.3–9.6 ms of between-activity idle per rank
within a roughly 415 ms request envelope. Internal graph idle remains opaque there.
Node traces expose internal streams, but heavy perturbation prevents precise native idle attribution.
Rank waiting cannot be separated from NCCL payload/protocol time with this evidence.

Removing *all* outside-graph work gives optimistic E2E ceilings of 6.20%, 5.15%, and 5.02%
at P4K, P8K, and P16K. Realistic gains would be smaller.
This does not motivate a host/sampling/staging architecture at the required 10–15% scale.

## 5. Critical-cost ranking and headroom

| Rank | Suspect | Evidence and decision |
|---:|---|---|
| 1 | Oversized KV backing-storage copy | Source, payload, repeated activity, and exact clean intervention confirm a large removable E2E cost |
| 2 | Expert/FFN and other local compute | Physical work exists. Full family critical fraction and removable architecture gain remain PARTIAL |
| 3 | TP all-reduce/rank synchronization | Repeats 40 times per forward. Transfer versus waiting and achievable removal remain UNRESOLVED |

Only rank 1 is a demonstrated optimization target.
Ranks 2 and 3 are investigation candidates, not proven removable bottlenecks.
Their ordering must not be treated as a precise fine-DAG ranking.

An optimistic complete-copy removal bound differs from an achievable control.
The control retains one logical copy and the fresh canvas write.
It retains input staging, attention, MoE, TP communication, block refresh, and sampling.
The measured effect below validates actual removability instead of substituting a byte oracle.

## 6. Architecture screening

| Layout | Physical mapping | Weight GiB/rank proxy | Feasibility and performance status |
|---|---|---:|---|
| TP4 | All four GPUs | 9.08 | Physically validated. New clean baseline and exact control measured |
| TP2×DP2 | (4,5) and (6,7) replicas | 16.35 | Weights fit in theory. Candidate KV/graph peak, B1, and B2/B4 throughput NOT_RUN |
| TP1×DP4 | One replica per GPU | 30.88 | Weights fit in theory. Long-context graph workspace and throughput NOT_RUN |
| PP2×TP2 | Stage0=(4,5), stage1=(6,7) | 9.08 | Layer-weight proxy only. Native diffusion transport/feedback and cache contract not validated |
| PP4×TP1 | Stages 4→5→6→7 | 9.08 | Same limitations. B1 has no independent refinement pipeline fill |

Weight capacity is not executable workspace capacity.
The full loaded-weight proxy is 33,155,296,256 bytes, including replicated/runtime tensors.
The checkpoint tensor metadata reports 32,511,286,784 bytes.
Native Stage A P16K allocates about 43.82 GiB/rank at peak.
Paired graph buffers remain co-resident. Their peak is not a minimum candidate deployment footprint.

For ideal balanced single-wave PP, B1 utilization is at most 1/2 or 1/4.
The ideal B4 wave figures are 4/5 or 4/7. These are dependency bounds, not measured throughput.
Successive refinements of one request cannot fill those bubbles independently.
PP stage imbalance, boundary transfer, and feedback must also enter any real model.

The architecture response model is PARTIAL_CALIBRATION_ONLY.
Lower-TP operator scaling and PP stage transport were not measured.
Numerical B1 or throughput predictions would therefore be unsupported.
We report PREDICTION_ONLY or ARCHITECTURE_NOT_AVAILABLE rather than invented speedups.
No alternative-layout engine was built.

The selected physical control keeps TP4 and addresses the strongest identified cost.
This decision avoids confusing replica throughput with B1 latency.
After the correction, no new parallel architecture has enough evidence for promotion.

## 7. One exact full-model control

The candidate performs a contiguous, independent clone of the logical layer KV view.
It writes fresh canvas K/V into that clone. The source cache remains unchanged.
It does not replace `slice_scatter` globally or claim equivalent stride metadata for arbitrary callers.
The replacement is scoped to this model's attention interface and captured native graph shapes.
Both arms use the same weights, graph policy, backend, device set, and sampler.

The control and operating contexts were frozen before candidate results.
No second candidate or threshold search ran.
Arm order is counterbalanced within each request and technical repeat.
The same model instance supports independent native and candidate output storage.
Held-out probes assert that KV and logit buffers do not alias.

| Cohort | N | Native clean E2E ms | Compact clean E2E ms | Mean paired reduction, 95% CI |
|---|---:|---:|---:|---|
| P4K calibration | 8 | 378.50 | 295.20 | 21.67% [20.32%, 22.85%] |
| P16K calibration | 8 | 806.82 | 481.52 | 37.86% [32.93%, 42.47%] |
| P16K held-out | 24 | 881.08 | 518.49 | 39.54% [37.33%, 41.54%] |

Held-out mean request-level output TPS increases from 69.32 to 113.74.
These are B1 output rates, not B2/B4 aggregate serving throughput.
Held-out coarse warm-block envelopes average 690.71 ms native and 328.53 ms compact.
These are event-lane envelopes, not clean warm-serving measurements.

Native output and NFE can differ across separate runtime sessions.
For example, the profiled coding request used 15 forwards in Stage A and 12 in the paired campaign.
The exact restart-level numerical cause was not resolved.
Within every paired comparison, outputs and NFE match exactly.
No cross-session absolute baseline or NFE distribution is pooled to estimate the control effect.

## 8. Correctness, limitations, and conclusion

Stage A passes 896 lane/rank checks for outputs, NFE, and retained acceptance updates.
The control passes every tested clean, coarse, and parity-only comparison.
All 40 regular-graph KV tensors and logits compare exactly on deterministic probes.
Native source values remain unchanged in the regression test.
Held-out output buffers are independent. No tolerance was relaxed after held-out execution.
The same output IDs and update schedule preserve EOS and bounded stop behavior.

This is not official GSM8K/HumanEval accuracy evidence.
Outputs are bounded at 64 tokens. No quality improvement or accuracy-speed trade-off is claimed.
Full cross-rank dependency reconstruction remains PARTIAL.
Fine family percentages, NCCL waiting, and internal graph idle are not fully resolved.
All actual replay occurrences and valid outer budgets remain available despite those limits.

The generic clone correction is useful. It is not a new multi-GPU architecture.
The native graph configuration is validated, not proven fastest among all engines.
Torch compilation, other KV backends, and newer library versions were not benchmarked.
The earlier CP component used separate KV allocations, not this monolithic backing-storage view.
Its small component timings were not an E2E estimate of the newly identified copy cost.
No prior frozen result was rewritten.
PyTorch already documents the underlying stride-preserving behavior.
FastPP, BlockServe, Denoising Surface, Epoch, FlashBlock, LoSA, and DCP constrain broader novelty.
See [prior_overlap.md](prior_overlap.md). No measured Epoch comparison ran.

The next action is to establish the compact-copy path as a stronger reference baseline.
Then run one supported TP2-versus-TP4 probe before proposing a new parallel scheduler.
Do not develop another CP, EP, or PP method from the uncorrected copy cost.

Branch: `flashvep/dllm-multigpu-e2e-atlas`.
Report: `poc_dllm_multigpu_e2e_atlas/report.md`.
Resolve this report's commit with `git log -1 --format=%H -- poc_dllm_multigpu_e2e_atlas/report.md`.
The final response records the full SHA after push.
