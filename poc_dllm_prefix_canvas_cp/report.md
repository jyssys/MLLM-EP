# dLLM prefix–canvas CP: final discovery report

## Decision

**Verdict: NO-GO for the tested exact four-GPU CP prototype.**

The native attention/KV path grew with prefix length. This observation did not
produce a useful CP implementation. The exact candidate was slower at every
tested length, even before prefix preparation entered its cost.

At P16,384, the candidate took 7.364 ms per 60 attention calls. Native attention
took 4.094 ms. A stronger exact local control took 2.880 ms. Candidate speedups
were 0.556× and 0.391×, respectively. These are physical component measurements,
not full-model latency measurements.

Stage C did not run. The 24 frozen held-out requests remain untouched.
No measured candidate E2E gain, TPS gain, or benchmark accuracy result exists.
This negative result does not exclude every possible future CP implementation.

## 1. Checkpoint, runtime, and context

The checkpoint is `inclusionAI/LLaDA2.0-mini`, not mini-preview.
The exact revision is `dad945cac317da394b390f82c7b40691d8a881ed`.
The loaded model supports 32,768 positions. No RoPE extrapolation occurred.

The existing dInfer revision is `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f`.
It contains pre-existing local changes. Their SHA-256 is
`7b148f52ab97e176ba84c57f28a0c40bacdc488e139135b8664569dbbab56e27`.
The completion audit checks that these changes remain unchanged.
The patch snapshot records provenance. It is not a new runtime modification.

The environment uses PyTorch 2.8.0+cu128, SGLang 0.5.3.post1, transformers 4.57.0,
and Triton 3.4.0. DeepEP 1.2.1+73b6ea4 is installed but inactive in the primary baseline.
The native attention backend is optimized Torch SDPA.
Profiler traces identify FlashAttention split-KV kernels during regular refinement.
Masked prefill uses CUTLASS memory-efficient attention.

The architecture has 20 attention layers, hidden size 2,048, 16 query heads,
four KV heads, and head dimension 128. All 20 layers use full attention.
`sliding_window=4096` exists, but `use_sliding_window=false` disables it.
RoPE uses theta 600,000 and rotary dimension 64.
The 19 sparse layers have 256 routed experts and top-k 8.

The old 4,096-token runner setting was not a checkpoint limit.
The tested logical prefixes were 512 through 16,384 tokens.
P32,768 was skipped because prefix plus output would exceed the trained limit.
P16,384 required native eager prefill because full-prefill graph capture exhausted HBM.
Refinement still used native CUDA graphs.

## 2. Physical four-GPU layout

Only physical GPUs 4, 5, 6, and 7 ran experiments.
They are H100 80GB HBM3 devices. The selected pairs report NV18 connectivity.
The recorded topology is an NVLink/NVSwitch fabric. Driver version is 570.211.01.

| Logical rank | Physical GPU | GPU UUID suffix |
|---:|---:|---|
| 0 | 4 | 6076e2f2-5b63-3761-5586-56ceb7df8139 |
| 1 | 5 | a1a1cfcf-93a1-3544-9a5e-e58144b68730 |
| 2 | 6 | e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28 |
| 3 | 7 | 4cc26b88-19fc-1988-f9e0-17858aa7a99b |

The native layout is TP4/EP1/DP1/PP1/CP1. All four ranks process one request.
Each rank owns four query heads and one KV head. Each KV-head owner retains
the complete key sequence. Native TP reductions restore the dense hidden state.
The LM head is replicated, not vocabulary-sharded.

This baseline does not use the older source-SP/DeepEP bridge.
That bridge would add unrelated communication to an attention study.
The source contains an SP helper, but this path does not establish source-SP4.
Per-rank loaded tensor shapes and UUIDs appear in `actual_layout.rank*.json`.

## 3. Prefix and canvas behavior

Native prefill already caches prefix projections. The candidate cannot claim
that saving. Current canvas Q/K/V remain fresh during each refinement forward.
Accepted positions inside the active block are not immutable prefix positions.

The trace sampled layers 0, 9, and 19 during three regular refinements.
It covered four ranks and six prefix lengths: 216 tensor-lifetime observations.
The sampled prefix K/V remained exactly equal. The current canvas changed.
Native code, rather than complete tensor probes, established commit invalidation.

The native cache adds 64 positions, aligns to 128, and rounds to a power of two.
For these prefixes, physical key capacity equals 2P. Regular refinement has
no attention mask. It therefore attends to native zero-padding slots too.
The fresh 32-row canvas replaces the final physical slots.
It does not necessarily occupy logical positions P through P+31.

This padding behavior predates the PoC. A native comment acknowledges possible
accuracy effects. We preserved it. Native observer parity does not establish
equivalence to an independent HF implementation with a different mask.

Native attention uses out-of-place `slice_scatter` for fresh canvas K/V.
Native graph preparation also copies cache inputs into static graph buffers.
The prefix is not reprojected. Logical copy sizes are not measured HBM traffic.
The KV-head expansion can be a zero-stride view, not a physical expanded copy.

## 4. Workload and measurement

The metadata-only freeze contains eight calibration requests and 24 held-out requests.
Calibration includes three coding, three math, and two document requests.
Backgrounds use real CPython source, GSM8K train examples, and WikiText test text.
Dataset revisions, file hashes, token hashes, and complete task text are frozen.
Background token ranges are disjoint. Article, source-file, and question disjointness are not guaranteed.

The primary contract is B1, block32, threshold 0.9, temperature 0, and output cap64.
Native EOS remains enabled. Actual suffix lengths range from 28 to 64 tokens.
The task remains complete at every prefix length. Context excerpts vary by length.
These bounded outputs are not complete benchmark solutions.

Each length has three warmups. Each calibration request has three clean repeats
and two coarse-event repeats. Trace and collective diagnostics use one frozen
request per length. One P16,384 request has a separate heavy profiler sample.

Clean timing uses minimal request/phase observers, not operator hooks or a profiler.
It includes native token-update bookkeeping for parity. Thus, it is not a
verified zero-observer serving measurement. The extra event observer is measured
against this minimal lane. Heavy trace wall times never enter headline means.

Cold E2E starts with an uncached request prefix on a loaded, warmed model.
It includes native prefill and generation. It excludes model loading, graph capture,
tokenization, network transfer, and server queueing.
Warm decode span excludes initial prefill but includes native cross-block refresh
and sampling. Pure refinement and commit wall times are not fully separated.
The internal first token-update event is not streaming TTFT.

Statistics first average repeats within each request. Bootstrap uses 2,000 request
resamples, not independent layers, tokens, or replays. Each length has N=8.
Components use one cumulative rank per request. No cross-rank maxima are added.
Absolute timestamps across ranks are not aligned.

## 5. Stage A: measured native latency

Means below describe real full-model native runs. TPS uses actual emitted tokens.
It is the mean of request-level TPS, not a ratio of aggregate means.

| P | Physical keys | Cold E2E ms, 95% CI | Prefill ms | Warm span ms | Warm TPS | NFE | TPF |
|---:|---:|---|---:|---:|---:|---:|---:|
| 512 | 1,024 | 270.26 [197.49, 336.71] | 40.33 | 229.10 | 308.10 | 38.00 | 1.821 |
| 1,024 | 2,048 | 274.58 [201.41, 338.37] | 11.85 | 261.73 | 265.14 | 40.63 | 1.691 |
| 2,048 | 4,096 | 298.82 [209.33, 375.18] | 17.74 | 280.19 | 254.98 | 39.00 | 1.813 |
| 4,096 | 8,192 | 383.84 [275.77, 476.10] | 33.12 | 349.30 | 198.30 | 40.00 | 1.738 |
| 8,192 | 16,384 | 547.83 [401.65, 672.36] | 73.83 | 472.56 | 143.46 | 42.25 | 1.612 |
| 16,384 | 32,768 | 868.07 [678.96, 1,037.60] | 194.43 | 672.03 | 94.88 | 42.75 | 1.482 |

P512 and P16,384 use eager prefill. The other primary points use graph prefill.
This choice follows the documented P16,384 graph OOM and its matched short control.
The cold curve is not a controlled estimate of token-length effects alone.
NFE and emitted lengths also vary with input content. All points use graph refinement.

The additional coarse observer costs 4.96%, 4.94%, 2.79%, 2.70%, 2.68%, and 1.77%, respectively.
Its 95% CI upper endpoints remain below the 10% invalidity threshold.
P512 and P1,024 exceed the preferred 5% ceiling at their upper CI endpoints.
The earlier dense-marker pilot failed this check and is not headline evidence.

### Attention attribution and Amdahl limits

The exact full-request, nonoverlapped targetable fraction is **UNRESOLVED**.
External graph events retain the last replay for each graph key.
They do not preserve every earlier operator interval.
Analysis multiplies these observed durations by actual native call counts.
That multiplication assumes stationarity. It is an estimate, not direct timing
of every attention execution.

The parent attention interval includes QKV, normalization, RoPE, attention,
and output projection. It is a generous superset of the CP target.
Per-call graph KV preparation is directly timed and added separately.
The following fractions and ceilings therefore remain conditional estimates.
Their bootstrap intervals do not capture attribution or stationarity error.

| P | Warm attention/KV superset estimate | Cold repeated-target estimate | Cold reduction ceiling: 1.5× / 2× / 4× / infinite target |
|---:|---:|---:|---|
| 512 | 25.15% | 20.54% | 6.85% / 10.27% / 15.41% / 20.54% |
| 1,024 | 29.03% | 27.43% | 9.14% / 13.72% / 20.58% / 27.43% |
| 2,048 | 35.67% | 32.93% | 10.98% / 16.46% / 24.70% / 32.93% |
| 4,096 | 44.97% | 40.10% | 13.37% / 20.05% / 30.07% / 40.10% |
| 8,192 | 56.79% | 47.80% | 15.93% / 23.90% / 35.85% / 47.80% |
| 16,384 | 70.28% | 52.38% | 17.46% / 26.19% / 39.28% / 52.38% |

At P16,384, the warm superset estimate is 70.28% [69.85%, 70.81%].
The cold estimate is 52.38% [47.19%, 56.79%]. These are not measured achievable savings.
The corresponding warm reduction ceilings are 23.43%, 35.14%, 52.71%, and 70.28%.
The JSON records both fractional latency reductions and Amdahl speedup factors.
For example, its 2× cold-target ceiling is approximately 1.355× at the mean fraction.

Stage A did not establish a precise critical-path PASS.
It justified only a bounded component kill-test, not E2E integration.
This is a limitation against the stricter Stage A promotion contract.
No unknown fraction was promoted to measured headroom.

### Kernel, communication, and memory limits

One P16,384 heavy sample identifies these rank-0 kernel-class totals:

- `memcpy128`: 123.25 ms over 560 events.
- Masked-prefill CUTLASS attention: 85.14 ms over 20 events.
- Fused MoE kernels: 32.75 ms over 570 events.
- NCCL BF16 ring kernels: 22.10 ms over 600 events.
- FlashAttention split-KV and its combine: 10.10 ms over 560 events.

These totals are service-time classifications, not independent critical-path shares.
NCCL annotations can overlap kernel events. Analysis never adds both as separate work.
`memcpy128` is a copy-class candidate, not proven exclusive KV-read traffic.
Exact prefix HBM reads, QKV-only wall time, current-canvas update wall time,
attention-specific TP critical time, and LM-head/sampling wall time remain unresolved.

Two hidden-state TP reductions per layer imply a logical payload proxy of
5 MiB per rank for a 20-layer, 32-row forward. This is not measured wire traffic.
No true bandwidth or independent CP-cost percentage is claimed for native inference.
Native CP is absent. Component CP measurements below include actual collectives.

One logical padded K/V copy across 20 layers ranges from 10 MiB/rank at P512
to 320 MiB/rank at P16,384. This counts elements, not physical HBM accesses.
At P16,384, mean peak allocated HBM is 69.80 GiB/rank. Reserved HBM is 72.36 GiB/rank.
These values include co-resident clean/event graph buffers. They are not minimum deployment requirements.
The full-prefill graph OOM remains in the raw evidence.

## 6. Stage B: physical exact attention kill-test

The component test uses synthetic BF16 tensors with native geometry and Q layout.
It is not a pretrained task execution. It preserves native unmasked regular-refinement
semantics, including physical zero slots. It does not accelerate masked prefill,
cross-block refresh, empty-key masks, or a sliding-window model.
Those cases would retain the native path. Their candidate correctness is not established.

All four real NCCL ranks participate. The candidate first transposes the fixed
KV head shards into sequence shards with physical all-to-all transfers.
It gathers fresh query heads and canvas K/V for each attention call.
Only rank 0 includes the canvas contribution.
An existing FlashAttention operator computes local output and logsumexp.
A global max and a weighted-output reduce-scatter restore the original head owners.
The merge uses FP32 and returns BF16 output. No new CUDA kernel exists.

This layout reuses four ranks. It does not claim that independent TP4×CP4 fits four GPUs.
Every timed candidate call includes packing, collectives, merge, and synchronization.
One-time prefix preparation is measured separately, then included in amortization estimates.

The native arm uses full-cache `slice_scatter` and optimized SDPA.
The stronger local arm updates only the fresh canvas rows in place, then uses the same SDPA.
Local and native outputs match bitwise in these probes.
The local arm is a component control, not an integrated serving implementation.

Each measured unit contains 20 layers and three full refinement traversals.
The order is step-then-layer, not three consecutive states within one layer.
There are three warmups and ten interleaved paired repeats.
Each repeat times ten CUDA-graph replays. Repeats are not independent task requests.
The table uses one cumulative rank for each prefix, not independent component maxima.

| Logical P | Native ms/60 calls | Exact local ms/60 calls | Exact CP ms/60 calls | CP speedup/native | CP speedup/local | Warm prefix preparation ms/20 layers |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1.652 | 1.296 | 4.883 | 0.338× | 0.265× | 4.910 |
| 8,192 | 3.278 | 2.489 | 6.707 | 0.489× | 0.371× | 4.249 |
| 16,384 | 4.094 | 2.880 | 7.364 | 0.556× | 0.391× | 6.035 |

At P16,384, per-call latencies are 0.06823, 0.04800, and 0.12273 ms, respectively.
The CP arm fails the 1.25× gate even with free prefix preparation.
Charging actual preparation makes it worse. First-process preparation includes
NCCL initialization and appears separately from warm per-block preparation.
It is not hidden or mistaken for recurring transfer cost.

The isolated component test does not reproduce every full-model cache-pressure effect.
It cannot establish E2E prediction accuracy from Stage A parent intervals.
Its direct conclusion is narrower: this prototype does not justify integration.
Collective and merge overhead exceed the saved local work in these measurements.
The experiment does not prove a hardware lower bound for all optimized CP designs.

### Correctness scope

Tolerances were frozen before candidate measurements: atol 0.03125, rtol 0.01,
and relative L2 at most 0.01. All 720 BF16 comparisons passed.
Maximum absolute error was 0.001953125. Maximum relative L2 was approximately 0.002993.
The evidence also includes 240 synthetic cache-version transition assertions.
Twenty-four independent FP32-formula comparisons passed the recorded error bounds.

Native clean/event/trace/collective runs passed 1,008 token, update-schedule,
and NFE comparisons across four ranks. The heavy profiler sample also preserved parity.
These checks do not establish candidate full-generation parity.
Stage C, candidate token/acceptance/router parity, and benchmark quality are NOT_RUN.
No accuracy-speed trade-off is accepted.

## 7. Failures and validity corrections

All failure logs remain in `results/raw/`. Later corrections do not overwrite them.

- An early overlapping cleanup matched another own PoC output too broadly.
  It affected an own smoke worker, not another user's process.
  Cleanup now scopes ownership to the exact output directory.
  A red/green regression checks this boundary with own CPU subprocesses.
- Dense event markers caused excessive observer overhead in an early pilot.
  Callback binding errors also broke that trace. CPU regressions cover both callbacks.
  Primary timing uses the corrected coarse observer.
- A shared RoPE module received repeated fine hooks in an early pilot.
  Hook installation now deduplicates the module. Old fine attribution is not used.
- P16,384 full-prefill graph capture failed in accumulated and fresh processes.
  Native eager prefill succeeded. The report does not assign an unproven OOM cause.
- The completed v6 campaign stalled during legacy SGLang graph/IPC teardown.
  All request, profiler, and parity files existed before interruption.
  The guard status remains interrupted, not successful.
  Cleanup restored the burners. Current reproduction code exits after synchronization,
  a four-rank barrier, and completion records. That cleanup correction was not GPU-rerun.
- Stage B v1/v2 are retained layout/order pilots. Final numbers use v3 only.
  Seed, native Q stride, and full-layer refinement order were validity corrections.
  They did not introduce a new candidate or change correctness thresholds.

Eight CPU tests pass. The completion audit checks raw hashes, preserved original
staged/dirty state, unchanged external runtime, untouched held-out IDs, and restored burners.

## 8. Prior boundary and next action

FlashBlock already exploits cross-step attention redundancy. LoSA already reduces
KV work through sparsity and stable-token reuse. Native DiffusionGemma caches
committed prefix KV. Ring Attention and distributed softmax already partition
attention across GPUs. See [prior_overlap.md](prior_overlap.md) for primary sources.

This PoC establishes no distinct dLLM-specific method contribution.
Fixed prefix plus fresh canvas is a valid runtime property, not novelty.
No secondary engine was built to rescue the negative result: SECONDARY_NOT_RUN.

Stop CP integration and do not create a new scheduler or custom kernel here.
The smallest separate follow-up would isolate exact local canvas-update and
graph-cache staging costs inside native inference. First require native mask parity
and clean E2E gain. This local question is not evidence for CP or a new method.

## 9. Required final fields

| Field | Result |
|---|---|
| VERDICT | NO-GO for the tested CP prototype |
| MODEL_RUNTIME | LLaDA2.0-mini dad945ca / existing dInfer 1ffeb961 + pinned pre-existing patch |
| SUPPORTED_CONTEXT | Trained 32,768. Logical P through16,384 executed. P32,768+output excluded |
| PHYSICAL_LAYOUT | GPUs4/5/6/7, TP4/EP1/DP1/PP1/CP1 native |
| PREFIX_CANVAS | Sampled prefix fixed. Canvas fresh. Native padding behavior retained |
| ATTENTION_CRITICAL_SHARE | UNRESOLVED exactly. Stationarity/superset estimates only |
| PREFIX_SCALING | Warm span229.10→672.03ms across P512→16,384, with content/NFE changes |
| AMDAHL | P16,384 cold2× target ceiling26.19%, not achievable measured gain |
| STAGE_B | Numerical component PASS. Performance FAIL |
| STAGE_C | NOT_RUN |
| MEASURED_E2E_GAIN | NOT_RUN |
| CORRECTNESS_QUALITY | Native observer parity PASS. Candidate task parity/accuracy NOT_RUN |
| PRIOR_NOVELTY | Generic CP/cache techniques overlap. New contribution not established |
| NEXT_ACTION | Stop CP integration. Optional separate native local-KV diagnosis only |
| BRANCH | flashvep/dllm-prefix-canvas-cp-poc |
| COMMIT | This report's commit. Resolve with `git log -1 --format=%H -- poc_dllm_prefix_canvas_cp/report.md` |
| REPORT | poc_dllm_prefix_canvas_cp/report.md |

The final response records the full commit SHA after commit and push.
