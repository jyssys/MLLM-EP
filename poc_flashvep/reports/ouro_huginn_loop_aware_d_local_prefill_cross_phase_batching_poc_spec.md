# Ouro/Huginn: SLO-Safe D-Local Prefill Deflection with Loop-Aware Cross-Phase Batching

**Status:** exploratory, fail-fast proof of concept (PoC); no performance or novelty claims assumed.  
**Intended repository path:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching_poc_spec.md`  
**Reference implementation:** `jyssys/MLLM-EP`, evidence parent `034b4ef9d2a8f0fe1a9d993370693d349cd03838` (`flashvep/loop-specific-native-split-deep-validation`).  
**Hardware:** H100 80 GiB, IDs 0–7 available; only an isolated 1P1D pair is eligible for a clean comparative serving run.

## 1. Motivation and research questions

The tested static Token/Loop/2D continuation paths migrate work in essentially every selected request (384/384 in each three-start medium cohort), even though the hypothesis of *instantaneously safe decode-side slack* was not validated. Native PD isolates decode and thus delivers better ITL in many cases. An idle-average or successful independent 16-token D-local probe is **not** evidence that a long, non-preemptible migrated operation can fit between decode steps. In the final marker-free medium cohort, Ouro's native ITL p99 was about 39.785 ms against a 65.083-ms bound; Loop3 produced 96.940 ms and Token3840 70.525 ms. Huginn's native ITL p99 was 126.472 ms against 192.821 ms; Loop31 produced 161.807 ms but Token1984 produced 207.278 ms. These values are per-run tail summaries, not deterministic per-step slack budgets.

This PoC **does not resume P-to-D partial-state migration**. It evaluates *whole-request, before-prefill D-local deflection*: a queued request is assigned either to P for Native PD or directly to D, where its entire prompt is processed in safe incremental units and its subsequent decode remains local. D-local assignment eliminates inter-GPU **prefill-state** handoff for that request but still needs the request input and local KV allocation; it trades the handoff for D-side compute and KV occupancy. No claimed net win is automatic.

Central scientific questions:

1. Can D execute any extra prefill in **actual online decode-safe windows**, rather than in windows inferred from aggregate utilization?
2. Does a baseline **generic SLO-aware D-local chunked-prefill** policy improve TTFT or SLO-goodput relative to Native PD? This is prior art, not the new mechanism.
3. Does **cross-phase, cross-recurrent-depth shared-core batching** deliver an *additional*, measurable improvement over equally tuned same-depth/generic co-batching and separate execution?
4. Are gains robust on **both Ouro R4 and Huginn R32**, and can they be attributed to tied-weight recurrence rather than ordinary batching, changed KV admission, a changed graph cache, or profiler overhead?

**Primary research comparison:** `generic D-local decode-priority deflection` **versus** `same deflection + loop-aware depth-mixed shared-core batching`. Improvements over Native alone are necessary but insufficient for a loop-specific novelty claim.

## 2. Prior-art and novelty boundary

- **Load-Aware Prefill Deflection (2026):** proactive whole-request placement on D; chunked prefill interleaved with D decode with TBT constraints; no P-to-D KV handoff for D-local requests. This is a strong *generic* baseline. https://arxiv.org/abs/2607.02043
- **Continuous Depth Batching (CDB, 2026):** loop-iteration batching and non-loop boundary scheduling. Depth-mixed batching by itself is not novel. https://arxiv.org/abs/2608.09444
- **FluidPD/FluidToken (2026):** pressure-aware token deflection/splitting and PPI/DPI; distinguish official algorithm from any local FluidToken-style reproduction. https://arxiv.org/abs/2610.06917
- **ExactLoop (2026):** loop-specific execution engine: exact kernel fusion, graph budgets, compact live KV planes, prefill/PD transfer optimization. Not itself a request-level shared-core D-local deflection method. https://arxiv.org/abs/2611.05150

**Required novelty control:** Show that combining ready states from **different recurrent depths** with the **same tied weights** improves efficiency over (i) separate execution, (ii) generic prefill/decode co-batching at compatible stages, and (iii) a depth-aware batching baseline analogous to CDB. If all improvements are achieved by ordinary D-local token chunking or same-depth batching, classify as `GENERIC_ONLY` and do not claim loop-specific novelty. Untied Transformer layers normally use different weights across layer depth, but grouped-GEMM and standard same-layer co-batching can already give substantial benefits; do not overstate this distinction.

## 3. Models, workload, contracts

| Dimension | Ouro | Huginn |
|---|---|---|
| Model | `ByteDance/Ouro-1.4B` | `tomg-group-umd/huginn-0125` |
| Revision | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` | `bb6621b65e90b6a4b9b29ef88dc83866d450470c` |
| Precision and exactness | Original BF16; no approximation, quantization, early exit, or skipped loops | Same; preserve Prelude, recurrent injection, Coda, and relevant KV planes |
| Recurrent depth | Fixed R4 | Fixed R32 |
| Prompt/output | 4096 / 32 tokens | 2048 / 16 tokens |
| Nominal QPS low / medium / heavy | 1.23460875 / 3.4569045 / 6.356 | 0.37903355455 / 1.06129395274 / 1.971 |
| Existing P / D slots | 2 / 8 | 2 / 8 |
| Frozen TTFT limit | 1112.993 ms | 1963.992 ms |
| Frozen request-average TPOT limit | 53.707 ms | 186.572 ms |
| Frozen token ITL p99 limit | 65.083 ms | 192.821 ms |

Keep the prior ShareGPT + LongBench length-controlled, fixed-output, ignore-EOS and prefix-cache-off contract for *comparability*. Use held-out prompts and multiple paired arrival seeds for final tests; report real offered/completed QPS, arrival-to-first-token TTFT, Service TTFT, queue/admission latency, TPOT and ITL distribution, maximum ITL, E2E, joint TTFT+TPOT SLO-goodput. Fixed token caps, rather than natural EOS, must remain explicit. Include at least one workload with *naturally variable decode occupancy* if feasible; do not infer generality from 128 deterministic prefixes.

**Decode safety:** retain frozen per-model ITL p99 and TPOT screen: additional work must keep request TPOT p95 ≤ 90% of frozen TPOT threshold AND ITL p99 ≤ frozen threshold. Also report per-request deadline misses and max ITL; an acceptable global p99 never authorizes starvation of individual decode streams. For a new, hard real-time claim, implement an additional explicit per-stream deadline contract; the original p99 screen alone is statistical.

**HBM/admission:** previous Huginn D used a 67.354-GiB KV pool and its full-depth reservation geometry admitted only about six full 2048-token requests despite eight D compute slots. New D-local prompts compete with decode KV. Measure admission credits, reserved blocks, HBM including graph pools and workspace, fragmentation, OOM, actual active decode count. Do not treat logically unused KV planes as free HBM without implemented compaction. ExactLoop's KV packing is a *separate*, optional control and must be applied symmetrically to baselines if used.

## 4. Distinguish observations from hypotheses

**Previously confirmed:** Native PD has better ITL for several tested static cuts; old independent 16-token D-local prefill probes were sometimes safe; qualified static cuts migrate work to D; accurate state handoff and fixed-depth forward are possible.

**Not yet confirmed:** the size/distribution of **contiguous** D-side safe execution windows under actual online decoding; whether a complete D-local prompt can be interleaved safely; whether mixed-depth batching reduces GPU time; whether a net TTFT or goodput benefit exists; and whether it is truly loop-specific.

A free-looking D interval can be exhausted by the next ready decode; CUDA work is generally not instantaneously preemptible by the host scheduler. Do not equate `ITL_limit - observed_native_ITL_p99` to a deterministic budget. Never declare D 'idle' using NVML mean utilization or by P kernel-union percentage.

## 5. Proposed execution architecture

At the **request admission boundary**, not after P has started prefill:

1. If P is backlogged, D KV credits are available, and decode-side prefill budget is safe, **assign the entire new request to D-local prefill**. Otherwise keep the request on standard Native PD P→D route.
2. D maintains decode-ready work at **strict first priority**. D-local prefill is executed in bounded slices selected from token chunks, recurrent-loop steps, or both. Maintain topological dependencies and exact per-request/per-depth KV plane semantics; do not run an arbitrary token×depth tile that depends on a predecessor not yet computed.
3. Between decode service opportunities, look at actual decode-ready backlog, per-stream last-token timestamps, forecasted next decode costs and safety guard. Launch a prefill slice only when its **calibrated upper execution bound + scheduling/graph overhead** fits the guarded window. Implement fallbacks for unexpected arrivals; count and report violations. Once a long non-preemptible launch starts, a newly arrived decode may still wait.
4. If D-owned prefill finishes, keep the request's decode on D; no P→D partial-state migration. Track exact local ownership, source leases where applicable, KV lifetime and ACK only on Native PD requests.
5. **Zero-deflection / high-D-load fallback** should follow the existing Native Atomic PD fast path without staged-connector overhead. Measure fallback parity.

### Shared-core cross-phase, cross-depth co-batching (the actual proposed loop-aware primitive)

For independent **ready** work items `i` that require the same recurrent core weights `W` but may have **different** recurrence indices and phases (prefill/decode), consider packing their dense projection rows:

`Y = concat(X_decode,depth=3, X_prefill,depth=1, ...) @ W_core`.

This may reduce repeated small GEMM launches and weight reads. It **does not** make attention, KV state, injections, positional semantics, or causal masks identical. Perform attention separately or use an explicitly validated mixed attention implementation with correct per-request/per-depth KV addressing; combine dense projections only where arithmetic and state barriers are genuinely equivalent. Split outputs back into their ownership-specific recurrent updates; run non-loop Prelude/Coda according to their own boundaries. Guard against CUDA graph changes and BF16 operation-order effects. Verify legal readiness of every input before packing. The existence of a shared weight pointer is **not** proof of a net speedup.

**Non-negotiable controls**:
- Separate execution of identical work (same workload and same D request placements).
- Generic decode-priority token-chunk D-local prefill (no recurrent-depth mixing).
- Loop-paced work but separate per-item forward (checks scheduling alone).
- Same-depth prefill/decode co-batching where feasible (generic opportunity).
- Cross-depth and cross-phase shared-weight co-batching (candidate).
- Optional grouped GEMM using distinct logical layer weights as a **systems counterfactual**; preserve numerical semantics and never replace the actual Looped model in quality runs.

An improvement in D-local prefill from *generic chunking* is recorded as such, not as a loop-specific gain. A cross-depth grouping that merely queues tasks more efficiently, with no measured shared-weight operation benefit, is also not automatically novel given CDB.

## 6. Experiment gates and sequence

### Phase 0 — source/readiness audit (CPU first)

Read prior reports, source hashes, profiler confounds, D scheduler hooks, KV reservations, graph-shape cache budget, existing D-local independent probe implementation, and paper baselines. Create a fresh isolated branch and evidence namespace. Verify quiet-GPU requirements and record original burner owner, staged-index, private runtime diff. Implement unit tests for state ownership, priority and safe fallback. Avoid modifying external runtimes in-place where process-local hooks suffice.

**Abort a method with `BLOCKED_CORRECTNESS` or `BLOCKED_RESOURCE` rather than presenting a timing result if necessary startup validation fails.**

### Phase 1 — measure **actual D decode-safe slack**

On the clean 1P1D pair, use Native PD with low, medium and stress arrival patterns, first at N128. Record per active decode batch: ready timestamp, start/end of each actual CUDA kernel or kernel sequence, graph replay, host/scheduler delay, whether D queue is empty, next-token deadline, KV occupancy, and prefill-insertion candidate windows. Use e.g. 16/32/64/128-token **D-local full-semantic probe slices**, at multiple recurrence depths, to measure p50/p95/p99/max execution and decode interference *with and without* arrivals during the slice. Distinguish idle wall, idle GPU, busy kernels and safe deadline slack. Kernel-union, SM cycles and memory bandwidth if profiler qualified; avoid attributing CUDA-event enclosures to exclusive compute. Counterfactual scheduling should replay observed decode arrivals, never assume known future arrivals online.

**Stop/redirect:** if there is no reliable SLO-safe unit larger than scheduling cost, the D-local method is not promising for this geometry. Report the observed safe granularity and its error bars.

### Phase 2 — D-local request end-to-end correctness and generic baselines

Route some **new** requests directly to D before P begins. Validate an entire D-local prefill plus its decode with N8→N32, full fixed-depth work, exact model outputs, per-depth KV, prompt-last hidden states, attention dependency and lifetime. Test P-origin Native PD and D-local paths for matched content and fixed controls; distinguish bitwise equality from floating-point batch-reordering tolerance. The preferred gate is identical generated tokens and bytewise valid state under matched geometry; if shape-induced numerics differ, log the exact deviation and do not silently relax the accuracy contract.

Measure (A) Native PD, (B) naive whole-prefill on D (diagnostic only, likely unsafe), (C) generic decode-priority D-local token-chunk prefill, (D) same placement + loop-paced *separate* operations. Candidate chunk sizes 16/32/64/128 and loop quanta 1/2/4, with valid full-prompt completion and bounded launch. Test only supported shapes and adjust safely using Phase 1; do not grid-explode. Count D-local *accepted vs attempted* requests, completed D-local prompt work, local KV blocks and no P→D handoff on such requests. A claim of 'no transfer' must exclude only intermediate P→D KV; the input still arrives at D.

**Go-to-Phase-3:** at least one correct, reproducible, SLO-safe D-local policy with meaningful work accepted in both models, OR model-specific conditional value with a clearly labeled single-model scope. If no safe work is admitted, do not build a full dynamic controller.

### Phase 3 — microbenchmark **incremental** shared-weight batching headroom

Before full runtime integration, collect ready hidden states and exact legal depth/context metadata from each model. Test pairs/triples of decode and prefill entries at the **same depth vs different depths**, and B1/B2/B4 geometries, for representative token chunk lengths. Compare on identical inputs:

1. Individual recurrent-core dense kernels (independent launches).
2. Generic same-depth prefill/decode batching.
3. Depth-mixed shared-weight packed GEMM(s), with separate correct KV/attention where required.
4. A depth-mixed *non-fused / individually dispatched* baseline to isolate scheduling benefits from arithmetic batching.

Report graph-capture viability, exact input/output state, bytes of redundant packing, weight-read traffic where measured, kernel count, achieved GEMM time, p50/p95/p99 and maximum latency, D decode stall cost, and successful fusion fraction. Warm all eligible geometries identically; report eager fallback (target zero). Run at least three independent warm repetitions, not only graph capture samples. Keep warmup cost separately.

**Novelty headroom gate (pre-declared exploratory goal):** at least ~10% reduction in the **affected dense-core service time** and >5% reduction in a correctly attributed full D-local slice time (or equivalently large reproducible measured savings), with no degradation in SLO safety, beyond **both** separate and same-depth/generic controls. These are targets, not statistical claims or required model physics. If there is no incremental cross-depth gain, mark `GENERIC_ONLY` or `NO_GO_LOOP_BATCHING` and stop before a deep implementation.

### Phase 4 — limited online integration and factorial comparison (only if Phase 3 passes)

Implement only the minimum validated cross-depth co-batching mechanism. Compare a 2×2 style controlled family at identical request assignments and arrivals:

- Native PD (no deflection / no mixing).
- Generic D-local decode-priority chunking (deflection, no mixed depth).
- Generic D-local + same-depth compatible batching.
- D-local + cross-depth shared-weight batching.

Add a forced-placement control so the cross-depth policy cannot gain by simply assigning fewer/higher-priority requests to D. For primary comparisons, fix D-local request placements across the three D-local arms, then evaluate policy-level scheduling separately. One model's permitted chunk or graph geometry must not silently differ across competing implementations. Run N128 with 3 paired independent arrival starts; only promising candidates get N512 persistence or additional starts. Use independent held-out prompt IDs and arrival seeds for confirmation; do not select cuts on the confirmation set. Report actual D-local completion share and abort/fallback counts. Online decisions must not use future arrivals. Do not oversubscribe 1P1D pair just to create apparent goodput.

**Timing validity:** marker-free is primary for TTFT, E2E, ITL and goodput. Instrumented traces are for diagnosis; use matched ON/OFF trials and bound local tails, not merely overall response span. Pair by arrival trace, seed, prompt ID, and model revision. Three starts are preliminary, not universal confidence; show start-level ratios and cautious CIs. No automatic equivalence from 'CI overlaps one'.

### Phase 5 — optional interaction with ExactLoop, only after a genuine gain

Test a strictly accuracy-preserving ExactLoop-style subset (e.g. exact graph management or compact live KV) **symmetrically** on Native and all D-local arms. Do not import approximate packed prefill or other accuracy-changing optimizations without a separate approved scope. The goal is a complementary ablation, not to claim generic runtime speedups as loop-aware scheduler benefits.

## 7. Required measurements, attribution and falsification

**Latency:** Request TTFT, Service TTFT, queue/admission, TPOT and ITL (p50/p95/p99/max), E2E, per-decode inter-token deadline misses. **Serving:** offered/realized/completed QPS, queue growth, SLO-goodput and decode screen, deflection acceptance, P backlog and D backlog, admission rejects. **GPU:** D kernel unions and per-kernel execution/wait, SM and bandwidth (qualified profiler only), graph capture/hit/miss, launch/submission counts, CPU scheduling, memory/HBM/KV credits, DMA/transport counters, output state exactness. **Mechanism:** same-depth vs cross-depth shared-core dense work, time saved, attention overhead, host packing/unpacking, unready work, granularity of actually executed slices and weight residency. Do not add component p95s or sum overlapping GPU enclosures; correlate per-request critical paths.

**Hypothesis falsification is required:**
- If D-local full request has no admission credit, report KV/HBM blockage, not model speed.
- If D-local chunking alone improves Native but cross-depth mixing does not improve generic deflection, report `GENERIC_ONLY`.
- If mixed-depth fusion accelerates dense kernels but host packing/attention/graph geometry erase online wins, report `MICRO_ONLY`.
- If a policy appears faster only when violating ITL or by dropping requests, declare it invalid.
- If benefits arise only from allowing different request placement, redo same-placement controls before claiming arithmetic benefit.
- If Huginn requires a fundamentally different scheduling ABI than Ouro, report per-model independent verdicts and restrictions.
- If lookahead forecasts are uncalibrated, do not call the controller SLO-guaranteeing; use a conservative measured gate and report misses.

## 8. Experiment budget, GPU usage and priorities

**First bounded PoC target:** ~4–6 wall-clock hours of actual GPU work, with a hard budget established by the executor. If the user has not set a hard deadline, choose and record a reproducible stop time; never continue long campaigns merely to 'get a positive answer'. Priority is Phase 1 → Phase 2 → Phase 3; Phase 4 conditional. Stop early on `NO_GO_SLACK`, `BLOCKED_RESOURCE`, `NO_GO_BATCHING_HEADROOM` or correctness blockers.

All H100 GPUs 0,1,2,3,4,5,6,7 are available subject to current ownership and approved burners. Choose 0/1 for Ouro clean 1P1D and 4/5 for Huginn clean 1P1D when ownership allows, but **do not run overlapping performance measurements** across pairs if the shared host/PCIe/CPU noise is uncharacterized. Other pairs (2/3 and 6/7) may handle **non-timing** correctness, warmup, isolated development tests, or serialized measurements after validating contention isolation; clean comparisons must use matched hardware. Record exact GPU mapping and prohibit destructive process cleanup. Restore authorized burner workers and verify process ownership at end. Keep all raw evidence under a distinct new namespace.

## 9. Decision framework

Report per-model and overall:

- `GO_LOOP_SPECIFIC`: correct full D-local serving; reproducible decode-SLO-safe online improvement **over best generic D-local deflection and same-depth/depth-scheduling control**; measured cross-depth tied-weight savings on the online critical path; nontrivial benefit verified on **both** models. Do not demand identical numeric effect, but identify common mechanism.
- `CONDITIONAL_SINGLE_MODEL`: convincing evidence on only one model, with the other not safely supported or below benefit threshold.
- `GENERIC_ONLY`: D-local chunking/deflection improves Native, but cross-depth tied-weight batching gives no extra benefit or reproduces known generic/CDB benefit.
- `MICRO_ONLY`: shared-weight microbench saves compute but no correct/online SLO-safe end-to-end gain.
- `NO_GO_SLACK`: no useful safe execution windows or D KV admission, despite correct workloads.
- `NO_GO_LOOP_BATCHING`: cross-depth path has no incremental gain or net negative cost after complete qualified tests.
- `INCONCLUSIVE`: correctness, scheduling support, profiler confounds or missing repeats prevent a fair test; do not convert to structural NO-GO.

A method that fails Decode ITL/TPOT gates cannot be declared GO even if TTFT improves. Full dynamic policy development requires at least some causal evidence that **shared-weight depth mixing** is valuable beyond *plain D-local deflection*.

## 10. Artifacts, figures and reproducibility

1. Write this English spec to the intended repository path.
2. Create separate descriptive evidence folder `poc_flashvep/reports/ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/`; save immutable configs, command lines, source hashes, exactness files, profiler settings, output token ID manifests, NIXL/KV/HBM receipts and failure inventory.
3. Final report: `poc_flashvep/reports/ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching_final_summary.md`.
4. Figures: (1) **D decode-safe slack windows** and added kernel lengths; (2) **D-local placement, KV credit and routing**; (3) **same-depth vs cross-depth microbenchmark** with attention/packing costs; (4) **Native vs generic D-local vs loop-aware** latency/Decode SLO/goodput; (5) **cause-of-gain waterfall** measured vs hypothesized, or an explicit missing-evidence panel when gated out.
5. Include a novelty comparison table vs Load-Aware Prefill Deflection, FluidPD, CDB and ExactLoop; identify which parts are borrowed. Avoid unsupported paper-quality conclusions.
6. Commit and push in a new project branch only after verifying no changes to the existing working index/private runtime; record local and remote SHA. Honor currently approved GPU burn/restore rules. Missing steps and timeouts are explicit `NOT_RUN`/`BLOCKED` rows, never implicit success.

**Final report must directly answer:** (i) *Why* Native PD's ITL survived while static Token/Loop interfered; (ii) whether D had **actual safe slack**; (iii) whether generic D-local deflection already captures the headroom; (iv) the **incremental** compute and end-to-end effect of cross-depth tied-weight batching; (v) whether this remains a distinct loop-specific method under ExactLoop/CDB/deflection prior art.

## 11. Executor amendment frozen before candidate GPU tests (2026-11-06)

The campaign starts at **08:21:36 KST**. The final deadline is **14:21:36 KST**.
Stop GPU experiments by **13:36:36 KST**. Do not start long jobs after **13:16:36 KST**.
These limits include source review, implementation, analysis, publication, and burner restoration.
The deadline does not restart after an interruption.

The publication branch is `flashvep/d-local-cross-depth-batching-poc`, based on
`034b4ef9d2a8f0fe1a9d993370693d349cd03838`.
The existing checkout, staged index, and private runtime remain in place.
New process-local hooks and evidence use a separate namespace.

The existing RLT `MixedPolicy` can already batch recurrent prefill and decode rows
at different depths. This is a reused mechanism, not a new implementation claim.
The matched-shape control must keep the dense input values, row count, and weight
matrix fixed. Changing a depth label alone must not be credited as a dense-kernel gain.
An additional scheduling opportunity must be separated from an arithmetic speedup.

Initial measurements reuse the previous pinned ShareGPT/LongBench prompt pool.
They are exploratory comparisons, not new held-out confirmation.
Record repeated prompt IDs and keep prefix caching disabled.
Select a new confirmation trace only if the Phase 3 gate passes.

Use three D-local placement controls: zero placement, all-local correctness, and a
fixed request-ID subset for interference tests. Placement occurs before P execution.
All compared D-local arms use the same subset. A failed local admission remains
queued with its original ownership. It must not silently switch to a different arm.
Keep native P-to-D reservations and ACKs unchanged for non-local requests.

Distinguish strict ready-decode priority from statistical deadline-budget insertion.
Strict priority can starve local prefill when decode is continuously ready.
If an insertion policy delays ready decode, label it as a measured statistical
SLO policy. Do not call it strict priority or a hard real-time guarantee.

CUDA-event durations enclose device work and possible gaps. They are not kernel
union, SM activity, or memory-bandwidth measurements. Report those counters as
`NOT_MEASURED` unless a separately qualified profiler supplies them.
Preserve local-tail interference tests. Overall throughput agreement alone does
not qualify a trace. Do not build Phase 4 when the incremental Phase 3 gate fails.

## 12. Recorded control corrections and additional microbenchmark checks

The original objectives, frozen SLOs, novelty gate, and absolute deadlines remain unchanged.
The original spec also remains in the evidence directory with its source hash.

V5 supplies the primary generic serving comparison.
Native and V5 use the same full-context decode-geometry warm-up and eight-entry graph budget.
V4 disabled that extra warm-up in D-local arms.
Keep V4 as a tuning diagnostic, not the warm-matched primary comparison.
Warm-up requests remain outside the measured cohort.

The first closed-loop control uses C32.
Its long solo-prefill tail can dilute a readiness advantage.
Add C128 and C248 microbenchmark controls before rejecting larger-chunk scheduling headroom.
The C248 control retains the existing 256-row sensitive-projection patch.
It does not change model arithmetic or precision.
Within each added control, all methods share a 512-token budget and eight-entry graph budget.
Both background streams produce a token before the new local prompt enters.
C248 must pass hidden-row and written-query-KV parity before its full-prompt microbenchmark.
These controls are not online serving results.

Separate cross-phase submission counts from cross-depth submission counts.
An accepted prefill/decode batch can still have only one recurrent depth.
Do not infer depth mixing from a method label.

Add a stronger same-depth control when a full-depth alignment wait limits that baseline.
Permit one legal prefill step when its current depth is one behind ready decode.
Charge that step to the measured execution.
Report its decode interference with the unchanged SLO.
This control is not strict ready-decode priority.

An optional output-credit follow-up may limit prefill submissions between observed output steps.
Qualify that path with mixed Native/D-local N8 and N32 requests before N128.
This throttle is not a hard-deadline predictor.
It retains the same fixed placement and frozen decode screen.

First-entry-only CUDA markers may measure Service TTFT with less instrumentation.
Marker-free serving remains primary.
Reject comparative timing claims when matched local-tail interference exceeds 5%.
Do not replace an unqualified kernel timeline with CUDA-event enclosures.
