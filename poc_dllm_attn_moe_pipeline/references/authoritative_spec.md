# dLLM MoE-EP Intra-Refinement Attention–MoE Pipeline PoC
## Dependency-Safe Early-Ready Token Groups and Prefix-Partial Attention

**Version:** 1.0 — 2026-10-03  
**Status:** Research hypothesis and execution contract, not a validated optimization.  
**Repository:** `/home/esjung/MLLM-EP-github`  
**Required server spec path:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/dllm_intra_refinement_attn_moe_pipeline_spec.md`  
**Allowed physical GPUs:** `2,3,6,7` only.  
**Primary model:** `inclusionAI/LLaDA2.0-mini`.  
**Primary execution regime:** One active request (B1), native block-diffusion refinement, real physical EP4.  
**Suggested output directory:** `poc_dllm_attn_moe_pipeline/`  
**Suggested branch:** `flashvep/dllm-attn-moe-pipeline-poc`

---

## 1. Executive decision to be made

Determine whether a B1 block-diffusion MoE forward has **recoverable waiting time at the boundary from MoE layer l to attention layer l+1**.

A token group whose complete layer-l output is available may be able to begin its next-layer normalization, QKV projection, and attention over already-valid prefix KV while other groups are still completing layer l. The current block remains bidirectional: the group's final attention output must still incorporate every permitted current-block key and value from the correct refinement iteration.

The proposed optimization changes the **schedule**, not the model. It does not predict acceptance, move expert weights, skip routed experts, reuse stale live activations, or change the attention mask.

This PoC must answer four separate questions:

1. Do token groups actually become ready at meaningfully different times, or can a cheap explicit split create useful early completion?
2. Can the next-layer consumer legally use those groups before all other groups finish, including the real TP/SP/EP layout and synchronization requirements?
3. Is there sufficient independent next-layer work, and does overlap remain beneficial after fragmentation, synchronization, merge, and hardware-contention costs?
4. Is any incremental benefit left beyond fixed token splitting and existing generic overlap techniques?

A positive answer only to question 1 is not enough. A dependency-only oracle is not a speedup. An improved local bridge is not automatically a new dLLM systems contribution.

---

## 2. Starting evidence and exclusions

### 2.1 Pinned existing evidence

Reuse the following repository evidence rather than rebuilding the model stack:

- `poc_dllm_ep_comm/report.md`, source commit `dc539d3222193ab0902e0a342679edc3610f54bb`.
- `poc_dllm_ep_comm/patches/dinfer_existing_bridge.diff` at the same commit.
- Existing acceptance/router/ownership validation and separate clean/timing/trace execution lanes.
- Existing replica studies may supply metadata, but their predictions and weight-movement mechanisms are out of scope.

The historical report describes BF16 LLaDA2.0-mini, 20 decoder layers including 19 sparse MoE layers, hidden size 2048, 256 routed experts per sparse layer, top-k 8, and a shared expert. It records dense TP4 / source-SP4 / physical EP4 / DP1 / PP1 on H100 80 GB GPUs. Revalidate these facts against the loaded runtime; do not infer the parallel configuration from the GPU count.

Pinned model revision: `dad945cac317da394b390f82c7b40691d8a881ed`.
Pinned dInfer upstream revision in the evidence: `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f`, with the existing local bridge patch. The report lists SGLang 0.5.3.post1, DeepEP 1.2.1+73b6ea4, PyTorch 2.8.0+cu128, and Triton 3.4.0. Record actual versions and any differences before running.

The bridge partitions replicated dense rows across source ranks, executes DeepEP, waits for combine completion, computes/adds shared-expert output, and gathers source rows back for the dense TP path. These are real prerequisites to audit, not communication steps that an oracle may silently delete.

### 2.2 Do not reuse the old promotion logic

The earlier 5.773x live-only communication amplification was an accounting reference, not measured Epoch performance and not recoverable latency. The earlier roughly 18% D+C/request-wall share was a stage measurement, not a guaranteed speedup ceiling for this new cross-operator transformation.

Neither number is a gate for this PoC. EP payload bytes may remain nearly unchanged even if this pipeline is useful.

The existing report also identifies eager execution, disabled TorchDynamo, an untuned small-M expert configuration, and default communication resource settings as limitations. Ordinary fixes to these limitations belong in the baseline, not in the claimed method gain.

### 2.3 Closed directions remain closed

Do not introduce cold or persistent expert replication, future-routing prediction, threshold tuning, approximate expert skipping, stale live-token KV, speculative next-refinement execution, or a new acceptance policy. Do not revive the previous replica family under a new name.

---

## 3. Prior-art boundary — mandatory before implementation

The source list in Section 25 is part of the contract. Inspect original methods, not only titles or abstract-level summaries. Record model, attention mask, request/batch regime, operator boundary, scheduling granularity, and actual evaluation scope.

| Prior | Established idea relevant here | Boundary this PoC must test |
|---|---|---|
| Epoch [R1] | Block-scoped expert support and token liveness control; fresh sparse work is merged into a full logical shard at the MoE boundary. | Does earlier next-layer work help beyond that boundary? Our old live-only accounting is not an Epoch reproduction. |
| MPK [R2] | Fine-grained dependencies, event-driven cross-operator scheduling, and a task representation that includes MoE/EP. | Generic dependency scheduling is already prior art. Does exposing prefix/current-block attention structure add anything beyond a generic task graph? |
| TokenWeave [R3] | Token splitting, including B1 sequence splitting, wave-aware granularity, and communication/compute overlap. | B1 alone is not novel. Its causal prefix-first schedule cannot simply be assumed correct for a bidirectional current block. |
| SGLang TBO/SBO [R4] | Attention–communication overlap across microbatches and same-batch overlap such as shared-expert work. | Separate these existing mechanisms from next-layer partial attention within one request. |
| COMET / Weave [R5,R6] | Fine-grained overlap and resource scheduling within distributed MoE execution. | Intra-MoE improvements must not be credited as cross-layer improvements. |
| Blockwise Parallel Transformer [R7] | Blockwise attention/FFN organization and fusion. | Block decomposition itself is not novel. |
| FlashAttention [R8] | Tiled, mathematically exact attention using stable online softmax accumulation. | Combining partial attention states is an established primitive, not the contribution. |

**Bounded literature assessment:** these sources establish substantial overlap in the broad motivation. This spec does not claim no identical paper exists. The narrow combination not established by this review is: measured EP-induced group readiness in B1 block diffusion, actual TP/SP consumer visibility, and early next-layer prefix-partial attention while retaining fresh bidirectional current-block dependencies.

Write `prior_overlap.md` with verified sections and unresolved comparisons. Label any reimplementation a functional adaptation, not a faithful reproduction, unless its full contract matches. If a verified source already implements the same mechanism under the same constraints and no new measurable bottleneck remains, return `NO_GO_DUPLICATE_MECHANISM` rather than merely renaming it.

---

## 4. Hypothesis: readiness, not merely stragglers

Use three distinctions throughout the experiment:

- **Rank straggler:** a GPU or a cumulative rank-stage finishes late.
- **Token dependency completion:** all selected contributions needed for a token's layer output have completed.
- **Consumer-visible readiness:** the complete token output is safely readable in the layout required by the next operator, with synchronization and memory visibility satisfied.

Different expert runtimes do not imply different token readiness. With top-k 8 on EP4, many tokens may depend on the same slow rank. A grouped expert kernel and a single combine event may also release all outputs together.

Define three hypotheses:

- **H1, natural readiness:** useful group-level completion skew exists before any artificial partitioning.
- **H2, legal overlap:** ready groups have usable next-layer work before other groups are ready.
- **H3, net gain:** exposing and consuming readiness costs less than the critical-path time removed.

It is also possible to create useful overlap through static splitting even if H1 is weak. Report that as `GENERIC_STATIC_PIPELINE`, not as evidence of native token-straggler exploitation.

---

## 5. Exact dependency contract

Use request r, diffusion block b, refinement t, transformer layer l, token position p, and group g as separate indices. Expert `(l,e)` is not the same tensor as expert `(l+1,e)`. No schedule may cross refinement t to t+1 until the native decoder has produced the next refinement input.

### 5.1 What is a complete layer output?

A token's final layer-l hidden state must include the native routed-expert sum, shared-expert contribution, residual path, and every required reduction/scaling/epilogue. Preserve the model's actual pre-/post-normalization order rather than substituting a generic Transformer formula.

Do not declare a token ready when just one selected expert or one remote rank finishes. Readiness is a join over all true dependencies.

### 5.2 Which next-layer work is token-local?

Given a complete layer-l hidden row in the required layout, next-layer normalization, QKV projection, and position-dependent transforms such as RoPE can be token-local. With TP, however, every necessary head shard must receive that row. An output available only at its source-SP rank does not automatically supply the next layer's other TP head shards.

Audit source-row all-gather, residual layout, QKV weight layout, output-projection reduction, and cache placement explicitly. A chunked replacement must charge communication and preserve collective ordering.

### 5.3 Prefix versus current block

For an eligible native block-diffusion invocation, separate the keys/values visible to a current-block query into:

- `P`: previous prompt/completed-block KV that the baseline declares valid and immutable for this invocation;
- `C`: all visible positions in the current bidirectional block, whose required K/V are produced from current-refinement layer states.

The proposed order is:

```text
Layer l: remaining groups still executing MoE/combine
                   ||
Ready group g: layout completion -> norm -> next-layer QKV
                   -> partial attention over valid P
                   -> wait for the required current-block C state
                   -> partial attention over C -> exact softmax merge
                   -> native output projection/reduction/residual
```

The first physical prototype should wait for all required current-block K/V before evaluating C. Do not add a second optimization that streams arbitrary C fragments until the simpler path is validated.

No prefix is eligible merely because its tokens are accepted. Accepted positions inside C may still need current-refinement hidden-state updates. Prefix validity must be proved from the actual cache version and attention mask. Cold prefill or refresh invocations with mutable prefix state use the serial fallback unless separate dependencies are demonstrated.

### 5.4 Partial attention must use one global normalization

For each query/head, let a key partition j produce maximum score `m_j`, denominator `z_j = sum exp(s-m_j)`, and unnormalized weighted value sum `u_j = sum exp(s-m_j)*v`.

For partitions P and C:

```text
m = max(m_P, m_C)
z = exp(m_P-m)*z_P + exp(m_C-m)*z_C
u = exp(m_P-m)*u_P + exp(m_C-m)*u_C
output = u/z
```

An implementation returning normalized outputs plus log-sum-exp may use the equivalent stable merge. Simply adding two separately normalized attention outputs is incorrect.

Handle empty P, masked keys, no-valid-key states, GQA/MQA head mapping, RoPE positions, scale factors, padding, and cache offsets. Follow native no-valid-key semantics; do not generate NaNs for the both-empty test case.

The transformation is mathematically equivalent in real arithmetic. Different reduction order can change floating-point results, so bitwise parity is not promised and numerical/decoding checks are mandatory.

---

## 6. Hardware and execution safety

Use exactly:

```bash
CUDA_VISIBLE_DEVICES=2,3,6,7
```

Logical ranks 0–3 should map to these physical GPUs; verify UUIDs each run. Do not infer permissions from prior availability. Audit competing processes and stop if the assigned devices are not available. Never kill another user's workload or use other GPUs.

If an existing, explicitly authorized utilization/burner process must be paused, record its owner, arguments, GPU mask, and original state; restore only that state in `finally`. Do not create a new background service or restore an unmasked default command.

Pin the working environment. Use an isolated branch/worktree when necessary. No global package upgrade, destructive `git reset/clean`, or overwrite of unrelated dirty/staged files. No whole-model or hidden-state tensor dump should be committed.

---

## 7. Experiment stages and effort bounds

This is a staged feasibility PoC, not authorization to write a new serving engine.

- **Stage 0:** prior and dependency audit; baseline validation.
- **Stage A:** readiness characterization and whole-request dependency headroom.
- **Stage B:** one two-layer physical seam, correct partial attention, measured split/overlap costs.
- **Stage C:** small full-model integration only after A/B pass.

A bounded two-layer or one communication seam is allowed. A full persistent-kernel compiler, replacement attention stack, new global scheduler, or framework port is not. If that is required before any test, return `CONDITIONAL_INTERFACE_BLOCKED` with the exact missing primitive.

Do not issue a research NO-GO merely because a helper library is missing. Conversely, do not spend unbounded implementation effort rescuing a small theoretical opportunity.

---

## 8. Stage 0: source audit and credible baseline

### 8.1 Required source map

Write `dependency_audit.md` containing a table for every operator from layer-l MoE input to the end of layer-(l+1) attention:

```text
operator | input layout | output layout | ready unit | stream/event
         | collective group/order | mutable state | eligible overlap
```

In particular inspect the existing bridge's:

```text
source-SP partition
-> router/top-k
-> DeepEP dispatch
-> local expert execution and reduction
-> DeepEP combine + completion wait
-> shared expert + add
-> source-row all-gather
-> layer epilogue
-> next-layer norm/QKV/attention
```

The shared-expert work may have no dependency on routed-expert completion. A safe shared-expert overlap is an existing generic optimization and must be measured separately if applicable. Do not count its gain as the new prefix pipeline.

### 8.2 Baseline ladder

Keep these lanes distinct:

- **B0, pinned native:** exact existing validated path, for continuity.
- **B1, strongest practical native:** B0 plus low-risk compatible tuning, reusable graph/capture paths, and safe existing overlap where supported. All changes documented and quality checked.
- **B2, serial decomposed:** same grouped/split attention operations as the candidate, but without overlap. This isolates split and merge overhead.
- **B3, fixed-group generic pipeline:** best calibration-chosen static two-/few-group schedule, correct for bidirectional C; QKV-only and prefix-enabled variants clearly separated.
- **B4, readiness-aware prefix pipeline:** candidate schedule using only currently available dependencies/readiness.

Do not deliberately disable a supported baseline optimization only for B1. Do not assume TBO or an arbitrary current SBO flag works in the pinned runtime. Inspect compatibility; an unavailable feature is `NOT_RUN`, not a zero-gain baseline.

MPK/TokenWeave paper comparisons are mandatory; complete ports are not required for the initial PoC. If they are not executed, final claims remain feasibility claims, not superiority over those systems.

### 8.3 Two frozen smoke probes

Verify physical EP4, expert ownership, actual inter-GPU operations, tokenizer/input identity, outputs, NFE, acceptance state, and correct prefix cache versions. One probe should have a nonempty valid prefix; one should test empty/short prefix or a cold boundary.

---

## 9. Workload and sample contract

Primary regime is one active request, not B4/B8 serving disguised as B1. Use block32 by default, native threshold 0.9, temperature 0, native cache/acceptance behavior. Block64 is a separately labeled robustness lane only when supported.

Reuse the earlier 32-problem cohort for continuity and code checks, but mark it **previously observed**, not fresh statistical holdout. Its prior 64-token-budget outputs were bounded prefixes, not complete task-quality evaluations.

For the new hypothesis, freeze a new 8-request calibration / 24-request held-out split, balanced between coding and reasoning if feasible. Use metadata-only selection and deterministic IDs; do not select by observed timing, routing skew, or candidate gain. A 4-calibration / 8-held-out subset is sufficient for an early kill test; expand only after the seam has positive evidence.

Include natural prompt-length variation. If natural cases provide no long valid prefix, use a separately labeled **shape-controlled microbenchmark**, not fabricated task data, at prefix lengths `{0,128,512,2048,8192}` subject to actual memory/context support. Random or repeated prefix tensors are not accuracy benchmarks. Native contexts cannot be padded with irrelevant text and then presented as real workload gains.

Start with short bounded generation for runtime characterization. Stage C quality evaluation needs native, sufficiently long completion limits and explicit truncation/EOS accounting. Record cold, hot, cache-write, and refresh-like invocations rather than treating them as interchangeable.

---

## 10. Stage A1: measure three readiness levels

For each sampled invocation and layer boundary, distinguish:

1. **Latent producer completion:** when the arithmetic contributing to a row or group has completed, if observable.
2. **Safe local publication:** when complete output bytes and an acquire/release or event guarantee are available to another consumer on that GPU.
3. **Next-layer TP visibility:** when all required head-shard consumers have the correct complete rows, after real layout communication.

For group g define `r_math(g)`, `r_publish(g)`, and `r_TP(g)` where measurable. Use `r_TP(g)` in an executable candidate model. Unknown fields must remain unknown, not replaced by an optimistic estimate.

Record `r_all = max_g r_TP(g)` on a properly aligned timeline. The readiness spread `r_all - percentile(r_TP,50)` is diagnostic; it is not a latency gain. Also record how many groups are ready early enough to run a calibrated nontrivial consumer task.

Do not compare raw device clock values across GPUs without alignment/error bounds. CUDA event time from unrelated rank streams is not automatically globally comparable. Prefer a profiler timeline with validated synchronization, or per-rank elapsed times tied to a common event protocol and report uncertainty. The uncertainty must be smaller than the window being claimed.

### 10.1 What old traces cannot establish

Per-rank routed counts, full grouped-GEMM durations, and one combine completion event do not establish per-token readiness. Do not divide a full-kernel time by token count or invent an expert completion order.

If the backend publishes only a whole buffer, the observed published readiness is whole-buffer readiness. Any finer times obtained from a modified kernel must be marked as a different diagnostic implementation.

### 10.2 Natural versus engineered early completion

Keep separate lanes:

- native readiness, no scheduling change;
- explicit deterministic group execution with measured overhead;
- analytical latent-readiness oracle, not a working interface.

One minimal diagnostic publication hook is allowed only if it preserves ownership, reductions, event ordering, and outputs. A timer inserted into a kernel is not a safe publication primitive by itself. Never read an in-flight buffer merely because a timestamp suggests the producer has finished writing part of it.

---

## 11. Grouping and collective ordering

Default candidate granularity is token **groups**, not one kernel or message per token.

Start with at most two groups; sweep a small calibration-only set such as 2 and 4 groups. At block32/source-SP4, the local rows can be very few. Include full-block/no-split fallback. More groups are not automatically better.

Preserve logical positions and source ownership. Define how each group is represented across all four ranks, including empty local contributions and padding. All ranks must issue matching collectives in a consistent globally agreed order. Independent rank-local ready-first reordering may deadlock and is prohibited.

A dynamic plan may use current router metadata and completed events, but not future layer timings, future refinement routing, future acceptance positions, or held-out optimal group sizes. Charge any cross-rank scheduling/control coordination.

If the only legal schedule is a fixed order because of the interface, record that limitation; do not label it dynamic token-readiness scheduling.

---

## 12. Stage A2: an honest dependency/headroom model

Construct a DAG over one complete refinement and, where useful, the full generated request. Include:

- all routed-expert dependencies and final combination;
- shared expert and residual/scaling;
- source-row gathering and TP reductions;
- next-layer QKV/RoPE;
- prefix partial attention and current-block K/V readiness;
- current-block attention, softmax merge, and output projection;
- later layers, native accept/controller time, cold/cache transitions.

Report three different quantities:

- **Dependency-only envelope:** a critical-path lower bound on completion time with deliberately optimistic resources; hence an upper envelope on potential improvement for the specified graph and cost assumptions.
- **Measured-cost schedule estimate:** a legal schedule using measured task sizes, split/merge/control costs, and measured or conservatively bounded overlap slowdowns.
- **Measured seam/full-runtime performance:** actual concurrent execution only.

Never present an envelope as a deployable oracle. Unknown costs must not be set to zero in the realistic estimate.

Calibrate the baseline DAG against the measured unsplit timeline. If reconstructed totals differ by more than 10%, fix the model or mark it unresolved.

Do not add independent per-layer gains, overlap the same saved interval twice, or sum separately maximized rank components. A global or dependency-aware per-rank timeline is required. The first/last layer and cold invocation need their own handling.

A useful sanity ceiling for one boundary is that overlap cannot save more than either the legally available waiting interval or the eligible consumer work. This is only a local ceiling; the whole-request estimate must still be computed with dependencies and resources.

---

## 13. Stage B: minimum physical two-layer seam

Proceed only if Stage A indicates material reachable headroom or one cheap seam can resolve a bounded uncertainty.

Choose early/middle/late MoE-to-attention boundaries using calibration data only. Do not claim a best-boundary speedup is the whole-model result. Sample routing/cache states from native executions, including later refinement states.

Use actual four-GPU expert execution, combine, and required layout transfers. A standalone attention kernel on already-copied tensors is a component benchmark, not physical EP overlap.

### 13.1 Four required paired runs

For identical inputs, weights, route metadata, cache snapshots, and hardware state:

1. `UNSPLIT_SERIAL`: native boundary.
2. `SPLIT_SERIAL`: candidate grouping plus split attention and merge, serialized.
3. `FIXED_OVERLAP`: best preselected fixed-group correct pipeline.
4. `READY_PREFIX_OVERLAP`: legally early group publication with prefix-partial attention.

Also measure `QKV_ONLY_OVERLAP` and `PREFIX_DISABLED` ablations if cheap. When there is no dynamically selectable ordering, B4 may collapse to B3; that is an informative negative result for readiness adaptation.

### 13.2 Charge everything

The seam interval begins before producer work and ends after the same next-layer attention output is complete. Include dispatch, grouped-GEMM fragmentation, combine, shared work, TP/SP changes, signaling, staging, QKV, partial attention, softmax merge, and all final waits. No precompleted candidate input may appear for free.

Use the same buffer lifetime and comparable graph/capture mode. Reused immutable prefix KV is allowed only because the native baseline has that same valid cache. Record additional memory allocations and peak HBM.

### 13.3 Contention measurements

Measure producer and consumer alone, serialized together, and concurrently. Log their duration changes and pair makespan. Overlap can hurt because both use SMs, registers, Tensor Cores, HBM, or NVLink. Do not assume separate CUDA streams or a faster consumer implies useful overlap.

No memory compression, lower precision, or disabled collective may be added to manufacture headroom.

---

## 14. Strong negative controls and attribution

At minimum include:

- **Fixed-group baseline:** rules out ordinary token splitting as the whole explanation.
- **QKV-only versus prefix-partial:** isolates the claimed new usable work.
- **Short/empty-prefix seam:** tests dependence on immutable prefix work.
- **Same split, no overlap:** exposes the true decomposition tax.
- **Best practical shared-expert/intra-MoE overlap:** separates an already-known optimization.

Optional shape-controlled dense-FFN or single-query counterparts can help explain the mechanism, but are synthetic controls rather than equal-quality model comparisons. An AR prefill with multiple queries may also benefit; do not claim the mechanism is impossible in AR merely because B1 AR decode has only one new query.

A strong result should explain which of these creates benefit: avoiding the full-shard barrier, token-readiness skew, prefix work availability, or generic asynchronous execution. If the gains come entirely from the generic part, return `GENERIC_OVERLAP_ONLY` for the proposed dLLM-specific motivation.

---

## 15. Epoch comparison protocol

Do not label the old `B_live_only` trace deletion as Epoch or use it to synthesize an Epoch timing row.

The initial baseline is the validated native dLLM runtime. If an existing faithful Epoch implementation is available at bounded cost, run both Epoch and Epoch-plus-pipeline with identical support/cache/refresh policies and workload. Compare the candidate to the corresponding Epoch reference, not to the unmodified model under a different approximation policy.

If Epoch is unavailable:

- record `EPOCH_RUNTIME_NOT_RUN`;
- compare the documented operator boundary in [R1] with the proposed boundary;
- optionally use a clearly labeled sparse-worklist seam to test scheduling sensitivity, without claiming Epoch output quality or E2E performance;
- defer any claim of measured gain beyond Epoch.

Sparsity can shorten the MoE tail and remove our overlap opportunity. Conversely it can increase irregularity. Neither effect is assumed. Algebraic compatibility does not prove runtime additivity.

---

## 16. Correctness and accuracy

### 16.1 Offline characterization

An unchanged trace run is not evidence that a future pipeline preserves accuracy. Record `CANDIDATE_NOT_IMPLEMENTED` until a candidate executes.

### 16.2 Seam-level correctness

Compare to native unsplit execution and an independent high-precision attention reference on small shapes. Check:

- partition union/disjointness and exact attention mask;
- prefix/cache version, layer ID, request ID, block ID, refinement ID;
- final MoE rows, normalization, Q/K/V, merged attention, and next-layer output;
- shared-expert and routed scaling, no missing/duplicated expert contribution;
- FP32 softmax-state merge and native dtype output;
- numerical max-absolute, relative-L2, and logit errors;
- completion/publication happens-before relationships;
- empty groups, all-to-one rank routing, mixed ready groups, padding, cold/refresh fallback.

Freeze dtype-appropriate tolerances using calibration numerical references, not after seeing failing held-out examples. Never relax tolerances to rescue the result. A future/current-block K/V substitution must fail a test.

### 16.3 Closed-loop full generation

Stage C must rerun actual generation, not merely replay a prerecorded acceptance schedule. For the deterministic frozen cohort compare:

```text
input IDs
final generated IDs
acceptance positions per iteration
NFE
router top-k IDs and weights within tolerance
EOS and truncation decisions
```

Demand identical discrete outputs, acceptance schedule, and NFE for promotion of the schedule-only PoC, unless baseline self-repeats demonstrate nondeterminism that is independently resolved first. This is a validation gate, not a claim that all mathematically equivalent floating-point executions are bitwise identical.

If a near-threshold numerical difference changes routing or acceptance, mark `CORRECTNESS_UNRESOLVED` and do not present an accuracy/speed tradeoff as success.

### 16.4 Task accuracy

If Stage C is reached, run a fixed complete-generation GSM8K exact-match and HumanEval unit-test subset for baseline and candidate at identical decoding limits. Report counts, exact outputs, truncations, and paired differences. Prior bounded 64-token continuations cannot establish task accuracy.

Execute generated code only in an isolated resource-limited sandbox with no network or credentials, never directly on the shared host. If a safe evaluator is unavailable, report `TASK_SCORE_NOT_RUN`; retain token-level parity evidence. Do not fabricate a zero accuracy delta or infer population-level equivalence from a small unchanged sample.

---

## 17. Timing, statistics, and reproducibility

Maintain separate clean-performance, event-only, and heavy-readiness trace lanes. Preallocate diagnostic buffers where possible. No `.item()`, CPU log I/O, device-wide sync, or per-token host callback inside the measured candidate critical path.

Use at least three warmups and five counterbalanced paired repetitions for each selected seam; use at least three paired whole-request repetitions if Stage C is reached. Record all runs, not only the fastest. Compile/capture warmup is excluded equally, and any runtime grouping decision or dynamic graph work is included.

For a projected 5–10% gain, an 8% observer overhead is not automatically acceptable. Target <=3% paired mean observer overhead for representative timing; if larger or differential overhead is uncertain, use clean runs for headline E2E and mark detailed attribution unresolved.

Request/problem is the independent statistical unit. Aggregate repeated measurements within request, then bootstrap request clusters with 2000 resamples. Do not treat layers, iterations, groups, or reused block32/block64 cases as independent observations. For controlled shape microbenchmarks, use repetition CIs labeled as microbenchmark CIs, not workload-generalization CIs.

Use latency reduction `1 - T_candidate/T_reference` and speedup `T_reference/T_candidate` as distinct fields. State the reference for every number. A modeled projection and measured full-request improvement must never share an unlabeled column.

---

## 18. Promotion and kill gates

Thresholds below are frozen research-investment heuristics, not laws or publication criteria. Small differences around a cutoff call for one bounded paired check, not repeated selection of favorable data.

### G0 — Validity

Real EP4, correct mask/cache state, safe publication semantics, trustworthy time alignment, and baseline parity are required. Missing instrumentation is `CONDITIONAL_MEASUREMENT`, not proof of no physical headroom.

### G1 — Dependency headroom

If even an optimistic, correctly dependency-constrained whole-request envelope is below 5% latency reduction on the predeclared target regime, stop as `NO_USEFUL_HEADROOM`.

An envelope >=10% supports a bounded seam test, not full implementation. A 5–10% envelope is borderline. If only impossible early visibility or a forbidden next-refinement dependency creates the opportunity, return `NO_LEGAL_EARLY_CONSUMER`.

### G2 — Reachable seam profitability

Require a positive physical seam makespan gain over `UNSPLIT_SERIAL` after all costs, with a target of >=10% on covered boundaries and a calibrated request-level projection of >=5%. The best fixed correct grouping is a required comparison.

If prefix-enabled readiness-aware overlap gives no additional gain over a correct fixed/QKV-only or generic pipeline, classify the narrower idea as `GENERIC_OVERLAP_ONLY` rather than promoting a new method story.

### G3 — Full runtime promotion

Stage C is justified only if G0–G2 pass and a bounded integration exists. Desired full-request outcome: >=7% mean clean latency reduction versus B1, CI above zero, plus a material improvement over the best executed generic-overlap baseline (target >=3% relative latency reduction). Report both thresholds without treating a single favorable case as sufficient.

If full integration is not run, verdict remains `CONDITIONAL_SEAM_ONLY` even after an excellent seam result.

### G4 — Correctness and attribution

Discrete generation/acceptance/NFE parity is required. The quality report must state exactly what was evaluated. If an existing technique explains all the gain, report a useful generic optimization without calling it a new dLLM-MoE-EP result.

### Valid terminal outcomes

- `GO_MEASURED_RUNTIME`: all required validity, runtime, comparison, and correctness evidence present.
- `CONDITIONAL_SEAM_ONLY` / `CONDITIONAL_INTERFACE_BLOCKED` / `CONDITIONAL_MEASUREMENT`.
- `NO_USEFUL_HEADROOM` / `NO_LEGAL_EARLY_CONSUMER` / `NO_NET_GAIN_AFTER_OVERHEAD`.
- `GENERIC_OVERLAP_ONLY` / `NO_GO_DUPLICATE_MECHANISM`.
- `CORRECTNESS_UNRESOLVED` / `RUNTIME_BLOCKED`.

Do not claim all Attention–MoE scheduling is impossible when only this narrow B1 path fails.

---

## 19. Required schemas

`readiness.csv`:

```text
request_id,problem_id,block_id,refinement_id,layer_id,source_rank,
group_id,group_size,phase,prefix_tokens,current_block_tokens,
completion_scope,natural_or_engineered,
r_math_us,r_publish_us,r_tp_visible_us,r_all_us,
clock_domain,clock_error_us,release_primitive,trace_mode
```

`dependency_nodes.csv`:

```text
node_id,request_id,block_id,refinement_id,layer_id,rank,group_id,
operator,input_version,output_version,predecessors,
measured_duration_us,resource_class,visibility_requirement,cost_source
```

`seam_metrics.csv`:

```text
request_id,boundary,variant,group_plan,prefix_tokens,repetition,order,
producer_us,layout_us,qkv_us,prefix_attention_us,current_attention_us,
merge_us,shared_us,final_collective_us,control_us,seam_makespan_us,
extra_hbm_bytes,correctness_status,clock_quality
```

`request_metrics.csv`:

```text
request_id,variant,prompt_tokens,returned_tokens,truncated,nfe,
e2e_clean_ms,denoise_span_ms,ep_dispatch_ms,ep_combine_ms,
tp_layout_ms,attention_ms,observer_overhead_pct,
token_parity,acceptance_parity,task_score,reference_variant
```

Unknown values are null with a reason, not zero. Separate logical traffic from measured wire counters.

---

## 20. Required figures and tables

Use distinct plots and consistent units. At minimum:

1. A measured timeline showing producer completion, safe publication, TP visibility, and next-layer work for a representative calibration case; no fabricated per-token times.
2. Group-ready fraction over time, including native and engineered lanes separately.
3. Dependency envelope versus measured-cost estimate versus actual seam/full gain.
4. Unsplit, split-serial, fixed-overlap, QKV-only, and prefix-enabled seam times.
5. Prefix length versus net gain, with natural workloads separated from synthetic shapes.
6. Per-layer/phase coverage and excluded-boundary reasons.
7. Closed-loop correctness and task-quality counts if Stage C runs.

Keep measured runtime and analytical/reference rows in separate tables. An Epoch row is permitted only if actually run. Leave unsupported or unmeasured entries as `NOT_RUN`.

---

## 21. Expected output tree

```text
poc_dllm_attn_moe_pipeline/
  README.md
  report.md
  prior_overlap.md
  dependency_audit.md
  references/authoritative_spec.md
  scripts/
    inspect_runtime.py
    collect_readiness.py
    build_dependency_dag.py
    estimate_schedule_headroom.py
    validate_partial_attention.py
    benchmark_two_layer_seam.py
    run_full_model.py              # only if promoted
    evaluate_quality.py            # only when valid/safe
    reproduce.sh
  tests/
    test_dependency_contract.py
    test_attention_state_merge.py
    test_publication_and_order.py
    test_no_future_state.py
    test_accounting_and_statistics.py
  results/<new_run_id>/
    environment.json
    design_freeze.json
    baseline_manifest.json
    readiness.csv
    dependency_nodes.csv
    headroom.json
    seam_metrics.csv
    request_metrics.csv
    correctness.json
    accuracy.json
    gate_audit.json
    summary.json
  figures/<new_run_id>/
```

Raw diagnostic captures may stay local with an integrity manifest and compact exported summaries. Do not upload model weights, secrets, hidden states, or full proprietary prompts. Preserve failed runs append-only.

---

## 22. Required final report questions

1. Is the signal native token/group readiness skew, engineered static splitting, or only rank imbalance?
2. At which exact operator does consumer-visible early readiness disappear?
3. Do all top-k dependencies and TP/SP layout transitions permit earlier QKV?
4. Which prefix KV are already valid, and how much legally movable work exists?
5. What fraction of boundaries/invocations are eligible?
6. Does bidirectional attention remain exactly represented by the split/merge?
7. What does the optimistic dependency envelope say, and what costs does it omit?
8. What is the actual split/merge/communication/signaling tax?
9. Does simultaneous execution beat serialized execution under actual resource contention?
10. Does the candidate beat a correct fixed-group pipeline, not only an eager baseline?
11. Is any benefit explained by existing shared-expert overlap or ordinary graph tuning?
12. Was actual Epoch run? If not, what remains unproven about additive benefit?
13. Are token IDs, acceptance, NFE, routing, and cache validity preserved?
14. Were complete task outputs scored, or were only bounded prefixes checked?
15. What was measured on four GPUs versus simulated or component-benchmarked?
16. What is the scoped verdict and the single smallest unresolved test?

---

## 23. Commit and handoff

The Codex execution task should preserve the original worktree/index, commit only PoC-owned files, and push a separate branch without force-pushing. Do not stage unrelated modifications. Include immutable spec/input/source hashes and reproducible commands.

Final handoff fields:

```text
VERDICT
NATURAL_READY_WINDOW
NEXT_LAYER_VISIBLE_WINDOW
LEGAL_PREFIX_WORK
DEPENDENCY_ONLY_E2E_ENVELOPE
MEASURED_SEAM_GAIN_VS_UNSPLIT
MEASURED_SEAM_GAIN_VS_FIXED
MEASURED_FULL_E2E_GAIN_OR_NOT_RUN
CORRECTNESS / TASK_QUALITY_SCOPE
EPOCH_BASELINE_STATUS
PRIOR_OVERLAP_STATUS
GPU_MASK
BRANCH / COMMIT / REPORT_PATH
```

Do not fabricate a future result or report work as completed if only scripts were written.

---

## 24. The narrow research criterion

The desired evidence chain is:

```text
B1 block-diffusion with physical EP
  -> nontrivial group-ready or safely created group-completion windows
  -> real next-layer consumers can observe the required state
  -> valid-prefix attention supplies independent work
  -> bidirectional current-block dependencies remain intact
  -> split/merge/layout costs and resource contention do not erase the gain
  -> performance exceeds a credible generic fixed-overlap baseline
  -> closed-loop output quality remains unchanged
```

If it holds, a possible contribution is a dependency-aware execution transformation for this regime. If not, a token-level straggler plot alone must not be promoted into a systems method.

---

## 25. Primary references and evidence links

Accessed 2026-10-03. These are source pointers, not claims of exhaustive coverage. Verify revisions and compatibility in the actual execution environment.

- **[R1] Epoch**, arXiv:2609.09748v1, Sections 4.3–4.4 and Algorithm 2. https://arxiv.org/html/2609.09748v1
- **[R2] MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs**, arXiv:2512.22219v2, Sections 1, 3–5, 6.4–6.5. https://arxiv.org/html/2512.22219v2 — venue page: https://www.usenix.org/conference/osdi26/presentation/cheng
- **[R3] TokenWeave**, arXiv:2505.11329v3, Sections 2.2 and 3.1–3.2. https://arxiv.org/html/2505.11329v3
- **[R4] SGLang expert-parallelism documentation**, computation/communication overlap. https://docs.sglang.io/docs/advanced_features/expert_parallelism
- **[R5] COMET**, arXiv:2502.19811. https://arxiv.org/abs/2502.19811
- **[R6] Weave**, arXiv:2609.21483v1. https://arxiv.org/html/2609.21483v1
- **[R7] Blockwise Parallel Transformer**, arXiv:2305.19370. https://arxiv.org/abs/2305.19370
- **[R8] FlashAttention**, arXiv:2205.14135. https://arxiv.org/abs/2205.14135
- **[E1] Existing physical-EP4 report.** https://github.com/jyssys/MLLM-EP/blob/dc539d3222193ab0902e0a342679edc3610f54bb/poc_dllm_ep_comm/report.md
- **[E2] Existing bridge patch.** https://github.com/jyssys/MLLM-EP/blob/dc539d3222193ab0902e0a342679edc3610f54bb/poc_dllm_ep_comm/patches/dinfer_existing_bridge.diff

