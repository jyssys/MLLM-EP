# PoC Specification v2 — Recurrent Depth × Tensor-Parallel Scaling: Four-Way Looped/AR Comparison

**Status:** Authoritative revised experiment specification; supersedes `looped_recurrence_tp_scaling_spec.md` for this campaign.  
**Date:** 2026-10-15  
**Server destination:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/looped_recurrence_tp_scaling_spec_v2_four_models.md`  
**Permitted physical GPUs:** **4, 5, 6, 7 only**. Never schedule GPUs 0–3.  
**Primary question:** *Does recurrent execution introduce a multi-GPU critical-path penalty or opportunity that standard tensor parallelism cannot amortize, beyond the cost of executing an equivalent number of ordinary Transformer layers?*

## 0. Scope and non-negotiable outcome

Implement and measure the following **four mandatory model/control families**:

1. **Native Looped:** pretrained `ByteDance/Ouro-2.6B-Thinking`, recurrent passes **R = 1, 2, 4**.
2. **Tied-Unrolled:** the **same pretrained Ouro** mathematically expanded into an explicitly unrolled logical sequence of 48 × R layer invocations, **still referencing the same parameter storage** and preserving per-effective-layer KV cache semantics; R = 1, 2, 4 where feasible. This is a controlled execution comparator, **not** a separately trained AR checkpoint.
3. **Real AR #1:** pretrained `Qwen/Qwen3-1.7B`, ordinary non-recurrent decoder-only AR.
4. **Real AR #2:** pretrained `Qwen/Qwen2.5-3B`, ordinary non-recurrent decoder-only AR.

**All four are in scope.** Do not silently drop either Qwen baseline. If tied-unrolled is infeasible without a costly engine rewrite, perform the bounded parity/scheduling diagnostic and explicitly mark `TIED_UNROLL_RUNTIME_BLOCKED`; do not fabricate a fourth successful physical baseline. In that case the causal comparison is incomplete and a loop-specific GO cannot be claimed.

**Optional, only after all four are addressed:** a heavier genuine AR model such as `Qwen/Qwen3-8B` to study compute scale (36 layers, hidden 4096, Q/KV heads 32/8, intermediate 12288). Treat it as **contextual scalability**, not a fifth mandatory arm and not a recurrence-matched control. Do not consume the campaign budget before the mandatory controls. Verify current revision, license, actual weight size and memory first.

The end-product is a **reproducible causal characterization** and **GO/NO-GO decision**, not a guaranteed new method. A measured native TP4 speedup of approximately 2.18× over TP1 on Ouro R4 **does not prove a removable gap to ideal 4×**. Count actual non-overlapped work, not just kernel service totals.

### Primary research hypotheses

- **H1 (depth-dependent TP scaling):** TP speedup or parallel efficiency changes as Ouro recurrent depth rises R=1→2→4.
- **H2 (recurrence-specific penalty):** after matching effective depth, mathematical operations, KV layout, model weights, tensor shapes, backend and GPU layout, native recurrence incurs additional TP-critical scheduling/synchronization/memory cost relative to tied-unrolled execution.
- **H3 (distinct from ordinary AR):** the residual overhead or exploitable recurrent opportunity is not explained simply by greater logical depth, different GQA/MHA head geometry, ordinary TP collective counts, or larger/smaller model compute.
- **H4 (physical importance):** any identified recurrence-specific mechanism has *measurable non-overlapped B1 E2E/TPOT headroom* sufficient to justify an eventual new distributed framework.

All hypotheses can fail. Failure is a valid scientific output; avoid moving goalposts.

---

## 1. Previous validated evidence and environment pins

From the committed previous physical PoC:

- Report: `poc_looped_multigpu_kv_placement/report.md`.
- Commit: `f1002a2f4baa6d1524e7e9b9bc8065bdc5c89a69`.
- Native Ouro pinned checkpoint: `ByteDance/Ouro-2.6B-Thinking@f1edd81e7ac41355db670500ceaf204e0f73af68`.
- Existing tested engine: vLLM `0.26.0+cu129` / PyTorch `2.11.0+cu129` / Transformers `5.16.1`, with FlashAttention 3 and CUDA Graphs. **Verify actual installed revisions rather than assuming these survive environment changes.**
- Ouro in loaded engine: **48 actually loaded physical attention layers × 4 passes = 192 logical attention executions**; 16 Q heads, 16 KV heads, head dimension 128; actual loop-specific KV index `loop * 48 + physical_layer`. Do not substitute conflicting descriptive numbers from a model card for runtime-validated geometry.
- Previous ~32K (32704 input + 64 forced output) B1: TP1 prefill 2638.904 ms, TPOT 34.631 ms/token, E2E 4830.185 ms; TP2 1568.244 / 24.531 / 3120.965; TP4 976.125 / 19.485 / 2213.117. Historical numbers only; **do not mix absolute times with a new campaign without a matched same-session reference**.
- Previous forward collected 384 logical layer reductions plus one embedding reduction at R4, using phase-specific NCCL/custom reductions. **Counts ≠ measured removable critical-path milliseconds.**
- Old EP and dLLM PoCs are not part of this experiment. No EP All-to-All in these dense Ouro/Qwen TP baselines.

### Actual environment audit

Before GPU work, record `nvidia-smi`, `nvidia-smi topo -m`, UUIDs, HBM free/used, users of selected GPUs, actual device visibility, driver, CUDA, Torch, NCCL, attention kernels, environment paths, model/config revision hashes, tokenizer revision, vLLM version, graph capture policy, KV cache dtype and pool capacity. Preserve other users' processes and all unrelated worktrees. The prior campaign used burner processes on available GPUs: coordinate with the current local policy; do not indiscriminately terminate burners or foreign jobs. Only stop/restore project-owned processes if authorized and tracked.

GPU mapping:

```text
TP1 → CUDA_VISIBLE_DEVICES=4              (logical 0)
TP2 → CUDA_VISIBLE_DEVICES=4,5            (logical 0,1)
TP4 → CUDA_VISIBLE_DEVICES=4,5,6,7        (logical 0,1,2,3)
TP8 → NOT_AVAILABLE; DO NOT use GPU0–3
```

A TP2 repeat on 6,7 is optional if inexpensive for pair/topology robustness. Do not call four 1-GPU independent replicas `TP4`; that is DP4.

---

## 2. Actual model geometry and matching matrix (verify checkpoints)

| Property | Native Ouro R4 | Tied-Unrolled Ouro R4 | Qwen3-1.7B | Qwen2.5-3B |
|---|---|---|---|---|
| Training provenance | Ouro pretrained | **Same exact Ouro weights** | Independent pretrained AR | Independent pretrained AR |
| Execution semantics | 48 physical × 4 loops | 192 explicit ordered layer calls, tied storage | 28 ordinary layers | 36 ordinary layers |
| Physical parameter footprint | Shared 48-layer core | Same shared core, **no copies** | ~1.7B | ~3B |
| Hidden size | Verify Ouro config | Identical to Ouro | 2048 | 2048 |
| Q heads | 16 | 16 | 16 | 16 |
| KV heads | 16 (MHA) | 16 | **8 (GQA)** | **2 (GQA)** |
| Head dim | 128 | 128 | 128 | Verify from hidden/heads (128) |
| FFN width | Verify exact Ouro checkpoint | Identical | 6144 | 11008 |
| Context upper bound | Native supported; previous total ≤32768 | Same as native | 40960 in currently inspected base config | 32768 in inspected base config |
| KV indexes | recurrent pass × layer | equivalent expanded logical slot | ordinary layer | ordinary layer |
| Sampling/model task quality | Pretrained Ouro | Must match Ouro at fixed R | Genuine AR performance | Genuine AR performance |
| Causal role | Main target | **Strongest matched runtime control** | Contextual real-AR comparison | Contextual real-AR comparison |

References (pin exact revisions at run time):

- Ouro: https://huggingface.co/ByteDance/Ouro-2.6B-Thinking/blob/main/config.json
- Qwen3-1.7B: https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json
- Qwen2.5-3B: https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/config.json
- Optional Qwen3-8B: https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json

**Do not confuse similar Q-head/hidden dimensions with matched FLOPs or KV layouts.** GQA differs substantially: TP4 on Qwen2.5-3B requires handling **two KV heads across four ranks**; this can entail KV-head replication or specialized partitioning. Verify the actual TP partition and bytes. Its scaling may differ for an entirely generic architectural reason unrelated to recurrence.

Important comparison types:

1. **Same-model / same-R / different TP:** meaningful TP scaling and B1 E2E within each family.
2. **Ouro native vs tied-unrolled at same R/TP:** best causal comparison for runtime loop overhead or graph schedule, subject to exact operation equivalence.
3. **Ouro native R=1 vs R=2 vs R=4 at same TP:** effect of recurrent execution depth; output quality changes with R; compare forced-token hardware lane.
4. **Ouro vs real Qwen AR:** independent, genuinely pretrained external reference; useful for context, **not a causal model-agnostic proof** because of architecture/parameter/kernel/tokenizer differences.
5. **Same executed-FLOP / matched-shape synthetic diagnostic (optional):** isolate generic deep-layer TP scaling when actual 192-layer, independently trained AR checkpoint is not available; synthetic hardware-only, never called genuine pretrained AR.

**Do not claim the listed genuine Qwen checkpoints depth-match Ouro R4. They do not.** Their 28/36 layers must be normalized and their kernel/communication structure inspected. Nor does native-vs-tied unroll inherently test parameter-sharing effects: both still share parameters; it primarily tests execution organization.

---

## 3. Stage 0 — Feasibility, correctness and decisive-control contract

### 3.1 Ouro R sweep

Verify `R ∈ {1,2,4}` truly runs all 48 physical layers per recurrent pass and allocates the matching `48R` independent KV cache indices. Check TP1/2/4 per-rank ownership and reductions. Preserve model weights, BF16, RoPE, masks and native cache/sampler policy. R=1/2 are **diagnostic altered-depth evaluations**, not quality-preserving speedups over pretrained R4.

### 3.2 Qwen3 / Qwen2.5 pretrained AR

- Load both official pretrained base checkpoints, *not random initialization*. Resolve exact commit SHA, license and compatible native serving executor.
- Verify actual TP1/2/4 head/weight partition, KV-head duplication at TP4 if present, attention backend (FA3/SDPA), CUDA Graph/eager mode, paged KV allocation, prefill chunking, prefix caching, quantization and sampling defaults.
- Prefer the **same vLLM installation/backend** for both Qwens and Ouro if this is valid. If backends differ, label `ENGINE_CONFOUNDED`; provide separate engine-normalized diagnostics, not a causal Ouro-vs-Qwen conclusion.
- Verify each AR checkpoint supports the tested total position count and actual greedy forced-token path. Qwen3 thinking-mode/chat-template variation should not be conflated with serving latency; for hardware controls use token IDs, no textual quality comparison.
- Record all successful and failed TP arms; never replace a failed AR run with random weights or silently switch to a different-precision model.

### 3.3 Tied-Unrolled Ouro: required concrete execution definition

**Purpose:** create an **explicit sequence of 48R logical layer invocations** using **the identical Ouro pretrained module objects/weight storages**, with identical input/output/KV semantics but without the native outer recurrent iteration dispatcher. It is an execution organization control, not an independent AR neural network.

Required implementation checks:

1. Build an *execution tape* of `R × 48` entries `(logical_layer_id, shared_module_ref, cache_slot, position/rope_state, residual_state_contract, runtime_kernel_signature)`.
2. Reuse the **same `torch.nn.Parameter` storage pointers** across corresponding repeated logical invocations. Verify base/data pointers and storage nbytes. **Do not clone or repeat entire weights** for primary tied control.
3. Make each logical KV slot independent exactly as in native `loop * 48 + layer` indexing; do not collapse caches simply because weights are shared.
4. Preserve Ouro recurrent-specific state updates and embeddings/normalization if present. An ordinary 192-layer `QwenForCausalLM` clone without recurrence-specific state is **not equivalent**.
5. Preserve masks, RoPE, output head, residual order, initial recurrent state, sampling, graph state and attention/caching policies.
6. At **R=1,2,4** (at least R4 if bounded effort prevents full grid) verify same prompt and forced decode produces matching logits, cached KV samples, token IDs and output-state trace. Freeze numerical tolerance before observing speed; ideally bitwise when identical kernel graph, otherwise report max abs/relative L2 and first divergence. BF16 TP differences across different TP degrees are a separate issue; compare same TP native vs tied.
7. Distinguish a genuine explicitly **unrolled graph/schedule** from changing Python source syntax while retaining the exact same captured kernels and replay boundaries. Record graph replay count, graph node count, collective count/order, cache allocation and low-overhead timing. If graph construction is impossible due to vLLM constraints, use a bounded forward microbenchmark, label `PARTIAL_TIED_COMPONENT_ONLY`, and do not invent full B1 E2E.
8. If a 192-layer monolithic graph OOMs, first try bounded chunked explicit unroll with a **documented** graph-boundary difference. Do not secretly disable CUDA Graphs only for one arm then call any speed ratio an effect of recurrence.
9. **Untied-Unrolled** (materializing R separate storage copies) is optional *diagnostic only*. It is neither one of the four requested families nor a pretrained model, and may greatly increase HBM. Do not let it replace tied-unrolled.

Exit on substantial unavoidable engine rewrite after a bounded feasibility proof; report the exact blocker. A fake unrolled runner whose output differs or whose kernels are not matched cannot support loop-specific claims.

---

## 4. Workloads and measurement matrix

### Primary four-family grid

| Family | R | TP | Prompt lengths | Generation |
|---|---|---|---|---|
| Native Ouro | 1,2,4 | **1,2,4** | 1K, 8K, 16K; 32K confirmation | 64 forced tokens (primary), natural secondary |
| Tied-Unrolled Ouro | 1,2,4 if feasible | **1,2,4** | at least 1K and 16K, 32K if equivalent and feasible | exact same Ouro forced continuation |
| Qwen3-1.7B | ordinary 28-layer AR | **1,2,4** | 1K, 8K, 16K; 32K where supported | 64 forced tokens, optional natural |
| Qwen2.5-3B | ordinary 36-layer AR | **1,2,4** | 1K, 8K, 16K; 32K where supported | 64 forced tokens, optional natural |

Suggested **low-cost sequencing** to prevent a sprawling campaign:

1. Stage 0 smoke: four families × TP1/2/4; Ouro R=1/2/4. Validate output/cache/reduction and graph feasibility.
2. Stage A1: eight frozen calibration requests, P=1K and P=16K, 64 forced tokens; three warmups and three measured repeats per feasible arm.
3. Stage A2: add P≈32K to validate prefill/KV heavy regime, plus 8K if it reveals distinct behavior; follow exact model context limits. Compare P+output ≤ configured max; do not forcibly extrapolate RoPE.
4. Stage B: tied-unrolled vs native matched traces/graphs; one/two representative requests at each R/TP, then paired requests if a stable gap exists.
5. Stage C: 24 frozen held-out requests **only if** there is a possible recurrence-specific, physically meaningful effect. If no effect, still finish all four-model small calibration matrix and report a clear NO-GO.
6. Optional Qwen3-8B: one modest matched context (1K/8K/16K), TP1/2/4 after all primary evidence is complete, only when memory and time permit.

#### Input controls

- Use frozen metadata-selected prompts (e.g. coding/math/document background), minimum 8 independent requests for discovery; do not choose requests based on observed latency or routing.
- Tokenizers differ: preserve a fixed **token count**, not a false promise of identical token IDs/semantics across models. Record tokenizer revision and prompt source. Equal token counts/sequence lengths are the hardware control; they are *not* equal semantic difficulty.
- Primary hardware lane forces or teacher-forces an **identical token continuation within each model** at every TP/R setting to fix decode count, avoid EOS, and avoid output-divergence timing confounds. Across Qwen and Ouro, decode token count matches but token IDs may differ because vocabularies/tokenizers differ.
- For Ouro tied/native, require *the same exact token IDs* and initial model state at fixed R/TP, not merely the same length.
- A separate natural-generation lane checks fixed-R pretrained Ouro/AR quality behavior, EOS, stop reasons and numerical divergence. **Do not compare R=1 accuracy with pretrained R=4 as if they were the same model quality.**
- Model training, LoRA, quantization, sampled token pruning and approximate attention are out of scope.

#### Fair runtime policy

- Same physical GPUs for each TP, same HBM utilization policy where executable, same precision BF16, same attention/cache/capture mode where feasible. Graph capture feasibility must be explicit. If one arm OOMs graph capture but can run eager, classify with `GRAPH_CONFOUNDED`, report separately, and never silently mix as fair speedup.
- Do not let a large static KV pool allocated by one arm prevent another from running. Shutdown/reload between configurations, preserve other users' state, and log any OOM.
- Counterbalance TP/model/config order where possible; record machine activity, clock/power observations and thermal drifts. Historical Ouro R4 numbers are context, not substitute for concurrent measurements.
- Main interest **B1 low-request latency**, not max server QPS. Optional B2/B4 only as distinct throughput subexperiment with matched total four-GPU budget.

---

## 5. Physical timing and instrumentation

### Separate clocks and ledgers

**Ledger W: Clean client E2E.** Client request time after loaded model, warmup and graph capture. Include prefill, full forced continuation, output handling, all device completion; exclude model-loading/setup unless separately reporting setup.

**Ledger E: Outer GPU event timeline.** Every prefill/decode forward, recurrence pass, graph replay/staging, final LM head where observable. Do not multiply one last replay by forward count; capture all actual occurrences.

**Ledger K: GPU kernel service/classification.** Attention, QKV, norm, FFN, all-reduce/custom TP reductions, logits gather, KV writes/reads, graph/copy/layout, idle. Service time can overlap other work and must not be reported as additive E2E percentages.

**Ledger C: Non-overlapped critical path (where validated).** Track per-rank readiness/completion, genuine collective waits vs transport/kernel time, graph boundaries, serial dependencies. If cross-rank clocks or graph internals are unresolved, report `PARTIAL`/`UNRESOLVED` and use verified component bounds rather than fabricated precision.

Collective counts and categories must include custom reductions, NCCL, embedding sync and logits gather. No EP D+C All-to-All exists in these dense model TP arms. For each kernel family, identify which part scales with R, with TP, and with KV length.

### Validity checks

- Counterbalanced warmup/measurement; no other jobs on the selected GPUs unless unavoidable and characterized. All requested processes use only 4–7.
- Event/profiler observer overhead compared with clean wall; preferred ≤5% and **>10% invalidates fine attribution**.
- GPU event timestamps must be resolved after request; avoid per-layer host `synchronize()`.
- Pin operation signatures and kernel backends; non-identical fused-kernel selection may be generic framework confound.
- Treat the independent unit as **request**. Use paired bootstrap ≥2000 resamples for each meaningful comparison, with confidence interval and n. Technical repeats, TP ranks, layers, recurrent passes and output tokens are not independent observations.

### Required per-config data

`model, revision, family, R, TP, GPU_uuids, prompt_tokens, output_tokens, actual_forward_count, logical_depth, dtype, head_partition, kv_partition, attention_backend, graph_mode, peak_HBM, KV_GiB_per_rank, weight_GiB_per_rank, prefill_ms, cached_TPOT_ms, clean_E2E_ms, output_TPS, TP_collective_count/kind, TP_collective_service_ms, validated_nonoverlap_comm_ms_or_UNKNOWN, QKV/attention/MLP/kernel-service, graph_replay_count, cache_slots, output_parity_status, measurement_validity`.

Compute all key ratios **per request before aggregation**:

```
Speedup(model,R,TP,P) = T(model,R,TP1,P) / T(model,R,TPk,P)
ParallelEfficiency = Speedup / TPk
ScalingInteraction(R,TP) = Speedup(Ouro,R,TP) / Speedup(Ouro,R=1,TP)
NativeVsUnrolledRatio(R,TP) = T(native Ouro,R,TP) / T(tied unrolled,R,TP)
UnrollTPInteraction(R,TP) = NativeVsUnrolledRatio(R,TP) / NativeVsUnrolledRatio(R,TP1)
NormalizedLayerCost = measured_phase_ms / executed_logical_layers
TPExcessVsIdeal = T(TPk) - T(TP1)/TPk  # diagnostic residual, NOT removable cost
```

Use **both prefill and cached-decode phase separately** for every ratio. E2E can hide opposite scaling regimes.

---

## 6. Causal questions and falsification tests

### Test A: Does TP efficiency decline with R?

Compare Ouro TP1/2/4 at **R=1/2/4** with paired fixed-token probes. A flat or *improving* efficiency curve is valid evidence against H1. If declining, quantify whether the gap is due to extra (48R) layers, cache size, communication counts, changing kernels, graph replay boundaries or true loop-specific state.

### Test B: Native Ouro vs computation-equivalent tied-unrolled

This is the **decisive control for recurrent execution organization**, not the genuine AR checkpoint comparisons. Verify identical operation tape, input/output, KV slots, parameter storage, per-layer collective signatures and GPU kernel shapes.

If native and unrolled tied have the same graph/kernel schedule and indistinguishable TP efficiency, conclude that syntactic recurrence alone causes no additional measurable TP penalty in this regime.

If they differ, identify *what actually changed* (e.g., graph replay boundaries, KV indexing copies, parameter rebind, launch segmentation, host dispatch, barriers). If the change is generic graph overhead rather than specifically exploitable recurrence, classify `GENERIC_RUNTIME_ONLY`.

### Test C: Contextual real pretrained AR baselines

Compare **Qwen3-1.7B and Qwen2.5-3B**, within each model's own TP1→TP2→TP4 speedup curves; add exact geometry/weights table beside the results. For reference, Qwen3 has 28 layers, Qwen2.5 has 36, and Ouro at R4 executes 192. Do not compare their raw 64-token E2Es and infer recurrence overhead.

Required confound checks:

- GQA vs MHA and KV-head replication (Qwen3=8, Qwen2.5=2, Ouro=16).
- Different FFN width, layer depth, vocab/LM-head size and operator FLOPs.
- Attention backend, CUDA Graph, KV cache allocator, chunked prefill, rope/masks and output gather.
- Prefix length and paged-cache allocation, weight resident bytes, and effective layer counts.
- Model compute scale and physical TP message size.

If a reviewer could explain a TP gap solely with these differences, do **not** call it loop-specific. External AR controls are useful sanity/market relevance baselines; they do not replace matched same-model C1.

### Test D: Additional depth-matched diagnostic if needed

If Qwen external contrasts appear dramatic, calibrate an **ordinary deep-Transformer hardware-only layer tape** with Ouro's exact tensor shapes, KV head geometry, kernels, and 48R logical executions; distinct parameter storage is optional if memory allows. Run TP1/2/4 with frozen tensors, without pretending these are pretrained AR weights. Label this `DEPTH_MATCHED_SYNTHETIC`. This helps isolate mere depth and GQA effects. Do not build a full 192-layer separate pretrained checkpoint or confuse this with model quality.

### Test E: Genuine *removable* cost

Only after Test B/C show a plausible special mechanism, estimate from **measured critical-path time** how much can be avoided while keeping output exact. Include communication, data movement, graph capture/replay, extra KV/weight storage and overlap. A purely hypothetical no-collective oracle or 4× ideal is NOT a proposed achievable gain.

The longer-term user goal is meaningfully better TP2/TP4 scaling at unchanged or near-unchanged accuracy; no prescribed speedup is assumed. To demonstrate a possible ≥10% improvement over optimized native TP4, identify at least that much realistically targetable non-overlapped wall time, with safety margin.

---

## 7. Decision gates

**Before claiming a loop-specific research opportunity**, require:

1. All mandatory family feasibility attempted; Qwen3 and Qwen2.5 run TP1/2/4 at common lengths, or a documented hard block.
2. Ouro recurrence grid established at R1/R2/R4 × TP1/2/4.
3. Valid same-TP Ouro native ↔ tied-unrolled operation/cache/output comparison, or explicitly limited `CONTROL_BLOCKED` verdict.
4. A robust non-trivial TP efficiency gap or recurrent implementation difference **that cannot be explained solely by ordinary executed depth, GQA, backend, cache policy or model parameters**.
5. A significant physical critical-path component is associated with that difference; counting 384 layer reductions is insufficient.
6. A plausible exact or quality-validated method could reclaim meaningful E2E/TPOT time after all added costs.

Suggested discovery triggers (not universal thresholds):

- >10% relative change of TP speedup/parallel efficiency across Ouro R, supported by paired uncertainty, to justify Stage B profiling.
- ≥10–15% E2E share of **non-overlapped physically targetable** recurrence-specific cost for a possible follow-up method.
- A bounded same-model native/unrolled differential that survives matched kernel/cache/graph controls.

**Verdicts:**

- `GO_LOOP_SPECIFIC`: validated recurrence-dependent removable critical-path cost and plausible physical recovery, with clear novel prior boundary.
- `CONDITIONAL_LOOP_SPECIFIC`: structural evidence but causal control or headroom not yet validated.
- `NO_LOOP_SPECIFIC_GAP`: same-depth matched execution eliminates apparent gap; TP behavior explained by ordinary Transformer depth/work.
- `GENERIC_TP_OR_RUNTIME_ONLY`: physical cost exists but also arises in AR / generic TP / graph / memory implementation.
- `CONTROL_BLOCKED`: cannot construct/validate tied unrolled or fair comparison; **no loop-specific GO**.
- `RUNTIME_BLOCKED`, `TIMING_INVALID`, `NO_GO`: apply when supported.

Avoid novelty-by-absence: a TP feature not implemented in a particular Looped engine is not necessarily a new academic mechanism. Do not create a new runtime or optimizer after negative evidence.

---

## 8. Quality and fidelity

- Same TP / same R native↔tied must preserve logits/KV/token outputs within frozen numerical criteria; provide tensor hash/stride/storage records.
- Different TP degrees can produce BF16 numerical differences; record first differing logit/token and do not silently assert bitwise equivalence. Repeat same TP multiple times to distinguish hardware nondeterminism from TP variation.
- Qwen and Ouro have different tokenizers, training and output distributions; quality metrics cannot be subtracted as if they are model variants. Official benchmark quality (if affordable) is **separate from hardware execution controls**.
- Diagnostic R1/R2 Ouro has altered logical compute depth; quality may differ from pretrained R4 and is not a valid output-preserving acceleration.
- Do not run generated code without sandboxing. Freeze problems before benchmarks; no output-length truncation counted as completed accuracy.
- Keep optional larger AR reference explicitly descriptive; it is not more causally matched merely because it is bigger.

---

## 9. Mandatory visualizations and report structure

Create:

1. **Four-family headline table** at each TP and context: model/control, actual layers, R, total depth, Q/KV heads, HBM, prefill, cached TPOT, E2E, TP speedup/TP1, CI, validation status.
2. `Ouro R×TP` heatmaps: E2E, prefill, TPOT, efficiency.
3. Ouro native vs tied-unrolled paired TP1/2/4 graph, same R, showing numerical parity and graph/kernel count differences.
4. Real AR `Qwen3-1.7B vs Qwen2.5-3B vs Ouro` **within-model normalized** TP scaling plots (NOT raw-latency superiority claims).
5. MHA/GQA/KV ownership panel for each family; Qwen2.5 KV-head TP4 caution.
6. Executed-logical-layer-normalized compute/collective cost charts, separate prefill and decode.
7. Per-loop (Ouro) and per-layer (Qwen) critical-path/timeline diagnostics with `VALIDATED/PARTIAL` markers.
8. Explicit confound ledger and physically targetable headroom chart; never label cumulative service time a direct E2E fraction.
9. Optional high-compute Qwen3-8B scaling plot only if all four mandatory families completed.

Report MUST answer:

- Are all **four mandatory families** measured and correct? Why is any arm unavailable?
- How does Ouro R affect TP speedup in prefill and decode separately?
- Is any extra TP cost due specifically to recurrence after controlling effective depth?
- Does tied-unrolling change graph scheduling, cache behavior, collective count, or only syntax?
- How do both genuine Qwen AR baselines scale, and how do MHA/GQA/FFN/depth differences confound interpretation?
- Can at least one measured, removable loop-specific critical-path cost motivate a new 4-GPU execution framework?
- Is a larger AR reference needed/feasible, and what independent evidence would it add?
- Does the evidence support GO, CONDITIONAL or NO-GO?

---

## 10. Deliverables, branch and completion

Recommended PoC directory:

```text
poc_looped_recurrence_tp_4models/
├── architecture_matrix.md
├── tied_unroll_contract.md
├── workload_contract.md
├── timing_contract.md
├── prior_overlap.md
├── reproduce.md
├── report.md
├── scripts/
├── results/
│   ├── env_and_pins.json
│   ├── native_ouro_r_tp.csv
│   ├── tied_unrolled_r_tp.csv
│   ├── real_ar_qwen_tp.csv
│   ├── all_config_per_request.csv
│   ├── parity_and_kv_validation.json
│   ├── kernel_and_collective_counts.csv
│   ├── profile_validity.json
│   ├── confounds.json
│   └── verdict.json
└── figures/
```

Suggested isolated branch: `flashvep/looped-recurrence-tp-4models-poc`.

Git rules: preserve all pre-existing dirty/staged changes. Never reset/clean unrelated work. Keep previous PoC artifacts unchanged. Commit and push **only this PoC's source, reports and small results**, not external checkpoints or giant profiler files. If heavy raw traces exceed appropriate repository size, retain them at an explicit stable local path and commit a manifest with hashes and reproduction steps.

**Final Codex response**: verdict; all four families and exact model revisions; R×TP matrix; Qwen AR TP1/2/4 matrices; native-vs-tied parity and timing; external confound assessment; validated critical-path evidence and remaining headroom; quality validity; optionally larger AR model status; physical GPU proof; branch, commit SHA, pushed report path. **Do not say “all four completed” when tied-unrolled control is only synthetic, partial or invalid.**

### Primary sources to verify/record

- Ouro model/config: https://huggingface.co/ByteDance/Ouro-2.6B-Thinking/blob/main/config.json
- Qwen3-1.7B config: https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json
- Qwen2.5-3B config: https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/config.json
- Optional Qwen3-8B: https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
- FlashLoop: https://arxiv.org/abs/2609.29812
- LoopSpec: https://arxiv.org/abs/2609.17184
- Parallel Loop Transformer: https://arxiv.org/abs/2510.24824
- Parallel Track Transformers: https://arxiv.org/abs/2602.07306
- vllm-rlt: https://github.com/ThinkFlowLab/vllm-rlt
- Existing verified native report: `poc_looped_multigpu_kv_placement/report.md`.

**Governing research rule:** If recurrent TP scaling behaves just like ordinary matched-depth TP, this study should CLOSE the claimed recurrence-specific synchronization pathology instead of designing an unsupported “loop-aware” parallel framework.
