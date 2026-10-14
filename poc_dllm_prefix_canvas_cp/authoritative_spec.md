# dLLM × Multi-GPU: Exact Prefix–Canvas Asymmetric Attention / Context Parallelism PoC

**Status:** research discovery contract, NOT a claim of a working method  
**Intended server file:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/dllm_prefix_canvas_cp_poc_spec.md`  
**Allowed physical GPUs:** **4, 5, 6, 7 ONLY** (`CUDA_VISIBLE_DEVICES=4,5,6,7`)  
**Primary objective:** batch-1, exact-semantics block-diffusion refinement latency and E2E output TPS  
**Primary checkpoint:** official `inclusionAI/LLaDA2.0-mini` (NOT necessarily `mini-preview`)  
**Secondary checkpoint (OPTIONAL):** `google/diffusiongemma-26B-A4B-it` with native vLLM implementation  
**Research style:** characterize a physical critical path, bound attainable benefit, then implement *at most* one minimal exact proof of concept if warranted.

## Executive summary

The existing MoE-EP PoCs repeatedly showed that more logical communication or routing imbalance does **not** necessarily imply higher physical latency. This investigation changes the target from MoE EP to **multi-GPU attention and KV movement** in block diffusion language models (dLLMs). A completed prefix is immutable while the current noisy/canvas block is repeatedly updated. Hypothesis: a *prefix-sharded, canvas-aware exact attention* decomposition can reduce repeated long-prefix KV access and/or redundant inter-rank communication relative to a strong native 4-GPU baseline, with sufficiently small distributed softmax merge overhead to improve whole-request latency.

**This is a falsifiable hypothesis, not an established bottleneck or a novel attention algorithm.** Cache reuse, attention split and context parallelism already exist. First establish (i) the actual stable/fresh KV semantics of the selected model/runtime, (ii) substantial attention-related physical E2E cost, (iii) a conservative attainable speedup on the real H100×4 topology, and (iv) exact attention semantics. If any major gate fails, stop without engineering a new runtime.

---

# 1. Research question and scope

**Primary question:** During each denoising/refinement forward for a block diffusion model, can the immutable prefix-KV region and changing current-block region be placed/computed *asymmetrically across four GPUs* to improve **actual** low-request serving latency/TPS over the best available native distributed baseline?

Distinguish several quantities:

- **Prefix length `P`**: previously accepted prompt + already committed blocks, valid under the runtime's effective context and per-layer attention windows.
- **Canvas/block length `B`**: current block's query positions (usually 32 for the primary LLaDA2 setup; verify supported B rather than hard-code).
- **Denoising iteration `t`**: updates canvas hidden states / Q,K,V each iteration.
- **Prefix KV `K_p,V_p`**: fixed *within the refinement of a block* only if experimentally verified at the chosen attention interface; can be shared between iterations. New committed blocks change the prefix and require cache updates.
- **Canvas `Q_c(t), K_c(t), V_c(t)`**: fresh each iteration; attention may be bidirectional within current block.

Do NOT assume every architectural layer uses identical attention (full/sliding, grouped-query, hybrid), that source-SP is equivalent to context parallelism, or that native caching fails to reuse fixed prefix. Do NOT reduce this research to generic attention caching or to generic ring attention repackaged with dLLM terminology.

**Out of scope for this PoC:** expert replication/placement, changes to router, skipping active tokens, approximation, fresh-token acceptance modifications, model retraining, learned predictors, speculative decoding, MultiBD, generic prefill–decode scheduling, or new large kernels before Gate A.

---

# 2. Context and prior boundary

Prior systems are strong and must be acknowledged:

1. **FlashBlock (ICML 2026):** exploits cross-step redundancy of block-external attention and caches/reuses attention results. This PoC must not claim discovery of prefix-attention redundancy or repeated KV reads as its novelty. https://proceedings.mlr.press/v306/chen26g.html
2. **LoSA (ICML 2026):** reduces dLLM KV loads by sparse attention and reuse for stable tokens; its approximate or locality-based token selection is distinct from the initial *exact* CP decomposition. https://proceedings.mlr.press/v306/xi26d.html
3. **vLLM DiffusionGemma native runtime:** already separates causal prefix/commit encoding from bidirectional canvas denoising, automatically caches the committed prefix, and batches different stages. Do not claim the baseline recomputes static prefix unnecessarily. https://vllm-project.github.io/2026/06/10/diffusion-gemma
4. **General context parallel / ring attention / FlashAttention distributed softmax:** sharding keys across GPUs and merging softmax summaries is established. The potential contribution (if any) would be in *dLLM-specific KV lifetime/layout, repeated canvas refinement, measured multi-GPU critical path, and a new superior execution schedule/layout*, not in the math of softmax splitting.
5. **Existing LLaDA2 and Diffulex/dInfer serving:** inspect native KV-cache, attention masks, source-SP/TP, full-source all-gathers, and kernel operations before proposing extra communication. Existing project EP PoCs are relevant for instruments and exactness only; MoE EP is no longer required.
6. **Long-context attention prior:** characterize the full-attention versus sliding-window layer mix. A claimed 32K length does not imply every layer reads all 32K keys.

Create `poc_dllm_prefix_canvas_cp/prior_overlap.md` with links, specific code-level baseline behavior, and exact claimed novelty *or explicitly say the novelty is unproven*.

---

# 3. Model and context verification: critical Stage 0

## 3.1 Primary model

Use official `inclusionAI/LLaDA2.0-mini` if loaded and verified. Its **current published model card** advertises **32,768 tokens context**; an older `LLaDA2.0-mini-preview` advertises **4,096**. Earlier team experiments configured a 4,096 maximum: do **not** infer that the official current model is limited to 4K, nor assume this installed runtime can serve 32K. Record the *actual* checkpoint SHA, config fields, RoPE limits, runtime max-context and KV allocator limits. Model card: https://huggingface.co/inclusionAI/LLaDA2.0-mini ; preview: https://huggingface.co/inclusionAI/LLaDA2.0-mini-preview

Prefer native existing inference path from this repo's prior PoCs, using valid BF16 checkpoint and original block sampler. Workload is batch-1 / low concurrency. It is acceptable for the primary engine to use TP/SP or EP as an implementation detail, but **research contribution and diagnosis must concern attention/CP, not MoE placement**.

## 3.2 Secondary model (strictly optional)

`google/diffusiongemma-26B-A4B-it` has an official vLLM runtime and a much larger advertised context. Its model uses **causal prefix encoding and bidirectional canvas denoising**. Important confound: the architecture has many sliding-window layers and a smaller number of full-attention layers. Profile layer-by-layer **effective** attended prefix, not just nominal context. Never port LLaDA2 assumptions blindly. References: https://vllm-project.github.io/2026/06/10/diffusion-gemma ; https://recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it

Do not spend primary PoC time rebuilding or converting a second engine. If native secondary inference or four-GPU mode is unavailable, mark `SECONDARY_NOT_RUN`.

## 3.3 Environment/GPUs

Use only:

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
nvidia-smi
nvidia-smi topo -m
```

Before running, validate physical UUID mapping to logical 0–3, HBM, compute utilization, NVLink fabric and conflicts. Do not kill/interrupt another user's jobs. Make no statement about performance of GPUs 2/3 or another node.

## 3.4 Audit actual attention contract

Write a short `attention_contract.md` answering:

- Is prefix KV computed only during prefill/commit? Is it immutable across denoising iterations? Identify actual tensors, storage owners, versions and invalidations.
- What positions are queries in each refinement forward: block-only or entire prefix+block?
- Is current block bidirectional and prefix causal? Any infill, padding, positional RoPE offsets, sliding window, sink tokens, grouped-query attention or ALiBi?
- Where exactly do TP/SP collective operations happen? Which tensors are replicated/sharded? Are complete source hidden states all-gathered before attention?
- What is the native attention backend (FlashAttention, Triton, fused kernel), caching behavior and step schedule? Are two kernels for different attention masks already used?
- Are attention time and KV bytes actually proportional to `P` or capped by a sliding window?

**Gate 0:** if the stable-prefix/fresh-canvas model is invalid in the measured path, or no working native 4-GPU inference/correctness baseline exists, return `SEMANTICS_BLOCKED` or `RUNTIME_BLOCKED` rather than fabricating a benchmark.

---

# 4. Precise baseline and fairness rules

Primary baseline = **best supported, unmodified, optimized native 4-GPU runtime on GPUs 4–7** using the same checkpoint, BF16, attention mask, block length, sampler, KV cache and request set. Name exact layout such as TP4, TP4 + source-SP4 / EP4, or the actual native hybrid; never call it TP4 if the geometry differs.

Secondary controls if cheap:

- Native single GPU and native TP2, to characterize overhead, **only if supported without OOM or algorithm changes**. Single GPU may not be memory feasible; no invented numbers.
- Existing native context-parallel/sequential distributed-attention mode where available, using *the same four physical GPUs*; assess whether it already achieves the target.
- Existing FlashAttention/SDPA attention-only microbench with exact same masks and dimensions, as a best-engineering baseline; synthetic tensors are suitable only for microbenchmark checks, not serving evidence.
- Cached-prefix ON vs OFF only as a diagnostic where the native implementation genuinely supports such a control, not as the method's primary baseline.

### Comparable four-GPU configurations

Do not assume `TP4 + CP4` uses four GPUs. Depending on the backend, `TP×CP` may require 16 ranks; hybrid rank reuse requires explicit support. Verify process groups, ownership and effective parallelism from traces, not configuration labels. If the only viable candidate replaces TP4 with TP1/CP4 or TP2/CP2, this **also changes dense-layer sharding**: report that confound and include an apples-to-apples overall-configuration comparison, not a false isolated CP speedup.

The proposed P/C scheme may be evaluated as a layer-level exact attention kernel replacement under an unchanged 4-GPU *outer model-parallel layout* if feasible. This is the preferred clean causal control.

---

# 5. Stage A: cheap physical profiling (no method implementation)

## 5.1 Freeze workload and supported context lengths

First read checkpoint and runtime limits. Tentative prefix-length ladder:

```text
P ∈ {512, 1024, 2048, 4096, 8192, 16384, 32768}
```

**Filter to legally supported `P + B + generation budget <= effective context`, plus realistic OOM/headroom constraints.** No silent RoPE extrapolation or runtime-context bump beyond model support. 32K may require a small B/output budget; record effective `P` including chat template overhead. The full 32K request may not be runnable; report unsupported cases rather than shortening silently.

Use a fixed primary `B` (default native 32 for LLaDA2), and a secondary supported larger B if cheap to expose Q-size sensitivity. Pre-fill workloads with deterministic actual text tokens (e.g. pinned public benchmark/document passages with recorded hashes), avoid synthetically identical tokens for headline performance. Frozen prefix contents matter for acceptances/NFE.

For Stage A, use **8 calibration requests** spanning code, math, and plain long-context QA/document continuation; **24 frozen held-out requests** only after gates. Long-form prompts must be valid and nontruncated. Pair each prompt across configurations and freeze prompt lengths and output budgets before looking at results. Quality scoring may use official GSM8K/HumanEval+/long-context QA with correct complete outputs; do not call truncated or artificial completions accurate.

If long input loading dominates, distinguish warm-prefilled generation from cold E2E. **Both** are required in the final analysis because the target is repeated refinement, while cold E2E is a real user metric.

## 5.2 Timing lanes

Three lanes:

1. **Clean** request timing (no hooks, no trace): wall E2E, cold prefill, commit, warm refinement, output TPS.
2. **CUDA-event only** with events around native operations and no per-layer GPU synchronization, validate observer overhead.
3. **Heavy routing/attention traces** separate from headline timing; never use instrumented wall as true serving latency.

Stage timing should capture **critical stream / cumulative rank for request** and report independently any asynchronous overlap. Do not simply add independent maxima or use sum of attention kernel durations as proof of E2E critical-path share when concurrent work overlaps.

Aim event observer overhead ≤5% (request mean); if >10% or upper CI >10% label fine-grained attribution `UNTRUSTED`. Preserve clean request latency.

## 5.3 Mandatory breakdown

By prefix length and denoising step, measure, where identifiable:

- full request latency, TTFT/first useful token, E2E, output TPS and time per final committed token;
- prefill/commit time vs repeated refinement time; forwards per output / NFE and TPF;
- attention **compute kernel wall/critical time**, attention input prep, QKV projection, RoPE/mask setup, KV update/cache lookup, prefix K/V read (bytes or validated proxy), canvas KV recomputation, attention output projection;
- TP/SP/CP communication time: all-gather, all-reduce, reduce-scatter, all-to-all as applicable; operation payload (measured where possible) and synchronization;
- MLP/MoE component, LM head/sampling and runtime/graph/bookkeeping time;
- per-layer sliding vs full attention effective key lengths, attention traffic and P sensitivity;
- HBM allocations peak per rank, attainable native context and OOM cases.

If attribution is difficult, use low-overhead NVTX + **one** bounded profiler sample (`torch.profiler`/Nsight Systems) rather than implementing new kernels. Distinguish overhead attributable to first prefill from recurring canvas refinement. Distinguish existing cached prefix reads from actually recomputing prefix K/V; do not count a cached read as a fresh prefix computation.

## 5.4 Stage A Amdahl analysis

Let `f_attn` be the **non-overlapped, critical-path** fraction of repeated refinement / E2E devoted to the *targetable attention + KV + attention-specific communication*. Compute an optimistic ceiling:

```
Speedup_E2E(S_target) <= 1 / ((1 - f_attn) + f_attn / S_target)
```

Evaluate S_target = 1.5×, 2×, 4× and infinite. Report cold E2E and warm generation separately. If `f_attn` is not separable, report bounded upper/lower estimates or `UNKNOWN`; do not mislabel sum of kernel durations as critical-path fraction.

**Stage-A kill conditions:**

- Targetable repeated attention-related critical-path share <~20%, *and* plausible maximal benefit is <10% E2E; or
- native attention time fails to grow meaningfully with effective prefix length (e.g., sliding-window dominated), or
- 4-GPU interconnect and sequence size imply unavoidable CP merge overhead larger than saved traffic, or
- correctness/validity cannot be established.

These are heuristic gates, not arbitrary proof of failure. If the primary model's context regime is genuinely too short to expose memory/attention pressure, it may be `CONTEXT_LIMITED` rather than a universal NO-GO. Secondary verified long-context model is optional, never required to rescue a negative PoC.

---

# 6. Stage B: exact prefix-sharded/canvas-aware attention microbenchmark

**Enter Stage B ONLY if Stage A identifies a substantial recurring attention bottleneck and feasible prefix/canvas semantics.** No change to model token generation or expert computation yet.

## 6.1 Baseline mathematical attention

For a query `q` from the current canvas, attention keys/values comprise the *eligible* prefix positions and the *eligible* current canvas positions under the exact model mask:

```
A(q) = softmax([ q K_prefix^T , q K_canvas^T ] / sqrt(d) + mask) [V_prefix; V_canvas]
```

Include correct RoPE/head scale, GQA KV head replication rule, logit biases/softcaps as appropriate to actual model. Respect sliding windows and any block-specific attention rules.

## 6.2 Candidate data layout

Maintain prefix KV sharded along **sequence/key dimension** across four GPUs, retaining only the keys each GPU is assigned. Treat the small current canvas separately (possibly replicated) while computing *exact* fresh Q,K,V each refinement. Do not multiply the canvas's contribution four times when merging results; e.g., merge prefix-shard partials then combine **one** canvas partial, or assign the canvas partial to one logical rank with a mathematically correct reduction.

For each allowed key shard `r`, compute a stable softmax summary:

```
m_r = max_j s_rj
l_r = sum_j exp(s_rj - m_r)
o_r = sum_j exp(s_rj - m_r) v_rj
```

Combine all key shards with the correct stable normalization:

```
m = max_r m_r
l = sum_r exp(m_r - m) * l_r
o = [sum_r exp(m_r - m) * o_r] / l
```

Masks and empty shards must be safe (`-inf`, zero l/o). The equation is already known distributed softmax, NOT a new algorithm. Ensure canvas partial counted exactly once. If one rank owns all Q or outputs, include actual gather/broadcast cost. For TP attention heads, the merge is only across shards sharing those heads; never mix distinct TP heads.

## 6.3 Minimum microbench arms

Use representative Q lengths `B ∈ {native block length, optionally larger native length}`, key lengths `P` from validated Stage A, exact attention masks and KV layouts:

- `B0`: native optimized 4-GPU attention path (or isolated native kernel including required collectives).
- `B1`: explicit prefix-sharded/canvas-owned exact partial softmax, with **all** required communication, data copies, synchronization and layout transforms.
- `B2`: optional balanced prefix replication / local attention for short P (where it fits in HBM), as a *fair* baseline.
- `B3`: optional native CP if supported.

Do not use naive PyTorch einsum/SDPA as the sole competitive native baseline if B0 is FlashAttention optimized. If candidate uses slower unfused ops, measure them honestly and stop; do not infer an optimized-speedup from theoretical FLOPs alone. Avoid writing complex custom Triton kernels until a real exact B1 design passes correctness and the conservative cost bound.

**Microbench control:** same physical device, same dtypes, head shapes, masks, RoPE, stream usage; include first-step warm-up, steady-state CUDA graphs if native uses them, and the actual repeated denoising shape. No excluded merge/collective/prep/reformat cost. Report measured per-call latency, effective HBM bytes proxy, NVLink bytes where measurable, and per-rank peak memory.

## 6.4 Numerical correctness

Compare candidate attention outputs against exact native attention within frozen BF16 error tolerance plus FP32 reference checks; set tolerances **before** running. Validate:

- different P/B and head sizes, causal prefix, bidirectional canvas, mixed/empty masks;
- arbitrary RoPE absolute positions, long offset, GQA groups and sliding window (if supported);
- per-layer correct KV versions across at least three refinement iterations and a block-commit boundary;
- no stale canvas K/V; no reuse of attention output across changing queries (unless proven exactly equal, which is unlikely);
- no duplicate canvas contribution or masked-key leaks; no numerical instabilities (NaN/Inf);
- output head/sequence ownership; eventual full logits acceptance and token sequence parity for identical configurations.

If model output is numerically sensitive near acceptance thresholds, fixed model/token/NFE/acceptance parity is a must to claim training-free exact inference. Distinguish mathematical exactness from bitwise kernel equality; BF16 floating-order differences can be non-bitwise. Numerical parity alone does not establish full generation parity.

## 6.5 Gate B (kill test)

Before any full-model integration, demand:

- candidate attention + all collectives ≥ **1.25× faster** than the *valid native attention path* for a relevant supported P/B regime, with uncertainty bounded;
- no significant increase in overall per-rank HBM footprint that invalidates long context;
- conservative expected **request E2E gain ≥10–15%** under measured Stage-A critical-path fractions, ideally ≥15%; and
- correctness pass.

If the measured B1 is slower/equal or only improves an unrealistic P not supported by the model, finish `NO-GO` / `CONTEXT_LIMITED`, not speculative `GO`.

---

# 7. Stage C (OPTIONAL): smallest exact full-model integration

Only after A+B pass. Integrate candidate at the **existing attention interface** with minimal layout changes. Keep all weights, samplers, block schedules, and nonattention kernels fixed. Reuse the runtime's KV allocator and native attention backend where possible. Avoid large engine rewrite, new scheduler, dynamic TP layout switching or generic CUDA-graph infrastructure.

### Comparisons

- `C0`: unmodified best native 4-GPU checkpoint/runtime at same supported prompt length and output budget.
- `C1`: identical model/runtime with only candidate attention path changed, all four GPUs identical.
- optional native CP / full prefix-replicated control if support exists.

Test short and long P including **the P where Stage B showed benefit**, with paired frozen requests. Every config should use same requested output cap and primary sampling parameters. Distinguish request-dependent actual output length; do not count raw padded/dummy rows as emitted tokens.

### Whole-system timing

- clean warm generation latency, clean cold request E2E, prefill/commit cost, refinement TPF/NFE;
- time per committed output token and TPS;
- exact per-request output + acceptance/NFE, official quality (where feasible);
- correct and robust multi-GPU rank participation, actual collective totals and peak HBM;
- per-request speedup and request-level paired-bootstrap 95% CI (>=2,000 resamples, 24 frozen requests preferred).

**Integrate without breaking native full-model TP/EP/SP state.** If changing layouts in one attention layer forces prohibitive all-gathers in other layers, those costs must appear in the measured E2E. Do not hide one-time precompute/replication/migration costs if repeated per request or cache miss.

### Gate C

Promising method territory only when clean **whole-request E2E latency** improves by **≥10%** (prefer ≥15% for strong signal) at meaningful P, with CI excluding 0, and no performance regression in quality/correctness. Faster attention microkernels without whole-system gain = `MICROBENCH_ONLY` / NO-GO for method development.

---

# 8. Specific failure modes / rigor controls

1. **False stable KV assumption:** verify actual cache version per layer and block, not only token acceptance; prefix KV at *a different layer* may be recomputed by some architectures.
2. **Double-counted canvas:** replicated canvas contributions added four times incorrectly; validate softmax normalization carefully.
3. **Bad TP/CP geometry:** adding CP4 to TP4 may require 16 GPUs, unavailable. Physical GPUs 4–7 only.
4. **Short-`B` communication dominance:** Q length ~32, prefix split softmax yields small per-GPU matmul but requires cross-rank normalization. There may be no win even for P=32K.
5. **Sliding-window layers:** nominal long context may be invisible to most layers, so key reads plateau. Profile effective P by layer.
6. **Native KV caching already optimal:** don't attribute prefix projection savings the native runtime already obtains. Differentiate KV write/read and query dependent attention computation.
7. **Approximations/Prior:** FlashBlock/LoSA already exploit redundancy; copying these techniques or generic CP is not a new research story.
8. **Poor baseline:** PyTorch naive attention vs fused FlashAttention is invalid evidence. Native baseline must be optimized and verified.
9. **TP change confound:** TP1/CP4 versus TP4 changes MLP/LM head ownership, requiring total-E2E comparison rather than single-cause assertion.
10. **Instrumentation distortion:** heavy hooks make attention appear expensive; use clean vs event-only observer test.
11. **Unequal generation:** changing the number of accepted tokens, NFE, lengths, quality or KV semantics makes TPS misleading.
12. **Prefill masking:** cold long-prefix prefill can dominate E2E; both warm generation and cold E2E must be shown.
13. **False multi-GPU:** exact rank topology and real GPU traffic must be measured; CPU/offline scheduling proxies insufficient.
14. **Accuracy overclaim:** 24-case accuracy cannot establish large-benchmark non-inferiority; retain correctness/quality as separate claims.
15. **Workload specificity:** long artificial prefix with tiny output might reveal microkernel cost but be unrepresentative. Include a real long-context continuation distribution and output length.
16. **No cost-free KV relocation:** dynamic placement/migration or persistent per-rank KV replication is not free; include actual bytes and per-request initialization.

---

# 9. Primary metrics and tables

For each `P, B, runtime-layout, backend` record:

| Metric | Definition |
|---|---|
| Supported effective prefix | Tokens actually eligible under per-layer mask and max context |
| `T_refine` | Clean repeated denoising wall time, excluding initial prefill and post-commit |
| `T_attn,target` | Bounded, nonoverlapped critical-path attention/KV/CP time |
| `f_attn` | Targetable critical-path fraction of refinement and E2E |
| `T_prefill`, `T_commit` | Other diffusion phases |
| `TPS` | Final emitted (post-EOS) output tokens divided by clean generation wall time |
| `E2E` | Full cold request, including prefill and all block commits |
| `TPF`, `NFE` | Useful final committed tokens per forward and forward count |
| `KV_bytes`, `CP_bytes` | Actual or carefully labelled logical payload; never substitute fake wire bandwidth |
| `GPU_peak_HBM` | Per physical rank allocated/reserved/peak |
| `S_attn`, `S_E2E` | Paired speedups vs native baseline |
| `Parity` | Same config token IDs, step acceptance, NFE (or an explicit bounded numerical caveat) |

Main presentation should include:

- **Figure 1:** per-prefix-length stacked wall/critical-path breakdown (prefill vs refinement, attention/KV, TP/SP communication, MLP/MoE, other) using valid non-overlap accounting.
- **Figure 2:** targetable attention fraction and Amdahl E2E speedup bound vs effective prefix length.
- **Figure 3:** native versus exact asymmetric attention per-step microbench latency with communication included.
- **Figure 4:** exact candidate vs native cold E2E and warm TPS across P, if integrated.
- **Figure 5:** KV/cache and per-rank HBM, full/sliding attention effective-P.
- **Figure 6:** numerical parity and correctness summary, if integrated.

A useful minimal discovery table:

| Model/runtime | P | B | Layout | Refinement ms | Attn+KV critical % | Collective critical % | 2× attention ceiling E2E | E2E ms | Outcome |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

---

# 10. Decision protocol

- **`SEMANTICS_BLOCKED`**: verified attention architecture does not fit prefix-fixed/canvas-fresh exact decomposition for the tested path.
- **`RUNTIME_BLOCKED`**: valid physical multi-GPU native inference or robust timing cannot be obtained without major engineering.
- **`CONTEXT_LIMITED`**: verified model/runtime context/window prevents meaningful attention stress testing; optional suitable second checkpoint may be considered separately.
- **`NO_ATTENTION_BOTTLENECK`**: targetable attention cost/critical-path share is small or effectively flat with prefix length.
- **`MICROBENCH_ONLY`**: accurate kernel-level win exists but insufficient clean E2E benefit or no bounded integration.
- **`NO-GO`**: physics, correctness, low speedup or strong native baseline excludes viable gain; do not implement final method.
- **`CONDITIONAL`**: exact physical problem is evident but measurement/realizability requires one clearly bounded missing test.
- **`GO`**: exact semantics, substantial physically measured targetable fraction, robust efficient candidate, material clean E2E gain and clear prior differentiation.

**Do not promote structural KV bytes, CUDA kernel percentages, Amdahl ceilings, simulated sharding, or static attention math alone to GO.** The project previously showed how such proxies can mislead.

---

# 11. Implementation plan / repo rules

Recommended repo output:

```text
poc_dllm_prefix_canvas_cp/
  README.md
  attention_contract.md
  prior_overlap.md
  reproduce.md
  scripts/
    inspect_checkpoint_runtime.py
    validate_native_multigpu.py
    freeze_requests.py
    profile_clean.py
    profile_event.py
    inspect_kv_lifetime.py
    inspect_attention_mask.py
    measure_collectives.py
    compute_amdahl_bounds.py
    microbench_exact_split_attention.py  # Stage B only
    validate_exact_attention.py          # Stage B only
    optional_full_model_integration.py    # Stage C only, if appropriate
    analyze.py
  results/
    environment.json
    model_runtime_contract.json
    frozen_manifest.json
    baseline_per_request.csv
    baseline_per_step.csv
    baseline_per_layer.csv
    critical_path_summary.json
    microbench.csv                       # only if Stage B ran
    correctness.json                     # as applicable
    headline_summary.json
    final_verdict.json
  figures/
    latency_breakdown.png
    attention_fraction_amdahl.png
    native_vs_candidate_microbench.png   # only if Stage B ran
    e2e_tps_vs_prefix.png                # only if Stage C ran
    kv_hbm_profile.png
  report.md
```

Copy/install **this spec** to the server intended path. Do not infer a server file exists merely because a local attachment exists. A GitHub branch name suggestion is `flashvep/dllm-prefix-canvas-cp-poc`. Respect unrelated dirty/staged files; do not `git reset --hard` or `git clean` shared work. Keep model weights/external engines and large raw traces out of git; retain revision hashes, commands, relevant small CSV/plots/log excerpts. Commit/push ONLY this PoC's files once finished. Do not spend GPU cycles on optional experiments after a hard no-go. No background task assumptions; the CLI agent can execute normally within its own session.

Reuse old instrumentation only where correct:

- `poc_dllm_ep_comm`, `poc_dllm_attn_moe_pipeline`: BF16 correctness, CUDA-event observer, request bootstrap, physical 4-GPU bridge.
- `poc_multibd_moe_ep`: clean/event/trace separation, context configuration, acceptance/NFE parity and deterministic cuBLAS caution.
- Avoid inheriting experimental bridge changes that alter attention semantics or add fixed buffer overhead.

Final report must explicitly answer:

1. Exact model ID/revision, original vs preview, verified maximum trained/context runtime and per-layer windows.
2. Exact rank/device topology and native TP/SP/EP/CP geometry.
3. Which prefix K/V is truly stable and where stored; which Q/K/V changes per iteration.
4. How much of **actual refinement and E2E critical path** is eligible attention/KV/communication, by P.
5. Whether existing native prefix caching already removes expected work.
6. Whether key length 4K→8K→16K→32K actually increases physical attention cost.
7. Amdahl ceilings with honest assumptions and uncertainties.
8. If B ran, exact attention kernel microbench vs optimized native with all comm and masks.
9. If C ran, paired clean E2E and TPS gain and BF16 full-generation correctness/quality.
10. Why any gain is *dLLM-specific and genuinely multi-GPU* rather than generic CP or caching prior.
11. Final verdict and single smallest next action, or clear closure.
12. Branch, commit SHA, report path, known limitations, and failures preserved.

---

# 12. Suggested frozen stopping thresholds

Use these as **pre-registered discovery heuristics** (not claimed universal requirements):

- **Stage A targetability:** repeated refinement has ≥25–30% *critical-path* targetable attention/KV/attention-collective share at an important P; a smaller share needs unusually strong evidence of ≥10% whole-request headroom.
- **Stage A E2E bound:** estimated attainable **cold E2E improvement ≥10%**, ideally ≥15%, after prefill/commit and communication accounted for.
- **Stage B real timing:** ≥1.25× **exact** attention-path speedup over strong native including cross-GPU merging; better ≥1.5×; no unacceptable HBM blowup.
- **Stage C end-to-end:** ≥10% clean cold E2E latency reduction with request-level CI excluding zero, while warm generation TPS improves and no output/acceptance/quality degradation is attributable to method.
- **Observer gate:** measured event-only overhead ≤5% preferred; >10% or upper CI >10% causes `UNTRUSTED` stage attribution.
- **If repeated attention is already negligible or long context unavailable:** stop immediately, retain useful characterization and mark the appropriate NO-GO/CONTEXT_LIMITED verdict.

**Success is not a pretty attention microbenchmark. Success is a credible exact 4-GPU execution improvement with a distinctive, defensible gap from FlashBlock/LoSA/general distributed attention.**
