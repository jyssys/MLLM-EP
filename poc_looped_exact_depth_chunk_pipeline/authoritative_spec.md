# Discovery PoC: Exact Recurrent-Depth × Sequence-Chunk Pipelining for Looped Transformer Inference

> **Revision v2 — 2026-10-15: Add official-checkpoint quality validation, reference evaluation reproduction, natural-generation and full-model correctness gates.**
> This document supersedes v1 for the revised experiment; all original dependency, cost, physical PP/TP, and GO/NO-GO safeguards remain in effect.

**Status:** Pre-method feasibility, dependency, cost, and physical-bottleneck characterization.  
**Primary purpose:** Determine whether an *exact*, training-free, multi-GPU pipeline over (recurrent depth × causal prompt chunks) can outperform an optimized conventional TP implementation **and an equally optimized generic pipeline-parallel (PP) control** for a pretrained Looped Transformer.  
**Authorized physical GPUs:** **4, 5, 6, 7 only**; `CUDA_VISIBLE_DEVICES=4,5,6,7`.  
**Model:** `ByteDance/Ouro-2.6B-Thinking`, preserve exact pretrained revision `f1edd81e7ac41355db670500ceaf204e0f73af68` unless the existing checkpoint has a justified new pin.  
**Starting runtime:** Existing working Ouro vLLM environment pinned in prior PoCs (vLLM `0.26.0+cu129`; validate actual versions).  
**Suggested server path:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/looped_exact_depth_chunk_pipeline_poc_spec_v2_quality.md`  
**Proposed output:** `poc_looped_exact_depth_chunk_pipeline/`  
**Suggested branch:** `flashvep/looped-exact-depth-chunk-poc`

---

## 0. Executive decision contract

This is **not** a request to build a large new engine by default. The experiment proceeds through gates:

1. **Stage 0:** Reconfirm actual model semantics, hardware, prior state and **official R4 model-quality configuration**; perform a bounded, natural-generation quality readiness check before treating this model/runtime as a research baseline.
2. **Stage A:** Build an **exact causality/recurrence dependency DAG** for chunked prefill; prove which tasks can overlap, and compare with ordinary depth-pipeline scheduling.
3. **Stage B:** Use **physical single-stage costs and physical inter-GPU transfers** to construct a calibrated schedule and conservative end-to-end estimate, including bubbles and full fixed-depth decoding.
4. **Stage C (conditional):** If credible headroom remains, run a **minimal physically correct** 2-stage prototype, with a 4-stage extension only after the 2-stage gate passes.
5. **Stage D (conditional):** Test clean E2E, natural-output and KV/logit parity, and **dataset-level quality noninferiority** under matched official model settings; compare optimized TP and generic PP.

An attractive DAG drawing, reduced theoretical arithmetic depth, a copy-only microbenchmark, or an idealized zero-cost pipeline oracle **does not justify GO**. If the proposed schedule is mathematically the same as ordinary PP on a 192-logical-layer unrolled Transformer, explicitly say so and **do not claim loop-specific novelty**.

**Primary experiment:** Fixed recurrence R=4 and long-context B1 **prefill**. Shorter R=1,2 are diagnostic only, not quality-preserving methods. **Decode remains native fixed-R and must be counted in E2E**; no speculative future-token prediction is part of this PoC.

---

## 1. Hypothesis and what would constitute a discovery

A pretrained recurrent model repeats a shared physical layer stack, adding a recurrent-depth dimension. For Ouro, inspect and verify:

- 48 physical Transformer layers per recurrent pass;
- 4 recurrent passes for the pretrained inference default;
- 192 logical layer invocations and independent recurrent-depth/layer KV slots;
- causal autoregressive attention with loop-specific KV indices;
- full recurrent boundary normalization and residual semantics.

For a long prompt split into **consecutive contiguous causal chunks**, a task at depth `r`, chunk `j` depends on:

1. **Depth predecessor:** chunk `j` has completed the entire required preceding recurrent depth `r-1`, supplying the correct hidden state to depth `r`;
2. **Causal prefix predecessor:** **all earlier chunks** of the prompt have populated the correct KV for the same depth `r` and for every attention layer actually executing within that stage;
3. **Intra-chunk causality:** the local attention mask allows only preceding positions within chunk `j`;
4. **Other precise dependencies:** any per-pass residual reset, final norm, positional encoding, shared state, multi-stage layer dependencies, and cache synchronization imposed by the actual model and runtime.

The central question is whether these dependencies leave useful **physical concurrency** such that different GPUs can work on different `(depth, chunk)` points while exactly preserving the pretrained full-depth outputs.

**Null hypotheses:**
- The dependency-respecting schedule is simply ordinary chunked PP over the 192 logical layers.
- Existing TP4 + FlashAttention3 is already faster than a 2/4-stage pipeline after stage compute, transfers, cache operations, and bubbles.
- Any prefill-only gain is too small relative to total E2E, or creates decode penalties.
- A new loop-KV transfer policy yields no gain when stage-local ownership already keeps each depth's KV local.

**Possible positive outcomes:**
- A physical **exact pipeline**, not just an oracle, reduces matched B1 long-prompt TTFT and **full E2E** versus optimized TP4, without worse accuracy; and
- A recurrent-specific scheduling or memory lifetime advantage survives comparison against the same generic PP execution graph and matched-cost baselines.

Winning only against an unoptimized serial single-GPU implementation or a generic unoptimized PP strawman is insufficient.

---

## 2. Novelty boundary — read before experiments

### 2.1 LoopSpec and WaveFront are already strong loop-aware decoding methods

- **LoopSpec**, arXiv:2609.17184 (2026), uses intermediate recurrence logits to draft *future* autoregressive output tokens and full-depth verification to preserve the desired output distribution. It is a **self-speculative decoding** algorithm; pipeline refers primarily to execution of speculative continuations and recurrent verification, not to a proven new multi-node depth/chunk prefill framework. Its public example starts a single-GPU SGLang-based server: <https://arxiv.org/abs/2609.17184>, <https://github.com/kaist-flexml-lab/loopspec>.
- **WaveFront Decoding**, arXiv:2609.23033 (2026), concurrently batches token-position/recurrence states for **speculative draft+verify** within the shared recurrent core. Reported primary Ouro/Huginn results use a single NVIDIA RTX A6000; the reported speedups are not physical multi-GPU TP/PP speedups: <https://arxiv.org/abs/2609.23033>, <https://github.com/summerbro-hhj/wavefront-decoding>.

Our primary test is **exact existing-prompt prefill without any draft, stale activation, tentative token, verification, or decoding trajectory modification**. This boundary must not be inflated into an automatic novelty claim.

### 2.2 Generic pipeline parallelism and chunked prefill are close prior work

Ordinary pipeline parallelism of an **explicit 192-layer network** can place layers 1–48 on GPU 4, 49–96 on GPU 5, etc., feeding sequential prompt microbatches. That is mathematically **the same computation dependency structure** as assigning Ouro loop 1–4 to four GPU stages with identical shared weights replicated per stage. The recurrent naming is not by itself a new parallelism axis.

Compare against generic:

- PP on logically unrolled 192 layers;
- chunked prefill with same causal attention semantics;
- TP/PP hybrid, including PP2×TP2 and PP4×TP1 if physically feasible;
- recurrent serving systems, e.g. <https://github.com/ThinkFlowLab/vllm-rlt>;
- distributed prefill KV layer-wise transfers and generic stage overlap, e.g. vLLM's disaggregated prefill documentation: <https://github.com/vllm-project/vllm/blob/main/docs/features/disagg_prefill.md>.

Include PipeFusion and DistriFusion as **image DiT analogies only**: their stale feature reuse is not exact causal AR prefill, and it cannot be copied into Ouro without correctness consequences: <https://github.com/PipeFusion/PipeFusion>.

### 2.3 KV-overlap hypothesis must be falsifiable

If each stage owns the complete KV for its assigned recurrent depth, ordinary attention reads **local KV**. No per-iteration remote KV fetch exists by default. A claim about hiding recurrent KV transfer must explicitly identify the true transfer (activation between stages, KV publishing to a second cache owner, cache migration, etc.), its producing event, its dependent consumer, and a valid overlap window. Do not count a zero-byte transfer or double-count materialized KV.

Compression (FlashLoop), residuals, stale features, speculative future tokens, dynamic early exit, and trained model modifications are outside primary scope.

---

## 3. Scope, hardware, and scientific controls

### 3.1 GPU authorization

Run GPU kernels only on physical GPUs **4,5,6,7**. Verify UUIDs, rank-to-device mapping, H100 80GB HBM3, and actual NV18/NVLink/NVSwitch connectivity. Never launch workers on 0–3. Other users' processes are never killed, paused, or modified. Project-specific burners on allowed GPUs may be scoped by explicit ownership only; restore as specified by the existing project execution policy.

### 3.2 Existing evidence / no repetition of known NO-GOs

Read before launch:

- `poc_looped_multigpu_kv_placement/report.md` — native Ouro TP1/2/4, KV ownership, 32K, throughput.
- `poc_looped_recurrence_tp_4models/report.md` — R×TP matrix; ordinary TP is adequate; native versus tied-unrolled is similar at 16K; retained-graph natural-output parity has an unresolved discrepancy.
- `poc_looped_recurrence_tp_4models/tied_unroll_contract.md` — math, loop and KV semantics.
- `poc_looped_recurrence_tp_4models/timing_contract.md` — valid timing boundaries and observer pitfalls.

Pin actual checkpoint and runtime revisions rather than relying on file names or previous claims. Do not overwrite or relabel earlier results.

### 3.3 Workloads

Start on frozen calibration, e.g. **8 prompts** spanning code/math/document and target lengths:

- `P=1,024` (short negative control),
- `P=8,192`,
- `P=16,384`,
- `P=32,704` (only if P + generated tokens <=32,768 and the engine has enough HBM).

Primary R=4, B=1, same 64 forced output tokens per request for **hardware-only** comparison; frozen input IDs. Use forced outputs for timing consistency, not accuracy. Separately check native greedy/EOS generation and validate exact candidate outputs. Stage C/D held-out: **24 new frozen requests** if candidate actually passes gates; otherwise not needed.

Record prefill engine envelope (TTFT-like), cached TPOT, end-to-end latency, accepted/generated output tokens, KV memory and all GPU allocations. Keep prefill versus decode separate and retain full-request comparison.

### 3.4 Principal performance baselines

**B0:** Native optimized Ouro TP4 (existing vLLM 0.26.0, BF16, FA3, CUDA Graph, full model R4; physical GPUs 4–7). Must be freshly paired in same session for any candidate claims.

**B1:** Native TP2, as available, and TP1 diagnostic for stage compute. Do **not** define a naive single-GPU run as the main multi-GPU baseline.

**B2:** Native TP4 chunked prefill (when supported) with comparable microbatch size and cache settings. The previous native campaign disabled chunking, so make a separate fair tuned native-chunk control before claiming superiority.

**B3:** Generic logically-unrolled PP2×TP2 and/or PP4; use the same exact 192-layer operator graph and schedule, with stage-local caches and actual inter-stage transfers. If expensive to build, use a **calibrated physical generic-PP cost model**, and mark the comparison *UNVALIDATED* rather than asserting it was beaten.

**B4:** The proposed depth-aware schedule only if it differs materially from B3. A diagram relabeling stages as loops is not a new method.

Optional single-GPU + DP controls are **throughput controls** and cannot be substituted for B1 latency comparisons.

---

## 3.5 Stage Q — **Mandatory checkpoint-quality and natural-generation validation** (NEW in v2)

### Q0. Audit published setup and distinguish three experimental protocols

Do this **before declaring any accuracy baseline valid**. Our earlier Ouro profiling PoCs used frozen 64-token *teacher-forced* continuations or short natural-EOS probes. Such tests are useful for hardware comparability but **cannot reproduce the model's reported reasoning accuracy**. This quality gate intentionally separates:

| Protocol | Purpose | Sampling and output | Can claim benchmark quality? |
|---|---|---|---|
| **H — hardware** | Exact prefill/TP/PP timing at matched shapes | Frozen pre-tokenized input plus 64 teacher-forced outputs; EOS ignored for the shape test | **No** |
| **D — deterministic correctness** | Verify mathematical equivalence of native vs chunked/tied/PP execution | Actual inference state, greedy decoding (`temperature=0`) and natural termination; separately inspect hidden/logits/KV | Can claim parity for tested cases, **not a published benchmark score** |
| **Q — natural quality** | Reproduce meaningful native-model quality and check candidate noninferiority | **Real generated tokens**, official chat template, appropriate reasoning output budget, true EOS, published-compatible sampling and a public grader | **Yes, within the documented evaluator and sample scope** |

Never mix H, D, and Q results into one accuracy or speedup metric. Do not count truncated outputs as correct completions or translate forced-token parity into benchmark accuracy.

**Official public-reference contract:**

- Checkpoint: `ByteDance/Ouro-2.6B-Thinking`, use the installed, already-tested revision `f1edd81e7ac41355db670500ceaf204e0f73af68` unless it fails a documented consistency audit. Save the exact tokenizer/checkpoint hashes and any remote-code revision.
- **Native baseline recurrent steps: `total_ut_steps=4`; `early_exit_threshold=1.0`** and verify that native vLLM runs all four passes. Alternative R=1/2 are **hardware sensitivity experiments**; they are not R4 quality-equivalent models.
- **Published Thinking-model evaluation uses `temperature=1.0`, `top_p=0.7` unless noted otherwise.** The official model card illustrates the same settings with an *unforced*, reasoning-length generation using `tokenizer.apply_chat_template(..., add_generation_prompt=True)`. Verify the *actual* checkpoint tokenizer's role delimiters/special tokens, any thinking tags, and stop/EOS IDs. Avoid inserting ad hoc assistant prefixes or sharing Qwen templates.
- The Looped LM paper's Ouro-2.6B-Thinking R4 results include **AIME 2024 pass@1 64.7%, AIME 2025 pass@1 50.3%, OlympiadBench ~76.44%, and GPQA ~52.7%**. These are **historical author-reported reference values** obtained with an **in-house prompt/harness and LLM-as-judge rubric**. They are *not* accuracy thresholds automatically expected under our public exact-match evaluation. Do not claim reproduction unless prompt, sample count/seeds, output length, parsing, evaluator, and judge policy truly match.
- Check paper/model-card discrepancy: public cards have described a smaller layer count, whereas our pinned physical weights/loader demonstrate **48 stored layers × 4 passes**. Inspect current pinned `config.json`, actual modules and paper/version. Report mismatch as a provenance limitation; follow runtime-verified 48×4 semantics without silently altering weights, recurrent depth or claimed reference setup.
- Model card recommends `transformers==4.54.1` for reference HF custom code; previous native experiments used vLLM 0.26.0 with newer packaged Transformers. **Do not downgrade the validated shared runtime** to match this recommendation. If needed, use an isolated compatible HF reference environment and compare outputs/semantics; record any incompatibility instead of assuming execution equivalence.

Official sources to pin in `quality_protocol.md`:

1. <https://huggingface.co/ByteDance/Ouro-2.6B-Thinking> (model-card inference example / recurrent steps).
2. <https://ouro-llm.github.io/> (authors' published scores and model family).
3. <https://arxiv.org/abs/2510.25741> (paper, reasoning evaluation protocol, depth ablations).
4. <https://github.com/ByteDance/Ouro> (if official evaluation scripts are accessible, prefer them).

If the full original judge rubric or source prompts are inaccessible, label published-score comparisons **`REFERENCE_PROTOCOL_NOT_IDENTICAL`**. This does not imply that Ouro is inaccurate. Instead establish a separately documented objective public-benchmark baseline.

### Q1. Make an actual-quality baseline before claiming pipeline validity

**Primary public score:** AIME 2024, 30 problems, via official/public question sources with frozen revisions, answer normalization and **deterministic exact-answer checking** where a definitive numeric/math expression can be extracted. The benchmark's ground truth must be frozen *before* reading generations. Use the checkpoint's chat template, natural output, reasoning SFT format, EOS and a sufficiently large `max_new_tokens` (start 4096; if many correct-looking solutions hit the cap, test 8192 as allowed by available context/memory). No artificial 64-token limit, forced tokens or manually supplied final answers.

**Sampling lanes:**

- `Q-published-style`: `R=4`, `temperature=1.0`, `top_p=0.7`, independent seeded samples. Start with three reproducible seeds on the full AIME24 30-problem set (90 generations); if outputs terminate and resources permit, extend to ten seeds (300 generations) for a more stable reported mean. Report **mean single-sample exact-answer accuracy**, seed variation and completion/cap rates. **Do not identify it with the paper's `pass@1` unless its definition and judge match.**
- `Q-greedy`: same 30 problems, `R=4`, `temperature=0`, natural generation. This is an **extra deterministic correctness and ablation baseline**, not a recreation of the published stochastic score.
- `Q-secondary`: choose one objective larger set (prefer **MATH-500** for math reasoning; optional **GSM8K** for inexpensive automated scoring). Freeze either 100 stratified instances as a discovery sample or the full 500/1319 when practical. Match prompts/answer extraction across native and candidate, record exact pass count, parse success and token cap. This secondary score is **new evidence**, not automatically an Ouro Thinking paper reference. For code tasks, optionally use HumanEval+/EvalPlus in an isolated resource-limited sandbox, never unsandboxed generated programs.

AIME2024 may be too small for high-precision statistical noninferiority, and output stochasticity can dominate minor changes. Do not claim R4 is "bad" from a small 3-seed deviation from the paper's in-house score. When exact answers are ambiguous, preserve raw output and mark `UNPARSEABLE`; do not use an unverified LLM judge as a deterministic ground truth.

Record per problem/seed: frozen problem ID and source revision, exact prompt/token length/hash, model+tokenizer revisions, chat template, sampling args, RNG seed and rank handling, EOS/stop reason, `max_new_tokens`, actual output length, full generated tokens, parsed answer, correctness, response latency, total position count, and whether any answer was manually reviewed. For score comparisons, use **problem-level** paired bootstrap or stratification (multiple seeds of one problem are not independent problems). Report exact numerator/denominator and confidence interval. Do not omit failures or capped generations.

### Q2. Independent implementation / engine sanity

- Run several fixed public problems through (a) the native Ouro vLLM TP1 or TP4 baseline and (b) the authors' official HF implementation **if a compatible isolated environment exists**; preserve `R=4`, weights, tokenizer and input tokens. Test a few single-forward full logits, selected layer/loop KV writes, greedy top tokens and naturally generated continuations. Kernel order may make BF16 outputs not bitwise identical across engines; use predeclared tolerances and separate token parity. Do not label stochastic samples a mismatch solely because they differ.
- Test **within-config seed/run reproducibility** before interpreting candidate changes. The previous tied-unroll study found a retained-graph natural-output mismatch on one configuration, later absent under ordinary graph policy. The baseline gate requires an audited ordinary-graph configuration and preserved logs. Treat the unresolved diagnostic as a real limitation.
- Resolve actual model vs model-card layer count discrepancy in an explicit table (`model_architecture_vs_reference.md`). Do not "fix" a model by changing `num_hidden_layers`, weights or cache policy to obtain a score.

### Q3. Stage-Q pass/fail and experiment routing

- **`QUALITY_BASELINE_PASS`:** A genuine R4 natural-generation evaluation finishes with trustworthy public scoring, acceptable completion rate, and a verified reproducible model/config. A large paper-score mismatch should trigger a configuration/source/evaluator investigation but does not by itself fail if protocols differ.
- **`QUALITY_BASELINE_WEAK`:** outputs are heavily truncated, prompts malformed, sampling differs unintentionally, scorer is not trustworthy, or run-to-run behavior is not controlled. Fix evaluation conditions before treating this as the model's representative accuracy. Stage A's *hardware-only* dependency profiling may proceed when numerically valid, but no quality-safe framework claim can be promoted.
- **`OFFICIAL_PROTOCOL_UNAVAILABLE`:** full authors' in-house judge/prompt recipe unavailable. Preserve the objective public-score baseline and label reference comparisons non-equivalent. This is expected and does not automatically stop the physics characterization.
- **`MODEL_REFERENCE_MISMATCH`:** checkpoint or inference semantics are actually inconsistent with the pinned official model and cannot be repaired without unvalidated changes. Stop method promotion and document exact blocking difference.

**No Stage C/D full-model speedup claim may be accepted unless the R4 native quality baseline is validated and the candidate's separate Quality Gate (below) passes.** Short hardware traces alone never certify the release checkpoint's performance.

### Q4. Candidate correctness and accuracy, if Stage C becomes warranted

**Tier 1 — component functional correctness:** Match native and pipeline at fixed R4 on the same prompt-token IDs, causal masks, RoPE positions, stage boundary norms/residuals, K/V slots and greedy next-token logits. Freeze numerical tolerances *before* measuring. For equal operation order and a matched CUDA graph, require bitwise when possible; otherwise report max absolute/relative error and rank-wise precision explicitly. Any divergent KV ownership/position must be resolved before moving on.

**Tier 2 — complete natural generation:** Evaluate with EOS enabled, long output budget and without forcing IDs. Compare greedy native/pipeline **generated IDs, final answer, EOS/stop reason, step count and complete outputs**, initially on at least 24 diverse frozen validation requests; include multi-chunk, long prefix, and previously troublesome retained-graph cases. If bytewise sequences differ, identify the first divergent forward, per-loop hidden/logit/KV cause and whether the sampler received equivalent distributions. Do not declare equivalence on the basis of four-token or 64-token probes.

**Tier 3 — objective quality and matched sampling:** Rerun the same Q1 benchmarks with unchanged dataset, prompts, R4, tokenizer, seeds, EOS, temperature/top-p and answer parser on native **and** pipeline. A stochastic sample from an exact, matched-logit implementation can diverge due to RNG scheduling; report **distributional and per-seed quality** instead of automatically requiring every sampled string to be identical, and audit RNG streams. Use paired problem-level confidence intervals and report accuracy/length/truncation. Pre-register a practically acceptable score difference (e.g. <=1–2 percentage points on a sufficiently large benchmark) and do not declare noninferiority when its confidence interval is too wide. On AIME24 (30 questions), noninferiority at a 1–2pp margin cannot be statistically established with only a few seeds; describe results as preliminary or augment with a larger objective set.

**Quality STOP:** If the candidate changes causal attention semantics, layer or recurrent depth, cache correctness, or significant accuracy, return `CORRECTNESS_BLOCKED` or `QUALITY_REGRESSION` instead of GO, even if it is faster. Keep R1/R2 **strictly diagnostic** (do not substitute their lower latency for valid R4 inference).

---

## 4. Stage 0 — Runtime and exact semantic audit

1. Verify pretrained Ouro R4, 48 physical layers, 192 distinct KV positions/layer slots, recurrent pass-boundary normalization/reset, RoPE, attention mask, causal context.
2. Verify per-stage exact hidden-state size, dtype, per-layer KV shapes and head/sequence ownership; four GPUs and appropriate topology.
3. Check whether CUDA Graph and compile already unroll native recurrence into an ordered GPU graph; earlier study suggests they do. Do **not** claim native Python-loop launch overhead without measured new evidence.
4. Verify model in TP4 B1 as the current physical reference; run mandatory Stage Q for published-setting R4 natural output and objective benchmark readiness before treating model quality as established; reproduce roughly comparable ranges, but always use same-session B0 for new candidate ratios.
5. Inspect implementation support for PP, stage groups, chunked prefill, distributed KV publication, and exact attention masks. Avoid porting an entirely new engine solely to get a performance number.
6. Record representative memory footprints. PP4 stage placement may replicate all 48 shared-layer weights across four GPUs, whereas TP4 divides weights; quantify actual HBM and graph workspace, not just theoretical weights.
7. Review primary prior and write `prior_overlap.md`, with explicit generic PP equivalence objection.

**Failure gate:** if the model's actual prefill has dependencies beyond those assumed, stop and amend the DAG; never use stale KV to hide invalid dependencies.

---

## 5. Stage A — Build a correct dependency graph, NOT a cartoon

Let `r ∈ {1,...,4}` represent recurrent pass, `j ∈ {1,...,M}` consecutive prompt chunks. Each chunk comprises a contiguous range of final input token positions. Let `F[r,j]` denote the complete recurrent-stack computation for chunk `j` in pass `r`, **including all 48 attention/MLP layers and native pass-boundary operations**. Model this macro-graph first, then validate against layer-level dependencies.

Potential dependency edges (verify in actual Ouro):

- `F[r-1,j] -> F[r,j]`: previous recurrent depth supplies chunk hidden state.
- `F[r,j-1] -> F[r,j]`: causal KV for earlier chunks at the **same recurrent depth**, across **all 48 layers**, must be committed before computing the next chunk.
- More fine-grained edges across `(r,j,l)` for `l=1..48` may permit streaming but are not free; model residuals, layer-by-layer KV availability, and cross-rank stages explicitly.

**Example of an exact schedule (conditional on verified edges):**

| Slot | Stage 1 / GPU 4 | Stage 2 / GPU 5 | Stage 3 / GPU 6 | Stage 4 / GPU 7 |
|---|---|---|---|---|
| 1 | (r1,c1) | idle | idle | idle |
| 2 | (r1,c2) | (r2,c1) | idle | idle |
| 3 | (r1,c3) | (r2,c2) | (r3,c1) | idle |
| 4 | (r1,c4) | (r2,c3) | (r3,c2) | (r4,c1) |
| ... | ... | ... | ... | ... |

**Do not conflate this with four-way parallelism for a single generated token.** It exploits multiple known prompt chunks; ordinary autoregressive decode of one future token has no known next chunk.

Required tests:

- Construct a topological order and verify every task's input hidden and all required per-layer same-depth KV are available.
- Implement a small CPU or single-GPU **reference scheduler/emulator** that executes valid tasks in DAG order, with exact masks and full recurrent output, to validate numerical semantics before any distributed implementation.
- Compare chunked exact prefill against native full prefill on frozen IDs: logits, hidden vectors, attention outputs at representative positions, written KV indices; state precision and *frozen* numerical error bounds. Exact bitwise parity is preferred when operator shape and kernel order match, but chunked FlashAttention may cause legitimate low-order BF16 differences: accept only predeclared narrow tolerance plus matching deterministic greedy outputs/NFE; report failures, never silently relax.
- Check **positional encoding**, causal mask across chunk boundaries, padding, RoPE position offsets, stage-local attention cache, and pass-boundary resets.
- Compare the graph to **generic PP on explicit 192-layer tied-unrolled execution**. If the graph and legal stage schedules are isomorphic, label `GENERIC_PP_EQUIVALENT` unless an independent recurrent-specific advantage exists.

**Gate A:** A valid exact DAG and a quantitative distinction (if any) from ordinary PP. If no independent loop-specific dimension exists, continue only as a capped generic-PP benchmark if it could yield useful performance; no loop-specific GO.

---

## 6. Stage B — Physical stage/transfer cost and dependency-constrained scheduling

This stage is **physical**, not zero-cost simulation. It can be run without a full multi-GPU Ouro serving engine.

### 6.1 Measure native reference components

At each P, trace clean native B0 and B1 to obtain:

- one-pass recurrent compute cost and split by `chunk`, `layer family`, `attention`/`MLP`, and cache writes;
- GPU service time of TP reductions vs observed completion time;
- prefill versus cached decode;
- actual KV memory per `(depth,layer,head,position)`;
- B1 graph shape, compilation behavior, CUDA graph replay, host gaps, and hardware occupancy.

Avoid profilers that perturb the measurement >10% for quantitative stage attribution. Clean `LLM.generate` is the headline E2E measurement; CUDA kernel service sums are not E2E percentages.

### 6.2 One-stage execution kernels

For hypothetical depth-stage pipelines, derive realistic times by executing one **full 48-layer recurrent pass on one GPU** for a causal chunk with correct same-pass prefix KV of the required length. Use actual pretrained weights, native precision and real attention backend, not synthetic arithmetic times alone. Also measure TP2 per-stage pass for candidate PP2×TP2.

Chunk sizes:

- 256, 512, 1,024, 2,048, 4,096 (skip invalid/capacity cases);
- `M = ceil(P/chunk_size)` actual chunks, including tail sizes.

Stage costs depend on accumulated context length and will not be constant across chunks; measure short/middle/late positions or directly all chunks when cheap. Estimate true per-task times with explicit extrapolation bounds if not all measured.

### 6.3 Stage-boundary physical communication

Use the actual GPU pairs and shape `[chunk_rows, hidden_dim=2048]` BF16 to benchmark:

- communication of chunk hidden state from stage `r` to `r+1`;
- size-dependent P2P send/recv, NCCL, optional packing/copy/unpacking;
- inter-stage readiness, CUDA-event handoff/synchronization, buffer ownership;
- any **real** KV cache publishing operations only if the candidate layout needs them.

Measure the **whole path including synchronization** and distinguish bytes from elapsed time. Do not claim overlap if the consumer must wait for an unproduced current chunk hidden state. Valid overlap is work on **different chunks** or transfer of an already-produced chunk during useful independent compute.

### 6.4 Calibrated physical scheduler

Build a DAG scheduler using:

- measured stage compute distributions depending on P and chunk index;
- measured inter-GPU transfers and completion dependencies;
- capacity/HBM feasibility on each stage;
- stream contention and resource constraints;
- startup and drain bubbles;
- exact attention cache barriers;
- plausible CUDA graph capture/recapture overhead and kernel efficiency effects;
- TP4 native measured input/output envelope.

Create at least three estimated schedules:

- **S0:** native TP4 full prompt execution;
- **S1:** generic PP2×TP2 or PP4 exact causal chunking;
- **S2:** if different, a recurrent-specific schedule or interstage communication-overlap variant.

**Crucial:** PP4 has one physical GPU per 48-layer pass; it does *not* have TP4 model-parallel acceleration within each pass. PP2×TP2 has two 2-GPU stages. Charge replicated vs sharded weights correctly and do not credit perfect utilization of all four devices before pipeline fill.

Make optimistic zero-transfer schedules **only diagnostic upper bounds**, labeled `ORACLE_NOT_ACHIEVABLE`. Primary promotion requires realistic end-to-end estimates and physically measured transfer/compute pieces.

### 6.5 Headroom test

Evaluate both:

- `prefill_reduction = (TTFT_native - TTFT_candidate) / TTFT_native`;
- `E2E_reduction = (E2E_native - E2E_candidate) / E2E_native`, including native decode and output-finalization for 64 tokens.

For each long context, show best/worst scenarios and paired uncertainty.

Suggested Stage B gates:

1. **Physical pipeline feasibility:** all exact dependencies satisfied; usable HBM and transfers.
2. **Prefill opportunity:** realistically projected >=20% prefill improvement versus optimized TP4 after costs.
3. **E2E opportunity:** realistically projected >=10% B1 E2E reduction at P>=16K (ideally >=15% at 32K) and not offset by decode degradation.
4. **Novelty:** opportunity not exhausted by generic PP/chunked prefill; otherwise classify generic and do not promote a loop-specific method.
5. **Uncertainty:** the lower-confidence scenario should not already exclude a meaningful improvement.

Failure results are useful and should end the PoC cheaply. Do not rescue the hypothesis by inventing a 4-GPU architecture with zero-cost KV movement.

---

## 7. Stage C — Conditional exact physical prototype

Proceed **only if** Stage A/B gates leave a credible improvement and no fatal prior-equivalence barrier.

### C0: minimal 2-stage validation first

- Test two-stage PP with TP2 per stage (`GPUs 4,5` and `6,7`) or two single-GPU stages (`GPU4` / `GPU5`) if that is the actual measured candidate. Choose the layout indicated by Stage B, not a convenient untested configuration.
- Preserve 48R logical layer calls, pretrained BF16 weights and exact native R4 outputs.
- Stage-local KV belongs to the corresponding depth; do not transmit it unnecessarily.
- Implement chunk hidden-state transfer with correct asynchronous event/stream dependencies.
- Explicitly audit chunk arrival order, output ownership and all 192 KV slots for the full input.
- Check no inactive GPU 0–3 is used.

### C1: four-stage extension if C0 works

- Map physical GPUs 4,5,6,7 each to one recurrent pass (or a justified alternative).
- Measure native TP4, generic PP4 and candidate under **same prompt length and timing conditions**.
- If candidate is identical to generic PP4 and has no recurrence-specific advantage, label result `GENERIC_PP_ONLY` even if faster.

### C2: decode/final output

Because a single autoregressively generated token must traverse recurrent depths in sequence, the four pipeline stages cannot all compute different *real* future tokens without speculation. Use a correctness-preserving native decode path, and charge actual KV layout transition if a prefill/decode transition occurs. If decoding reuses the pipeline layout, charge the sequential underutilized path. **Never treat prefill-only speedup as full E2E gain.**

### C3: parity and quality

- **Follow Stage Q4's three tiers in full**: component/KV/logit correctness, complete greedy natural generation and a paired objective-quality benchmark against validated native R4.
- Same P, R=4, TP/PP mathematical semantics, native attention mask, RoPE, residuals, pass norms, KV cache policy, BF16, tokenizer, chat template and stop conditions.
- Verify that the final natural generated response is not clipped by the 64-token hardware benchmark. Use sufficient output budget and natural EOS, and log all caps.
- On stochastic official-style settings, `temperature=1.0, top_p=0.7`; compare score distributions and RNG/protocol details rather than claiming sample IDs must be identical across different random streams. Also retain a separate greedy parity lane.
- Compare native and candidate on **the same public AIME24 and larger MATH-500/GSM8K subset**, with per-problem score and valid CI. A 30-problem AIME subset alone cannot establish strict 1–2pp noninferiority.
- If any candidate numerical discrepancy is not explained under frozen tolerances, `CORRECTNESS_BLOCKED`; if quality deteriorates beyond predeclared tolerance, `QUALITY_REGRESSION`; do not promote latency gain alone.
- Preserve R1/R2 only as physics diagnostics; no claims of accuracy equivalence to pretrained R4.

---

## 8. Quantitative reports and plots

Required tables:

**T1 — Baseline**
`P | R | TP | B1 TTFT | Cached TPOT | E2E | HBM/rank | KV/rank | Graph policy`

**T2 — Exact scheduling**
`P | chunk B | #chunks | legal tasks | critical path | stage idle/bubble | total task compute | interstage bytes | memory budget`

**T3 — Stage economics**
`B | #cache keys | 48-layer pass time on TP1 / TP2 | activation transfer ms | KV publish ms (if any) | merge/layout ms | sync wait ms`

**T4 — Candidate comparison**
`P | B0 TP4 | B2 native chunk TP4 | B3 generic PP | B4 candidate | predicted/observed TTFT | predicted/observed E2E | quality parity | verdict`

**T5 — Published-setting R4 native quality**
`Benchmark | N problems | seeds | checkpoint/R | chat template | temp/top_p | cap | EOS% | parse% | exact numerator/denominator | score/CI | published reference | protocol comparability`

**T6 — Quality/correctness vs native**
`Configuration | Greedy full-output parity | logits/KV parity | natural EOS/completion | AIME24 score | larger objective score | paired score delta/CI | result`

Plots:

1. Exact DAG of `chunk × recurrent depth`, including **causal prefix-KV edges**, not just vertical depth edges.
2. Four-GPU resource-limited Gantt chart with actual stage costs, transfers and bubbles.
3. Context/chunk-size sweep: estimated and measured TTFT/E2E; distinguish oracle and realistic.
4. Pipeline stage utilization; service+wait without double-counting.
5. Weight / loop-indexed KV / communication workspace per GPU versus native TP4.
6. Sensitivity to generated-token budget: 1/64/256 output tokens, with speculative decoding disabled.
7. `generic PP vs any genuinely new recurrent-specific control` to establish novelty boundary.

All plotted values must carry provenance (`MEASURED_CLEAN`, `MEASURED_COMPONENT`, `CALIBRATED`, `ORACLE`, `UNVALIDATED`).

---

## 9. Statistics, reproducibility, and evidence quality

- Use paired frozen requests as independent statistical units; technical repeats and chunks are not independent requests.
- At least 3 warmups and 3 clean repeats per condition when physically measured; request-level >=2,000 bootstrap draws for candidate improvements (8 calibrations are insufficient for broad inference).
- Resolve different device timing clocks before calling cross-GPU intervals a critical path; otherwise report task DAG bound and local-rank intervals only.
- Heavy profiling must be separate from clean E2E; measure observer overhead and disallow quantitative attribution when overhead >10%.
- Keep every actual CUDA graph replay, kernel occurrence, buffer-copy and collective count; do not multiply a last-replay operator time by NFE and claim it was directly observed.
- Compare against vLLM's existing optimized FlashAttention3, TP and chunking, not eager PyTorch.
- Preserve previous dirty/staged changes and all unrelated paths. Avoid global `git reset --hard`, `git clean`, or broad process killing.
- Keep pinned checkpoint, exact commands, environment, GPU mapping, original baseline, profiling validity, raw tensors/hashes, and failed runs.
- The previous tied-unroll PoC recorded a retained-graph natural-output mismatch. **Do not assume exact output parity simply because same logical layer tape is used.** Restore graph policy parity and report this confound.

---

## 10. Explicit decisions

Possible verdict labels:

- **`DEPENDENCY_BLOCKED`** — proposed legal overlap impossible under actual causal/recurrent KV dependencies.
- **`GENERIC_PP_EQUIVALENT`** — legal pipeline is just ordinary PP over the unrolled model, no recurrent-specific work saved.
- **`NO_PHYSICAL_CROSSOVER`** — legal but after real transfers, stage compute, bubbles, and decode it does not beat optimized TP4.
- **`PREFILL_ONLY`** — prefill improves but E2E opportunity too small or decode wipes out gains.
- **`CORRECTNESS_BLOCKED`** — exact output/cache semantics not established.
- **`QUALITY_BASELINE_WEAK`** — official/pretrained R4 accuracy baseline still unvalidated due to run protocol, truncation, or scoring issues.
- **`MODEL_REFERENCE_MISMATCH`** — validated native model differs materially from intended checkpoint/settings.
- **`QUALITY_REGRESSION`** — candidate materially reduces benchmark quality, even if physical E2E is faster.
- **`RUNTIME_BLOCKED`** — runtime capability cannot be validated without a major unapproved engine rewrite.
- **`TIMING_INVALID`** — timing methodology cannot separate artifact from real improvement.
- **`GENERIC_PP_ONLY`** — physically beneficial but explained entirely by ordinary PP; useful engineering, not claimed looped novelty.
- **`CONDITIONAL_LOOP_SPECIFIC`** — plausible recurrent-only advantage remains; no validated full-model gain yet.
- **`GO_LOOP_SPECIFIC`** — novel recurrent-specific mechanism, validated exact full-model E2E/quality gain versus optimized TP4 **and** generic PP.

Strongest final scientific decision rule:

> **A legal multi-GPU pipeline is not novel merely because we label its stages “recurrent loops.”**
> Prior work already provides chunked PP, and the 4-loop fixed Ouro forward is an explicit 192-layer graph. Continue only if a **physical, recurrent-specific gain beyond that generic graph** survives matched controls and full E2E accounting.

---

## 11. Required repository deliverables

```text
poc_looped_exact_depth_chunk_pipeline/
├── report.md
├── reproduce.md
├── dependency_contract.md
├── prior_overlap.md
├── correctness_contract.md
├── quality_protocol.md                  # NEW: official-vs-public evaluation contract
├── model_architecture_vs_reference.md   # NEW: official card vs pinned loaded architecture
├── timing_contract.md
├── stage_memory_ownership.md
├── scripts/
│   ├── inspect_ouro_semantics.py
│   ├── build_exact_dag.py
│   ├── check_causal_kv_dependencies.py
│   ├── measure_stage_compute.py
│   ├── measure_stage_transfers.py
│   ├── schedule_cost_model.py
│   ├── run_native_baselines.py
│   ├── run_official_style_quality.py      # NEW
│   ├── score_public_reasoning.py         # NEW
│   ├── audit_hf_vllm_parity.py           # NEW (optional)
│   └── optional_exact_pipeline.py
├── results/
│   ├── environment.json
│   ├── frozen_workload.json
│   ├── dag_tasks.csv
│   ├── measured_stage_costs.csv
│   ├── measured_transfer_costs.csv
│   ├── schedule_bounds.json
│   ├── baseline_requests.csv
│   ├── verdict.json
│   ├── native_quality_r4.json            # NEW
│   └── candidate_quality_r4.json         # NEW conditional
└── figures/
    ├── exact_depth_chunk_dag.png
    ├── four_gpu_gantt.png
    ├── chunk_size_sensitivity.png
    ├── predicted_prefill_e2e.png
    └── memory_ownership.png
```

Do not create fictitious paths: write only the actually produced artifacts and list missing conditional deliverables as `NOT_RUN` in the final report.

Commit/push only the new PoC files, using `flashvep/looped-exact-depth-chunk-poc` if branch policy permits. Return exact commit SHA, branch, report path and verdict, with the *measured* optimization headroom and comparison to generic PP, not an unsupported proposed speedup.

---

## 12. Required 18-question report checklist

1. What are the exact pretrained model and runtime revisions, GPUs and topology?
2. What exact causal/prefix-KV/recurrent dependencies constrain chunk overlap?
3. Is the `(r,chunk)` pipeline identical to generic 192-layer PP?
4. Which parts, if any, use weight sharing to improve communication/compute **beyond** generic PP?
5. Can R4 prefill be executed exactly from streaming chunks and preserved KV indices?
6. How many physically runnable tasks are available at each schedule time?
7. What are the per-chunk, per-depth stage compute durations on one/two GPUs?
8. How much activation data crosses stages, and how long does it take physically?
9. Is any KV actually transferred between ranks, or is it stage-local?
10. Where can communication **legally** overlap useful independent computation?
11. How much of the four-GPU schedule is startup/drain bubble?
12. Does all-layer causal attention force serialization not shown by the macro DAG?
13. Does a calibrated schedule actually beat native optimized TP4 with chunked prefill?
14. What is its best physical E2E headroom at 16K and 32K, including native decode?
15. Does real stage HBM/weight replication fit without stealing KV capacity?
16. Did native R4 complete the public official-style reasoning quality evaluation, and are published scores protocol-comparable? Did candidate exact natural generation and KV/logit correctness and paired benchmark quality pass?
17. Do any savings survive generic PP and prior-work baselines?
18. Is the final conclusion GO/CONDITIONAL/NO-GO (including QUALITY_* or MODEL_REFERENCE_MISMATCH as appropriate), and what one follow-up, if any, is justified?
