# Ouro Reasoning Fidelity & Real-Workload Multi-GPU Profiling
## Authoritative Discovery PoC Specification — v1.0

**Project:** MLLM-EP / Looped-Transformer discovery  
**Date:** 2026-10-16 (Asia/Seoul)  
**Primary target:** `ByteDance/Ouro-2.6B-Thinking`  
**Checkpoint revision:** `f1edd81e7ac41355db670500ceaf204e0f73af68`  
**Permitted physical GPUs:** **4, 5, 6, 7 ONLY** (four NVIDIA H100 80 GB; previously NV18/NVSwitch-connected)  
**Work type:** faithful model evaluation, debugging/validation, and physical multi-GPU characterization; **not** a new method implementation by default.

> **Research questions.** (Q1) Can we establish a trustworthy, reproducible Ouro R4 reasoning-quality baseline and explain its discrepancy from the authors' reported AIME2024 pass@1 64.7? (Q2) Under that validated *actual long-reasoning* workload, what are the real latency, KV-capacity, and four-GPU serving tradeoffs of TP1/TP2/TP4 versus TP×DP, and is there an empirically identifiable, non-generic bottleneck worth a future method PoC?

## 0. Essential evidence and scientific constraints

### 0.1 Facts versus unresolved assumptions

- The authors report **Ouro-2.6B-Thinking R4 AIME2024 pass@1 64.7% and pass@10 90.0%** in *Scaling Latent Reasoning via Looped Language Models* (Table 9), using **temperature 1.0, top-p 0.7**, an **in-house evaluation harness** and **LLM-as-judge**. The public paper does **not** establish that the exact AIME prompt, grader, number of samples, every sampling filter and output cap are reproducible from available artifacts. **Do not represent a public exact-match evaluation as an exact author reproduction.** The paper also mentions `max_new_tokens=8192` for an Ouro-Thinking safety experiment; **do not assume that statement defines the AIME paper protocol.**
- The pinned Hugging Face repository carries a merged 2026 tokenizer/chat-template fix. The template supports `enable_thinking=True`, appending `<think>` at the assistant-generation boundary; by default this is not injected. The previous PoC quality script called `apply_chat_template(..., add_generation_prompt=True)` **without explicitly enabling thinking**. This is a **high-priority diagnostic hypothesis, not a proven cause of the accuracy gap**. Verify against the **exact pinned** `tokenizer_config.json`, current installed tokenizer behavior, the decoded prompt bytes and token IDs. Avoid relying on an unpinned `main` revision.
- The model card recommends `transformers==4.54.1`, R4 and `early_exit_threshold=1.0`. It lists 24 layers, while the actual pinned loaded model in earlier physical audits had **48 physical layers × four iterations**, **192 KV cache slots**. Preserve the measured architecture and document the inconsistency rather than editing it to fit text in a card.
- Earlier public evaluation at `max_new_tokens=8192`: AIME2024 sampling **39/90 completed correct = 43.33%**, with **48/90 cap-terminated**; greedy **11/30 = 36.67%**, with **18/30 capped**; frozen MATH-500 100-subset greedy **85/100 = 85.0%**, 13/100 capped. This was `QUALITY_BASELINE_WEAK`, not evidence that the pretrained checkpoint is intrinsically poor. AIME completion-conditioned accuracy is **selection-biased** and not pass@1.
- Earlier comparisons observed **12/100 MATH prefixes diverging** when only the output-token cap changed, and a native-vs-tied **retained-graph** natural-output discrepancy in another PoC. Quality reproducibility and numerical equivalence require explicit checks; do not claim perfect bitwise parity across TP degrees, backends, or different cap/graph settings.
- Earlier one-shot workload was chiefly **16K/32K prompt + 64 forced outputs**. This is **not representative** of many AIME/MATH reasoning tasks, whose prompt is relatively short and completion can be thousands of tokens. Never use forced outputs to report accuracy.

### 0.2 Authoritative sources (read and archive provenance)

1. Ouro paper, Section 5.2, Table 9 and Appendix C: <https://arxiv.org/html/2510.25741v2>
2. Authors' project/reported benchmark values: <https://ouro-llm.github.io/>
3. Pinned Ouro model card: <https://huggingface.co/ByteDance/Ouro-2.6B-Thinking/blob/f1edd81e7ac41355db670500ceaf204e0f73af68/README.md>
4. Pinned tokenizer and config: `https://huggingface.co/ByteDance/Ouro-2.6B-Thinking/tree/f1edd81e7ac41355db670500ceaf204e0f73af68`
5. Merged tokenizer fix / `enable_thinking` discussion: <https://huggingface.co/ByteDance/Ouro-2.6B-Thinking/discussions/7>
6. Previous validated user experiments, **read these before changing anything**:
   - `poc_looped_multigpu_kv_placement/report.md`
   - `poc_looped_recurrence_tp_4models/report.md`
   - `poc_looped_exact_depth_chunk_pipeline/report.md`
   - `poc_looped_exact_depth_chunk_pipeline/quality_protocol.md`
   - `poc_looped_exact_depth_chunk_pipeline/scripts/run_official_style_quality.py`
   - `poc_looped_exact_depth_chunk_pipeline/results/native_quality_r4.json`

These references are baseline context, not independently reproduced facts in the new run. Retain exact raw revision/file hashes and note unavailable author evaluation instructions. Do not silently substitute a reimplementation or unpinned checkpoint.

## 1. Scope, freeze rules and hardware safety

### 1.1 Nonnegotiable baseline

- Checkpoint exactly pinned above; **no retraining, LoRA, weight edits, hidden precision reduction or benchmark-answer-specific tuning**.
- Recurrent steps `total_ut_steps=4` and `early_exit_threshold=1.0` (fixed full-depth R4). A later R3 study may be *separate research* and must not masquerade as R4 reproduction.
- Native vLLM installation carried from the previous PoCs: approximately `vLLM 0.26.0+cu129`, `torch 2.11.0+cu129`, `Transformers 5.16.1`; audit exact installed versions. Preserve BF16, FA3, default/ordinary CUDA Graph policy. The vLLM model card says adaptive exit is not supported: verify every requested forward actually executes R4.
- Separate isolated official-style HF baseline environment with **pinned Transformers 4.54.1** as recommended by model card (or the closest exactly functioning reference with an explicit version-drift note). Do not change the working vLLM environment to satisfy HF reference dependencies.
- Physical GPUs **4, 5, 6, 7 only**. TP1 uses GPU4; TP2 GPUs4,5; TP4 GPUs4,5,6,7. Four TP1 replicas use GPUs4/5/6/7; two TP2 replicas use pairs 4–5 and 6–7. Confirm actual UUID/order at load. No TP8 runs or extrapolations.
- Only stop **identified project-owned burners** on assigned GPUs; protect all other users and their processes including GPUs0–3. Restore project burners after success, failure, timeout. Preserve dirty/staged files, unrelated branch changes and historical experiment directories; no `git clean/reset` of unrelated state.
- Keep total sequence length **<=32,768** for primary comparisons (SFT context and proven serving cap) even though config may advertise 65,536. A larger context is an *independently validated optional* experiment, never assumed.
- Do not claim to have externally matched the original paper's closed-source evaluator; state `REFERENCE_PROTOCOL_NOT_IDENTICAL` unless original author materials establish equivalence.

### 1.2 Freeze before outcomes

Before scoring any new full-benchmark output, freeze: exact model/tokenizer hashes; benchmark rows and order; prompt templates; whether an explicit instruction suffix is present; generation semantics; `enable_thinking` arms; sampling parameters and seed stream; model context cap; EOS/stop handling; requested output caps; answer parser; grading logic; comparison protocol; and all reporting denominators. Save a machine-readable `protocol_lock.json` and SHA256 hashes. Predeclare small *diagnostic* prompt trials distinctly from a *final evaluation* cohort. No per-question prompt edits or replacing hard cases.

## 2. Stage 0 — Model, tokenizer, EOS and official protocol forensic audit (REQUIRED)

**Goal:** Find concrete mismatches **before spending hours on benchmark generation**.

1. Re-audit model and tokenizer revision, on-disk safetensors hashes, `config.json`, `generation_config.json` if any, `tokenizer_config.json`, `special_tokens_map.json`, `modeling_ouro.py`, `auto_map`, RoPE positions, KV indexing, loaded physical modules, 48R execution count and exit behavior.
2. Inspect `bos_token`, `eos_token`, `pad_token`, IDs, native vLLM effective `eos_token_id`, `stop_token_ids`, `ignore_eos`, tokenizer `skip_special_tokens` behavior and actual finish reasons. **Never patch EOS purely to increase the score.** Explain if the model naturally emits `<|im_end|>` but the runtime fails to stop, or vice versa. Confirm EOS and `<think>` placement by ID and exact rendered text.
3. Render and persist side-by-side *exact* text, escaped repr, last 32 IDs, hashes, token counts for:
   - `P0`: old pinned script `add_generation_prompt=True` with default `enable_thinking` behavior.
   - `P1`: `add_generation_prompt=True, enable_thinking=True` if the pinned tokenizer truly supports it.
   - `P2`: model-card/HF official example or author-documented format, if materially different from P0/P1.
   Do not assume a keyword option works merely because it is accepted: verify rendered `'<think>'` and corresponding token ID. Assert tokenizer versions do not silently produce mismatched IDs. Store exact prompt text and ID sequences used by both engines.
4. Read the paper's Table 9 and Appendix C, identify **published** versus **not-disclosed** evaluation properties. Record AIME2024 dataset (30 AIME I/II questions), temp=1.0, top-p=0.7, R4, the paper's pass@1/pass@10 definition if disclosed, judge rubric availability, number of seeds/samples, stop rules, and whether any official 8192 cap pertains to AIME or another section. All unreported settings remain UNKNOWN.
5. Audit frozen dataset provenance from the previous PoC, correct-answer field, deduplication, whitespace/tokenizer drift and public evaluator behavior. AIME answer is an integer 0–999. Include parser self-tests and wrong-answer controls. Prefer exact mathematical grading with a separately labeled **non-author** optional judge audit for ambiguous final-answer formats; never invoke a paid LLM judge without authorization and never change decisions after seeing desired scores.
6. Build a short sanity suite: one short arithmetic, one moderate AIME-style question, one known long reasoning case; ensure coherent `<think> ... </think>` and final-answer rendering, correct stop reasons, no mismatched BOS/EOS, and R4 execution. Record final answer independently from internal reasoning text.

**Exit:** `PROMPT_EOS_AUDIT_PASS`, or `MODEL_TOKENIZER_CONTRACT_BLOCKED`. A successful audit does not certify accuracy.

## 3. Stage Q1 — Independent HF ↔ vLLM numerical and natural-generation fidelity

**Goal:** Establish that a high-quality official-style reference and the optimized multi-GPU runtime implement the same pretrained model, without requiring impossible cross-backend bitwise equality.

- Bring up HF official remote code with recommended standalone environment on **one permitted GPU**, ideally physical GPU4. Pin package versions and checkpoint. Verify HF source's cache implementation, residual resets, per-loop norms, sequence positions and complete R4 logical KV slots. Do not count deprecated/broken HF cache behavior as model failure.
- Run native vLLM TP1 on GPU4, TP2 on GPUs4/5 and TP4 on all four sequentially. Match **exact input token IDs** (do not allow engine re-tokenization), R4, BF16, EOS, max positions and decoding policy.
- For at least 8 frozen prompts spanning easy math, complex math and MATH, do teacher-forced same-prefix probes at (a) initial prompt, (b) 1, 4, 32, 128 cached-token steps where affordable, and (c) a late reasoning prefix (>1024 tokens) sampled from frozen pre-evaluation data. Compare pre/post final norm full logits, top-k ranking/margins, relevant hidden states, per-loop KV indices and selected cache values, and completed token counts. Test implementation parity separately for HF full vs HF chunk and vLLM TP1/TP2/TP4 if helpful; **do not conflate cross-runtime differences with same-runtime numerical errors**.
- Measure max absolute, relative L2 and normalized logit differences; establish *predeclared* floating-point tolerances based on observed baseline variability and operator differences **on a calibration set**, not post-hoc accuracy changes. Same-model/TP/graph repeats should ordinarily have tight numerical agreement; cross-TP BF16 bitwise equality is not required. Identify the earliest divergent layer/pass/token if a material discrepancy is discovered.
- Natural outputs: deterministic greedy decode first (at least 8 prompts; 256-token probes plus >=2 longer completion cases), then temperature 1.0/top-p 0.7 with controlled per-request seeds. Compare EOS, output lengths and final-answer correctness, not only token IDs under stochastic sampling. For a fixed engine and exact configuration repeat seeds/batching to probe reproducibility. Repeat the prior `retained-graph` diagnostic as an **isolated optional debugging lane** only if relevant; never let it replace native ordinary-graph scoring.
- Check whether varying only `max_tokens` changes already emitted prefixes. If it does, record first divergence and whether native batching/graph scheduling, RNG state, token limit or bad cache semantics explains it. Do not ascribe all score gains to censoring when trajectory prefix differs.

**Exit categories:** `REFERENCE_RUNTIME_PASS`, `REFERENCE_RUNTIME_PARTIAL`, `HF_REFERENCE_BLOCKED`, `NUMERICAL_MISMATCH_UNRESOLVED`. Do not require every stochastic sample to be identical across backends. A brief 4-token match is **insufficient**.

## 4. Stage Q2 — Reasoning quality protocol calibration and full public reproduction

**Goal:** Reproduce *plausible model quality* with verified protocols; do **not** optimize the test set to hit 64.7%.

### Q2-A. Small, frozen protocol diagnostic (NOT the final benchmark)

- Use a *predeclared* 6–10-question diagnostic subset or separate validation problems; log overlap with full AIME. Both P0 (legacy) and P1 (explicit thinking) run with the **same checkpoint, R4, seed(s), output budget, backend and evaluator**. Compare actual `<think>` rendering, EOS rate, mean generation length, cap rate and answers. An apparent accuracy gain on these prompts is exploratory and cannot justify reporting the better arm as unbiased AIME pass@1.
- Check the author-visible prompt instructions. Prefer the most **author-faithful documented** prompt for the primary protocol, not whichever yields a higher AIME score. If the original harness cannot be retrieved, predefine an explicit default/Thinking split and report both on the final benchmark if budget permits.
- Predeclare `temperature=1.0`, `top_p=0.7`, `top_k` explicit (currently prior code `top_k=-1`/no limit) and explain differences from HF defaults. Preserve deterministic greedy as a **separate, non-author** quality/correctness lane. Do not inadvertently enable extra stopping strings or disable EOS.

### Q2-B. Output cap and final-score policy

- Primary author-aligned candidate is **R4, sampling temp=1.0, top-p=0.7**. The primary cap is initially **8192** (for comparison to previous run). Conduct a **frozen full-lane** extension to **16384**, and **24576** only when needed and within the 32768-token total-position bound, with adequate HBM/KV capacity. `max_new_tokens = min(requested_cap, 32768 - prompt_token_count)`; retain raw requested vs effective cap. Never silently truncate the prompt or extend RoPE/context beyond the validated window.
- Compare within identical samples/requests using **fixed per-request seeds**, same batch sizes and engine config where possible; test common-prefix stability. Record that a changed max_tokens may change sampling trajectories on some runtimes. Run *entire predetermined cohorts* for each cap: do not rerun only previous failures or select lucky samples.
- Reasoning quality includes **all generations**, including capped ones, in the denominator. A generation capped before a valid terminal answer is incomplete, never silently counted as correct. Keep separate fields: raw final-answer extractability, correct-without-EOS diagnostic, EOS/natural termination, parsed correct completed answer, cap hit and repetition/reasoning loops. Report correctness conditional on completed outputs only as a **selection-biased diagnostic**.
- First full evaluation: all **30 AIME2024** problems × **3 prespecified seeds** = 90 independent sampled completions nested within 30 problems. Next, if runtime/quality gates justify, expand to **10 seeds per problem** = 300 outputs, estimate pass@1 as average correct fraction and pass@10 as the fraction of problems solved at least once out of their ten sampled outputs. Preserve the seed list and cohort hierarchy. The 64.7% official figure is a **reference**, not a pass/fail exact target; with only 30 problems and an unmatched judge, uncertainty is substantial.
- Include **greedy AIME2024 30/30** as a distinct baseline. Secondary MATH-500: use the previous immutable metadata-stratified 100-problem subset with exact math-verify first, then the full **500** if feasible and quality stable. Never call 100-item subset 'MATH-500 full accuracy.' Add AIME2025 (30) as secondary *if evaluation setup is trustworthy*; it is independently reported by the authors (50.3%) and helps detect benchmark-specific overfitting.
- Optional higher-fidelity author-judge reproduction only if actual rubric/prompt and access are known and approved. Public integer exact match remains the primary transparent grader for AIME; keep all judge disagreements with blinded adjudication principles (no score chasing).

### Q2-C. Quality decision and statistical analysis

Report by prompt protocol, engine and output cap:

1. pass@1 completed-correct / all (including capped); seed-wise and problem-wise;
2. pass@10 when ten samples exist; ***never*** infer pass@10 from three seeds;
3. capped, EOS, parser-success, completion-length P50/P90/P95 and max;
4. sampling & greedy separately; answer parsing disagreement ledger and clean frozen graders;
5. cluster bootstrap 95% CI **resampling problems, carrying all seeds for a problem**; uncertainty with only 30 AIME problems must be described conservatively;
6. published 64.7% and 90.0% alongside a prominent `REFERENCE_PROTOCOL_NOT_IDENTICAL` label and exact sources of mismatch;
7. whether longer cap changed common prefixes, and if score changes are not solely completion effects;
8. HF vs vLLM same-protocol quality at matched seed/temperature where affordable, with stochastic trajectories understood;
9. no post-hoc selecting the arm closest to 64.7% as the main outcome.

**Operational quality gate:** `QUALITY_BASELINE_READY` only if checkpoint/prompt/EOS are validated, outputs are meaningfully completed in a predeclared cohort (aim >=90% natural termination, or otherwise explicitly justify why enough complete outputs exist), scoring/reproducibility are credible, and HF↔vLLM comparisons do not reveal an unexplained consequential model bug. Numerical closeness to 64.7% is *supporting evidence*, **not the sole gate**. Other labels: `QUALITY_REFERENCE_APPROXIMATE`, `QUALITY_BASELINE_WEAK`, `MODEL_REFERENCE_MISMATCH`, `EVALUATOR_MISMATCH`, `OUTPUT_CAP_BLOCKED`, `HF_REFERENCE_BLOCKED`. Even if the accuracy stays below 64.7%, preserve honest results and diagnose; do not tune until it matches.

## 5. Stage P — Real long-reasoning multi-GPU physical characterization (after credible quality)

**Core contrast:** earlier 16K/32K *prompt + 64 output* versus genuine *short/medium prompt + thousands of generated reasoning tokens*. Unlike forced-output hardware tests, **the primary P lane must generate real tokens to natural EOS** and record each answer's correctness.

### 5.1 Fixed GPU-budget serving layouts

Measure physical, simultaneous engines (not sum of separately measured TPS):

| Layout | Physical placement | Use case |
|---|---|---|
| `TP1` | GPU4 | Single request baseline and quality reference |
| `TP2` | GPUs4,5 | Single request baseline |
| `TP4` | GPUs4,5,6,7 | Lowest latency candidate for B1; prior short-output baseline |
| `TP1×DP4` | Four independent TP1 engines on GPU4/5/6/7 | Aggregate throughput, capacity, tail latency |
| `TP2×DP2` | Two TP2 engines on GPU4–5 and GPU6–7 | Aggregate throughput, capacity, tail latency |
| `TP4×DP1` | One TP4 engine on GPUs4–7 | Four-GPU comparator |

Use same R4, prompt protocol and sampling configuration. Where stochastic natural outputs differ, compare *distribution-level* quality and latency. Add a separate **trace/replay lane** with frozen output IDs and matched output length for hardware causal interpretation; do not use replay results as natural quality or replace primary natural-lane results. No TP8.

### 5.2 Workloads and workload fairness

- **W-real-AIME:** frozen AIME2024 reasoning requests with actual full completions (source, question ID, seed, prompt length, output length, finish reason, answer correctness). Prefer quality-protocol-confirmed settings and predeclared seed cohort. Do not report only successful completions.
- **W-real-MATH:** at least 30 diverse frozen problems from the prior MATH subset, with actual full completions and corresponding correctness. Use a bounded larger cohort if feasible.
- **W-length-matched:** replay *the same tokenized input and output token trace* for each TP/layout, preserving identical model work shape and fixed context; specifically distinguish this diagnostic from genuine generation.
- **W-history-control:** selected previous 16K prompt + 64 frozen output workload, using clean new measurements; historical absolute times are context, not paired baselines. This quantifies how the bottleneck shifts between long-prefill/short-output and real reasoning.
- Prompt lengths may be ~100–2048, output lengths can be 512, 2048, 4096, 8192, 16384+; use **actual measured bins** rather than assume all tasks reach a prescribed length. Label completed and censored runs separately. Avoid silently batching dissimilar workloads or choosing the fastest short completions as the representative cohort.

### 5.3 Per-request and physical metrics

Collect a timestamped per-request ledger (warm load and graph capture outside request wall unless explicitly noted):

- **TTFT** (client/scheduler distinction); **cached TPOT / inter-token latency** and its distribution along output-token index; total natural **E2E**, output TPS (real emitted tokens only); per-request **P50/P90/P95** where sample size justifies, including queue+service decomposition.
- Decode intervals for token counts `1–512`, `513–2048`, `2049–4096`, `4097–8192`, `8193–16384`, `>16384` when data exist. Track TPOT versus live KV length, cache-page usage and input/output lengths. Do not mix times from different request compositions without stratification.
- GPU memory: real weights unique bytes, preallocated KV-pool bytes, **filled logical KV** and head ownership, actual memory allocated/reserved/peak HBM on each rank, workspace/graph cache; distinguish them from device-wide `nvidia-smi`. Verify model R4 still has `192` independent logical cache slots, Q/KV head sharding TP1=16, TP2=8, TP4=4 per rank.
- Theoretical exact BF16 full-slot Ouro R4 KV payload: `192 slots × 2 (K,V) × 16 heads × 128 dims × 2 bytes = 1,572,864 bytes/token` **aggregate** (=1.5 MiB/token); at TP4 each rank logically stores one-quarter (=393,216 bytes/token), subject to allocation/layout overhead. Validate against actual tensor ownership. KV grows with prompt + generated-token positions.
- Physical critical path: actual kernel service, GPU busy union, graph replay periods, TP all-reduce (including native custom reductions), attention, GEMM, KV paging, rank readiness/waits only where validated. **Do not add overlapping kernel service times and call that E2E.** Prefer carefully bounded profiler spans (observer inflation <10%), and use clean wall for headlines. No global cross-rank time alignment unless explicitly validated.
- Concurrency sweep `C={1,2,4,8}` offered (and more only if legal and evidence justifies). Measure completed-request wall, aggregate *real generated* TPS, per-request E2E, queue time, actual simultaneous resident requests, OOM, admission/eviction and prefix/KV capacity. Each four-GPU layout must complete exactly the same frozen offered cohort or be marked capacity-blocked. Record all incomplete/capped cases; do not artificially improve throughput by early stopping.
- Operational fairness: identical checkpoint, four physical GPUs available to each layout, total HBM budget, R4, dtype and reference-quality policy. Engine memory reservations may differ; report actual per-device capacities. TP×DP replicas must be **really concurrent**. Engine count effects, CPU contention and neighboring users on GPUs0–3 are limitations. Alternate/counterbalance session order if feasible.

### 5.4 Measurement quality and analysis

- Three warmups and >=3 technical repeats per request/config for **short hardware lanes**; long natural generation may be expensive, so prespecify feasible independent requests/seeds rather than multiplying runs blindly. Do not treat technical repeats, seeds, layers or tokens as independent tasks. The primary independent units are **problems**; for repeated sampling, bootstrap clustered by problem.
- Report uncertainty and sample sizes honestly. For finite-cohort throughput, report repeat cohort-level ranges; a request-bootstrap alone is not an independent cohort confidence interval.
- For natural workloads with different generated outputs/lengths, distinguish matched-shape efficiency from outcome-conditioned latency. Present both *actual latency-to-final-answer* and *per-token service at comparable context length*. Do not assert TP4 is faster solely because it outputs fewer tokens, or TP1×DP4 is more accurate solely due stochastic noise.
- Matched TP quality comparisons include correct/incorrect and completion/truncation denominators. For stochastic decoding, exact token parity is not mandatory across ranks; reproducibility/numerical drift must be tracked.

## 6. Stage N — Bottleneck discovery and optional low-risk improvement screen

Stage N is **analysis-first**. It is *not* permission to implement a large novel method without evidence.

1. Establish Pareto frontier across `TP1/2/4`, `TP1×DP4`, `TP2×DP2`, `TP4×DP1` for **quality, completed-request E2E, throughput, P95 and max feasible concurrency**.
2. Determine whether the reasoning workload moves the bottleneck from prefill-dominated to **decode/KV capacity/queueing**; quantify actual critical-path milliseconds and how they grow with recurrent depth's 192 KV slots. Compare with a genuine AR control only if budget permits and normalize for architecture/hidden/KV/head/depth differences; no causal recurrence attribution from raw speed.
3. Test whether recurrent-depth-specific KV **access order or lifecycle**, beyond ordinary AR and generic paged KV, creates a *measured* physical overlap or capacity opportunity. Do not re-propose simple remote KV placement or generic TP×DP routing as a novel framework. Prior boundaries include FlashLoop, CDB, vllm-rlt, LoopSpec, WaveFront, generic paged KV and continuous batching.
4. If a **simple implementation correction** is discovered (e.g., wrong EOS/stop token, redundant tensor copy, generic graph issue), isolate its output/quality parity and E2E benefit, report it explicitly as **runtime-only/generic**, and preserve the corrected baseline for subsequent comparisons. Do not call it loop-specific novelty.
5. Only recommend a new multi-GPU *method* if there is (a) robust quality baseline, (b) measurable removable **non-overlapped** critical-path or capacity headroom, (c) a mechanism demonstrably tied to recurrence rather than a generic AR problem, (d) strength relative to the best native 4-GPU layout, and (e) real implementation feasibility with accuracy preserved.

### Decision labels (report independently, not a single ambiguous verdict)

**Quality:** `QUALITY_REFERENCE_APPROXIMATE`, `QUALITY_BASELINE_READY`, `QUALITY_BASELINE_WEAK`, `MODEL_REFERENCE_MISMATCH`, `EVALUATOR_MISMATCH`, `OUTPUT_CAP_BLOCKED`, `HF_REFERENCE_BLOCKED`.

**Correctness:** `REFERENCE_RUNTIME_PASS`, `REFERENCE_RUNTIME_PARTIAL`, `NUMERICAL_MISMATCH_UNRESOLVED`.

**Systems:** `REAL_WORKLOAD_PROFILE_COMPLETE`, `PARTIAL_PROFILE_ONLY`, `NO_RECURRENCE_SPECIFIC_HEADROOM`, `GENERIC_RUNTIME_ONLY`, `CONDITIONAL_LOOP_SPECIFIC`, `GO_FOR_NEXT_METHOD_POC`, `TIMING_INVALID`, `CAPACITY_BLOCKED`.

**No forced GO:** failure to reproduce the exact 64.7 does not automatically imply the model is defective; equally, a plausible 64.7 does not establish novelty or method speedup. Even when quality is weak, preserve a **small, explicitly labeled exploratory hardware audit** rather than launch a full expensive profiling campaign.

## 7. Deliverables, reporting and Git safety

**New output directory:** `poc_ouro_reasoning_fidelity_multigpu/`  
**Suggested new branch:** `flashvep/ouro-reasoning-fidelity-multigpu-poc`  
**Source spec location on server:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/ouro_reasoning_fidelity_real_workload_multigpu_poc_spec.md`.

Required files:

- `report.md` — executive decision, published-vs-public evaluation protocol matrix, all quality and systems findings, falsification cases, uncertainty, explicit unmet conditions.
- `reproduce.md` — exact isolated interpreter environments, CUDA_VISIBLE_DEVICES, installed versions and hashes, staged commands, sample schedule, data hashes.
- `model_tokenizer_eos_audit.md`, `author_protocol_comparison.md`, `protocol_lock.json`.
- `hf_vllm_fidelity.md`, `quality_protocol.md`, `quality_scorecard.md`, `grader_tests.md`.
- `real_workload_contract.md`, `timing_contract.md`, `memory_kv_contract.md`, `prior_overlap.md`.
- Machine-readable request-by-request raw input/output token IDs, decoded strings, seeds, EOS/cap reasons, correctness, length, logits/KV probes (as permitted by storage), independent timing and GPU ownership records. Store long raw outputs safely; add SHA256 manifests and note off-Git evidence where necessary rather than bloat Git.
- Tables and figures: P0/P1 prompt rendering; cap-vs-completion-vs-quality; AIME seeded pass@1 + pass@10 (if valid); HF-vLLM numerical drift; token-length distributions; token-index–TPOT curves; rank HBM/KV filling; 4GPU E2E/TPS/P95/quality frontier; critical-path attribution validity.
- A high-level **advisor summary** (5–10 grounded sentences): what made previous baseline weak, whether this was corrected, how public evaluation compares to authors, real long-generation TP/DP tradeoffs, and whether a loop-specific motivation survived.
- Track run/failure ledgers and preserve negative attempts. The agent must **commit and push only its newly owned PoC files**, verify remote SHA and provide branch, commit SHA, report path. Do not reset/clean unrelated state or touch other users' GPUs.

### Suggested execution order and early exits

`Stage 0 audit → Q1 HF/vLLM reference fidelity → Q2 frozen protocol calibration → Q2 full AIME/MATH quality → Stage P physical real-workload TP/DP → Stage N headroom/novelty decision`.

If Q1/quality cannot be validated, stop before broad P and report `QUALITY_BASELINE_WEAK` with precise causes. If broad P is completed and only generic TP/DP or reference-runtime improvements are discovered, report `NO_RECURRENCE_SPECIFIC_HEADROOM`, not an invented method. Never promise future measurements: run and report feasible experiment stages within this campaign, including explicit NOT_RUN entries and reasons.

---

## Appendix A — Minimum protocol matrix

| Lane | Protocol | R | Backend/layout | Cap | Purpose |
|---|---|---:|---|---|---|
| P0 | Legacy default chat template | 4 | HF + vLLM TP4 | 8192 | Reproduce previous behavior |
| P1 | Explicit `enable_thinking=True`, verify rendered prompt | 4 | HF + vLLM TP4 | 8192 | Test evidence-based prompt discrepancy |
| P1-long | Same frozen P1 | 4 | vLLM TP4 | 16384; conditional 24576 | Output-completion sensitivity |
| Fixed-judge AIME | Predeclared author-faithful prompt | 4 | vLLM TP4, plus small HF corroboration | Primary declared cap | Credible 30×3, optional 30×10 public accuracy |
| Greedy | Same declared prompt | 4 | TP4, small TP1/TP2 verification | Declared cap | Deterministic runtime correctness |
| MATH100 | Same declared prompt | 4 | TP4 | Declared cap | Existing subset continuity; optional full 500 |
| Reasoning B1 | Natural EOS | 4 | TP1/2/4 | Real termination within cap | Real-world latency and KV behavior |
| 4GPU serving | Natural EOS | 4 | TP1×DP4, TP2×DP2, TP4×DP1 | Real termination within cap | Four-GPU throughput and tail latency |
| Frozen trace | Identical forced IDs/length | 4 | TP/DP configurations | Exactly fixed | Hardware normalization **not accuracy** |

## Appendix B — Explicit out-of-scope actions

Do **not**: tune prompts on the 30 AIME test questions to hit 64.7%; train or quantize weights; alter R4 for the official score; enable model-specific benchmark leakage; override EOS to force earlier correct answers; misreport censored runs as correct; add `max_tokens` beyond verified context; assume a particular speculative/PP method is novel; infer all-reduce critical-path time by summing kernels; run GPUs0–3; call a conditional theoretical oracle a measured real improvement; hide unresolved numerical or output prefix divergences.

