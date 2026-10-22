# Ouro Chunk–Loop Prefill: C/G Characterization and SLO Sensitivity Discovery PoC

**Status:** EXPERIMENT SPECIFICATION — new experiments NOT_RUN  
**Date:** 2026-10-22  
**Intended repository:** `/home/esjung/MLLM-EP-github`  
**Required specification destination:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/ouro_chunk_loop_slo_sensitivity_poc_spec.md`  
**Expected report folder:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/ouro_chunk_loop_slo_sensitivity_discovery/`  
**Compute restriction:** physical NVIDIA GPUs **4, 5, 6, 7 only**. Single GPU first: GPU4.  
**Research boundary:** The separately paused custom native-vLLM Ouro Loop PP program MUST NOT resume without the user's exact phrase **"vLLM looped PP 구현 재개하자"**. No new Ouro TP/PP runtime development is authorized here.

## 0. Executive decision and rationale

The previous fixed-quantum experiment demonstrated a genuine **distributional trade-off**, not an across-the-board gain or a joint-adaptive result. At Ouro-1.4B BF16, static R4, H100 GPU4, long prompt ~8,177 tokens, closed-loop concurrency 8, configured token chunk capacity **C=1024**, comparing native atomic G4 with grouped G2:

| Observation (10 held-out cohort means) | Atomic G4 | Grouped G2 | Interpretation |
| --- | ---: | ---: | --- |
| Long-request TTFT | 924.384 ms | 1116.074 ms | G2 worse |
| Decode ITL p95 | 38.510 ms | 139.216 ms | G2 substantially worse |
| Decode ITL p99 | 364.162 ms | 205.288 ms | G2 43.63% lower, reported 95% paired CI [34.61%, 49.12%] |
| Output token throughput | 172.820 tok/s | 163.030 tok/s | G2 lower |
| SLO-qualified *drained burst* rate, TTFT <= 5s AND per-request max ITL <= 100ms | 0.169 req/s | 0.159 req/s | G2 lower **for that particular SLO and load** |
| ITL events > 100ms | 253 / 8890 | 483 / 8866 | G2 creates more moderate stalls |
| ITL events > 200ms | 194 / 8890 | 104 / 8866 | G2 creates fewer extreme stalls |

All cells above are **HISTORICAL_MEASURED** from immutable commit `c458f6a2a764f667a0da6784bcb2592b36626837`. They are **not** new results, not steady open-loop SLO throughput, and the p95/p99 entries are *means of cohort-level quantiles*, not pooled token-interval quantiles. All 128-token bounded generated requests reached the cap with no EOS; full benchmark accuracy was not established. Isolated numerical tests passed 31/31, but 112/120 batched capped output sequences matched exactly. The old verdict `CONDITIONAL_FIXED_QUANTUM_ONLY` must be preserved, not rewritten as GO or NO-GO.

**New question:** What are the independent and interacting effects of token chunk capacity **C** and prefill recurrent group size **G** on prefill efficiency, decode stall frequency and severity, TTFT, memory, and latency-SLO **attainment over a range of arrival rates and SLO values**? Does any interior/static or online-selected (C,G) combination extend the Pareto frontier beyond thoroughly tuned atomic C-only, split G1, and existing RLT Mixed scheduling?

**Goal:** Characterize and explain the behavior before selecting or writing an adaptive controller. Do not prematurely reject the research solely because one C1024/G2 setting loses at a 100ms per-request-max-ITL SLO. Do not presume joint scheduling wins. Completion of scientific characterization and a defensible negative finding is an acceptable outcome.

## 1. Definitions and conceptual comparison

- `C` = **maximum prefill tokens in one chunk**; NOT the model's number of loops or the active request count. Actual selected tokens can be lower because of remaining prompt tokens, per-batch token budget, or the presence of decode rows.
- `G` = maximum number of **consecutive recurrent loop iterations per prefill scheduling submission**. Here R=4 always, and allowable G is `{1,2,4}` initially. G does not indicate the number of token chunks and DOES NOT reduce total model depth.
- `R` = four required recurrent passes of 24 tied physical layers. Prompt prefill and decode remain **full static R4** in the core discovery; never conflate G adaptation with *model early exit*.
- `N` = concurrent client requests, separate from C.
- `Q=(C,G)` = prefill scheduling quantum design. The execution cost is **not assumed proportional to C*G** because recurrent attention, existing context, batching, hidden staging, graph execution, and runtime overhead vary.
- `Atomic G4`: each token chunk traverses R1–R4 before yielding. **This IS conventional token-chunked prefill, not unchunked prefill.** With ~8K prompt and C1024 there are approximately eight token chunks.
- `Grouped G2`: the *same* token chunks execute R1–R2, yield, then R3–R4, while completing all four passes in total.
- `Split G1`: the *same* token chunks yield after each loop. Existing RLT feature.
- `Mixed`: RLT stock combined recurrent prefill/decode batches. Do **not** replace it with a weaker G1 baseline or compare its timings as pure G effects; actual batch packing/graph fallbacks may differ.
- `No Chunking` is an additional control with sufficiently large prefill token budget to admit the whole prompt and G4, **only if genuine without artificial truncation and within safe memory**.
- Scheduling **opportunity** is not an unconditional decode execution; actual chosen work and any deferral must be logged.

## 2. Pinned provenance and safe workspace

### Existing immutable experiment

- Repository: `jyssys/MLLM-EP`, feature branch `flashvep/ouro-chunk-loop-prefill-joint-poc`, commit `c458f6a2a764f667a0da6784bcb2592b36626837`.
- Historical outputs: `poc_flashvep/reports/ouro_chunk_loop_prefill_joint_discovery/{analysis.md,benchmark_summary.md,correctness.md,implementation_scope.md,profiling.md,failures.md,results_raw.csv,paired_comparisons.csv,ITL_stall_counts.csv,...}`.
- Reusable patch: `poc_flashvep/ouro_chunk_loop_prefill/patches/rlt_grouped_prefill.patch`.
- Model: base `ByteDance/Ouro-1.4B`, revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, BF16, no retraining, no quantization, no stale activations; four recurrent passes over 24 shared physical layers and 96 independent attention/KV namespaces.
- Engine: separate vLLM-RLT pinned `4a350344027960a48384626aa646f1765fda6d4f`. This engine is NOT native vLLM and must not be represented as such.
- Attention: previously compared Triton and FA4. Use one pinned backend for primary comparisons; optional repeat on the other for robustness.
- Existing GPU burners on physical GPU4–7 must not be interrupted without confirmation of ownership. Stop only the user's **recognized** burners when starting authorized tests; never kill unrelated jobs. Restore only recognized burners after tests if existing operational procedure permits.
- Create an isolated new feature branch and report folder; keep prior history/results intact. No `git reset --hard`, `git clean -fd`, global environment rewrite, or force push. Commit only newly generated files and the small scoped patch if needed; observe existing approved remote push policy.
- Capture `nvidia-smi` UUID/PCI topology and `CUDA_VISIBLE_DEVICES` mapping in every run: initial `CUDA_VISIBLE_DEVICES=4`. Avoid accidental logical-to-physical GPU index mismatch.

### Primary source anchors (verify actual checked-out code and commit hashes before work)

- RLT scheduling boundaries: https://github.com/InferMatrixLab/vllm-rlt/blob/4a350344027960a48384626aa646f1765fda6d4f/vllm_rlt/core/scheduling_policy.py
- RLT token budget/chunk frontiers: https://github.com/InferMatrixLab/vllm-rlt/blob/4a350344027960a48384626aa646f1765fda6d4f/vllm_rlt/core/scheduler.py
- RLT executor grouped loop count: https://github.com/InferMatrixLab/vllm-rlt/blob/4a350344027960a48384626aa646f1765fda6d4f/vllm_rlt/worker/executor.py
- RLT configuration: https://github.com/InferMatrixLab/vllm-rlt/blob/4a350344027960a48384626aa646f1765fda6d4f/vllm_rlt/config.py
- RLT docs: https://github.com/InferMatrixLab/vllm-rlt/blob/main/docs/scheduler.md
- CDB reference: https://arxiv.org/abs/2608.09444 and https://github.com/kschwethelm/continuous-depth-batching

## 3. Literature-based SLO and metric contract

### A. Layered Prefill, MLSys 2026 — direct paper definitions

Paper: https://proceedings.mlsys.org/paper_files/paper/2026/file/c0f460c6d63599ea870ba9db63dc96a9-Paper-Conference.pdf (Table 5, Figure 3, section 5.2).

- Request attains SLO iff `TTFT <= threshold_TTFT` **AND** **every** generated-token `TBT <= threshold_TBT`.
- SLO attainment (%) = 100 * (# requests satisfying both) / (# eligible offered requests). Include failures/timeouts in the offered denominator by a predeclared policy; never silently drop failed requests.
- Effective service-capacity criterion: **at least 90% request SLO attainment**, swept across different **open-loop request arrival rates**.
- Paper TBT limits: Qwen3-30B-A3B = 125ms for both ShareGPT/arXiv; GPT-OSS-20B = 100ms; paper TTFT limits = 5s for ShareGPT, 10s for arXiv. **These are NOT universal Ouro limits**; do not copy the values blindly.
- Paper states TBT target is approximately 5x a no-prefill-interference decode latency reference at 4K context with a batch of 32; TTFT selected for model/workload operating point. Log calibration method and exact measured reference before using any derived limit.
- Include **combined SLO attainment and decomposed TTFT-only and TBT-only attainment**, matching their Figure 3/Figure 4 logic; present all to avoid masking which side fails.

### B. Sarathi-Serve, OSDI 2024

Paper: https://www.usenix.org/conference/osdi24/presentation/agrawal; full discussion https://arxiv.org/abs/2403.02310 .

- Evaluates P50 TTFT, P99 TBT and serving capacity under a latency constraint.
- Strict/relaxed P99 TBT caps use 5x/25x no-prefill-interference decode iteration time at 4K and batch size 32 for a given hardware/model pair. This is one published methodology, not a binding standard.
- A run is not sustainable when queues continue to diverge. Sarathi also checks queue delay (median scheduling-delay bound in its paper).

### C. Current 2026 context and appropriate comparisons

- **SLOWeave** (Sept 2026): https://arxiv.org/abs/2609.07883 — deadline-aware online **adaptive C-only**; evaluates goodput under per-output-token objectives, e.g. 25ms and 10ms; simulated and iteration-level GPU studies. It is an important **adaptive C-only methodological baseline**, but not automatically a drop-in executable RLT baseline. Distinguish faithful implementation, modeled benchmark, and algorithm-only imitation.
- **FlowPrefill** (Feb 2026): https://arxiv.org/abs/2602.16603 — operator-level preemption and event-driven scheduling targeting TTFT goodput, related to choice of yield boundaries and control-plane overhead.
- **P-PAS / P-PAS+** (2026): https://arxiv.org/abs/2608.15171 — adaptive prefill token budgets based on load and measured mean E2E latency; compare its pressure idea as a C-only variant where feasible.
- **DuetServe** (ICML 2026): https://proceedings.mlr.press/v306/gao26m.html — dynamically isolates prefill/decode via SM partitioning, measured throughput/TBT SLO, distinct GPU spatial multiplexing mechanism.
- **Continuous Depth Batching (CDB)** (2026): https://arxiv.org/abs/2608.09444 — dynamic decode depth and recurrent scheduling, not evidence that joint adaptive C/G SLO optimization has been demonstrated.
- **RLT** https://github.com/InferMatrixLab/vllm-rlt — already supports atomic G4, split G1, mixed recurrent prefill/decode and chunked prefill. The exact G2 patch is **not** a new fundamental mechanism by itself.
- **Layered Prefill** (MLSys 2026): https://proceedings.mlsys.org/paper_files/paper/2026/hash/c0f460c6d63599ea870ba9db63dc96a9-Abstract-Conference.html — schedules layers for MoE weight-traffic reasons. Our Ouro Dense R4 does not inherit MoE expert-reloading benefits; compare scheduling ideas without attributing MoE-specific memory traffic.

**No verified exact prior publication establishing the proposed online adaptive `C,G` controller for Ouro was located in this audit. This is NOT a claim that none exists. Perform a targeted recent-work update before making novelty statements.**

### D. Quantile versus request-level compliance

- `TTFT_i = first_delivered_token_i - scheduled_arrival_i` (use client monotonic timing).
- `TBT/ITL_i,j = observed_token_delivery_i,j - observed_token_delivery_i,j-1` for consecutive actual output tokens; record sampling/client chunk effects. No imaginary individual ITLs inside multi-token delivery chunks.
- `TPOT_i = (last_delivery_i - first_delivery_i)/(num_output_tokens_i - 1)`, defined only for two or more output tokens; do not substitute mean TBT for request max TBT.
- `p50/p95/p99 TBT`: quantiles over observed token intervals, or explicitly report per-request quantiles and cohort-level aggregates separately. Report the exact aggregation. A pooled p99 over tokens is NOT the same as 99% of requests passing a per-request max-TBT SLO.
- `request_TBT_pass_i(b) = max_j TBT_i,j <= b` for the strict *all-token* criterion used by Layered Prefill.
- `request_TTFT_pass_i(a) = TTFT_i <= a`; combined pass is their conjunction.
- `SLO_attainment(a,b,lambda) = passing_offered_requests / total_offered_requests`; `SLO_goodput = passing_offered_requests / offered_arrival_window_seconds`, including failures and timeouts in the denominator/failed count.
- `capacity90(a,b)`: highest **stable** offered arrival rate for which combined SLO attainment is >=90%, ideally accounting for statistical intervals and backlog stability. Label unbracketed rates and inadequate repetitions as exploratory.
- `interference-only tail`: interval distribution of already decoding requests during admitted prefill; and full request lifecycle latency metrics are both needed.
- Separate `drained-burst qualified req/s` from arrival-window goodput and sustainable capacity. Never call burst throughput steady-state serving capacity.

## 4. SLO registration — no cherry-picking

**Stage 0: decode-only calibration (before comparing C/G candidates).** For Ouro BF16 full R4 on GPU4, measure per-token decode iteration/observed client ITL with no interfering long prefill, at representative context lengths near 2K and 8K and resident batch counts {1,4,8}, optionally 16 if memory-safe. Use multiple warmed independent starts. Report `t_dec_baseline(P,N)` with actual mean, median, p95 and measured GPU executor duration; do not pretend RLT B8 equals Sarathi's B32 calibration. Freeze an explicitly chosen reference configuration and physical values.

**Relative three-tier SLO grid** (proposed calibration-based exploratory tiers, not official standards):

- Strict: `TBT_SLO = 5 * t_dec_baseline`.
- Moderate: `TBT_SLO = 10 * t_dec_baseline`.
- Relaxed: `TBT_SLO = 25 * t_dec_baseline`.

Also publish a **fixed absolute sensitivity grid** `{75,100,125,150,200,250,500} ms`, where physically meaningful; these candidate points are analytical probes, not claims of user/product requirements. The 100/200ms distinction is specifically motivated by the previous event counts, but predeclare the whole grid before evaluating new candidate workloads. If measured natural decode-only performance already exceeds a candidate SLO, mark it intrinsically infeasible (not an algorithm failure).

**TTFT limits:** preregister a calibrated grid: e.g. `1.25x/1.5x/2x` the native warmed **no-interference** long-prompt TTFT at each prompt bin, plus fixed 2s/5s/10s reference limits where relevant; do not derive thresholds separately for each candidate. Calibration-based thresholds should be reported in numeric milliseconds. Use a compact primary grid (such as strict/moderate/relaxed TBT x two prespecified TTFT limits) and the full grid as sensitivity supplement, avoiding selective claims.

For **all** SLO curves, simultaneously report `(i) combined request pass, (ii) TTFT-only pass, (iii) TBT-only pass, (iv) pooled TBT p95/p99, (v) event violation rates at 100 and 200ms, (vi) latency distributions, (vii) raw counts and confidence intervals`. Under any strict per-request-max-ITL rule, one violating token is enough to fail that request. Explicitly contrast this with Sarathi-style aggregate p99-TBT threshold, which is a **different service contract**.

## 5. Workloads and experimental matrix

### Model / runtime constants

- Main: Ouro-1.4B Base BF16, Static full R4, physical H100 GPU4.
- Native RLT's grouped patch from prior commit; verify full input numerical parity. Preserve model and trained weights. No early-exit depth policy, no loop skipping. No native vLLM port and no Loop PP.
- Pin attention backend (use qualified FA4), native CUDA Graph setting, token budget, KV reserve, resident engine, HuggingFace tokenizer, EOS and stop-string handling. Make no performance-changing config changes between compared arms.
- Compare both server and in-process API only when backend identical; characterize in-process first and add proper client arrival/streaming clocks for primary SLO. In-process benchmark cannot be labeled HTTP latency.

### Fixed-grid factors

| Dimension | Minimum settings | Extension only if justified |
| --- | --- | --- |
| C (configured token chunk capacity) | 128, 256, 512, 1024 | 2048 / whole-prefill control after memory and graph validation |
| G (prefill loop group) | 1, 2, 4 | Optional dynamic per-group selection after discovery |
| Prompt bin | ~2K and ~8K | Genuine realistic 1K/4K/16K if safe |
| Concurrent decode requests | 4,8; include C1 isolated control | 2 and 16 when safe |
| Workload | existing frozen mathematical-prompt concat stress; natural long prompt arrivals into ongoing decode | ShareGPT-like conversations and arXiv-like long text, with frozen provenance |
| Decode output | bounded 128-token legacy replay for exact historical comparison; **complete natural EOS** with cap >=512, ideally 4096 in selected cells | More full-answer quality if resources allow |
| Serving load | closed-loop reproducibility plus **open-loop Poisson and burst-trace** | varied prompt/output mix |
| Random seeds and startup | >=3 independent engine starts initially, varied execution ordering | >=5 independent starts if tail CI unstable |

Strictly separate the historical 128-output cap lane from the **full natural EOS quality/SLO** lane. A test in which all requests hit the cap should not be described as full response completion or answer-quality validation. Preserve frozen prompts and evaluation recipes. Use a second representative workload rather than relying exclusively on 8K concatenated math question text. Ensure no prompt/output tokens are silently modified or fixed by teacher forcing in main natural generation.

### Required policy families

- `NC`: true no-chunk full-prefill control, when the engine/limits permit without memory bias.
- `C_ONLY_FIXED`: G4 with full C sweep; optimize C separately **per calibration set and SLO**, not on held-out evaluation.
- `G_ONLY_FIXED`: fixed C (both C1024 and one calibrated representative C); G1/G2/G4 sweep. Strong current RLT split baseline G1 included.
- `CG_STATIC`: full fixed C x G grid, analyzed as a response surface and Pareto frontier; do **not** label this an adaptive result.
- `MIXED_RLT`: unmodified stock RLT Mixed G1 with its own C sweep; record actual batch composition. Exclude grouped Mixed unless safely separately validated.
- `STAGED_G4`: same staged prelude/state path as Grouped, G4, to isolate staging/graph effects from G itself. Control should cover repeated cohorts, not one trace.
- `C_ONLY_ADAPTIVE` (CONDITIONAL): faithful minimal SLO-aware adaptive chunk budget derived from a validated monotone runtime cost model (SLOWeave-like or transparent heuristic); strong comparison before any joint online method.
- `G_ONLY_ADAPTIVE` (CONDITIONAL): fixed C, choose G among {1,2,4} using causal, online queue state and the same cost predictor/decision frequency.
- `CG_ADAPTIVE` (CONDITIONAL): only if a measurable joint operating region survives tuned controls; same online information and total scheduling budget. All conditions must be explicitly predeclared and tested without future-oracle knowledge.

No claim of independent variable isolation if Mixed changes batch packing or graph mode. Report actual effective C, G, prompt rows, decode rows and launched graph buckets. For grouped G2 at fixed C, ensure R1–R4 total work and all prompt KV rows are preserved. If an existing grouped patch is absent, restore that minimal patch rather than rewriting the executor.

## 6. Core causal hypotheses to test

H1: Smaller C with G4 shortens very long prefill interruptions, but incurs more chunks and potentially smaller GEMMs / per-chunk overhead; these must be evidenced by actual events/kernels, not assumed.

H2: Smaller G at fixed C reduces per-submission duration but increases the number of opportunities to interrupt decode. It may reduce >200ms stalls while increasing 100–200ms stalls, as the old G2 C1024 condition suggests.

H3: The joint CxG response surface is not fully explained by the product C*G or a single scalar time budget. If all performance points collapse onto one simple cost axis, an adaptive 2D controller may have little additional value over strong adaptive C-only.

H4: The trade-off changes with P, N, active decode batch width, actual KV context lengths, graph capture eligibility, and available HBM/staged hidden capacity.

H5: For some preregistered SLO/workload cells, a static interior (C,G) configuration lies outside the Pareto envelope of individually tuned C-only, G-only, and Mixed. If **no** such cells exist, adaptive joint superiority remains unjustified.

H6: A causal online policy based on calibrated expected quantum duration, decode deadlines/slack, current batch, prefills remaining, staging memory, and per-token SLO can select an appropriate C/G without future outputs. This is **not** permission to implement a controller before H3/H5 evidence exists.

### Estimation model to measure (not assume)

For candidate q=(C,G) and state x=(actual prompt rows, prior KV length, active decode rows, graph bucket, resident state, memory pressure):

`T_quantum(q,x) = T_launch + T_state_save_restore + T_GEMM_attention(q,x) + T_control_sync + T_queue_effect`.

Estimate `T_quantum` empirically. Report prediction error on held-out shapes as p50/p95 absolute and relative error. Distinguish elapsed GPU stream/wall time, host scheduling interval, and user-visible inter-token gap. Do not sum overlapping interval durations as if they were serial.

## 7. Phases and decision gates

### Phase 0 — Historical data reuse and audit (mandatory, no new GPU)

1. Load **raw** previous `results_raw.csv`, `paired_comparisons.csv`, `ITL_stall_counts.csv`, `natural_output_parity.csv`, traces, experiment manifests and failures. Verify hashes/provenance and identify warm/cold/rejected/JIT-order lanes before aggregation.
2. Recompute C-only/G-only performance maps from previous discovery grid. Mark one-repeat points as exploratory, repeated held-out points separately; do not use a single discovery cell as a final dominance claim.
3. If per-request ITL/TTFT time series exist, recompute **all prespecified absolute SLO grids** (both all-token request-pass and pooled quantile limits); identify missing raw fields. Never infer request success from aggregate counts alone.
4. Audit 112/120 batched sequence mismatch requests; write which mismatch mechanisms are unresolved and the tests required prior to claiming quality equivalence.
5. Review source and prior-work coverage; record engine/backend/commit lock and actual comparison fairness.

**Gate G0:** Evidence audit complete with a `missing_data.md` record and no fabricated SLO counts. If raw per-request timestamps are missing, schedule new captures.

### Phase 1 — Calibrate thresholds and instrumentation

1. Record warmed decode-only iteration/ITL by context and batch width, and isolated full-prefill TTFT for ~2K/~8K; confirm GPU4 physical ID and sufficient HBM.
2. Freeze relative Strict/Moderate/Relaxed SLOs and TTFT primary thresholds **before** method comparisons. Fix an independent calibration split and a distinct held-out workload split.
3. Build client monotonic request arrival/token timestamp recorder (streaming if available). Log scheduled arrival, actual send, first token, each delivered token, EOS/stop/cap, completion, errors, dropped requests, timeout, queue length over time, and system utilization.
4. Validate light instrumentation overhead vs no-observer runs; place Nsight/NVTX traces in separate lanes, never mix heavy traces into headline latency.

**Gate G1:** No clock/denominator ambiguity; demonstrably sufficient request/TBT samples for target percentile. Small sample p99 remains `EXPLORATORY`.

### Phase 2 — Fixed C/G characteristics with controlled workloads

1. Run C-only G4 sweep, fixed-C G1/G2/G4, and full static CxG map under *the same* FA4, graph mode, reserve and 4-request/8-request arrivals.
2. Include tuned Mixed G1, true no-chunk if valid, and staged G4. Profile latency per quantum, frequency of submitted quanta, achieved actual C, active prefill/decode rows, CUDA graph buckets/hits/fallbacks, host submission gaps, KV page pressure, staged hidden memory, and remaining prompt length.
3. Produce per-request ITL survival function / CCDF and a **stall severity distribution** split into <=50, 50–100, 100–200, >200ms (also continuous full distribution); report event rates **and** per-request max ITL compliance. Do not make the bins the only analysis.
4. Compare prefill-only P1 baseline vs mixed requests; isolate whether regression is staged-state cost, graph cut, batch packing or more frequent prefill interference. Use repeated matched controls and Nsight when causal attribution remains ambiguous.
5. Randomize/counterbalance method order across independent cold engine starts, equal warm-up per method, freeze the graph-capture cache policy, and log first-use JIT. Do not repeat the previous Atomic-first order confound.
6. Repeat only pre-registered shortlisted cells with >=3 separate server starts and >=10 paired cohort repetitions; expand as necessary for tail CI. Use stratified/hierarchical bootstrap by server-run and request/cohort, not independent tokens or kernels.

**Gate G2 (characterization):** A completed explanatory characterization can be PASS even when no policy improves all metrics. Keep independent `G2_motivation_headroom` flag distinct. Specifically test whether the G2 >200ms-stall benefit persists against smaller-C tuned Atomic and stock Mixed and whether its cost can be quantified. No automatic adaptive implementation from a p99-only success.

### Phase 3 — SLO sensitivity and actual sustainable serving rate (mandatory)

1. Take C/G/Mixed shortlist from independent calibration, freeze on held-out. Run **open-loop** Poisson scheduled arrivals plus a fixed seeded burst-trace. Workloads must include overlapping long prefill arrivals and active decode.
2. Determine initial lambda0 from an independent, stable native RLT C-only calibration, **not from the candidate**. Consider `[0.25,0.50,0.75,0.90,1.00,1.10] * lambda0`, refine near transition only when safe. The target system is a single GPU; rates must be empirically calibrated, not invented from earlier burst req/s.
3. Plot attainment versus offered `lambda`: combined; TTFT-only; TBT-only (Layered Figure 3/4 style). Show throughput, goodput, queue delay, unfinished backlog, timeout/OOMs, p50/p95/p99 TTFT/TBT, per-request maximum TBT and request E2E.
4. Count all **offered requests**, not only successful or completed; allow a declared drain window, and explicitly distinguish completed-in-window, drained, missing, errors. Show both offered-arrival-window SLO goodput and drained-cohort throughput with distinct denominators.
5. Require stable queue behavior. If the backlog grows over measurement intervals, mark arrival rate **UNSUSTAINABLE**, not a valid capacity point. At chosen 90% attainment, use uncertainty-aware estimates and upper/lower rate brackets.
6. Evaluate **multiple preregistered SLO settings**. No selecting 200ms as primary solely because old G2 improved >200ms event count. Compare curves; optional show max-ITL request pass vs pooled-P99 TBT compliance as separate panels.
7. Report sensitivity when TTFT is tight versus TBT is tight; document which resource and phase cause each failure. Qualify any capacity improvement with complete natural EOS and model-output quality checks.

**Gate G3:** Request-rate-versus-attainment curves are repeatable, include uncertainty, sample sizes and no hidden queue divergence. This is the primary scientific deliverable irrespective of whether adaptive optimization proceeds.

### Phase 4 — Adaptive policy only if evidence supports it (conditional)

**Prerequisites:** Fixed CxG has identifiable nontrivial headroom beyond tuned adaptive C-only and RLT Mixed; G2 rare-stall advantage survives valid comparisons; cost predictor is measured and stable; no unexplained correctness failure.

1. Start with simple *online* C-only (G4), G-only (C fixed), and C+G policy using the **same** deadline estimator and process budget. Candidate includes `G∈{1,2,4}` and `C∈{128,256,512,1024}` clipped by actual feasible token count and memory.
2. At every legal boundary, decide based on observed queue state, currently outstanding decode deadlines, warmed predicted processing time and measured memory availability. No future request arrivals, output tokens, true future EOS, future measured runtime, or oracle exit depths.
3. Use same engine, kernels, buffer strategy, graph rules and scheduler fairness; do not quietly switch Mixed/No-refill policy. If this is technically impossible without substantial refactor, mark BLOCKED and retain the characterization result.
4. Include fixed tuned C-only; adaptive C-only; fixed tuned G-only; adaptive G-only; best static CxG; RLT Mixed; adaptive joint. Evaluate across strict, moderate, relaxed SLO grids, low and medium loads and bursts.
5. Repeat held-out runs to show an uncertainty-bounded improvement in 90%-attainment serving capacity or SLO goodput without unacceptable TTFT/E2E or output-quality loss. Distinguish `MODELED` oracle envelope from real online `LIVE_MEASURED` results.

**Gate G4:** GO only for clearly repeatable client-visible gain versus **strongest** tuned C-only and Mixed controls under frozen SLO/workload; if a 1D policy matches joint, report `GENERIC_ADAPTIVE_CHUNK_SUFFICIENT` and stop.

### Phase 5 — Multi-GPU portability (only if the relevant previous gates pass)

- Start single GPU physical4; extension uses 4/5 (two independent RLT replicas), then 4/5/6/7 (four replicas) with the same frontend/load balancing and same workload **per-gpu plus cluster-wide** metrics.
- Replicas provide **DP-style multi-GPU serving capacity**, not multi-GPU inference for an individual request. Be exact in tables/captions.
- Optional TP2/TP4 inside a single request requires independent verified support and may be BLOCKED. **Do not restart the paused native vLLM Loop PP program** or rebuild a TP/PP engine for this PoC.
- Measure normalized SLO-capacity/GPU, replication load skew, GPU memory and cost, and whether the same scheduling policy remains beneficial. Do not claim linear scaling without actual measurements.

## 8. Figures, tables and publication-oriented outputs

**F1: Motivation — distribution instead of one chosen tail statistic.** Empirical ITL CCDF / survival curve with marked 100ms and 200ms, Atomic G4 / G2 / stock G1 / Mixed. This explains why p99 can improve while p95 worsens.

**F2: C x G response surfaces.** Heatmaps for quantum duration mean/p95, submissions/1000 prompt tokens, TTFT, background ITL p95/p99, and staged hidden bytes; show *actual C* vs configured C for Mixed.

**F3: TTFT–TBT Pareto frontier.** Points for fixed C-only G4, G-only at fixed C, all fixed CxG, RLT Mixed, with confidence intervals; flag dominated/non-dominated points. Do NOT claim a joint advantage because G2 is an interior configuration absent from baselines.

**F4: Layered-Prefill-style open-loop SLO attainment vs offered req/s.** 90% reference line. Separate TTFT-only and TBT-only attainment and combined. Different SLO grid settings in separate panels; use the same arrival trace and prompt population per policy.

**F5: Goodput–SLO and sustainable-capacity plots.** Absolute or calibrated TBT threshold on x-axis, highest safe req/s with >=90% combined request compliance on y-axis, separate TTFT constraints; show sample adequacy.

**T1: Latency/efficiency.** P50/P95/P99 TTFT, token-level ITL, per-request max ITL quantiles, TPOT, E2E, GPU kernel/host time, output tokens/s, prefill tokens/s; include prompt/output length and observed N.

**T2: Correctness and capacity.** Full R4 issued work, numerical probes, batched output parity, GSM8K/MATH accuracy (if run), HBM peak, KV pool/staged activation, graph memory when available, failures.

**T3: Prior-work positioning.** Source-backed feature matrix for CDB/RLT, Layered Prefill, SLOWeave, FlowPrefill, P-PAS, DuetServe; no claims of nonexistent related work.

**Figure integrity:** no illustrative made-up measured data. No p99 from 24 requests as qualified final. Make a distinction between means-of-cohort quantiles and globally pooled quantiles; preregister primary statistic.

## 9. Statistical and fairness controls

- Predeclare the primary workload, primary SLO grid and stopping decision before held-out candidate runs. A sensitivity sweep is valid exploratory research; clearly label it exploratory rather than retroactively claiming a primary 200ms SLO.
- Use common prompt IDs, sampling seeds, arrival timestamps, generation cap and EOS policy across arms. Different natural outputs are allowed but must be logged. For repeated same-prompt request families, bootstrap at the request-family/engine-run level; do not bootstrap every token independently.
- Use >=3 fully independent engine startups; >=10 randomized, paired cohorts for shortlist; more for stable p99 and capacity. Aim >=1000 completed requests per arm/rate for p99 and preferably 5000 when reasonable, but scale safely and label all sample shortfalls. Use 5000+ bootstrap replicates with fixed analysis seed.
- Use same FA4 backend, graph setting, token budget and memory headroom, unless assessing those variables as separately labeled ablations. Each policy may have different actual batch composition; measure it, not falsely claim identical kernels.
- Run warmed trials and alternating randomized policy order. Separate JIT/cold captures, warm steady-state and genuine per-request dynamic captures. Do not drop normal request-bound graph overhead from client latency.
- Include full stopped/no-chunk control only when memory and runtime limits permit; no changing max context unilaterally. Expose difference between token-chunk count, recurrent-loop group count, and number of *actual submitted work batches*.
- Candidate beneficial thresholds must not be selected after seeing held-out performance. Include negative results in reports.

## 10. Correctness and safety gates

1. **No changed outputs by fiat.** No teacher-forced output or future oracle in primary Serving; model BF16 and official weights unchanged. Prompt full-R4 and decode full-R4; G only changes prefill execution granularity.
2. Exactness probes for C128/256/512/1024 and G1/G2/G4 on P128/512/2048/8192 including logical KV namespaces and hidden state, logits and actual output-token traces; note full batched BF16 differences separately.
3. Specifically reexamine legacy 112/120 batched output mismatch: identify first divergent token and whether it is batch arithmetic, graph/backend, race or lifecycle before claiming universal parity. Require no unexplained *placement-specific* quality degradation; full reasoning accuracy only through completed naturally stopped answers.
4. Test interrupted/résumed prefill states at R1/R2/R3, C boundaries, preemption/cancellation, out-of-order arrivals, request-ID reuse, memory pressure, EOS, timeouts, and safe graph buffer reuse. Never access uninitialized KV or overwrite other depths' history.
5. Keep all GPU activity within physical GPU 4/5/6/7, no unauthorized removal of burners, no OOM storm or interference with other jobs. Initial capacity runs use safe headroom and abort on unsafe memory states.
6. If large-sample open-loop traffic or complete response quality cannot be attained within safe runtime, report `BLOCKED`/`PARTIAL`, not fabricated results.

## 11. Required experiment outputs

Write to `poc_flashvep/reports/ouro_chunk_loop_slo_sensitivity_discovery/`:

- `analysis.md` — causal interpretation, supported claims, unresolved alternatives, verdicts.
- `baseline_audit.md` — legacy source, data hashes, what was already run.
- `literature_slo_contract.md` — Layered vs Sarathi and recent works; links and precise threshold/aggregation definitions.
- `slo_registration.md` — measured no-interference calibration values and frozen strict/moderate/relaxed and TTFT grids **before** held-out comparisons.
- `benchmark_summary.md` — fixed C/G and SLO sensitivity; confidence intervals.
- `slo_attainment_by_rate.csv` — offered arrival rate, SLO tiers, TTFT-only/TBT-only/combined successful offered requests, denominators, stable backlog criterion, confidence ranges.
- `request_events.parquet` or `.csv` — request ID, scheduled arrival, actual admission, first token, every token time, EOS/stop, completion, exceptions, timeout; preserve complete raw trace.
- `quantum_events.csv` — per-submission actual C, G, recurrent depth ranges, prompt/decode active rows, batch size, start/end, graph replay/capture, resource use.
- `cg_response_surface.csv` — aggregate per C/G/workload and uncertainty.
- `pareto_frontier.csv` — nondominated candidates under fully specified metrics and SLO.
- `itls_by_request.csv` — distribution, per-request max, event counts, quantile aggregation type.
- `correctness.md` — parity failures, all 96 namespace test coverage, complete-answer quality scope.
- `profiling.md` — staged-state, GPU kernel vs submission times, graph/JIT, copy, pressure.
- `adaptive_feasibility.md` — only if Phase3 justifies Phase4; estimate whether 2D improves beyond C-only.
- `experiment_matrix.csv`, `reproduce.md`, `failures.md`, `decision_gates.md`, `run_manifest.json` and exact source hashes.

Make small figures programmatically if this materially helps interpretation, but do not fabricate missing data; no user-facing model weights or binary caches.

## 12. Terminal assessment requirements

Write two independent decisions:

1. `CHARACTERIZATION`: PASS / CONDITIONAL / BLOCKED — whether causal C/G features and multiple SLO boundaries are adequately characterized.
2. `ADAPTIVE_JOINT_HEADROOM`: GO / CONDITIONAL / NO_GO / NOT_RUN — whether a 2D policy is justified over tuned C-only, G-only and stock RLT Mixed.

`MULTIGPU_PORTABILITY` is another separate GO / CONDITIONAL / BLOCKED / NOT_RUN, not bundled into the main research verdict.

In the final report, answer clearly:

- Where does C-only lose efficiency? Is the cause actually small GEMMs, launch overhead, cache reprocessing, or batch packing?
- Where does G-only lose responsiveness or TTFT? Is it state/graph overhead or longer single-loop quantum?
- Does C*G describe quantum runtime adequately, or is there measurable interaction?
- Does the optimal **fixed** pair (C,G) change with TTFT/TBT SLO or workload? How much beyond tuning C?
- Which latency threshold matters: per-request all-token TBT 100ms, 200ms, calibrated 5x/10x/25x, or aggregate P99? Explain that these are different objectives.
- Is the strong original RLT Mixed / adaptive C-only result already sufficient?
- Can a fully online feasible Q(C,G) policy improve the **sustainable 90%-attainment serving capacity**? If not tested, do not speculate as if proved.
- What extra staged activation, graph and implementation cost would be required?

A strong **negative result** with full C/G sensitivity and explicit evidence is preferable to rushing into an expensive new scheduler or multi-GPU runtime.

---

### References for source audit (not measured PoC results)

1. Layered Prefill, MLSys 2026: https://proceedings.mlsys.org/paper_files/paper/2026/file/c0f460c6d63599ea870ba9db63dc96a9-Paper-Conference.pdf
2. Sarathi-Serve, OSDI 2024: https://www.usenix.org/conference/osdi24/presentation/agrawal
3. SLOWeave, Sept 2026: https://arxiv.org/abs/2609.07883
4. FlowPrefill, Feb 2026: https://arxiv.org/abs/2602.16603
5. P-PAS, Aug/Oct 2026: https://arxiv.org/abs/2608.15171
6. DuetServe, ICML 2026: https://proceedings.mlr.press/v306/gao26m.html
7. CDB, Aug 2026: https://arxiv.org/abs/2608.09444
8. vLLM-RLT: https://github.com/InferMatrixLab/vllm-rlt
9. Existing Ouro C/G PoC analysis: https://github.com/jyssys/MLLM-EP/blob/c458f6a2a764f667a0da6784bcb2592b36626837/poc_flashvep/reports/ouro_chunk_loop_prefill_joint_discovery/analysis.md
