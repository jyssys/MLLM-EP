# PoC Specification: Predictive Asynchronous Recurrent Pipeline Parallelism (PARPP)

**Status:** Research PoC specification (not an implementation or measured result)  
**Date:** 2026-10-17  
**Priority:** New multi-GPU parallel execution architecture; **not** reinforcement-learning policy optimization  
**Hardware constraint:** **Only physical NVIDIA GPUs 4, 5, 6, and 7** may be used  
**Default checkpoint:** `ByteDance/Ouro-1.4B-Thinking`  
**Primary scope:** Exact fixed-depth R4 baseline; approximate recurrent-stage parallelism with **warm-up = 1 and 0**; *accuracy before distributed engineering*.  
**Secondary scope:** Dynamic-depth/early-exit compatibility **only after primary quality and feasibility gates pass**.

---

## 0. Executive decision and non-negotiable requirements

**Research question.** Can a low-cost, causally available prediction of intermediate hidden states relax the *within-recurrent-pass* layer-stage dependencies of a looped transformer enough to enable multi-GPU asynchronous execution with near-baseline mathematical reasoning accuracy and real end-to-end latency improvement?

We seek an **AsyncDiff-inspired change to the execution dependency graph**, not a new queue-aware DP scheduler or simple work stealing. The implementation must expose a genuinely concurrent GPU schedule rather than producing approximate hidden states while still executing every stage serially.

**Non-negotiable requirements:**

1. **Accuracy is a first-class output.** Report GSM8K and MATH-500 accuracy for the exact baseline and every substantive approximate method. Use actual free-running autoregressive decoding, not just teacher-forced logits. Preserve outputs, sample IDs, prompt templates, parsing rules, failure examples, and statistical uncertainty.
2. **No fabricated measurements.** Record all sample counts, exact hardware/software versions, runtime errors and OOMs. Label analytical upper bounds, simulated schedules, teacher-forced analysis, and real GPU timings distinctly.
3. **No model-weight modification or low-precision quantization** in the initial PoC. Use BF16 if supported and stable; document any fallback. A tiny predictor or calibration coefficient is allowed if its data, cost, and causal inputs are explicit; do not train or fine-tune Ouro itself.
4. **Depth-indexed KV by default:** independent KV state indexed by `(recurrent_depth, physical_layer, token_position)`. No single-slot shared-KV shortcut. Match the exact checkpoint's cache and attention semantics.
5. **Warm-up must be 0 or 1.** Warm-up 1 is the first practicality baseline. Warm-up 0 is a required feasibility attempt with explicit quality results or a clearly evidenced impossibility/NO-GO; do not use 2+ exact warm-up loops as a hidden rescue.
6. **Fixed R4 first**, with exactly the same tokenization, prompts, decoding, prefill configuration and KV policy as the approximate implementations. Dynamic-depth results are a separate extension, not a substitute for the fixed-depth primary result.
7. **GPU isolation:** restrict all CUDA/NCCL code to physical GPUs 4,5,6,7. Under `CUDA_VISIBLE_DEVICES=4,5,6,7`, logical `cuda:0..3` correspond to physical `4..7`; print and verify the mapping before distributed tests. Do not change or terminate unrelated GPU workloads.
8. **Avoid a misleading theoretical speedup.** A 4-stage/4-loop toy schedule suggests up to 16/7=2.29x for warm-up 1, and up to 16/4=4x for warm-up 0, **only against a purely serial 16-stage reference with equal stage costs, zero communication/prediction/synchronization overhead, and successful approximations.** It is **not** a measured speedup or an expected gain against optimized TP4/DP4.

### Deliverable decision states

- **GO_ARCHITECTURE:** full quality gate + real or cost-grounded concurrency speedup gate pass; multi-GPU prototype demonstrably overlaps stages.
- **QUALITY_FAIL:** approximation materially harms task accuracy even if hidden states appear similar.
- **FEASIBILITY_FAIL:** predicted-input path is too costly or an executable causal dependency graph cannot be constructed.
- **PERFORMANCE_FAIL:** feasible/accurate path has inadequate measured advantage over the strongest fair baseline.
- **INCONCLUSIVE:** data are missing, confidence intervals are too wide, official baseline fails validation, or GPU resources cannot support a fair test.

---

## 1. Evidence and implementation audit

### 1.1 Read and pin the original sources

- AsyncDiff (NeurIPS 2024): https://arxiv.org/abs/2406.06911 and https://github.com/czg1225/AsyncDiff
- Official Ouro-1.4B-Thinking checkpoint/model card: https://huggingface.co/ByteDance/Ouro-1.4B-Thinking
- Official Ouro implementation (`modeling_ouro.py`): https://huggingface.co/ByteDance/Ouro-1.4B-Thinking/blob/main/modeling_ouro.py
- CDB (depth-adaptive looped LM serving): https://arxiv.org/abs/2608.09444 and https://github.com/kschwethelm/continuous-depth-batching

AsyncDiff uses *features from an adjacent denoising step* to relax dependencies among partitioned model components; its stride optimization is separate. We are translating the **dependency-relaxation research idea**, not claiming to reuse diffusion-specific quality properties. No claim of equivalence between diffusion timesteps and language-model recurrent loops.

### 1.2 Critical Ouro implementation facts to verify in the checked-out commit

- The model card describes 24 reused transformer layers and default `total_ut_steps=4`.
- The official recurrent model forward iterates `current_ut` through all physical layers and produces per-loop hidden states/gate scores; the CausalLM implementation can **select** an output at an earlier loop after full-loop computation. **Selecting an earlier output does not by itself prove actual recurrent GPU compute was skipped.** Verify this separately when testing dynamic depth.
- There is a terminal normalization operation at a recurrent pass boundary. Instrument the exact tensor entering each physical stage and the normalized state passed to the next recurrence; do not conflate pre-norm and post-norm.
- The official cache uses recurrent-depth-specific physical layer indexing. Verify attention reads, writes, causal masks, cache positions, and current-token KV update order.
- CDB warns that single-slot KV can severely degrade Ouro GSM8K accuracy. Do **not** choose it merely for speed or ease of pipeline implementation.

Pin checkpoint revision, model code revision, PyTorch, Transformers, CUDA, driver, GPU model, and optional CDB revision. Save source file hashes. If a library patch is required to make Ouro cache work, log the exact diff and prove reference equivalence before running experiments.

### 1.3 Repository location

Suggested project root **if** the existing repository is `/home/esjung/MLLM-EP-github`:

`/home/esjung/MLLM-EP-github/poc_async_recurrent_pp/`

**Never assume the repository path exists.** Inspect the actual workspace, reuse existing conventions, and place the spec at `poc_async_recurrent_pp/reports/spec.md` after copying it into the remote checkout. Do not create a separate repository if there is already a suitable project structure.

---

## 2. Formal model and the exact dependency graph

Let `t` index an **autoregressively generated token position**, `r ∈ {1,...,R}` a recurrent depth, `j ∈ {1,...,P}` a pipeline stage, `P=4`, `R=4` for the primary experiment, and `x[t,r,j]` the residual-state tensor after stage `j` of recurrent step `r` (with exact normalization conventions noted).

The exact recurrence is schematically

```text
x[t,1,0] = Embed(token_at_position_t) with model-specific preprocessing
x[t,r,0] = BoundaryNorm(x[t,r-1,P])                 for r > 1
x[t,r,j] = F_j(x[t,r,j-1]; KVpast[t,r,j], position_t)
```

where `F_j` contains contiguous physical transformer layers; KVpast is **the correct historical token cache of this recurrent depth and layers**. Include any actual boundary normalization at its exact location per the pinned code (the equations are schematic, not permission to insert extra norms). For Stage 1, the input at the start of recurrent round `r` is available when the preceding recurrent round has completed. For Stage `j>1`, its exact input at recurrent round `r` requires the **current** round's preceding stage output, creating a serial edge `F_(j-1)(r) → F_j(r)`.

**Baseline DAG:** with one token and P=4, R=4, there are 16 dependent stage executions (ignoring fine-grained intra-stage kernels). Splitting weights across GPUs does **not** remove the serial path.

**Target relaxed DAG:** in a given recurrent round, predict each stage `j>1` input from information **available before that round's concurrent stage execution**, and evaluate `F_1(r),...,F_P(r)` in parallel, then finalize the round and advance to `r+1`. This is *approximated within-round stage parallelism*, **not** concurrently executing `R1` and `R2` of the same token without respecting the recurrent-round boundary.

### Dependency invariants

- No predictor may inspect the current token's exact `x[t,r,j-1]` unless it has already been computed in the legal schedule; doing so is **oracle leakage**, not a valid asynchronous inference mechanism.
- A previous-loop same-stage activation is available for `r>=2` **only after that loop completed** and only if retained.
- The current token's input (`x[t,r,0]`) is available at round start **only after** preceding recurrent output/normalization is ready.
- For `r=1`, a zero-warm-up predictor must work without any previous-loop activation from the **current** token. It may use the current embedding, positional features and prior-token states already produced in the causal prefix. Do not use future-token or baseline-only hidden states.
- Approximate activations modify the Q/K/V and hence the persisted current-token KV of a stage. Free-running evaluation must propagate this effect into later tokens; teacher-forced probes alone are insufficient.
- Record whether gates operate on exact/approximate states and whether any rollback or exact correction occurs. All corrections and communication must enter the latency model.

---

## 3. Models, datasets, workloads, and splits

### 3.1 Models

**Primary:** `ByteDance/Ouro-1.4B-Thinking`, 24 physical layers reused up to R4, four contiguous physical stages (`L1–6`, `L7–12`, `L13–18`, `L19–24`) initially. Verify 1-indexed layer labels correspond exactly to 0-index code. Later consider load-balanced alternative cuts **only if** profiling shows >15% stage time imbalance; report both cuts.

**Optional follow-up:** Ouro-2.6B-Thinking only after primary results. Do not replace the primary model mid-study without documentation.

Use `model.eval()`, frozen weights, consistent dtype, deterministic greedy decoding (`temperature=0`, `do_sample=False`) and matching tokenization/chat template across every arm. Explicitly fix EOS handling, `max_new_tokens`, stopping criteria, attention backend, flash attention mode, and cache implementation. Prefer `max_new_tokens=2048` for math accuracy (or more if necessary and feasible); disclose truncations and rerun at a higher cap if truncated outputs meaningfully bias results. **Do not** hide generation-length differences, since a method can appear faster by terminating early or truncating.

### 3.2 Accuracy workloads

- **GSM8K:** canonical held-out `test`, up to **1,319** examples. Evaluate exact numeric answers with a documented normalized/flexible-match parser. Preserve model generations, valid/missing final answers, errors and per-example correctness.
- **MATH-500:** canonical 500 examples. Define a deterministic boxed-expression/math-equivalence evaluator; validate the parser against a hand-audited subset. Report parsing failures separately from mathematically incorrect answers.
- Optional lightweight sanity tasks: exact fixed-depth vs patched-code output invariance, reasoning/non-reasoning mix, longer-context stress. These are **diagnostics**, never replacements for math task accuracy.

**Split discipline:** use an independent non-evaluation **calibration set** to fit predictor parameters/hyperparameters. Do not tune, choose thresholds, or inspect example correctness on the final GSM8K test or MATH-500 set. Keep a separate held-out pilot set where possible. Never use generated labels, exact stage activations, or answer labels from the evaluation sets as an inference-time feature.

### 3.3 Progressive workload schedule

1. 16–32 hand-checked smoke prompts: run integrity checks and parser validation only; numbers **not** used as performance/quality claims.
2. ~128 independent held-out math prompts for fast **screening**, with explicitly wide uncertainty and paired per-item comparisons.
3. Full GSM8K-1,319 and MATH-500-500 for **any GO-quality claim**. If capacity does not allow full evaluation, report actual n and **INCONCLUSIVE**, not GO.
4. Latency: decode batch/concurrent requests 1, 2, 4, 8; context lengths 512, 2,048, optionally 8,192 when memory permits; fixed output cap and matched output length replay for capacity measurements. For realistic E2E, additionally measure natural-output workloads with actual generated lengths.

Keep the model's exact full-depth prefill for **all initial decode-only comparisons**. The work targets decode first; a fully approximate prefill is a separate, explicitly out-of-scope extension. Both exact and approximate arms must start from equivalently initialized historical KV state.

---

## 4. PoC stages, execution order, and exit criteria

### Stage 0 — Audit and certify exact Ouro baseline **[mandatory]**

1. Detect the actual GPU model, HBM, CUDA driver, NCCL, interconnect (`nvidia-smi topo -m`), visible IDs, other processes, available disk/RAM. Run only on physical GPU 4 for first single-device checks if free, otherwise another in the allowed set. Do not interfere with other users.
2. Load pinned official checkpoint + tokenizer and verify loop count, gate semantics, layer partition plan, norm positions and depth-indexed KV indexing.
3. Implement an **instrumented exact forward** (`Exact R4`) that permits reading/writing stage-boundary activations without altering computation. Compare with unmodified official inference using the same kernels/dtype and seeds on a deterministic bank: exact token IDs and EOS behavior, logits/logprobs discrepancies, KV positions and lengths. If a numerically different code path is unavoidable, characterize the source and do not proceed without a credible invariance check.
4. Evaluate a sufficiently large baseline sanity bank, record GSM8K/MATH-500 baseline pilot score and check that accuracy is plausible under the **same** evaluation template/decode cap; avoid treating unrelated published model scores as direct apples-to-apples targets.
5. **Do not claim dynamic-depth compute savings** merely because a different loop's logits were selected. Confirm executed recurrent layers by kernel traces/counters.

**Stage-0 gate:** reproducible exact reference, correct KV and generation, full metadata; otherwise STOP `BASELINE_INVALID`.

### Stage 1 — Recurrence and boundary-state measurements **[mandatory]**

On exact fixed-R4 decoding, capture for a **bounded** calibration subset:

- `x[t,r,0]` (actual recurrent-round input), stage inputs and outputs at `j=1..4`, post-boundary norm; optional Q/K/V norms; per-layer causal KV indices and bytes.
- Same-position previous-loop similarities and the *actual stage-input forecasting target*, with clear pre/post norm labels.
- Cosine similarity, relative L2 (`||prediction-target||/(||target||+eps)`), normalized RMSE, drift magnitude/direction, stage-wise sensitivity/JVP proxies if economical.
- Top-1 logit agreement, KL divergence and logit margin **only when a candidate predicted input is actually injected into an otherwise well-defined execution path**.
- Stratify by recurrent step, stage boundary, token index, input length, and easy/hard question type. Log finite numbers, timing, VRAM, sample count.

**Mandatory distinction:** similar `x[t,r,j]` across recurrences does **not** prove that substituting the state is safe. The target for predictor `F_j(r)` is its correct stage input `x[t,r,j-1]`, not an arbitrary similar output.

Storage policy: stream summary statistics, sample bounded activations (e.g., 1–4K tokens), cap GB, store in fp16/BF16 as *diagnostic copies* only; never accidentally quantize actual model compute.

### Stage 2 — Offline predictors and causal input audit **[mandatory]**

Implement at least these candidates (separate CLI switches and result files):

- **E0 exact:** original fully sequential R4; zero prediction overhead.
- **E1 stale previous-loop:** for `r>=2`, stage `j>1` input estimate uses the *same-position previous recurrence's corresponding stage input*. This is the simple AsyncDiff-inspired naive comparison, not a guarantee of accuracy.
- **E2 delta residual forecast:** for `r>=2`, stage `j>1` predictor estimates

  `z_hat[t,r,j] = x[t,r-1,j-1] + alpha[r,j] * (x[t,r,0] - x[t,r-1,0])`.

  Here `z_hat` approximates exact input to `F_j(r)`, while `alpha[r,j]` is a calibration-only scalar or tiny diagonal mapping fitted **off** evaluation data. Validate shape/norm consistency. If this particular drift formula is poor, test a bounded number of predeclared similarly cheap causal alternatives; avoid open-ended tuning on test sets.
- **E3 W0 initialization:** for **first recurrence** `r=1` and stage `j>1`, use a cheap estimator based only on `x[t,1,0]`, known position/embedding and *causal prior-token* recorded state if available. Start from simple identity/linear/low-rank calibration and measure inference cost. No current-token exact stage activations from the baseline. Training the predictor on a calibration split is permitted; training Ouro is not.

For **warm-up 1**, run R1 *exactly*, then use E1/E2 on subsequent recurrences. For **warm-up 0**, use E3 on first recurrence **and** E1/E2 for R2–R4. All four stages in a relaxed recurrent round must receive inputs already available at round start. After round completion, advance the recurrent boundary normally.

Include a **strict availability trace**: timestamp, source tensor ID, token index, loop index, stage index, producer completion timestamp, consumer start timestamp. Reject a predictor if it needs a tensor not yet produced by that time. An *offline teacher oracle* may be computed separately as an **unimplementable diagnostic upper bound**, clearly labeled `ORACLE_NOT_EXECUTABLE` and NEVER mixed with valid latency/accuracy results.

### Stage 3 — Executable functional emulation and **quality-first** screening **[mandatory]**

Start on one allowed GPU (or CPU for tiny unit tests). Build an **exact execution-semantics emulator** of the predicted-input DAG. Execute stages serially *only as a numerical emulator*, but use *the inputs they would have received concurrently*; do not accidentally update stage-j inputs with current-round stage-(j-1) outputs. Maintain separate pending/committed stage outputs and depth-indexed KV slots.

**Subtle but essential:** serial emulation is valid for **accuracy** only if every simulated concurrent stage receives its predicted input and historical KV equivalent to the real concurrent schedule. It is **not** valid evidence of latency speedup.

Run paired comparisons:

| ID | Recurrence | Warm-up | Stage input | Prefill | Decode | Main use |
|---|---|---:|---|---|---|---|
| A | R4 fixed | N/A | Exact sequential | Exact | Exact | **Primary accuracy/perf baseline** |
| B | R4 fixed | 1 | Previous-loop stale | Exact | Approximate | Naive quality diagnostic |
| C | R4 fixed | 1 | Delta predictor | Exact | Approximate | **Primary W1 candidate** |
| D | R4 fixed | 0 | Causal initialization + stale/delta | Exact | Approximate | **Primary W0 candidate** |
| E | R4 fixed | 0/1 | Teacher exact hidden | Exact | Oracle-only | **Non-executable upper bound, never GO** |

For each real candidate run:

1. Paired **teacher-forced** next-token analysis using a fixed exact prefix **only as a diagnostic**; report logit KL (both directions if helpful), top-1 agreement, logit margin, hidden-state errors by boundary, gate input error (if relevant).
2. Paired **free-running greedy generation**, where newly generated tokens and approximate KV are fed into all subsequent tokens. Evaluate task accuracy with the identical baseline prompt/template and max-output rules; include EOS and truncation status.
3. Run ablations: predict stages `[2]`, `[3]`, `[4]`, combinations; one approximate recurrent round vs multiple; W1 vs W0; alpha=0/stale vs calibrated alpha; with/without a **measured** cheap correction. Document achieved concurrency as a DAG property, not merely number of substituted stages.
4. Report **per-prompt** exact/approx final answers and correctness transitions: right→wrong, wrong→right, both right, both wrong; failure patterns, hallucinated outputs, numerical instability, mean/quantiles of generated tokens.

**Do not** use temperature sampling for the main paired comparison. If outputs are truncated or the parser misses a valid format, report counts transparently and inspect before final accuracy claim.

### Stage 4 — Accuracy gate **[mandatory before expensive distributed prototype]**

Run **all** canonical GSM8K test (1,319) and MATH-500 (500) for candidate(s) that survive pilot screening; at minimum exact A vs best W1 and best W0 candidate. Avoid cherry-picking one dataset.

For each dataset/method, report:

- `n_total`, `n_completed`, `n_valid_answer`, `n_correct`, exact-match/equivalence accuracy percentage, `n_truncated`, `n_generation_error`, `n_parse_fail`.
- Absolute and percentage-point accuracy difference vs A (same per-item IDs), paired correct/incorrect transitions, 95% **paired bootstrap CI** for the accuracy difference (resample **prompts**, not tokens; >=2,000 resamples, fixed seed), plus optionally McNemar discordant-pair counts/test.
- 95% CI for absolute accuracy; if confidence too broad, return `INCONCLUSIVE` rather than pretending point estimates are conclusive.
- `max_new_tokens`, prompt/response length summaries, TTFT/decode-time preliminary values, generated tokens/s and any unexpected early termination.

**Pre-registered practical quality targets (project decision thresholds, not claims from literature):**

- GSM8K absolute accuracy loss **≤ 1.0 percentage point** relative to the locally measured exact R4 baseline.
- MATH-500 absolute accuracy loss **≤ 2.0 percentage points**.
- Paired 95% bootstrap lower confidence bound for `candidate - exact` should be above **−3 pp (GSM8K)** and **−4 pp (MATH-500)**, indicating that a materially larger regression is not plausible under the study's sampling uncertainty. Record CI and limitations rather than blindly thresholding extremely small deltas.
- Unchanged numerical stability, no exploding NaNs/Infs; no systematic severe wrong-answer regressions. No “quality maintained” claim based solely on KL/cosine/token agreement.

If only some thresholds are passed, report the accurate result and its uncertainty; do not silently relax thresholds. **W0 may fail while W1 passes**: that is a valid and important finding, not a reason to hide W0 results. If both fail, stop with `QUALITY_FAIL` and an evidence-backed explanation; skip distributed implementation.

### Stage 5 — Multi-GPU stage-latency measurements, DAG simulator and prototype **[quality gate required]**

**5a. First produce cost-grounded bounds without a full engine.** On allowed GPUs, measure with CUDA events and `torch.cuda.synchronize()`:

- stage-wise exact compute `T_Fj` by batch/context/dtype, predicted-input compute `T_pred`, intra-node transfer bytes/times between physical 4↔5↔6↔7, host scheduling latency, cache memory and synchronization overhead;
- stage partition load balance and bandwidth; `nvidia-smi topo -m` topology and actual rank mappings;
- separate prefill time, decode prelude/round execution, gate/head, postprocessing, KV write/copy costs;
- replay at least 100 timed runs after proper GPU warm-up; publish p50/p95 and dispersion. Timed tests must not share uncontrolled GPU load.

Use measured costs in an event-driven simulator that respects the **predicted-input** dependency DAG and all correct KV/normalization/round-barrier events. Present both `ZERO_COST_OPTIMISTIC` and `MEASURED_COST_REALISTIC` as **simulated** results. A simulator speedup does not constitute real implementation speedup.

**5b. Only if accuracy + realistic oracle look good:** implement a small working 4-process/4-GPU prototype with stage partitions:

```
CUDA_VISIBLE_DEVICES=4,5,6,7
logical rank 0 -> physical GPU 4 -> L1-6
logical rank 1 -> physical GPU 5 -> L7-12
logical rank 2 -> physical GPU 6 -> L13-18
logical rank 3 -> physical GPU 7 -> L19-24
```

At each recurrent round, verify concurrent stage kernels across all four ranks with CUDA/Nsight timeline or equivalent profiling. Predict stage inputs prior to GPU stage launch; stage 1 input comes from prior round/embedding. Preserve depth-indexed KV, model norm, EOS, and historical token semantics. Do not use a hand-tuned time sleep to simulate overlap. Communicate activation tensors efficiently, and count every transfer in elapsed time. Record whether predictor input distribution/correctness in this implementation matches Stage-3 emulation.

**Fair hardware baselines**, as actually feasible in the environment:

- A1: exact R4 on one allowed GPU (replicated data-parallel **single-request** baseline).
- A2: exact R4 on 4-GPU partitioned PP4, with actual transfers.
- A3: TP4 exact R4 **only if a correct model-supporting implementation is available**. Otherwise clearly document `NOT_AVAILABLE` instead of asserting a result.
- A4: independent DP4 replicas with consistent request arrivals and decoded outputs for throughput comparisons; never present aggregate DP throughput as latency of a single request.
- P1/P0: actual predictive async PP4, W1/W0 respectively if their quality gate passed.

Measure **per-request** TTFT, decode TPOT, E2E latency, median/p95 under batch/concurrency 1,2,4,8, throughput, GPU utilizations, GPU HBM peak, critical path, and output token count. For decode-only comparisons, isolate decode. For end-to-end results, include **unmodified exact prefill** in both baseline and candidate; never present decode-only gains as complete E2E gains. Report prefill+decode fractions and amortization with output length.

**Performance threshold for GO_ARCHITECTURE:** after quality pass, real measured ≥15% decode-latency reduction at concurrency=1 **versus the strongest available fair exact four-GPU latency baseline**, and nonnegative meaningful E2E gain after prediction/communication costs, without severe degradation at concurrency 2/4/8. This is a **project decision threshold**, not a proven literature standard. If no fair 4-GPU comparator could be implemented, report `INCONCLUSIVE` for the relative-architecture GO decision even if a single-GPU comparison looks favorable.

### Stage 6 — Dynamic-depth extension **[conditional, not a blocker for fixed R4 discovery]**

Only if some W0/W1 fixed-depth candidate passes a real quality gate:

1. Enable **actual** dynamic early exit (a gate that reduces executed recurrent computation), not post-hoc selection of earlier logits while still calculating R4. Confirm with kernel count/trace.
2. Compare **exact dynamic depth** against **approximate dynamic depth** with matched exit threshold, depth-indexed/last-exited semantics and truly skipped loops. Never compare approximate dynamic only to fixed R4 and claim its overall speedup comes entirely from parallelism.
3. Capture per-token exit-depth distribution (R1..R4; check any minimum-depth guard), mean loops, gate disagreement, earliest error point, quality, exact and approximate KV memory use and additional parallelism available before exit.
4. First use **round-barrier exit decisions** (no next round launched until Gate resolves). Optional speculative preparation/cancel can be tested later and must count wasted work, rollback and KV state.
5. If using CDB, distinguish its replayed exit trajectories from actual online-gate inference and do not inherit the default single-slot KV when quality suffers.
6. Report four paired accuracy/performance arms where possible: exact fixed R4, approx fixed R4, exact dynamic, approx dynamic. State whether improvements arise from fewer executed loops versus faster recurrent-stage execution.

---

## 5. Tests required for a credible concurrency and cache claim

Provide automated tests that fail loudly for:

- **Reference parity:** instrumented exact R4 vs official R4 generate: prompt IDs, logits/greedy output, cache length and per-layer/recurrent index.
- **Stage partition identity:** sequentially evaluating `F1→F2→F3→F4` with exact inputs is numerically equivalent to the unsplit physical-layer recurrence (incl. boundary norm).
- **Round input freeze:** perturb a current-round prior-stage output *after* stage inputs have been frozen; that output must NOT influence a concurrent subsequent stage in the same round.
- **Causal predictor:** inject sentinel/corrupted future-stage states; predictor cannot access them. Log producer-completion/consumer-launch ordering.
- **KV uniqueness:** exactly one active logical cache write per `(token, recurrence, physical layer)` unless a documented exact correction/rollback is explicitly enabled; no accidental cross-depth aliasing; no cross-request leakage.
- **Long decode:** multiple subsequent tokens observe the approximate committed KV (not exact-oracle KV), and after EOS no spurious writes occur.
- **Gate semantics:** selecting loop-2 logits from a four-loop forward does not count as an actual two-loop execution. Profile executed layers to establish savings.
- **Reproducibility:** same seed/config produces the same sample list and evaluation outputs, unless a known deterministic kernel limitation is documented.
- **Prompt/data leak check:** calibration IDs disjoint from held-out evaluation IDs; no serving-time access to exact reference hidden states.
- **Physical GPUs:** process mapping shows only NVIDIA devices 4/5/6/7; no `CUDA_VISIBLE_DEVICES=0,1,2,3` command is allowed for execution on this shared host.

---

## 6. Required artifacts and report organization

Suggested files (adapt to existing repo conventions without creating needless duplication):

```
poc_async_recurrent_pp/
  reports/
    spec.md                       # THIS SPEC copied into remote repo
    environment.md                # revisions, software, GPUs, topology
    execution_dag.md              # exact and relaxed dependency timelines
    baseline_validation.md        # exact parity and baseline accuracy
    activation_analysis.md        # statistics + figures
    accuracy_report.md            # paired FULL math benchmark results
    latency_oracle.md             # measured costs vs optimistic simulation
    final_report.md               # concise findings, constraints, verdict
  configs/
    main.yaml                     # immutable baseline and per-arm settings
  scripts/
    inspect_ouro.py
    validate_exact.py
    collect_activations.py
    fit_predictor.py
    eval_quality.py
    measure_stage_cost.py
    simulate_dag.py
    run_distributed.py           # conditional after GO quality
    aggregate_results.py
  tests/
    test_reference_parity.py
    test_causal_inputs.py
    test_kv_depth_indexing.py
    test_stage_partition.py
    test_eval_parsers.py
  outputs/
    manifests/
    per_sample/                  # prompt-id, completions, parses, diffs
    traces/
    metrics/
    plots/
```

**Exact script names may differ; the functional coverage must not.** All large intermediate dumps should be bounded, compressed or cleaned up only with a documented retention policy. Never delete user data, baseline checkpoints, or unrelated processes.

### Main accuracy table (must be filled with measured values)

| Arm | Warm-up | GSM8K correct / n | GSM8K Acc % | Δ pp [95% paired CI] | MATH-500 correct / n | MATH Acc % | Δ pp [95% paired CI] | Parse fail / truncated | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Exact R4 | – | TBD | TBD | – | TBD | TBD | – | TBD | TBD |
| Previous-loop stale | 1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Delta predictor | 1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Causal initialization + delta | 0 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Main latency table (must not mix simulation with measurements)

| Arm | Measured vs simulated | GPUs | Context / concurrent req. | TTFT ms | TPOT ms | E2E ms | P95 E2E ms | Peak HBM GiB | Quality gate | Notes |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| Exact 1GPU | MEASURED | 1 | TBD | TBD | TBD | TBD | TBD | TBD | Exact | TBD |
| Exact PP4 | MEASURED | 4 | TBD | TBD | TBD | TBD | TBD | TBD | Exact | TBD |
| Exact TP4 (if supported) | MEASURED | 4 | TBD | TBD | TBD | TBD | TBD | TBD | Exact | TBD |
| W1 Async PP4 | MEASURED / UNAVAILABLE | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| W0 Async PP4 | MEASURED / UNAVAILABLE | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| W1/W0 DAG | SIMULATED | 4 | TBD | – | – | TBD | – | – | Depends on arm | optimistic/realistic labeled |

### Required plots

1. Adjacent-loop hidden-state cosine similarity and relative L2 **by (r,j)**, violin/box + sample count.
2. Candidate predicted-stage-input error vs true target, plus downstream logit KL/top-1 mismatch.
3. Paired accuracy changes and 95% CI for each arm/dataset (both FULL and pilot distinctly labeled).
4. Decode critical-path and CPU/GPU timeline; exact PP vs async PP with individual ranks and communication shown.
5. Speedup vs request concurrency and context length, with distinct markers for analytical/simulated/measured data.
6. Peak KV/HBM footprint and late-token drift/error accumulation.

### `final_report.md` format

- **One-line verdict** (`GO_ARCHITECTURE / QUALITY_FAIL / FEASIBILITY_FAIL / PERFORMANCE_FAIL / INCONCLUSIVE`) and highest completed Stage.
- What was **actually executed** (GPU IDs, checkpoint hash, n prompts, scripts, sample outputs).
- Highest-value measurements: exact vs W1/W0 GSM8K & MATH-500 score and paired CIs, plus verified measured vs simulated latency.
- Specific counterexamples where approx switched a baseline-correct answer to incorrect; link to prompt IDs and dumps.
- The causal availability audit and KV-layout correctness evidence.
- Known unresolved blockers, alternative hypotheses, and **one concrete next experiment** only if justified.
- Explicit **NO-GO** when quality or achievable speedup is insufficient; do not write speculative wins as conclusions.

---

## 7. Execution commands (examples; validate CLI flags before running)

```bash
# Only the physical GPUs authorized for this project.
export CUDA_VISIBLE_DEVICES=4,5,6,7
export TOKENIZERS_PARALLELISM=false

nvidia-smi -i 4,5,6,7
nvidia-smi topo -m
python -c 'import torch; print("logical GPU count:",torch.cuda.device_count()); [(print(i,torch.cuda.get_device_name(i))) for i in range(torch.cuda.device_count())]'

# Example ONLY after implementation and local CLI inspection:
# python scripts/validate_exact.py --config configs/main.yaml --device cuda:0
# python scripts/collect_activations.py --config configs/main.yaml --device cuda:0
# python scripts/eval_quality.py --config configs/main.yaml --arm exact_r4 --dataset gsm8k
# python scripts/eval_quality.py --config configs/main.yaml --arm warmup1_delta --dataset gsm8k
# python scripts/eval_quality.py --config configs/main.yaml --arm warmup0_delta --dataset math500
# torchrun --standalone --nproc_per_node=4 scripts/run_distributed.py --config configs/main.yaml
```

**Important:** the above script flags are **not** evidence that the scripts already exist. Implement and check `--help` before running. On the chosen environment, torchrun local rank `0..3` maps to the four devices in `CUDA_VISIBLE_DEVICES` order. Log that mapping and confirm with system GPU process IDs.

---

## 8. Work discipline and incremental milestones

- **M0:** source audit, environment report and exact-baseline parity; stop if not certified.
- **M1:** activation and dependency analysis with real stage-target inputs; include predictor availability audit.
- **M2:** W1 and W0 serial functional emulator, full cost attribution, pilot *free-running* quality and parser validation.
- **M3:** full GSM8K/MATH-500 paired accuracy and confidence intervals; strong quality GO/NO-GO.
- **M4:** cost-calibrated latency oracle and optional real multi-GPU prototype **only after quality GO**.
- **M5:** dynamic-depth extension if architecture shows promise.

Each milestone should write one short dated report, runnable commands, configuration hash, metrics JSON and git commit. If remote `git push` fails or credentials are missing, retain local commits and disclose the exact error; never claim a push succeeded when it did not. Do not ask to widen GPU access. Avoid continuous polling or noisy profiling on a shared machine.

---

## 9. Research caveats that must be preserved in interpretation

1. **Different science from AsyncDiff:** neighboring diffusion timesteps and recurrent computation in an autoregressive LM may have completely different feature dynamics. Similarity is a hypothesis to test, not an established property.
2. **Warm-up 0 can be especially hard:** current-token first-loop intermediate activations do not yet exist. Warm-up 0 results based on future/current exact states are invalid.
3. **Dynamic depth is not the central contribution:** investigate fixed R4 first, then assess whether gated early exits reduce parallel opportunity or introduce extra pipeline benefits. Existing CDB batching/lookahead remains a relevant prior baseline.
4. **KV is a model correctness constraint, not just a memory optimization.** Incorrectly shared or overwritten KV invalidates accuracy claims.
5. **Accuracy and GPU acceleration are two independent gates:** excellent KL or cosine without full math accuracy is not a success; theoretical overlap without actual or measured-cost-supported speedup is not an architecture success.
6. **No inference-time oracle:** any fixed-depth candidate that uses exact reference activations or answer labels to choose its branch is diagnostic only, never deployable.

**Final instruction:** execute the work in the order above. Prefer a rigorous `QUALITY_FAIL` or `PERFORMANCE_FAIL` supported by real math accuracy and measured costs over an impressive-looking but invalid asynchronous pipeline diagram.
