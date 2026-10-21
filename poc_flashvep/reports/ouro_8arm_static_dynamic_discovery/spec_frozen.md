# Ouro-1.4B: Eight-Arm Static/Dynamic Multi-GPU Fidelity, Quality, and Serving Discovery

**Status:** execution specification; not an experimental result  
**Date:** 2026-10-21  
**Working directory (on research server):** `/home/esjung/MLLM-EP-ouro-request-pp`  
**Target specification path (on research server):** `/home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_8arm_static_dynamic_fidelity_serving_spec.md`  
**Allowed physical GPUs:** **4, 5, 6, 7 only**  
**Model:** Ouro-1.4B, one verified and pinned checkpoint, BF16 for comparable inference arms  
**Research mode:** measurement/discovery first, not predetermined Loop-PP advocacy  
**Primary target:** 4 parallelism configurations × 2 depth policies, with independent numerical and task-quality assessments

---

## 0. Executive contract

Build and evaluate these eight configurations:

| Arm ID | Parallelism | Physical GPUs | Depth policy |
|---|---|---|---|
| S4 | Single GPU (TP1/PP1) | 4 | Static R4 |
| T4 | TP4 | 4,5,6,7 | Static R4 |
| L4 | Layer PP4 | 4,5,6,7 | Static R4 |
| O4 | Loop PP4 | 4,5,6,7 | Static R4 |
| SD | Single GPU | 4 | True Dynamic, max R4 |
| TD | TP4 | 4,5,6,7 | True Dynamic, max R4 |
| LD | Layer PP4 | 4,5,6,7 | True Dynamic, max R4 |
| OD | Loop PP4 | 4,5,6,7 | True Dynamic, max R4 |

**Non-negotiable distinctions:**

1. Mathematical TP equivalence is not guaranteed to be bitwise equality under BF16. The existing strict TP numerical gate failed; do not retroactively call it a pass or assume a quality regression.
2. A gated numerical diagnostic failure must **not automatically cancel task-accuracy measurement**. If inference is stable and a bounded correctness audit permits a clearly labeled exploratory lane, measure quality there, separately from qualified serving results. Never launder an unqualified arm into a proven speedup.
3. Static R4 and **real compute-skipping** Dynamic R<=4 are two different quality/latency conditions. Post-hoc gate selection after executing all loops is NOT Dynamic.
4. Dynamic exit changes algorithmic computation; changing parallelism alone must not silently change gate, KV fallback, tokenizer, checkpoint, accuracy protocol, or exit threshold.
5. Each performance result must state the number of GPUs used, execution engine, batching regime, prompt/output lengths, and whether generated tokens/exit traces were matched.
6. The deliverable is a table of eight arms, with truthful `PASS`, `QUALIFIED_WITH_NUMERICAL_WARNING`, `EXPLORATORY`, `BLOCKED`, or `NOT_RUN` statuses. No fabricated fill-ins.
7. Use full **request-level** prompts; **no intra-prompt token chunk partitioning** in the PP comparisons. Different concurrent requests may overlap on different stages.
8. **Do not pursue fixed R2-only as a standalone final target.** The fixed workload is R4; R2/R3 exits are naturally observed within real Dynamic R<=4.

---

## 1. Verified source baseline and required reading

The committed baseline **actually checked** for preparing this specification is the following pinned remote Git revision:

- GitHub repository: `jyssys/MLLM-EP`, commit `6f986ed826a2eaccec5664b38f9a78e217d10e01` on `flashvep/ouro-tp4-parallelism-discovery`.
- [TP4 analysis](https://github.com/jyssys/MLLM-EP/blob/6f986ed826a2eaccec5664b38f9a78e217d10e01/poc_flashvep/reports/ouro_tp4_parallelism_discovery/analysis.md)
- [TP4 benchmark summary](https://github.com/jyssys/MLLM-EP/blob/6f986ed826a2eaccec5664b38f9a78e217d10e01/poc_flashvep/reports/ouro_tp4_parallelism_discovery/benchmark_summary.md)
- [TP4 numerical correctness](https://github.com/jyssys/MLLM-EP/blob/6f986ed826a2eaccec5664b38f9a78e217d10e01/poc_flashvep/reports/ouro_tp4_parallelism_discovery/correctness.md)

On the **local research server**, first inspect the exact working tree, current branch and commit. Do **not** assume the local folder path is identical to the remote repository name, or that `main` contains the experiment. Inspect these related previous-run files as well:

- `poc_flashvep/reports/ouro_request_pp_discovery/analysis.md`
- `poc_flashvep/reports/ouro_request_pp_discovery/benchmark_summary.md`
- `poc_flashvep/reports/ouro_tp4_parallelism_discovery/{analysis.md,benchmark_summary.md,correctness.md,environment.md}`
- Existing reproduction, transport-control, logs, numerical diagnostics, source scripts, and original frozen 21-row workload descriptions.

**Evidence known at specification time:**

- Custom TP4: actual four-rank head/weight sharding and NCCL All-Reduce; 192/192 *bounded operator checks* PASS.
- Custom TP4 short P128/O8 eager and graph probes PASS; P512/O32 full-model aligned numerical contract FAIL: max logit absolute error `5.210938`, max per-step logit mean error `0.802111`, max hidden relative L2 `0.469359`, rank-0 full owner-KV relative L2 `0.024168`.
- All 32 P512 aligned greedy argmax IDs matched on four ranks; this is NOT a benchmark accuracy guarantee.
- Unsharded custom-cache control matched HF reference bitwise at P512 for all 32 steps. Attention SDPA diagnostic head-geometry control reduced max hidden L2 to `0.098947` without qualifying the model. Error peaks were observed at zero-based decode steps 18,19,26,27.
- Native vLLM 0.26.0 TP4 P512/O32 rank-0 max hidden relative L2 `0.498974`; free-running bounded outputs matched 72/72 reference tokens over three probes. vLLM may differ in execution backend and is **not** a certified gold numerical oracle.
- TP4 vs Loop-PP4 serving performance: **NOT_RUN**; 21 R4 workload rows remain unqualified.
- PP C4/P128/O32 median makespan in the updated legacy control: Layer `1.4412s`, Loop `0.8538s`, Layer/Loop `1.688x`.
- Same shape, four-process control: Layer `0.6326s`, Loop `0.5424s`, Layer/Loop `1.166x` (**only three repeats**, MP Layer-PP range `0.6289-1.0529s`). These are PP-only measurements, not vs TP4 or vs Single GPU; the ratio change is not attributable solely to multiprocessing.
- C1 four-process PP was not universally faster than legacy. Process count by itself does not guarantee better speed.

**Do not silently change prior acceptance thresholds:** historical TP4 gate included hidden relative L2 `<= 0.03`, logit max abs `<= 0.75`, logit mean abs per step `<= 0.08`, KV relative L2 `<= 0.03`, stable argmax mismatches 0. This historical gate retains FAIL at P512.

## 2. Goals, hypotheses, and exclusions

### Primary questions

1. What is the task accuracy for each of the 8 arms on **the same exact evaluation examples**, and the percentage-point accuracy delta relative to Single-Static R4 and relative to the same-depth-mode Single GPU?
2. What is the request-level latency and four-GPU cohort throughput for each arm? For each Static/Dynamic mode separately, how much faster/slower is each multi-GPU arm than its corresponding Single-GPU arm on an identical workload?
3. Does the TP4 internal divergence correspond to changed outputs or reduced task accuracy? Is it a TP-specific discrepancy or already present in TP1/backend comparisons?
4. Is Loop-PP's advantage over Layer-PP preserved with the same capable runtime after accounting for stage scheduling, GPU computation, transport, and MP variance?
5. How do true per-token recurrent early exits affect quality, KV correctness, per-rank load, and pipeline utilization? Is any Loop-PP advantage robust to those effects?

### Optional hypotheses, not assumed facts

- Ouro's compact shared weights make complete weight replication affordable and permit an R4 loop-wise pipeline with relatively few stage handoffs.
- For sufficient request concurrency, reduced stage-boundary overhead may offset the loss of finer-grained stage parallelism.
- TP4 may incur substantial repeated collective synchronization across recurrent invocations, but GPU H100/NVLink collectives can be fast; determine critical-path contribution experimentally.
- Dynamic early exits can improve compute efficiency and change rank utilization, in potentially different ways for TP, Layer-PP, and Loop-PP.

### Out of scope for the primary study

- New adaptive/elastic placement or RL scheduler.
- Cross-loop KV compression, quantization/pruning/fine-tuning, model weight changes.
- Prompt-token chunk pipeline, unannounced microbatch aggregation, other GPU IDs.
- Benchmarking fixed R2/R3 as main configurations.
- Claiming a general architecture win or task-accuracy equivalence based only on 32/72 matching token probes.

---

## 3. Execution environment and reproducibility

1. Limit GPU visibility to `CUDA_VISIBLE_DEVICES=4,5,6,7` for four-rank jobs; for Single GPU use `CUDA_VISIBLE_DEVICES=4`. Explicitly record **physical** GPU IDs alongside logical ranks 0-3.
2. Enumerate GPU SKU, free HBM, NVLink topology, process ownership, utilization, active competing jobs, driver/CUDA/NCCL/PyTorch/vLLM versions, FlashAttention backend, and installed/reproducible environments. Use only GPUs the user authorizes. Do not terminate or interfere with other people's jobs.
3. Record model identifier, weight revision/hash and tokenizer hash, config and exact checkpoint mapping. CDB's example Ouro checkpoint may not be byte-identical to the currently used checkpoint; never substitute without checking tensor equivalence.
4. Record `git rev-parse HEAD`, branch, dirty status, source hashes, experiment-config JSON, library lockfile/venv identity. Create an experiment-specific branch without overwriting prior branches, results, or reports.
5. Keep precision BF16 for measured comparable inference; FP32 diagnostic runs are **not** BF16 performance baselines.
6. No prefill token chunking in request-level PP; the full prompt is a stage item. Preserve true cross-request GPU overlap (verified by kernel timelines) for concurrency>1.
7. Use independent, *dedicated* benchmark lanes: (i) clean runtime clock, (ii) heavy Nsight/CUPTI profiling, (iii) numerical correctness instrumentation, (iv) task evaluation. Do not combine observers into headline timings.
8. Record startup, model load, graph capture, warmup separately. State whether they are inside or outside service timing and keep rules consistent across arms.
9. Preserve failures and reproducible scripts; no silent fallback to single-GPU execution on a purported four-GPU arm.

## 4. Parallelism implementations

### S4 / SD: Single GPU baseline

- Physical GPU4. One full 24-layer Ouro weight set, four recurrent iterations for Static; true CDB-compatible dynamic exit for SD.
- Serve the identical request cohort and evaluation questions. Use both a compatibility-matched custom implementation, if needed for exact numerical comparisons, and an optimized TP1 external control when meaningful; label engine variants separately.
- Test possible OOM/memory-limit cases, do not substitute smaller prompts without reporting.

### T4 / TD: true Tensor Parallelism

- GPU4-7, four ranks. Shard QKV heads, intermediate MLP (SwiGLU) and appropriate projection tensors; perform genuine collective reduction. Confirm rank-local shape/ownership and actual NCCL kernels.
- Prefer established **native vLLM 0.26.0 Ouro TP4** as the external reference implementation. Confirm the pinned exact commit/API/version before running; the inspected file uses `QKVParallelLinear`, `MergedColumnParallelLinear`, `RowParallelLinear` and unique `(loop, physical-layer)` attention slots.
- Maintain custom TP4 as an explicitly labeled diagnostic/engine-matched candidate. Do not assume its numerical-contract failure proves TP is impossible. Native vLLM may have different kernels/scheduling/cache layout.
- For Dynamic, vLLM 0.26.0's static Ouro forward alone is insufficient. An actual dynamic-aware TP adaptation is required, or mark TD **BLOCKED** and explain why rather than relabel static vLLM as Dynamic.

### L4 / LD: physical-layer PP4

- GPU4: L1-6; GPU5: L7-12; GPU6: L13-18; GPU7: L19-24. For each recurrent iteration, route the request across all four ranks and back to GPU4 for the next iteration.
- Store per `(loop,layer,request,position)` KV at owner rank; never replace missing dynamic cache with a different policy than other arms.
- Use validated four-process event-driven workers as the leading PP control, but keep the legacy runtime as a *sensitivity control*, not a fourth/fifth parallelism category.

### O4 / OD: loop-level PP4

- GPU4: R1; GPU5: R2; GPU6: R3; GPU7: R4. Each GPU has the full 24-layer physical weights. Each rank owns its particular loop's KV per request/token and a one-way hidden-state handoff to the next loop.
- Maintain whole-prompt request-level stages, cross-request overlap, and exact loop norm/residual logic.
- For Dynamic exit, allow empty/token-ragged downstream stages. Ensure each token's result/logits/gate and its historical KV semantics match SD/TD/LD.

### Key fairness restrictions

- All four *parallelism* categories use the same checkpoint, tokenizer, prefill/decode semantics, precision, recurrent cap, EOS and generation settings.
- Preserve the same exit policy and cache semantics across **all** Dynamic categories; any engine unable to support it gets a blocker, not a silently modified approximation.
- Equal algorithmic work is assessed in a separate **aligned replay lane**. Actual free-running generations may have different lengths and exit depths and must report those differences.
- Four-process workers are a robust implementation choice, not an inherent requirement of PP. Avoid presenting a process-layout effect as a loop-placement effect.

---

## 5. Gate N: numerical qualification and TP triage (mandatory first)

Do not repeat the prior generic statement that TP4 is broken. Investigate the actual cause.

### N0: controlled exact definitions

- Separate `HF single`, `custom unsharded`, `custom TP4`, `native vLLM TP1`, and `native vLLM TP4`; where practical add custom TP1/TP2 to isolate scaling effects.
- Align checkpoint, positional IDs, tokenizer, RoPE, causal mask, epsilon, residual/layernorm order, prefill cache population, output/logit extraction, and token feedback. Save the *exact* observed output tensor shapes/position indices.
- Preserve historic C2 thresholds and its P512 FAIL in an append-only historical audit.

### N1: first-divergence localization

- Run locked prompts P128/P256/P512/P2048 with matched forced output positions (at least 32 decode steps at P512), multiple distinct inputs and seed records.
- Save per-loop R1..R4, per-physical-layer L1..L24, pre/post attention/norm/MLP deltas at sampled positions; distinguish full-tensor, **last-token-only**, and decoded-step hidden error.
- At each boundary measure hidden relative L2, RMS error, logit mean/max absolute error, logit JSD/top-1/top-k agreement, stable-margin argmax mismatch, and KV relative L2 per loop/layer/head.
- Pay special attention to P512 zero-based decode steps 18,19,26,27 and the prior SDPA-head-geometry sensitivity. Determine whether the divergence begins in attention kernel/head layout, row reduce, quantization/casts, a cache issue, or accumulation.
- Compare TP1 vs TP4 **within the same vLLM build/backend**; then compare each to HF. This isolates engine differences from TP-specific effects.
- Use FP32/high-precision operation references for suspect layers on bounded inputs to assess whether the TP sum or the single-backend computation is closer to a numerical oracle. Respect GPU/time limits.

### N2: prospective numerical qualification

- Do not rewrite the failed 0.03 historical threshold. Before looking at **held-out** probes, define a documented new analytical criterion (if justified by TP1 vs TP4 calibration and FP32 references). Use a calibration set disjoint from GSM8K/MATH test sets.
- Record `HISTORICAL_STRICT_C2=FAIL` where applicable, regardless of the newer analysis outcome.
- Evaluate two independently reported axes: **Numerical fidelity** (tensor and distribution errors) and **behavioral/task fidelity** (stable argmax, free-running generation, actual benchmark accuracy).
- Exploratory TP4 task-accuracy generation is permitted if all outputs are valid/finite, model/sharding/weights are audited, and results are explicitly `NUMERICALLY_UNQUALIFIED_EXPLORATORY`. Such results must never be used as proof that exact TP correctness was established. Stop and mark blocked for model-level illegal outputs, unusable generation, fatal backend bugs, or obvious wrong weights.
- Preserve failed probes and do not tune numerical thresholds on final evaluation test labels.

**Gate result:** (i) fully qualified; (ii) behaviorally qualified but numerically unresolved; (iii) exploratory only; or (iv) blocked. Only (i) yields an unqualified serving comparison. Other categories have visible caveats.

---

## 6. Gate D: true dynamic-exit semantics (independent from TP gate)

### D0: choose and lock one existing method

Preferred reference: [Continuous Depth Batching (CDB)](https://github.com/kschwethelm/continuous-depth-batching) and its [paper](https://arxiv.org/abs/2608.09444). The repository implements genuine depth-adaptive inference and continuous batching for Ouro/Huginn and explicitly says it is a research baseline, not a full serving engine. Inspect the **exact Ouro paths**, model weight IDs, exit scoring, supported minimum depth, threshold, special tokens, available KV policies, prefill and decode behavior, refill/no-refill scheduling, and any reported evaluation prompts.

Alternative: Enabling Dynamic Computation / best-available-KV, if its implementation is accessible and sufficiently validated. This is a **different experiment**: do not mix its gate/KV semantics with CDB across arms.

### D1: lock fidelity and quality knobs

- Max recurrent steps **R4**. Use actual gate-based token-level exits, possibly R2/R3 according to the original method. Record gate precision, threshold and any min-depth or lookahead constraints; never invent or tune it on test answers.
- KV handling for an early-exited historical token must be exactly the chosen reference method's rule, documented for `(loop,layer,token)` missing entries. Verify ongoing deeper loops consume the same effective past context. No silent zero-fill, no implicit R2 KV reused as R4 unless explicitly the reference policy.
- Refilling/no-refill and batching policy must be specified and held consistent. Single Dynamic defines the reference output/exit trace for correctness; all Dynamic 4GPU arms implement the **same algorithm**, not just the same nominal threshold.
- Validate boundary cases: every token exits early, none exits early, a mixed-depth prompt, one token exits before a later token continues deeper, EOS, cache reuse across decode positions, request completion, and overlapping requests.
- Dynamic computational saving must be physically observable in GPU kernel/loop counts. A post-hoc selection after full R4 is FAIL.

### D2: policy calibration (train/validation only)

- Use a separate small calibration split (e.g., training partition or manually held-out prompts not included in evaluation). Choose the original CDB paper/checkpoint threshold if it can be reproduced; otherwise calibrate a quality-vs-average-depth curve **before** locking the threshold.
- Report the selected threshold and resulting mean exit depth, histogram, residual kernel counts, and quality on calibration. Keep that one threshold across SD/TD/LD/OD.
- If a backend's floating-point difference changes threshold decisions, report depth-trace mismatch and gate-score margins. Do not quietly force matching depth in natural generation. Use trace-locked replay in a separate diagnostic lane only.

### D3: dynamic correctness gate

- Dynamic Single GPU must pass a reference CDB output/exit/KV policy audit before porting to TP4 and both PP4 placements.
- For TD/LD/OD compare same-policy reference gate/exit decisions, final logits, output token IDs, masked/ragged KV ownership, and memory safety on bounded prompts; replay expected exit traces to isolate execution routing from gate-numerical drift.
- If true dynamic execution is not yet feasible for one arm, continue with completed Static arms; report Dynamic rows `BLOCKED` or `NOT_RUN`, never substitute fixed R2 or static R4.

---

## 7. Benchmarks: data, prompts, and scoring

### E0: evaluation datasets

Primary public test suites (subject to confirmed evaluation protocol for this exact checkpoint):

| Dataset | Official evaluation size | Task and scoring | Suggested stages |
|---|---:|---|---|
| GSM8K test | 1,319 problems | Final numeric answer exact match after documented canonicalization | 64 pilot -> 256 interim -> full 1,319 |
| MATH-500 | 500 problems | Math-expression equivalent answer matching using a standard verified grader, record ambiguous/failed parses | 50 pilot -> 100 interim -> full 500 |

- Explicitly verify dataset repo ID, revision, split, hashes, licensing/access, exact question IDs and order. No cherry-picking cases.
- The paper's evaluation numbers are **external sanity checks**, not substituted for this run's Single GPU measured accuracy.
- Verify model variant: base vs thinking, tokenizer chat template, few-shot instructions, BOS/EOS, reasoning format, stopping, max generation budget and answer extraction. Follow an existing official Ouro/CDB evaluation protocol where appropriate, and archive the exact generated prompts. **Do not silently mix different checkpoint/protocol results**.
- Freeze evaluation configuration before comparing arms. For precision diagnosis, separately use synthetic and calibration prompts; do not count those as GSM8K/MATH test samples.
- If full 8×1,819 problem evaluation is infeasible, complete the pilot **with labeled sample size** and report full results as NOT_RUN. Do not extrapolate accuracy.

### E1: free-running quality lane (primary accuracy)

- Each arm independently generates with identical deterministic decoding configuration (greedy, temperature 0 where applicable), EOS/stopping rules, and maximum new tokens; no frozen output tokens for quality scoring.
- Store one row per example and arm: prompt checksum, generated text/IDs, number of tokens, stop reason, final extracted answer, score (0/1), parsing status, runtime errors, dynamic depth trace summary, and timestamp.
- Include sample-level paired comparison to Single Static and Single Dynamic. Record exact answer changes and cases where numerical deviations change or do not change correctness.
- Do not count forced-length frozen-token hardware probes as task accuracy.

### E2: accuracy statistics

Define for arm `a` in depth mode `d` and dataset `q`:

`Accuracy[a,d,q] = correct_count / evaluated_count * 100`.

**Primary task-quality delta (percentage points):**

`DeltaAcc_pp_vs_S4[a,d,q] = Accuracy[a,d,q] - Accuracy[S4,static,q]`.

**Parallelism-only delta within depth mode:**

`DeltaAcc_pp_vs_mode_single[a,d,q] = Accuracy[a,d,q] - Accuracy[Single,d,q]`.

Also report optional relative percent change with explicit denominator, but never confuse `%` with percentage points (`pp`). Report paired answer-change counts and, where sample size allows, paired bootstrap/appropriate confidence intervals. If variants complete different example counts, compare only a documented **common completed set** and separately show failures/missingness; do not bias by dropping hard examples silently.

---

## 8. Performance methodology and single-GPU speedups

### P0: two complementary lanes

**Lane P-A (real generation; primary user experience):** same exact questions, prompts, max-new-token cap, EOS and decode policy, each backend freely generates its own answer. This lane measures actual wall latency and total task completion/throughput. Output lengths and depth traces can vary; report them rather than assuming constant work. Do not interpret a shorter incorrect answer as an architectural speedup.

**Lane P-B (controlled identical-workload replay; hardware isolation):** use frozen prompt IDs and **frozen output token IDs** in Static R4 for Single/TP4/LayerPP4/LoopPP4. For Dynamic, use a **trace-locked exit schedule** from a verified Dynamic Single reference and charge any simulated-policy work honestly; also run a separate fully live gate path. This lane isolates placement/runtime effects under matched work. Mark any numerical failure flags alongside measurements.

### P1: speedup definitions

For each mode `d` separately, compare matched workload `w`:

`Speedup_E2E[a,d,w] = E2E_Single[d,w] / E2E_a[d,w]`.

`Speedup_Cohort[a,d,w] = Makespan_Single[d,w] / Makespan_a[d,w]`.

In addition provide an explicitly **cross-mode** overall gain:

`OverallGain[a,d,w] = E2E_SingleStatic[w] / E2E_a[d,w]` (or makespan analog, labeled separately).

In live generation, ensure paired prompts are identical but state when token counts differ. Provide time-to-answer and output length alongside ratios. If outputs differ materially, show controlled replay ratios as the cleaner architecture comparison.

Single uses **1 GPU**, while TP4/PP4 uses **4 GPUs**. Consequently a speedup >1 does **not** imply superior four-GPU resource efficiency. Add throughput per GPU, and, if time permits, **optional DP4 four-GPU capacity baseline** in a clearly separate table (do not expand the 8-arm primary contract). When Single GPU OOMs, mark the speedup undefined and report the capacity difference instead.

### P2: workload matrix and protocol

- Start with existing reproducible Static R4 21-row frozen hardware matrix from the earlier PP experiment; re-run it freshly on the current engines, and never mix old absolute latencies with new engine measurements.
- For first synchronized pilot: concurrency C=1,4,8,16; prompt lengths P=128,512,2048; output caps O=32,128 as feasible. Include a few mixed-length cohorts. Expand toward C=32/64 subject to HBM. Prefer explicit recorded subset if workload is too large.
- For task lane, use the pinned GSM8K/MATH examples and the same concurrency/cohort composition in each arm; report the full per-arm workload and queueing timeline.
- For Static and Dynamic, report TTFT (including queue), TPOT (decode with contention), per-request E2E P50/P90/P99 as supported, cohort makespan, completed requests/s, output tokens/s, total generated tokens, latency per generated token, GPU utilization/service, input throughput where informative, and stall/idle causes.
- Use **at least 5 clean independent repetitions** for the headline pilot where feasible; interleave/randomize arm run order, separately warm each arm, log thermal/clock changes. At minimum, report repeats, median/min/max and variability; bootstrapped CI with only 3 repeats remains weak.
- Ensure *actual cross-request GPU kernel overlap* for both PP categories, and actual NCCL collectives on all TP ranks. Record GPU memory for weights, logical KV, graph scratch, allocator reserved/allocated, NVML peak, and OOM threshold where safely measurable.
- Isolate NCCL all-reduce/launch/communication vs P2P transport and worker scheduling. Use Nsight/CUPTI separate heavy lane and do **not** sum overlapping kernel/copy times as exclusive wall-clock components.
- Record single-rank baseline speed, GPU clock, scheduling/graph preparation costs, and cost to serve same 4GPU total workload with optional DP4 before calling a method throughput-efficient.

### P3: optimized vs matched backends

Create at least two explicit comparison tables if both are available:

1. **Architecture-controlled/custom:** custom Single/Custom TP4/MP LayerPP4/MP LoopPP4 where kernel/batching/math can be aligned as much as possible.
2. **Production-oriented:** native vLLM TP1/TP4 (Static only if vLLM lacks Dynamic) versus the best qualified PP runtime; list different CUDA Graph, KV, batching, process topology and attention backend settings.

Do not silently mix the fastest result from each unrelated run campaign without disclosing engine differences. Four-process is recommended for PP production-controls, but it is not a correctness or architectural prerequisite.

---

## 9. Staged execution plan and gates

### Stage 0: preflight, source audit, manifests (mandatory)

- Locate exact local Git root and previous scripts/logs at required working directory. Confirm source revision and GPUs. Verify model checkpoint, tokenizer, libraries and dataset scripts. Snapshot existing reports.
- Write `preflight.md`, `environment.json`, `arm_manifest.json`, `dataset_manifest.json`, `numerical_protocol.md` before modifying the runtime.
- Reproduce bounded previous PP/TP correctness controls, without overwriting prior results.

**Gate 0:** actual checkouts, executable environments and GPU ownership available, or `BLOCKED` with a precise workaround.

### Stage 1: static numerical study (first priority)

- Execute the N0-N2 controlled numerical diagnosis, native TP1/TP4 comparisons, first-divergence probes and FP32 narrow references.
- Log historical strict gate as FAIL, prospectively classify current arm. Do not defer all accuracy evaluation merely because hidden L2 exceeds a historical heuristic, but ensure the exploratory label and safety of outputs.

**Gate 1:** qualified Static multi-GPU runtime, or documented `NUMERICALLY_UNQUALIFIED_EXPLORATORY` quality lane; unqualified serving performance cannot be promoted as final baseline.

### Stage 2: Static R4 eight-arm subset (four Static arms)

- Run GSM8K64/MATH50 pilot on S4,T4,L4,O4 with actual generation. Save exact sample-level output and accuracy deltas.
- Run controlled hardware replay on matched Static R4 at the same time, collecting TTFT/TPOT/E2E/makespan and numeric flags; then scale dataset and workload as resources allow.
- Diagnose anomalies; full 1,819×4 Static task runs after pilot gates pass.

**Gate 2:** complete, labeled Static four-arm table and explanation of any gaps.

### Stage 3: Dynamic Single reference fidelity

- Audit and pin CDB Ouro settings, checkpoint and KV policy. Implement SD first, compare real per-token exit traces, logits/outputs/KV to reference and verify actual compute skipping.
- Calibrate policy only on a disjoint validation split. Choose one threshold/policy and freeze it for all Dynamic arms.

**Gate 3:** a validated Dynamic Single baseline or a transparent Dynamic blocker. Do not use fixed R2 as replacement.

### Stage 4: Dynamic TP4, LayerPP4, LoopPP4

- Port the same reference policy to TD/LD/OD; enforce identical per-token cache/exit semantics, test ragged KV and multi-request overlap, and record differences from SD.
- Run the shared GSM8K64/MATH50 pilot for all four Dynamic arms; extend to full suites subject to qualified correctness and resource budget.
- Collect free-run SLO/latency/accuracy, trace-locked hardware replay, depth histograms and per-rank busy/idle/stage queues.

**Gate 4:** full Dynamic comparisons where feasible, otherwise explicit per-arm BLOCKED/NOT_RUN; no fake dynamic results.

### Stage 5: interpretation and go/no-go

Produce separate conclusions for:

- **Numerical TP fidelity:** reference/backend versus sharding, with unresolved flags.
- **Task quality:** Static R4 versus true Dynamic changes, independent of TPU/PP implementation.
- **Hardware speed:** Single-normalized latency, same-GPU-budget throughput, and controlled-replay behavior.
- **Loop-PP architectural value:** advantage surviving matched optimized controls, and whether observed performance is explained by stage-boundary host overhead, communication or actual GPU compute.
- **Research decision:** `GO`, `CONDITIONAL`, `NO-GO`, `BLOCKED`. A system can be blocked on one arm and provide valid results for the others. Do not exaggerate novelty.

---

## 10. Required result schemas

### Table A. Final quality (one row per dataset × arm)

| Dataset | Arm | Depth | N eval / N correct | Accuracy % | Delta vs S4 (pp) | Delta vs mode-Single (pp) | Avg tokens | Avg depth | Fidelity status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| GSM8K | S4 | Static R4 | ... | ... | 0 | 0 | ... | 4.0 | ... |
| GSM8K | T4 | Static R4 | ... | ... | ... | ... | ... | 4.0 | ... |
| GSM8K | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| MATH-500 | ... | ... | ... | ... | ... | ... | ... | ... | ... |

### Table B. Performance (one row per scenario × arm)

| Mode | Scenario | Arm | GPUs | Engine | Speedup vs mode Single | Overall gain vs S4 | Median E2E s | TTFT/TPOT ms | Cohort makespan s | Output tok/s | Peak per-rank HBM | Status |
|---|---|---|---|---|---:|---:|---:|---|---:|---:|---|---|

### Table C. Numerical trace

| Engine | TP | P/O | First divergent (loop,layer,op) | max hidden rel-L2 | last-token rel-L2 | logit MAE/max | KV rel-L2 | stable argmax agreement | old strict gate | new prospective verdict |
|---|---|---|---|---:|---:|---|---:|---|---|---|

### Table D. Dynamic exit and work

| Arm | Dataset/scenario | Exit R1/R2/R3/R4 hist | Avg depth | Actual skipped loops | GPU busy/idle/queue | KV policy | Output drift vs SD |
|---|---|---|---:|---:|---|---|---|

*All unfinished cells shall explicitly say NOT_RUN, BLOCKED, or N/A; never manufacture numeric predictions.*

---

## 11. Artifacts and deliverables

Create a **new** report directory under the separate working tree:

`/home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_8arm_static_dynamic_discovery/`

Required deliverables (files may be split with clear manifest):

- `README.md` / `status.md`: commands, state of each arm/gate and reproducibility details.
- `analysis.md`: decision with explicit reference/engine/quality/architecture qualifications.
- `numerical_tp_diagnosis.md`, `first_divergence.csv`, `numerical_gate.json`: error location, FP32 comparisons, historic and prospective rules.
- `dynamic_policy_contract.md`, `dynamic_reference_validation.md`, `exit_trace_summary.csv`: exact gate/KV/refill semantics, benchmark alignment and real skipped work.
- `dataset_manifest.json`, `eval_protocol.md`, `evaluation_outputs.jsonl`, `accuracy_by_arm.csv`, `paired_accuracy_deltas.csv`: actual dataset IDs, prompts, outputs, scoring and CIs.
- `serving_by_arm.csv`, `serving_summary.md`, `timelines/`, `comm_profile.csv`, `memory_by_rank.csv`, `gpu_activity.csv`: clock-lane and heavy-lane separated results.
- `run_manifest.json`, `commands/`, `logs/`, `versions.txt`, `failures.md`, `validation_tests.md`, `source_hashes.json`.
- A compact `eight_arm_matrix.md` with **all 8 rows including honest status flags**, plus Static and Dynamic separate benchmark tables.
- Tests for Single/TP/PP numerical correctness, dynamic active-token routing, ragged KV and cache ownership, complete-request overlapping, and faithful metric formulas.

Preserve previous commit `6f986ed...` and prior report folders. Prefer a **new dedicated branch**. Commit completed changes and include reproducible CLI commands. Push only if explicitly authorized under the working repository policy; do not rewrite or merge old branches. Save paths as actual local paths, not assumed GitHub repo names.

---

## 12. Stop conditions and quality safeguards

- Stop a particular arm if its actual computation is wrong, its checkpoint is inconsistent, it generates invalid/nonfinite outputs, or it cannot implement the selected KV/exit policy. Preserve reason and full debug trace; do not stop other valid arms automatically.
- Retain historic TP strict-gate failure in reports, even if independent new criteria later qualify its accuracy evaluation.
- Do not infer an accuracy percentage from 32/32 or 72/72 aligned/natural matching tokens.
- Do not measure unqualified four-GPU TP latency and present it as a verified architecture win. Optional exploratory timing must be watermarked as such.
- Do not confuse one full-eval accuracy number with equivalence; use paired correctness, intervals and exact total samples.
- Do not use output length reductions alone as an algorithmic placement speedup. Report live and identical-work controlled replay separately.
- Do not equate `KV bytes`, PyTorch peak allocated, and NVML device memory or claim a new KV capacity limit without a measured OOM/capacity sweep.
- Do not equate PP handoff-count reduction with a proportional E2E wall-time improvement; previous Nsight P2P service totals were small relative to wall time.
- Do not assume TP communication is All-to-All in dense Ouro: expect frequent projection-related All-Reduce/collective patterns; profile actual kernels.
- Do not claim that a four-process runtime is architecturally obligatory. Measure runtime architecture impacts with matched workload.
- Protect GPU4/5/6/7 ownership; fail cleanly if another user's workload compromises accuracy of measurements.

## 13. Final decision template

Answer with explicit evidence:

1. Was the numerical TP1-vs-TP4 divergence localized? Did prospective checks qualify TP4 despite the historical gate remaining failed? What did actual GSM8K/MATH quality show?
2. For Static R4, which parallelism achieves best single-request E2E, finite-concurrency makespan, and output throughput relative to Single GPU? At what memory footprint and with how much resource cost?
3. For True Dynamic, is the selected reference policy numerically and behaviorally equivalent across parallelisms? How much did it change **accuracy** and **average recurrent depth**?
4. How much of Loop-PP's benefit over Layer-PP remains with comparable multiprocess execution, and how much is stage-boundary/gap versus actual P2P critical path?
5. Does Loop-PP beat qualified TP4 or merely exhibit a prototype-specific advantage? Does this persist on a strong same-four-GPU baseline?
6. What is the smallest grounded next experiment if something is blocked? Label each answer as MEASURED, INFERRED or UNKNOWN.

**Success is an honest eight-arm evidence table, not necessarily a win for Loop-PP.**

