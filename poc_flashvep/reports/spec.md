# Looped MoE Expert-Efficiency Headroom: Three-Model, Single-GPU Discovery PoC

**Document:** `spec.md`  
**Version:** 1.0 — 2026-10-26  
**Required target path on the research server:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/spec.md`  
**Study type:** Measurement-first, falsifiable systems research PoC  
**GPU inventory:** CUDA devices `0,1,2,3,4,5,6,7` are potentially available; verify free memory/health before scheduling.

---

## 0. Executive decision

Investigate whether recurrent execution in *looped sparse MoE language models* creates expert-computation efficiency opportunities beyond those of conventional per-layer MoE execution. The immediate goal is **not** to claim a novel acceleration technique or to implement EP. Instead, establish the runtime feasibility, loop-wise routing/output behavior, rigorous accuracy-vs-work tradeoff, and *measured* single-GPU latency headroom.

**Exactly three pretrained models must be used in every phase and every PoC:**

1. **IFM MoE Ouro, 500B training tokens:** `IFM/LoopedLM-P1-moe-ouro-500b`; 8,780,889,600 resident parameters.
2. **IFM MoE Huginn, 500B training tokens:** `IFM/LoopedLM-P1-moe-huginn-500b`; 8,783,251,968 resident parameters.
3. **Foil-1, 100B training tokens:** subfolder `Foil-1` of `ShourenWSR/how-to-loop-moe`; **563.2M** parameters, one repeated MoE layer with 64 experts, 16 loop passes, and *untied attention per pass*.

No other pretrained test model, proxy small model, model substitution, quantized copy, or alternate checkpoint is allowed. Other *workload datasets* are allowed. Different internal execution engines are expected; therefore, **never compare raw latency between different models as if it were a controlled experiment**. Compare changes against each model's own validated baseline and disclose hardware/runtime differences.

The TEAM paper is **inspiration for measuring temporal and spatial routing consistency**, not evidence that denoising-step reuse works for recurrent Transformer passes. An observed top-k expert overlap alone is not evidence of redundant expert computation.

### Primary questions (all three models)

- Q1: Is per-token expert selection stable between successive passes at the **same physical recurrent MoE site**?
- Q2: Conditioned on selecting the same expert, are the *inputs*, expert outputs, and weighted routed contributions sufficiently similar for reuse, or do they change materially?
- Q3: Can expert contributions be selectively omitted or replaced without unacceptable next-token loss, generation changes, or downstream-quality loss?
- Q4: What portion of **measured end-to-end latency** is actually removable expert execution (versus attention, router, dispatch/gather, memory operations, and nonrecurrent computation)?
- Q5: Does a concrete single-GPU implementation provide speedup at batch 1 / low request counts after accounting for decision, storage, routing, and gather overhead?

### Strict exclusions for this initial PoC

No multi-GPU EP or TP/PP execution; no LoRA/retraining; no precision reduction; no model pruning or persistent weight changes; no external draft model; no reliance on an oracle at deployment time; no accuracy claims based solely on greedy-output agreement on a handful of prompts. EP extension is a later independent decision after a single-GPU GO.

---

## 1. Authoritative upstream references and runtime assumptions

| Model | Upstream checkpoints/runtime | Key precautions |
|---|---|---|
| Ouro | https://huggingface.co/IFM/LoopedLM-P1-moe-ouro-500b and https://github.com/ifm-ai/xllm-loop | Native BF16 xLLM artifact; use `xllm.paper_part1.native_inference.load_native_model`, **not** Transformers `AutoModel` |
| Huginn | https://huggingface.co/IFM/LoopedLM-P1-moe-huginn-500b and https://github.com/ifm-ai/xllm-loop | Same native-loader and manifest constraints; prelude/recurrent/coda architecture must be identified from instantiated modules |
| Foil-1 | https://huggingface.co/ShourenWSR/how-to-loop-moe and https://github.com/SR-A-W/how-to-loop-moe | Use model's `Foil-1` subfolder, `trust_remote_code=True`; code is structurally different; 563.2M parameters, 16 passes |
| TEAM analogy | https://proceedings.mlr.press/v306/wei26r.html | Temporal/spatial consistency motivation only; diffusion denoising is not recurrent-pass execution |

**Important IFM support constraint.** The published xLLM-Loop installation references Python 3.12, PyTorch 2.11.0+cu128, CUDA 12.8, and FlashAttention 3, originally tested on H200. These are reference versions, not proof the target machine supports them. Preflight must inspect actual GPU architecture, driver, CUDA, `nvidia-smi`, FA3 availability and a working full-model forward. If required FA3 kernels cannot run on the available GPU architecture, check for a *legitimate functional attention backend* supported by the same code path and verify token/logit parity before using it. If not possible, mark that model `BLOCKED_RUNTIME`, explain the incompatibility, and **do not silently fall back to an unrelated implementation or fabricate measurements**.

**Important Foil correctness constraint.** `AutoModelForCausalLM.from_pretrained(..., subfolder='Foil-1', trust_remote_code=True, dtype=torch.bfloat16)` is the advertised loading pattern. The model card warns that (i) padding masks are not supported and (ii) labels are **not shifted internally**. For pure next-token NLL, calculate CE directly using `logits[:, :-1]` against `input_ids[:, 1:]`; do not use `labels=input_ids` as a generic language-model loss measurement. The returned model loss may include routing auxiliary losses. A tokenizer or evaluation harness mismatch is a validity failure.

Record exact repository Git commits, checkpoint revision/hash or manifest, host OS, Python, torch, CUDA runtime/driver, attention backend, GPU name/HBM and model configs. **Do not guess** the IFM number of recurrent passes, router top-k, expert count, routing normalizations, or whether the released inference API supports dynamic exit: inspect the actual release config/code and record them. A fixed, published inference procedure is the default; no adaptive-depth intervention unless explicitly validated.

---

## 2. Deliverables and required layout

Place this specification at the exact target path above. Keep implementation isolated within a new research directory rather than modifying unrelated `poc_flashvep` experiments:

```text
/home/esjung/MLLM-EP-github/poc_flashvep/
  reports/
    spec.md                                # this specification
    looped_moe_headroom/                   # generated artifacts
      README.md
      environment.json
      hardware_inventory.json
      run_manifest.json
      baseline_summary.csv
      routing_summary.csv
      representation_summary.csv
      timing_summary.csv
      oracle_summary.csv
      candidate_summary.csv
      go_no_go.md
      plots/
      raw/                                 # compact run-level JSONL/NPZ; no full hidden-state dumps
      logs/
  looped_moe_headroom/                     # new reproducible source/scripts
    README.md
    requirements.lock.txt                  # after validated install, not speculative pins
    config/
    adapters/                              # ifm, foil and common normalized record schema
    scripts/
    tests/
```

Implementation filenames may differ if existing conventions are stronger, but the three-model coverage, machine-readable outputs, plots, Go/No-Go report, and reproducibility requirements are mandatory. Do not overwrite previous experiment reports. Commit research code/spec/reports **only after tests pass**, and do not commit checkpoint weights, private datasets, massive `.npz` tensors, generated virtual environments, or credentials. If remote push is unavailable, report the local commit/hash and exact error, without claiming a push succeeded.

---

## 3. Multi-GPU resource strategy: eight independent single-GPU workers

### 3.1 Rules

- **Each model forward runs on exactly one GPU** (`CUDA_VISIBLE_DEVICES=<physical_GPU_ID>`). Parallelism here means independent process-level experiments across GPUs, **not** EP, DP throughput claims, FSDP, or model sharding.
- Acquire a per-GPU lock and enforce `CUDA_VISIBLE_DEVICES` for each spawned worker; inside worker map to `cuda:0` as visible device. Never let a default `cuda:0` unintentionally target physical GPU 0 for all workers.
- Check `nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap,pstate,temperature.gpu --format=csv` or available equivalent and live processes before launching. Respect other users' processes. Do not terminate them.
- IFM BF16 weights alone are ~17.6 GB decimal per model; 24-GB GPUs may be tight after CUDA kernels, temporary buffers, KV cache, and trace hooks. Try the smallest valid workload first and reject OOM paths explicitly. **Never secretly quantize, CPU-offload, or split a model** to force it to fit.
- Download each large checkpoint **once** into a manifest-compatible local directory on a filesystem that supports it; do not create eight concurrent downloads or xLLM Hub-cache symlink trees. Avoid simultaneous model loads saturating shared storage; stagger launches. Use a shared read-only checkpoint path with independent per-process GPU copies.
- If GPUs are heterogeneous, record hardware and report within-hardware or appropriately stratified normalized comparisons; do not pool raw measurements blindly.

### 3.2 Initial workload assignment (conditional on feasibility)

| Physical GPU | Primary worker | Example experiment dimension |
|---|---|---|
| 0 | Ouro | Prefill routing + hidden/input sampling |
| 1 | Ouro | Decode routing + quality/traces |
| 2 | Ouro | Baseline timings + microbench/analytical oracle |
| 3 | Huginn | Prefill routing + hidden/input sampling |
| 4 | Huginn | Decode routing + quality/traces |
| 5 | Huginn | Baseline timings + microbench/analytical oracle |
| 6 | Foil-1 | Prefill + decode probes |
| 7 | Foil-1 | Timings, quality + candidate/oracle tests |

This is a **scheduling template**, not a claim that all GPUs can fit all IFM models. Actual jobs must be generated from a manifest and dispatched only to eligible GPUs. When Foil finishes, reassign GPU 6/7 to eligible IFM jobs or additional Foil replications; after first-wave completion, rotate representatives across equivalent GPUs where practical to detect hardware bias. All three models must undergo **every experimental arm** even if a model has fewer dedicated initial workers. A blocked run cannot be counted as completed.

For timing studies, avoid concurrent jobs that saturate PCIe/CPU/storage or produce GPU cross-talk; capture idle-vs-parallel sensitivity on a small subset. The latency baseline and variant for any pair must use the same model, same GPU (or same verified GPU class), same inputs, and analogous process conditions. Use warmup, CUDA synchronization/CUDA events, repetition, confidence intervals, and baseline–variant interleaving where practical.

---

## 4. Study protocol: common workloads and phases

### 4.1 Dataset design, all three models

Use fixed, provenance-tracked, non-training evaluation samples with a deterministic sampling seed (`20261026` by default) and save prompt IDs/text/hash. The same *raw prompts* appear for all three models, but token counts differ by tokenizer and must be reported separately.

- **D1 / language modeling:** 128 held-out, general-domain passages covering factual, narrative, procedural, and code-like text. Obtain a documented publicly usable held-out source. Avoid known training-overlap claims unless verified. Compute full teacher-forced token NLL/perplexity proxy, token agreement, and reference-vs-variant logit divergence.
- **D2 / arithmetic-reasoning:** 128 GSM8K test examples where permitted by local dataset and tokenizer availability. Run exact, fixed decoding configuration; report exact-match when the answer parser is valid and report baseline quality for every model. A weak Foil baseline does **not** invalidate its NLL/logit-fidelity measurements.
- **D3 / difficult reasoning:** 128 MATH-500 problems where permitted, or a predeclared difficulty-balanced and reproducible substitute if unavailable. Preserve the same evaluation protocol and parser across baseline and variants. Record valid/invalid parse counts.
- **D4 / synthetic stress (not scored for semantic accuracy):** deterministic snippets with repeated tokens, unique tokens, copy patterns, mixed-length contexts, and frequent expert transitions, to isolate mechanisms and stress stability.

Split sampling into smoke (8–16 items), **calibration/tuning** (e.g. 32 or 64 items) and **held-out evaluation** (all 128 items per domain or a disjoint predeclared evaluation set). Never tune any thresholds or select oracle opportunities on the final evaluation examples and then call the result held-out. If limited compute requires changing sizes, document it before runs and clearly label the pilot as underpowered.

For **all three models**, run the same measurement matrix where supported:

- Prefill-only: batch `B=1`, input lengths approximately `128, 512, 2048` tokens (ensure actual tokenized lengths are valid; Foil max context 4096).
- Decode-focused: `B=1`, generated `16, 64` tokens with input length `128` and, if feasible, `512`; greedy / deterministic decoding.
- Additional low-load concurrency sensitivity: `B=2,4` only when naturally supported by the native runtime and memory. **B=1 is the primary acceptance setting.**
- Control execution: identical recurrence depth/count, mask, cache usage, tokenizer, numerical precision, attention backend, kernel dispatch, and stopping logic between corresponding baseline and candidates.

### 4.2 Mandatory phases — every phase includes all three models

| Phase | Goal | Required output | Advance condition |
|---|---|---|---|
| P0 | Environment/model correctness | 3 validated model adapters, configs, baseline output parity, measured GPU feasibility | Each model can execute baseline correctly or `BLOCKED_RUNTIME` is recorded |
| P1 | Routing + hidden/output discovery | Loop-wise routing stability, conditioned similarity, confidence, contribution magnitude, uncertainty | Mechanism and counterexamples are quantitatively characterized for all 3 |
| P2 | E2E profiling + conservative oracle | MoE time fraction, candidate removable-work curves, full-model quality after interventions, Amdahl bounds | There is measurable quality-constrained E2E headroom |
| P3 | Minimal actual single-GPU candidate(s) if justified | Fully rerun runtime that *actually skips* expert matmul/reuses computed output, measured speedup and overhead | GO/NO-GO applies based on real E2E, not analytical reductions |

If P0 is blocked for one model, finish other models but mark global 3-model evidence incomplete. If P2 shows a decisive NO-GO, do not spend substantial time on P3. A small non-optimized correctness prototype is acceptable but must be separated from a performance result.

---

## 5. P0: native baselines and correctness locks

1. Inspect server architecture, CUDA toolchain and availability. Record `nvidia-smi -L`, topology (`nvidia-smi topo -m`), free HBM, processes, NVIDIA driver, torch, and git revisions. Verify FA3 or explicitly vetted official fallback.
2. Clone/fetch upstream `ifm-ai/xllm-loop` and `SR-A-W/how-to-loop-moe` into version-pinned external subdirectories. Use isolated environments if their dependency requirements conflict. Do not patch their baselines merely to make a chart look positive.
3. Download both IFM model directories directly with `hf download ... --local-dir ...` so the native artifact manifest is intact. Verify artifact_manifest and loading. Download Foil-1 via documented model-card pattern; pin revision.
4. Load each model with its native recommended method. Preserve BF16 parameter precision. Register adapters that emit identical probe schema while respecting different forward signatures and outputs.
5. Generate deterministic reference outputs for smoke prompts; if a faithful native reference exists, compare instrumented vs uninstrumented forward/token outputs (greedy token parity, finite logits, selected logit tolerances). Check that installing no-op hooks does not change results beyond documented floating-point tolerance.
6. Identify and log each *physical* MoE block, loop pass index, logical layer/pass identity, top-k, router weights, active expert count, attention sharing, prelude/coda status, and whether KV/cache state is pass-specific. Do not mix non-recurrent modules into temporal-overlap denominators.
7. Verify baseline prompt lengths and exact generated token counts. Test that Foil's next-token CE is explicitly shifted and no padding mask is passed. Check all baseline metrics for finite values and sensible latency.
8. Implement a reversible probe enable/disable switch and verify that disabled probes use unmodified baseline execution.

**Correctness stop rule:** if instrumented baseline and untouched native execution cannot be reconciled, `BLOCKED_CORRECTNESS`. Never interpret a routing probe from a changed or misaligned execution as native-model behavior.

---

## 6. P1: scientific discovery probes

Define a recurrent MoE site by **physical expert/router identity** `s`, a token position `i`, and recurrent pass `r`. Use the same `s` and `i` when comparing `r` vs `r-1`. For Ouro/Huginn, derive actual loop boundary and physical module IDs from config/runtime. For Foil-1, compare its one recurrent expert layer across 16 passes; do not conflate untied attention parameters with expert sharing. Record D4/generation separately from teacher-forced prefill.

For each sampled `(model, example, s, r, i)` capture only what is needed:

- Pre-expert hidden vector `h[r,s,i]`, selected top-k expert IDs `E[r,s,i]`, gating probabilities/weights `w[r,s,i,e]` and raw router logits when accessible;
- Per-selected-expert input `x[r,s,i,e]` (respect dispatch transforms), output `f_e(x)`, and **weighted contribution** `c[r,s,i,e] = w[r,s,i,e] * f_e(x)`; distinguish router weight normalization and aggregation;
- Post-MoE output, residual/h-state, and next-token output on selected probe indices where instrumentation overhead is tolerable;
- Measured expert kernel/dispatch/gather timings in a **separate low-overhead profiling pass**, not from massive Python tensor hooks.

Always record tensor dtypes and shape; handle routing and reuse keys correctly under tokens with multiple assigned experts. For memory safety, sample e.g. up to 32 sequences x 64 token positions in first-wave probes, and store aggregated per-pass histograms / scalar statistics; save dense vectors only for small audited subsets. Do not serialize every hidden state for every token, expert and pass.

### 6.1 Primary metrics (report distribution, not only means)

1. **Same-token consecutive-pass expert overlap:** `J = |E_r ∩ E_(r-1)| / |E_r ∪ E_(r-1)|`; also report intersection fraction `|E_r ∩ E_(r-1)|/k`, exact set match, and top-1 match. Compare by pass number, task, context length and prefill/decode regime.
2. **Gate-weight stability:** selected-expert rank transitions, absolute weight deltas, router entropy, max softmax probability, margin between top-1/top-2 (when applicable), and top-k assignment churn.
3. **Expert-input similarity conditional on same selected expert:** cosine similarity, relative L2 change `||x_r - x_(r-1)||/(||x_r||+epsilon)`, and absolute norm change; report *with* and *without* output normalization.
4. **Actual contribution similarity:** cosine, relative L2 change and norm ratio for `f_e(x)` and `w_e f_e(x)`, conditioned on expert match. Relative L2 with near-zero denominator must be thresholded or accompanied by absolute errors.
5. **Importance/contribution diagnostics:** distribution of `||w_e f_e(x)|| / (sum_j ||w_j f_j(x)||+epsilon)`, with a control for cancellation and residual streams; low router weight alone is not proof an expert is safe to skip.
6. **Spatial consistency:** compare adjacent token positions `i,i+1` *within the same recurrent pass* separately from temporal consistency at fixed `i`. Condition results on shared expert and known repeated-text patterns. Separate genuine semantic similarity from repeated routing driven by frequent tokens.
7. **Control/nulls:** shuffled loop-pass pairing; random positions matched on sequence/length; random expert identity with same activation frequency; where appropriate, prelude/coda *within the same model* as nonrecurrent reference sites. Preserve causality and the observed top-k marginals in null construction.
8. **Autocorrelation:** similarity vs pass gap `Δr ∈ {1,2,4,8}` when present, showing whether opportunities persist beyond adjacent passes.

**Important distinction:** high temporal overlap merely means the same expert weight is executed repeatedly, not that its input/output can be reused. Need evidence for unchanged or safely approximable outputs and quality preserved after rollout.

### 6.2 Outputs

Provide per-model, per-workload, per-pass plots: overlap/Jaccard; expert input/output similarity conditioned on expert match; routing entropy; estimated skippable fraction under candidate rules; and counterexamples where matched experts have large output changes. Report 95% bootstrap confidence intervals over independent **sequences**, not individual correlated token/pass observations. Summarize at least the median, P10/P90 and overall token-weighted mean with denominator/sample size.

---

## 7. P2A: measure the latency denominator before declaring headroom

Baseline measurements must isolate actual runtime cost, with CUDA synchronization and steady-state repeated timings. Exclude model download and cold initialization from primary steady-state latency but report one-off loading separately.

Measure, if identifiable without distorting the run:

- Prefill latency; per-output-token decode latency (TPOT) and full request E2E latency for low-request serving;
- Router computation, dispatch/scatter/sort, expert GEMM+activation, gather/combine, attention, nonrecurrent dense blocks, recurrent control overhead, memory/KV work, launch overhead;
- Per-loop breakdown and total cumulative **removable expert computation** `T_exp`, vs total `T_e2e`; time proportions must not double count overlapping kernels or be inferred naively from summed nested spans;
- Model memory/HBM peak, per-token optional cache bytes and allocations; compilation/cold vs warm timings if relevant;
- Variability: ≥5 warmup iterations and ≥20 measured repetitions for a microbenchmark when feasible; at least 3 independent macro repetitions, report median, P10/P90 or bootstrap 95% CI. Where runtime is too expensive, record the smaller sample and its limitation.

**Ideal zero-cost skip bound (diagnostic, not measured speedup):** for baseline fraction `f_exp = T_exp / T_e2e` and feasible expert compute fraction removed `q`, report

`S_ideal(q) <= 1 / (1 - f_exp * q)`

under an explicit non-overlap, fixed-other-cost approximation. In parallel/overlapped executions, this is only a coarse upper-bound model; use trace replay or controlled kernel-level substitution for a more appropriate bound and distinguish empirical wall-clock response from modeled estimates. Router/gather/attention remain paid unless actually eliminated. Static routing overlap itself does not imply `q>0`.

Benchmark one-recurrent-site expert work separately at realistic per-expert token counts to expose tiny GEMM / launch overhead and whether a token-level shortcut maps into meaningful kernel reductions. Interpret predicted savings against the achievable dispatch granularity and launch costs.

---

## 8. P2B: controlled oracle and quality-work Pareto study — all three models

The word **oracle** refers to offline knowledge unavailable in deployment. Do **not** report oracle results as an implementable online system. Separate three increasingly realistic levels of evidence:

1. **Analytical ideal bound:** remove an exact fraction of expert kernel time with zero decision/cache overhead, keeping all other costs fixed. This is a *timing ceiling only*; does not assert safe accuracy.
2. **Hindsight intervention oracle:** select items to modify using baseline outputs/contribution norms or actual future-loop information, then **rerun full forward/generation with those interventions** and score outputs. Such hindsight selection is inadmissible online; report quality versus omitted FLOPs/calls and modeled kernel savings.
3. **Causal deployable candidates:** rules use only quantities available *before* the target expert computation (e.g. current router weights, prior pass saved output, prior hidden state, or current pre-expert input). Rerun full model; actually avoid corresponding expert compute where possible; include all overhead.

### 8.1 Required test arms

- **B0 / Exact native baseline**: same weights, depth, tokenizer, precision, attention backend and decoding for every paired run.
- **O1 / Contribution-informed drop oracle (hindsight):** after reference execution, rank candidate routed `(token,pass,site,expert)` items by baseline weighted contribution norm or measured downstream sensitivity (if available). In a *new forward*, suppress selected expert contributions without recalculating those expert outputs, at target budget `q = 0.05, 0.10, 0.20, 0.30`. Track skipped **expert FLOPs / kernel work**, not just number of router indices. Default no renormalization; test explicit renormalization as a separate arm.
- **O2 / Same-expert reuse oracle (hindsight):** on repeated expert IDs across passes for the same token and physical module, substitute previously computed expert output from a previous pass; optionally choose oracle-safe reuse occurrences using baseline output error. Distinguish storage as **raw expert output** vs **weighted contribution**, since routing weight may change. Count eligible replacement operations and cache byte cost. Rerun model with substitutions and quantify drift.
- **C1 / Online router-threshold skip:** decide from current top-k weights, entropy/margin, and/or per-pass policy before expert evaluation. Explicitly benchmark whether removing a sparse token/expert assignment actually reduces GPU work with the used kernel; static top-k kernel may not. Report forced-fallback costs.
- **C2 / Online same-expert stale reuse:** when the previous pass for the same `(request, token position, physical expert)` exists, reuse that cached raw expert output with current gate weight. Compare optional cheap input-delta/score guard, and a baseline unconditional reuse arm. Count comparison, lookup, gather, cache-write and bytes.
- **N1 / Randomized control:** drop/reuse the same FLOP budget using a matched random choice; demonstrate any quality/latency gain is not explained by merely touching fewer activations.

If some arms prove meaningless due to model topology, define the nearest faithful per-model analogue, document why, and **do not silently omit a model**. All three must have outcomes (PASS, FAIL, or explicitly BLOCKED with evidence) for each required experiment.

### 8.2 Accuracy and fidelity safeguards

For every arm and every model, measure:

- Held-out shifted next-token cross-entropy (`ΔNLL` in nats/token), perplexity change where stable, and top-1 next-token agreement on teacher-forced contexts;
- KL divergence `KL(P_baseline || P_variant)` on matched next-token logits; appropriately avoid numerical overflow; optionally top-k-logit margin deltas;
- Deterministic autoregressive generation exact-token agreement, divergence onset/token count, and valid answer exact match on GSM8K and MATH-500, **with absolute baseline accuracy counts and parser validity**;
- Full autoregressive rollout with modified state, *not only single-step replay under pristine teacher-forced baseline hidden states*;
- 95% paired bootstrap CIs over examples for answer-accuracy difference and NLL or generation fidelity. Do not interpret a tiny weak-baseline sample as evidence that a candidate preserves accuracy.

A skipped MoE expert must have an unambiguous implementation semantics: omit only its weighted contribution (zero out) unless a separately specified renormalization is tested. For stale reuse, cache by correct token position, physical expert ID, loop pass and request, with no invalid cross-request / across-position reuse. Use correct numerical accumulation and reset caches for new requests. A mask applied **after** performing the original GEMM is an **accuracy probe**, not a real computation-saving implementation.

### 8.3 Pareto and selection logic

Present Pareto plots per model of (i) removed expert FLOPs or avoided expert kernel calls, (ii) measured / modeled E2E speedup, (iii) `ΔNLL`, (iv) task-score loss with CIs, (v) peak cache bytes. For hindsight oracles, label curves **unrealizable offline upper bounds**. Calibrate online thresholds on the calibration split only and freeze before held-out test. Require an explicit breakdown of how much of an ideal gain disappears due to prediction mismatch, GPU work granularity, and bookkeeping.

---

## 9. P3: minimal actual acceleration prototype (conditional on P2 evidence)

Select at most one or two clearly favorable **causal** mechanisms, and implement them in a localized, reversible way for **each of the three models**. Retain native exact baseline with feature flags off. Do not fork a new attention stack unless indispensable, and do not claim `speedup` for an implementation that still executes all expert GEMMs.

Measure full steady-state baseline vs candidate on the *same GPU* with fixed model, dataset and seeds:

- `T_prefill`, `T_decode/token`, `T_E2E`, throughput tokens/sec at B=1 (B=2/4 secondary), and candidate/baseline **ratio**;
- Actual executed expert work (kernel count, active tokens per expert, estimated GEMM FLOPs) and avoided dispatch/gather operations as observed;
- Bookkeeping/branching/selection overhead, memory allocation/cache reads/writes, kernel launches and GPU memory peak;
- Workload-by-workload accuracy degradation and paired bootstrap confidence intervals;
- Performance across early/mid/late recurrent passes, prompt lengths and generated lengths;
- At least one matched random control and one instrumentation-off baseline repeated in the same process regime.

Log the *effective* (not requested) omission/reuse fraction and account for output tensor materialization. Explicitly address fragmentation: skipping sparse independent token-expert edges may reduce logical FLOPs while producing no time benefit if fixed GEMM dimensions or kernels are unchanged. Favor prototype scope and measurement rigor over clever but unvalidated kernel rewrites.

---

## 10. Statistical discipline, pitfalls, and publication-grade interpretation

- Independent resampling unit is **request/sequence**, not route event. Use paired CIs for baseline/candidate within prompt.
- Report matched inputs, tokenizer differences, GPU/hardware strata, and native implementation differences. No cross-model absolute-latency leaderboard.
- Foil's 100B-token training budget, smaller size and untied attention make its routing statistics non-identical in meaning to IFM's 500B-token 8.78B models. Treat it as a generalization/control model, not an architectural twin.
- Do not confuse expert identity reuse (already guaranteed by shared weights) with expert *output* reuse (not guaranteed).
- Do not confuse high top-k overlap with lack of learning/progression; many recurrent systems deliberately refine representations with the same experts.
- Do not confuse masking after GEMM with a reduction in compute, nor number of removed edges with actual GPU wall-clock time.
- Report negative results and examples where same experts produce dissimilar outputs. Detect if `f_exp` is so low that even perfect elimination cannot yield a meaningful E2E win.
- Separate prefill and decode: prefill exposes many tokens and may pack GEMMs efficiently; B=1 decode may be launch-bound. Mechanisms may work only for one regime.
- Report numerical reproducibility: BF16, router/logit dtype, deterministic flags, seeds, FA backend, CUDA graphs/compilation where applicable.
- Preserve legal attribution and model license requirements; no unapproved dataset publication.

---

## 11. Decision gates: explicit verdicts

The following are **predeclared heuristic research gates**, not established empirical outcomes. Include uncertainty and qualify any unsupported gate.

### P0 operational gate

- `READY`: all three models load, run, preserve reference correctness, and fit at least the shared B=1 smoke matrix on eligible single GPUs.
- `BLOCKED_RUNTIME`, `BLOCKED_HARDWARE`, or `BLOCKED_CORRECTNESS`: describe issue, attempted legitimate remediation, affected model/GPU, reproducible logs. Continue independent work where feasible, but never claim global three-model completion.

### P1/P2 scientific headroom gate

- `NO_GO_HEADROOM` when even a generous *quality-constrained* ideal E2E gain remains below **1.10x** across substantive low-request cases in both IFM models, or when perturbations sufficient to produce headroom cause unacceptably large quality drift.
- `CONDITIONAL` if one IFM model crosses 1.10x or Foil supports a mechanism but the two 8.78B IFM models disagree, or if analytical gains cannot be implemented under native kernel granularity.
- `GO_PROTOTYPE` if an accurately replayed, quality-constrained oracle indicates **>=1.15x ideal E2E** in at least one meaningful B=1 regime for **both** IFM models and Foil has been evaluated (Foil may be a disconfirming control). An ideal gain alone never establishes final GO.

Default *quality-constrained* operating target: point estimate `ΔNLL <= +0.02` nats/token on held-out language modeling **and** <=1 percentage point absolute drop on reliably parsed downstream tasks, with CI interpretation and sufficiently competent model-specific baselines. These are sensitivity targets, **not** safe/meaningful acceptance if the baseline is near chance, confidence intervals are wide, or generation has catastrophic outliers. Always provide a stricter exact-output / zero-drop frontier.

### P3 implementation gate

- `GO_SINGLE_GPU`: reproducible **>=1.10x measured E2E speedup** in at least one meaningful B=1 setting for **both** IFM models, with acceptable held-out quality and all three model outcomes recorded; preferably larger 1.15x+ gains or clear generality.
- `MODEL_SPECIFIC_ONLY`: a robust gain on only one model. Do not frame as a general Looped MoE systems result.
- `NO_GO_IMPLEMENTATION`: only FLOP/route-count gains without positive end-to-end gains, or accuracy failure, or high additional memory/overhead.

Report each model's own P0/P1/P2/P3 verdict, plus an overall verdict. Explain whether differences arise from recurrent topology, router statistics, kernel implementation, or scale; avoid claiming causality without an ablation. **Multi-GPU EP is not authorized by this spec**, but may be proposed as a separate follow-up only after single-GPU evidence supports it.

---

## 12. Required figures and tables

**Table A — Environment and checkpoint verification:** model ID, exact revision, params, loops, physical recurrent MoE sites, experts/site, top-k, trained tokens, dtype, attention backend, architecture, GPU/compute capability, runtime and status.

**Table B — Representative baseline:** per model/workload B=1 prefill, decode TPOT, E2E, `T_exp/T_e2e`, `T_attention/T_e2e`, `T_router+dispatch+combine/T_e2e`, peak HBM. Include confidence intervals; define measurement windows and avoid overlapping-span double count.

**Table C — Routing/representation:** adjacent-pass top-k Jaccard, exact-set retention, routing entropy, expert-input cosine/relative L2, expert-output cosine/relative L2, weighted-contribution norm, early-vs-late-loop differences and null baselines.

**Table D — Oracle/candidate tradeoff:** each model, arm B0/O1/O2/C1/C2/N1, targeted vs actual reduced expert work, ideal-bound speedup, **measured** speedup when applicable, ΔNLL, token agreement, GSM8K/MATH-500 accuracy and CI, cache memory.

**Figures:** loop-index curves for routing overlap, hidden/output similarity, contribution norms, distribution of expensive cases, Pareto frontier with separate oracle vs causal curves, Amdahl headroom vs quality-constrained `q`, prefill vs decode stacked timing, workload/length sensitivity and confidence intervals.

**Narrative:** a short 1-page executive `go_no_go.md` backed by machine-readable raw runs. Must clearly say whether there is a **loop-specific** effect or merely generic MoE behavior.

---

## 13. Mandatory automated tests / integrity checks

Implement and run tests for:

- Same public model/checkpoint and exact BF16 settings; model adapters don't reinitialize parameters or change inference depth inadvertently;
- Instrumentation disabled = reference baseline; instrumentation on but no intervention produces matching reference outputs;
- Correct physical recurrent-site identity, token-position mapping and no prelude/coda misattribution;
- Exact preservation of model-specific routing weights, top-k and normalization when probes are disabled;
- Foil shifted token labels, no invalid padding mask;
- Cache keys never mix tokens, requests, sites or experts; resetting between requests works; shared-expert IDs alone never trigger unconditional equivalence;
- `q=0` intervention matches B0 within specified tolerance; random control budget matches treatment budget;
- Actual candidate skips expert arithmetic, not just suppresses its result after compute;
- Per-workload output counts, denominator sanity, no empty-task accuracy, no contaminated tune/test split;
- Raw run logs identify physical GPU and validate one model process per device;
- Calculated speedups derive from paired wall-clock times; modeled ideal speedups are separately labeled.

---

## 14. Execution sequence for the coding agent

1. Read this entire specification. Audit existing repo conventions, recent experiment scripts and reports without overwriting them.
2. Inspect all eight GPUs and environment; enumerate per-GPU eligibility for the two IFM models and Foil. Check FA3/native runtime compatibility **before** long downloads/benchmark runs.
3. Pin upstream sources and exact 3 checkpoints; create per-engine isolated environments if necessary. Download each once and verify.
4. Implement 3 native model adapters and smoke-level parity checks. Persist environment/run manifests.
5. Execute P0 for **Ouro, Huginn and Foil-1**. Fix legitimate runtime problems with verified procedures; publish partial P0 blockers rather than fake an all-green run.
6. Schedule P1 probes on 8 GPUs using eligible worker slots and lightweight recordings. Produce tables/plots, including nulls and counterexamples, for **all three models**.
7. Measure P2A baseline timing fractions independently and compute Amdahl-style ideal bounds; establish whether expert compute contributes enough time to matter.
8. Execute P2B, always including B0/O1/O2/C1/C2/N1 or a documented blocked-equivalent for **all three models**. Run actual perturbed forwards, autoregressive rollout and quality checks; keep oracles unmistakably labeled.
9. If P2 gate passes, implement minimal P3 causal candidate(s) for all 3; otherwise stop costly optimization and finalize quantitative NO-GO.
10. Emit environment, raw results, summary CSVs, figures, Go/No-Go and instructions to reproduce. Verify no file is fabricated or reported complete without execution evidence.
11. Run tests, review `git diff`, commit only generated code/spec/small reports when appropriate; never assume push permissions. Give the user an honest concise summary: exact commands, 3-model coverage, eligible GPUs used, key measured metrics, negative findings, unresolved blockers, and revision hash.

### Minimum run-manifest record

Each run row must include: `model_id`, `checkpoint_revision`, `upstream_code_commit`, `local_code_commit`, `gpu_physical_id`, `gpu_name`, `compute_capability`, `torch_version`, `cuda_version`, `attention_backend`, `dtype`, `workload_id`, `sequence_ids_hash`, `split`, `input_length_actual`, `new_tokens`, `batch_size`, `recurrent_passes_actual`, `experiment_arm`, `target_q`, `effective_q`, `seed`, `warmup_count`, `timed_repeat_count`, `time_window`, `baseline_ms`, `variant_ms`, `speedup_measured`, `speedup_modeled`, `quality_metrics`, `cache_peak_bytes`, `status`, `failure_log`.

### Suggested source entrypoints (names may be adapted to local project style)

```bash
# Inspect first; never presume installed packages or hardware compatibility.
nvidia-smi -L
nvidia-smi topo -m
python -V

# After dependencies/models are validated:
python -m looped_moe_headroom.scripts.preflight --gpus 0,1,2,3,4,5,6,7
python -m looped_moe_headroom.scripts.run --phase p0 --models ouro,huginn,foil1 --gpus 0,1,2,3,4,5,6,7
python -m looped_moe_headroom.scripts.run --phase p1 --models ouro,huginn,foil1 --gpus 0,1,2,3,4,5,6,7
python -m looped_moe_headroom.scripts.run --phase p2 --models ouro,huginn,foil1 --gpus 0,1,2,3,4,5,6,7
python -m looped_moe_headroom.scripts.summarize --output poc_flashvep/reports/looped_moe_headroom
```

The entrypoints above are **requested implementation interfaces, not existing commands**. The coding agent must implement/test them or update README with the actual, working equivalents. Never print fabricated benchmark numbers or assert a successful remote execution from document creation alone.

---

## 15. Success criteria in one sentence

**Demonstrate with the same three public, unmodified BF16 models that loop-wise expert routing, input/output redundancy, and true single-GPU E2E acceleration are causally and quantitatively linked—or conclude NO-GO with high-confidence measurements identifying why not.**
