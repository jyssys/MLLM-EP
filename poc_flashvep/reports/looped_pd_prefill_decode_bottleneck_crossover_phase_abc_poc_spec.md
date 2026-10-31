# PoC Specification: Prefill–Decode Bottleneck Crossover and Adaptive Recurrent Partitioning

**Study:** 2-GPU capacity characterization, bottleneck phase diagram, and conditional Dynamic A evaluation (Phases A/B/C)  
**Target systems:** Dense Looped LLMs Ouro Base 1.4B (BF16, fixed R4) and Huginn-0125 (BF16, fixed R32), plus a diagnostic conventional dense-AR control.  
**GPU pool:** physical GPU IDs 0–7 available for isolated experiments. **Primary comparison: 2 GPUs, 1P+1D vs DP2**. No required 4/8-GPU pooled-serving campaign.  
**Budget guideline:** 10–12 hours for one discovery campaign if feasible, with transparent gate-based partial completion; correctness and isolation take priority. This is an experiment budget, not a promise of completion.  
**Prior code/evidence:** `jyssys/MLLM-EP`, branch `flashvep/adaptive-loop-token-pd-2gpu-w1-w4`, source/evidence commit `d2050debde78c33c4a77fce5fe3ef342fc5f3694`.  
**Spec target:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/looped_pd_prefill_decode_bottleneck_crossover_phase_abc_poc_spec.md`.

## 1. Scientific motivation and non-claims

The previous 2-GPU campaign completed 56/56 W1–W4 cells and 2,272 clean requests across main, repeats, and protocol controls. It qualified Loop/Token transfer correctness against the existing numerical reference. Yet primary Loop continuation occurred in only 5/112 Ouro and 2/112 Huginn requests, all in W2; repeated W1 online policies used full-PD in 48/48 requests for each model. In the Loop-Dynamic primary run, sampled P GPU utilization was around 3–13%, while D utilization was around 67–94%; Huginn W2/W3 showed severe D KV-reservation queues. These data **do not** show that recurrent architecture inherently causes a decode bottleneck. They characterize one implementation, 1P1D allocation, and long-generation workloads.

Two distinct issues must be separated:

1. **Per-request time:** total Decode generation interval can far exceed Prefill interval, even though a single Prefill execution delays one Decode step.
2. **Pool capacity:** the P-only service capacity `mu_P` and D-only completion capacity `mu_D` (requests/s, for a fixed length distribution) determine which stage becomes congested. Memory-admission and host-scheduling capacity can be different bottlenecks.

Do **not** label a workload prefill-heavy solely because input tokens exceed output tokens, because a request's Prefill is slower than one Decode token, or because TTFT is high. Instead quantify service capacity, queue growth, reserved KV credit, and Decode-side SLO slack. Distinguish `P_COMPUTE_LIMITED`, `D_COMPUTE_LIMITED`, `D_KV_ADMISSION_LIMITED`, `HOST_CONTROL_LIMITED`, `BALANCED`, and `INCONCLUSIVE`.

The central question is whether a **measured** regime exists in which Prefill is the limiting resource and D has sufficient room to borrow some of that work. A negative answer over the declared grid is a valid outcome. No post-hoc relaxation or cherry-picked success-only claims.

## 2. Related work and the scientific boundary

- DistServe (OSDI 2024): separate P/D resources and optimize TTFT/TPOT-SLO goodput, not an assumption that per-request Prefill wall time exceeds total Decode.
  https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
- DuetServe (ICML 2026): its 2×H100 Qwen3-8B 8K-input/200-output motivation uses the official vLLM PD demo setting and illustrates the **prefill-saturated 1P1D** case. Importantly, its *method* is not restricted to prefill-heavy: default co-location and **SM-level P/D isolation only when TBT is threatened**.
  https://proceedings.mlr.press/v306/gao26m.html
- FluidPD (Oct 2026): `FluidToken` borrows D capacity for transient P pressure; `FluidRole` reassigns roles for sustained P/D imbalance. Dynamic borrowing or role shifting alone is established prior art.
  https://arxiv.org/abs/2610.06917
- TaiChi (2025): aggregation/disaggregation/hybrid selection changes with TTFT/TPOT priorities.
  https://arxiv.org/abs/2508.01989
- NVIDIA *Beyond the Buzz* (MLSys 2026): disaggregation has especially strong opportunities in prefill-heavy traffic; decode-heavy can make colocated execution competitive; mapping, model scale, and SLO strongly affect crossover.
  https://research.nvidia.com/publication/2025-06_beyond-buzz-pragmatic-take-inference-disaggregation
- Load-aware Prefill Deflection (2026): D-side prefill borrowing with explicit Decode SLO protection is a strong generic comparison.
  https://arxiv.org/abs/2607.02043

**Novelty restriction:** Neither P→D prefill migration, utilization-based admission, nor recurrent-boundary selection alone proves loop-exclusive novelty. At the same boundary, a same-model generic logical-layer implementation can be equivalent. This campaign characterizes the bottleneck and tests Loop-vs-Token under matched conditions; it does not declare exclusive loop novelty without independent evidence.

## 3. Immutable model, engine, hardware, and safety contract

- Ouro: exactly the qualified Ouro Base 1.4B checkpoint and fixed R4; Huginn: exactly the qualified Huginn-0125 checkpoint and fixed R32. BF16 weights/KV, no approximation, no early exit, no weight modification. Reuse `down_norm` and `down_norm_adapter` numerical corrections from the qualified release.
- Preserve previous branch, commits, actual source archives, and private RLT runtime diff; use independent worktree/branch and unique output namespaces. Record Git commit SHA, private engine HEAD+diff hash, checkpoint revisions, CUDA/NCCL stack, per-GPU UUID/topology, GPU memory/clock, applied overlay hashes, CPU affinity, and all configuration files.
- Available GPUs 0–7. **During clean primary timing, execute one isolated 2-GPU lane at a time** unless an independent fabric/CPU-contention test validates parallel timing. Non-timing correctness/compilation tasks may use other GPU pairs subject to ownership checks. No force push; do not touch other users' processes. Restore only validated owned burners after experiment.
- Set equal memory and graph policies for comparable conditions. The previous RLT config had 4 compute slots, a fixed D KV pool, 256-token chunk/budget on PD, and different full-prompt atomic DP behavior. Record and vary these instead of silently treating them as algorithmic properties.
- Never describe the existing no-Mixed DP+Chunked as a production-quality Sarathi-Serve implementation. Shared-GEMM Mixed remains disabled in causal partition-axis experiments; an optional **matched** optimized Mixed comparison is a separate supplementary experiment, not a new contribution.
- Explicitly distinguish the native PD_G1 chunk-streaming Full-PD handoff from the full-cut snapshot used by Loop/Token connectors. Include same-boundary transfer-protocol controls wherever declaring gains over PD_G1.

## 4. Controlled workload grid and datasets

Use real full-text ShareGPT prompts where they meet desired length bins. Prefer a **real long-context QA/extraction** dataset for short answers (e.g., appropriate LongBench tasks), with exact source/task named. Never call a length-matched synthetic sample a genuine trace. For controlled length experiments, use `ignore_eos=True`, report that it is a stress workload, and preserve original prompt IDs, source hashes, tokenizer IDs, exact output targets, and seed. A small natural-EOS validation can be supplementary. Calibration/test prompts must be disjoint.

**Core grid, primary 2-GPU characterization:**

| ID | Input tokens | Output tokens | Purpose |
|---|---:|---:|---|
| G1 | 2,048 | 128 | Moderate prefill; 16:1 |
| G2 | 3,072 | 128 | Long-input candidate; 24:1 |
| G3 | 3,500 | 128 | High-input candidate; 27.3:1 |
| G4 | 3,072 | 192 | Medium generation; 16:1 |
| G5 | 3,500 | 192 | Long prompt with longer generation; 18.2:1 |
| G6 | 3,072 | 256 | Increased Decode demand; 12:1 |

**Controls:**

- C1: 1,536 input / 448 output: Decode-heavy control from prior setting.
- C2: ~1,900 input / 448 output: prior W1-like mixed/long-generation control.
- Optional *diagnostic only*: 3,500/32 or 3,500/64, only if the primary 128–256 grid does not exhibit the hypothesized crossover; label separately, do not replace the primary grid.
- Optional faithful DuetServe 8K/200 test **only on a dense-AR model with supported context and matching GPU assumptions**, not Huginn. Never equate that result with controlled same-model Looped comparison.

**Context gate:** Huginn hard max context 4,096. Verify `actual_prompt_tokens + target_output_tokens <= 4096` using the model tokenizer for *every* request; no silent trimming. Use a common-support prompt subset when comparing models; include per-model tokenizer lengths. 3,500+256=3,756 satisfies this constraint.

Prefer 24–32 requests per cell in screening; use >=64 at selected critical points with repeating independent server starts and an open-loop Poisson rate sweep. Budget reduction must be predeclared and symmetric across comparators, not cherry-picked. W1–W4 previous results remain historical controls, not rerun as substitutes for this grid.

## 5. Phase A — independent service capacity and architecture diagnosis

### A1. Honest P-only measurement

Run a Prefill-only worker with real Ouro/Huginn inference, no D KV reservation gate, no associated Decode worker blocking its admission. Measure prompt Prefill service demand and sustainable throughput over different queueing loads, for token budgets/chunks {256, 512, 1024, 2048} subject to valid execution. Measure batch sizes {1, 2, 4} (and 8 only if KV allows), CPU submission, actual CUDA kernel busy, GEMM shapes, end-to-end service time *excluding* waiting/Decode. Avoid using host TTFT containing D reservation as `P_time`.

### A2. Honest D-only measurement

Initialize valid Prefill KV and any model-specific hidden/injection/RNG state for the requested prompt under the original full-depth model. Load/seed the D cache from a qualified snapshot or an equivalent exact state. **Exclude the one-time preparation/transfer from timed Decode-only service**, but preserve a separate preparation/transfer-cost measure. Produce the requested full 128/192/256 tokens at original R4/R32; verify output parity and KV correctness against the same baseline. Sweep active Decode concurrency {1,2,4} and KV pool capacity/headroom; record batching and exact D reservation capacity. Never count a fabricated KV state as a valid decode microbenchmark.

### A3. Capacity estimation and dense-AR diagnostic control

Estimate `mu_P` and `mu_D` (requests/s) at **the same prompt/output length distribution and legal concurrency** with explicit intervals and capacity definition. Generate estimated `mu_P/mu_D` and uncertainty. Separate compute vs KV-admission vs CPU control constraints. Measured throughput near a single rate is not necessarily stable service capacity; check completion/drain/queue trends.

Add at least one real **conventional dense AR** control (for example Qwen3-1.7B, exact checkpoint/config verified first) using the same supported input/output, GPU types, BF16 and controlled generation; optionally add Qwen3-8B if capacity/time allow. It is a structural **diagnostic**, not a latency-normalized head-to-head model race. Note differences in architecture, KV footprint, number of logical layers, model parameters, kernels, and engine (vLLM vs RLT). Do not attribute differences to recurrence without additional matched-kernel evidence. Record raw and normalized capacities for each model.

### A4. Outcome

A condition is **P-service-limited candidate** if credible evidence supports `mu_P < mu_D` with matched resource assumptions and no dominant KV/admission restriction that invalidates D spare. If all candidates have `mu_D <= mu_P`, report `NO_P_SIDE_CROSSOVER_ON_GRID` with evidence; proceed with Phase B diagnosis but do not manufacture a P-heavy result.

## 6. Phase B — true 1P1D bottleneck crossover

### B1. Resource matching and admission instrumentation

Run the same exact request streams on 1P+1D and DP2. Ensure state and pool capacities are documented. Record for every request: arrival, coordinator admission, D KV reservation attempt/success, P queue entry, P begin/finish, transfer begin/end/ack, D ready, first-token delivery, individual token events, completion, failures; log reasoned waiting. Also collect actual token batch sizes, per-worker CPU scheduling/submission, CUDA activity, KV pool allocated/reserved/free, active decode, GPU power/utilization, P/D queues, and completed bytes. Do not count D-KV-reservation waiting as P compute bottleneck.

### B2. Rate sweep

For each G1–G6 and C1/C2 at least one initial clean diagnostic rate, then focus on two or three promising candidates. Sweep offered rate relative to independently measured capacities at factors e.g. {0.50,0.75,0.90,1.05,1.20}×min(mu_P,mu_D), subject to finite-run budget. Include points near `mu_P` **only if D has a credible reserve**. Run same prompt IDs and arrival seeds for all arms; W3-style 3–5× prefills bursts are optional supplementary. Track queue slope/age and drain; 24–32-request cohorts cannot establish infinite-horizon stability. At selected points, use >=64 requests and >=3 independent starts where feasible.

### B3. Causal bottleneck classification

Predeclare phase classification:
- `P_COMPUTE_LIMITED`: P actual service near measured capacity + P compute queue growth; D has real CPU/GPU/KV headroom and ITL slack.
- `D_COMPUTE_LIMITED`: D service/Decode queue or token age dominates; P has spare.
- `D_KV_ADMISSION_LIMITED`: D lacks valid KV credit, blocking P before Prefill regardless of P GPU utilization.
- `HOST_CONTROL_LIMITED`: reservations/scheduling/dispatch dominate despite unsaturated GPU compute.
- `BALANCED` or `INCONCLUSIVE`: no single reliable bottleneck.

Use GPU traces and queue trends, not solely NVML GPU utilization. Produce a phase diagram over input length, output length, arrival rate, model and batch size. Show a 2D map for each model and add an explicit AR diagnostic comparison. Document kernel and memory reasons if AR and Looped differ, not merely a different ratio.

### B4. A controlled SLO study

Freeze SLOs from an independent calibration set before held-out candidate runs. Keep distinct TTFT + request-average TPOT (the prior primary criterion) **and** a predeclared ITL p95/p99/maximum-interruption target for interference sensitivity. Report both, never silently switch the target based on a favorable result. SLO-goodput = number of offered requests meeting the criteria divided by arrival-to-drain wall time; also show offered rate and completion throughput. Finite-cohort SLO-goodput is not proven sustainable capacity; define a stability-qualified criterion if sufficient data exist.

## 7. Phase C — conditional Dynamic A headroom

**Run only when Phase B identifies a verified P-compute-limited case with D compute/KV slack.** A result with D-KV reservation blocking P is not an eligible P-limited regime. Use the same frozen prompt/output and arrival stream across all conditions.

### C1. Main comparison arms

1. DP2 (two replicas) — strong real-system control.
2. DP2 + separate-phase Chunked Prefill — explanatory baseline with no new Mixed claims.
3. PD_G1 native streaming — full-prefill PD control.
4. Full Cut R snapshot — matched transport-only control for other snapshot arms.
5. Loop-Static: fixed cut chosen using disjoint calibration (Ouro 0–4, Huginn 0/8/16/24/28/32; include R endpoint).
6. Loop-Dynamic: **live** P/D state-based cut with a logged effective nonfull cut, reservation recheck and safe source fallback.
7. Token-Static: valid prefix/suffix migration with matched tuning budget.
8. Token-Dynamic: live token-boundary selection with matched telemetry, request streams, and equivalent normalized work fractions.

The user's core original seven arms are preserved; Full-Cut snapshot is an explicit confound control. **No intra-loop Layer Split and no 4/8-GPU pooled-serving expansion in this study.** Co-batching/shared-GEMM Mixed is off for matched axis comparison. Do not disable or artificially degrade the strongest real PD_G1 control; report protocol mismatch where unavoidable.

### C2. Fix the previous conservative cost-model pathology

The prior online policy estimated D continuation from `median(host TTFT / input_tokens)` and constrained it using `0.5 × (TPOT_SLO − max_decode_token_age)`. This conflates queueing with GPU service and uses a stricter next-token-age surrogate than its **request-average TPOT** primary SLO. Replace or ablate it using measured **exclusive D GPU service time** and predicted queue/critical path, plus explicit KV credit and SLO headroom. Compare `old_conservative`, `new_SLO_safe`, and `forced_static` policies on the same trace. No unconstrained threshold relaxation, no future arrivals or true future output length. A dynamic policy that chooses Full-PD for all requests must report `NO_EFFECTIVE_PARTITION` rather than an algorithmic speedup.

### C3. Quantify real mechanistic changes

Count for each arm/model/workload: full-PD choices, initial nonfull choices, accepted D continuations, D admission rejections, source-side fallbacks, actual Loop/Token work migrated, KV/hidden/control bytes, P and D GPU service, queue, TTFT, TPOT, ITL, E2E, and both SLO goodputs. Attribute protocol-only speedups to protocol; exclude results where no new method executed. At matched load compare Loop-Dynamic vs Token-Dynamic and best Static, PD_G1 and DP2. Analyze performance at neighboring D-heavy controls to ensure safe fallback.

### C4. Follow-up optional micro-mechanism

If Stage C reveals worthwhile unused P/D overlapping execution windows, profile chunk-frontier recurrent continuation as a separate *optional* concept. Do **not** implement it before establishing the basic crossover or assert novelty from pipeline overlap alone.

## 8. Runtime budget and stopping policy

The previous 10–12h user budget is a **guideline** for initial discovery; preserving all three phases may require a second authorized campaign. A hard deadline (if set) must be enforced with the priority **correctness → A capacity map → B crossover → conditional C**; never fake or extrapolate unmeasured GPU data. Suggested initial planning blocks: A 2–3h, B 3–4h, conditional C 3–4h, cleanup/analysis 1h; these are allocation ceilings/targets, not task-completion estimates. Prefill-only/D-only engines may need nontrivial implementation and parity work, which takes priority over speed.

Prioritize at least one representative dense AR diagnostic and both Looped models in A/B. If the time budget becomes tight, declare a frozen reduced grid G2/G3/G5 + C1 with equal coverage for both models, log `BUDGET_REDUCED`, and defer other cells. If no P-headroom is confirmed, do not waste the remaining GPU budget running full C methods; use it to diagnose D KV reservation, CPU control, and model-specific capacity differences. Close GPU leases safely and restore owned burners; report `NOT_RUN_TIME_BUDGET` and `GATE_NOT_MET` for unexecuted cells.

## 9. Measurement requirements / quality gates

- Baseline output/token IDs, stop reason, per-depth issued row counters, valid KV, final state, transfer hashes on finite gates. Huginn's Prelude/Coda/injection/suffix initialization remains qualified. Original HF and downstream accuracy are separate and not established by internal parity.
- Profiling runs segregated from clean latency; host E2E component shares are not exclusive GPU kernel/DMA shares. Do not sum overlapping GPU work as a critical path.
- Independent starts, controlled seed/prompt ordering, per-request raw output (or a compressed lossless file), no missing/failed request exclusion from denominator. Record CIs at the independent-start level, and note small N.
- Same GPU budget, equivalent model memory constraints, meaningful chunk/batch tuning for all comparators; never declare a static cut globally optimal based on 4 calibration prompts.
- Overlap/actual bottleneck classification includes observation windows that demonstrate enough ongoing work; short finite throughput should never be reported as a max stable capacity.
- Report two dimensions separately: `prefill vs decode workload demand` and `TTFT vs TPOT/ITL SLO stringency`.

## 10. Deliverables / names

Save self-identifying snake_case reports in `/home/esjung/MLLM-EP-github/poc_flashvep/reports/`:

- `looped_pd_phase_a_independent_capacity.md` + machine-readable capacity/latency CSVs.
- `looped_pd_phase_b_bottleneck_crossover.md` + per-model crossover phase diagram (CSV, optional PNG).
- `looped_pd_phase_c_dynamic_a_headroom.md` + run- and request-level comparison data.
- `looped_pd_prefill_decode_bottleneck_final_summary.md` containing model-wise `P_COMPUTE_LIMITED`/`D_COMPUTE_LIMITED`/`D_KV_ADMISSION_LIMITED`/`INCONCLUSIVE` cases and GO/NO-GO for C.
- `looped_pd_prefill_decode_bottleneck_reproduction.md`, configurations, command ledger, source hashes, failed runs and proof of GPU restoration.

Use a new isolated code subdirectory `poc_flashvep/looped_pd_bottleneck_crossover/` unless existing verified code can be safely reused without modifying protected sources. Changes limited to new artifacts; only commit owned changes and push normally after validating source/index protection.

**Decision:** `GO_DYNAMIC_A_DISCOVERY` only if a defensible P-compute-limited/D-spare regime and nontrivial fixed-cut benefit occur; `GO_ONLINE` only if live nonfull migrated work produces repeatable SLO-goodput/latency improvements beyond tuned controls; `NO_P_HEADROOM_ON_TESTED_GRID` if none is found; `NO_GO_PERFORMANCE` if actual migration cannot beat controls; `INCONCLUSIVE` for missing validity/stability. Document both successful and unsuccessful cells without selection bias.
