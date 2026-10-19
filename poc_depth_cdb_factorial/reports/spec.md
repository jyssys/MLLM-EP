# PoC Specification — Multi-GPU Ouro: CDB Refill × Cross-GPU Hindsight Scheduling

**Version:** 1.0 | **Date:** 2026-10-19 | **Language:** English  
**Required server destination:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/depth_cdb_factorial_hindsight_spec.md`  
**Physical GPUs:** **4, 5, 6, 7 only**. DP1 uses physical GPU 4; DP4 uses physical GPUs 4, 5, 6, 7.  
**Required predecessors to read on the server:** Existing `reports/decode_first_depth_aware_dp_spec.md`, related v3 `final_report.md`, `decode_oracle.md`, `live_dp4_baselines.md`, trace files, scripts and implementation; examine v3 branch `flashvep/decode-first-depth-aware-dp-v3`, commit `0eb9c04d1eef6af33c30a0ca2c0244fb26e41950` if accessible. The prior file locations reported by the user include `MLLM-EP-decode-first-depth-dp-v3/poc_decode_first_depth_aware_dp_v3/reports/final_report.md`. Do not assume that path exists locally without checking.  
**Research goal:** Quantify (a) the benefit of intra-GPU Continuous Depth Batching (CDB) **refill**, (b) the independent/additional benefit of clairvoyant **cross-GPU request placement**, (c) their empirical interaction, and (d) whether refill merely absorbs/saturates depth-heterogeneous work or whether any actual cross-GPU coordination is present. **Finish all five rows even if improvements are tiny or negative.** Do not train an RL predictor.

## 0. Decision and non-negotiable scope

The earlier v3 PoC reported real depth heterogeneity and cross-GPU imbalance among 128 requests, a model-based C64 scheduling estimate of 12.77% latency reduction, and a **live hindsight** C64 reduction of 4.41% with a wide reported 95% interval of [-8.30%, 13.34%]. Existing prompt-only prediction did not capture oracle gains; batch-to-batch token parity was not established. These are **historical user-provided observations**, not independently reproduced measurements. They must be preserved and used to motivate, **not predetermine**, this factorized ablation.

This PoC is deliberately **diagnostic, not an attempt to inflate speedup by weakening the baseline**. A large improvement versus no-refill but little improvement versus refill remains a valuable negative/interaction result. Primary comparisons are **within the same GPU count**. The paper-style 1GPU-normalized 5-row table is an additional absolute multi-GPU scaling view, NOT a substitute for the strongest DP4+CDB baseline.

**Finish-all-rows policy:** No GO/NO-GO threshold on gain may terminate the study. Complete the five runnable configurations at all required primary workload cells, including accurate quality and elapsed-time numbers. If a row truly cannot run because of an irreparable correctness issue, environment blockage or OOM after bounded mitigations, label `NOT_MEASURED` with cause; **never fabricate a number or substitute simulated times for measured times**. Explicitly enumerate attempted mitigations. Skip optional large-C if unsafe; keep core C8/C16/C32/C64. Do not spend open-ended effort building a new serving engine; pursue parity-preserving minimal switches in the same base code first.

## 1. Factorial arms — exactly five required rows

| ID | Paper-style name (exact display label) | Devices | Dynamic exit | Local CDB refill | Cross-GPU Hindsight |
|---|---|---|---|---|---|
| A0 | **Naive Ouro (w/ dynamic depth) GPU1-DP1** | physical 4 | Yes | No | No |
| A1 | **Naive DP4 Ouro (w/ dynamic depth)** | 4,5,6,7 | Yes | No | No |
| A2 | **Naive DP4 Ouro (w/ dynamic depth) + Hindsight** | 4,5,6,7 | Yes | No | Yes |
| A3 | **DP4 Ouro + CDB** | 4,5,6,7 | Yes | Yes | No |
| A4 | **DP4 Ouro + CDB + Hindsight** | 4,5,6,7 | Yes | Yes | Yes |

- **Terminology:** Here `Naive` means **online dynamic-depth with exited items removed and the local recurrent batch allowed to shrink, but no immediate refill**; *not* fixed R4, not a kernel that merely selects an exit after executing all R4, and not a naive depth-agnostic KV cache. A0/A1/A2 should use the **same no-refill dynamic execution primitives** as A3/A4 where possible. If repository code names this `CDB(no-refill)`, say so and retain the paper-style requested five labels.
- **CDB** for this ablation is the **single-GPU loop-step scheduler WITH REFILL**; it may also include shared queue/kernels/coda/prelude components in both modes. Identify which extras are common to no-refill/refill. Do not accidentally switch model, gate, KV, CUDA graphs, speculative features or kernel family only for a subset of arms.
- **Hindsight** means clairvoyant knowledge of **future per-request decode depths and/or output length for rank assignment before that request's prefill**, using a fixed algorithm independently applied to both no-refill and refill. `+ Hindsight` does **not** mean migrating live KV every step. This is not an online, causal method and must never be named an implemented RL policy.
- For A1/A3, implement at least one matched non-clairvoyant placement rule (e.g., `queue/load-aware`) across the four independently running replicas. Use exactly the **same base placement rule in A1 and A3**; document any feasible round-robin and strong cost-aware alternative as extra comparison, not a hidden change to the 5 requested rows.
- For A2/A4, use **the same deterministic hindsight method** and constraints. Run a second stronger hindsight rule (depth-sum vs trace-derived rank-time heuristic) **if technically easy**, label it as sensitivity; do not quietly pick the winner post hoc.

## 2. Frozen shared model, KV, decoding and dataset protocol

**Preferred already validated engine:** previously selected CDB/Ouro stack for **Base `ByteDance/Ouro-1.4B`**, BF16, **online adaptive decode** q=0.5, minimum R2, maximum R4, **`last_exited` depth-aware KV** (or the exact selected version in the existing v3 reports if it differs). Full-depth prefill; first generated token from prefill must be tracked separately. Exit gate is unchanged across arms. Set greedy decoding consistently (including temperature, token IDs, EOS, tokenizer/chat template, prompt format). Pin model/tokenizer revision, CDB repo commit, device mapping and env. No gate retraining or RL.

**Quality warning inherited:** earlier 32-per-domain pilot Base Ouro-1.4B reported dynamic GSM8K 21/32 and MATH-500 19/32 (fixed R4 22/32 and 24/32). Retain this warning. Do not make the full-R4 gap a new scheduling stop, but measure identical adaptive settings for all five arms.

Quality corpus: **same, version-locked GSM8K and MATH-500 samples, identical across five arms**. Aim for full GSM8K test N=1,319 and MATH-500 N=500 for final accuracy when runtime permits. At minimum use a pre-registered, identical **>=128 per domain** sample set for the completed study if full runs are impractical; print N prominently and preserve prior 32-sample values as historical, not the new main measurements. A separate held-out workload corpus (ShareGPT/Alpaca/GSM8K etc.) may be used for load and speed, but the model and decode settings must be identical. For speed tests, do **not** let Hindsight know benchmark correctness labels; it may look only at recorded true execution traces as its deliberately clairvoyant future-information input.

**Parity requirement:** Same task prompts and scoring protocol -> report correct/N, accuracy, EOS/truncation/parse failures **for each arm**, plus token-exact agreement, greedy top-1 disagreement, depth-trace agreement and first divergence case **versus A0** and pairwise A1↔A2 and A3↔A4. Be honest if variable batching changes bit-level numerics or exits. Verify gate/KV correctness in no-refill mode before its performance comparison. If strict parity cannot be achieved, give both measured accuracies and explain the confound; no claim of quality preservation without data.

## 3. Fairness rules for 1GPU and 4GPU comparisons

### 3.1 What exactly is 1.00×?

For a **fixed finite cohort** of N identical requests, with identical prompt+generation workloads and fair admission rules, measure wall-clock elapsed time from submission of first eligible request until last completion:

`T_cohort(Ax, N) = t_last_finished - t_first_submitted`  
`Speedup_1GPU(Ax, N) = T_cohort(A0, N) / T_cohort(Ax, N)`.

Therefore A0 is **1.000×** and the main five-row speed-up column measures **cohort completion/makespan speedup**, not a single-request latency improvement and not necessarily a fixed-load tokens/s ratio. Provide raw cohort seconds next to the × number. Under a fixed identical cohort, request throughput N/T gives the same ratio, but decoded-token throughput requires accounting for token-count drift (report both). For `online` arrivals, do NOT reuse finite-cohort speedup: report stable goodput req/s or tokens/s **with the same arrival rate, censoring policy and latency SLO**, and show normalized throughput ratios separately. Always expose request queueing, P50/P95 E2E, TTFT, ITL/TPOT, KV usage, actual token count, executed loops and hardware utilization.

**Crucial:** Our method uses **four GPUs versus one**. An A4 ≈4.0× absolute cohort speedup might result almost entirely from owning four replicas; it does not imply our Hindsight scheduling gives 4.0×. **Only A4/A3 is the incremental CDB+Hindsight gain**, with matched 4 GPU resource and workload. A2/A1 measures Hindsight's incremental no-refill gain.

### 3.2 Same work, GPU capacity, and concurrency

- Define `C` as **system-wide simultaneous eligible/admitted requests across all GPUs**, not C requests per GPU. At C64 and DP4, balanced assignment is ~16/GPU; Hindsight may assign uneven **counts** if rank capacity allows. At C64 and DP1, all 64 are part of the same eligible workload but may wait in the single GPU's bounded queue; do **not** forcibly resident-allocate 64 long KV caches if that causes OOM.
- Use identical finite request cohorts, prompt IDs, arrival times (burst first; rolling/open-loop separately), output caps, EOS rules and max **per-GPU active token or KV budget** across all arms. Queue excess work. A single GPU is not expected to serve the four-GPU total workload within the same wall-clock deadline; include all its queue time and completion to make absolute speedup fair.
- Fix physical GPU4 for A0. Keep CPU/model download/cache effects controlled; 4GPU tests should not oversubscribe user GPUs, and the same rank-local scheduler budget must apply in no-refill and refill. Fix request order and seed across trials.
- Primary cells: `C ∈ {8,16,32,64}` plus C4 low-load diagnostic. Try C128/C256 **only as conditional memory-safe sensitivity** after recording HBM cap, maximum active slots, queue policy and measured OOM risk. Do not treat C128/C256 as required if unsafe. Distinguish `total cohort N`, `active C`, `max resident requests per GPU`, `decode core batch size`.
- Primary standardization: same prompt-length strata, output-length distribution, and workload mix (including **matched-length but varying mean-depth groups**). For fixed cohort speedup C is the number of requests released together; for rolling/open-loop C is the target eligible/in-flight concurrency, define precisely. Use at least **3 independent paired repeats**, preferably ≥5, alternated/interleaved arm order, report median and paired bootstrap 95% CI. If benchmark variance is large increase repeats rather than claiming certainty.
- Control start/stop synchronization, prefill overlap, CUDA warmup, CUDA synchronization, stream semantics, Python overhead, CPU scheduler bottlenecks, possible shared NVLink contention and `nvidia-smi` occupancy. Include real per-rank local batch shapes; avoid using `nvidia-smi util%` as a sole utilization proxy.

## 4. Step-by-step execution (all stages mandatory, no early stop for low gain)

### Stage 0 — Reuse, inspect and freeze

1. Read historical v2/v3 specs, branch reports, code, raw depth traces, seed/sample lists and previous 1.046× claim; verify how 4.41% was defined (latency reduction vs throughput increase); preserve originals and dirty/staged changes.
2. Locate the authoritative CDB library, benchmark script, no-refill/refill switch and lookahead/scheduler modes. CDB paper distinguishes its **no-refill** local batch-shrink execution from **refill** loop-level queue scheduling. **Confirm in code** that a no-refill toggle does not silently change physical exit, KV, model outputs or admission policy.
3. Confirm online gate execution, true R2–R4 physical skipping and `last_exited` KV correctness across both modes. If test shows replay-depth scheduling, label it REPLAY, not live; live-quality tests must use actual gate.
4. Record four physical GPU health/topology, independent DP ranks, memory limits, software hashes and original report evidence.

### Stage 1 — Minimal five-arm functional smoke

- Construct all **A0–A4** including 1GPU A0 and 4GPU A1–A4. Include a `5 × 4` verification matrix: physical GPU IDs, q/min/max, KV read/write policy, actual kernel work, local refill on/off, Hindsight on/off, output checksum and pass/fail.
- Verify A0 vs A1 no-refill, A1 vs A2 (placement only), A3 vs A4 (placement only), and A1 vs A3 (refill only) on a small matched prompt cohort. Collect token-output parity and per-request depth traces. If not token-identical, explain whether it is numerical drift, different batching, KV policy mismatch or a bug.
- Rule out the error where A0 uses standard full-depth CB while A1–A4 use adaptive depth: **all five must physically skip early-exited steps**.

### Stage 2 — Live factorized throughput/latency evaluation

- Measure **five arms** for C4, C8, C16, C32, C64, with **C8–C64 the required core**, using identical frozen cohort IDs. Evaluate burst cohorts first (consistent wall-clock makespan/Speedup), then rolling/Poisson with defined arrival/SLO if available; do not conflate the latter metrics.
- For each arm and C: wall-clock cohort makespan, requests/s, output tokens/s, TTFT, per-token ITL/TPOT, P50/P95 E2E, GPU per-rank busy/idle, number of active requests, recurrent batch-size histogram, queue length time series, overhead, actual emitted tokens, loop distribution, peak allocated/reserved KV/HBM, OOM/retries.
- Do **not** stop because Hindsight is <1.05× or slower. Negative results must be recorded in the same main tables and analyzed.
- For A2/A4, use the **same hindsight placement algorithm** generated from earlier frozen traces, not tuning on measured outcomes within the evaluation run. If a trace changes due to different batch shape, report prediction/realized work divergence explicitly.

### Stage 3 — Joint explanation: why does CDB reduce or fail to reduce Hindsight's extra benefit?

**3.1 Determine CDB's true scope:** inspect authors' scheduler and deployed local engine. Identify whether code coordinates workloads across DP replicas, tracks other GPUs' queues or moves requests across GPUs. **Do not infer multigpu-awareness solely from good 4GPU balance.** If replicas are independent, mark `LOCAL_ONLY`. If shared queue-aware ingress is present, distinguish the **outer DP request-placement policy** from CDB internal scheduling. Explain whether lower imbalance is a side effect of refill or a conscious global coordinator.

**3.2 Mechanism decomposition:** With exact same incoming requests:
- 4GPU rank-level `sum(actual recurrent steps)`, `sum(generated tokens)`, request count, and **real GPU core milliseconds**, per rank and per time window.
- Local core batch-size distribution and **batch shrink** when no-refill; **refill event count**, average empty slots, pending recurrent queue and coda/prelude occupancy when refill.
- Time when one rank is idle while another has runnable work; separate `idle-with-backlog`, `idle-no-local-backlog` and `blocked-on-KV/memory`.
- Depth-work skew metric (CV and max/min), actual GPU-time skew, per-rank makespan, queue/coda costs, fraction of total E2E from prefill vs decode.
- Are Hindsight placements simply smoothing a workload sum? Compare estimated vs realized GPU-time rankings. Did local batching nonlinearity or request finish-time dynamics erase the predicted gain?
- Direct paired difference between A2–A1 and A4–A3; inspect whether local CDB refill **absorbs** (not coordinates) the imbalance and/or nonlinearly reduces the benefit of better placement.

**3.3 Cross-C interaction:** Plot C8/C16/C32/C64 over the identical five-arm data: normalized makespan, refill utilization, queueing, imbalance, KV peak, difference in incremental Hindsight speedup. Avoid declaring a trend based on one noisy C.

**3.4 Comparison to earlier v3:** Rebuild or audit the C64 12.77%-model / 4.41%-live hindsight gap. Is identical estimator/metric/workload used? Were upper-bound depths assigned to prefill admission vs migration? Recompute speedups from raw time with CI. Explicitly mark anything not reproducible and never equate a model oracle with live measured performance.

### Stage 4 — Accuracy, robustness and paired statistical checks (ALL five arms)

- Use identical pre-registered GSM8K and MATH-500 questions, same prompt/seed/token cap and grader. Report `correct/N`, accuracy percentage, paired 95% CI for accuracy differences **relative to A0 and relative to A3 for A4**. Include EOS/truncation/parse-fail counts and total generated tokens.
- Output quality of adaptive model should be similar across arms; a significant difference caused by admission ordering or local batch shape is a **confound to report**. Do not hide behind the common fixed-R4 vs dynamic quality gap.
- If generating all five arms on full benchmark is prohibitive, at least achieve N>=128/domain on identical pre-registered items and include coverage percentage, with a detailed transparent reason for missing full evaluation. Do not present the old 32-question smoke as completed five-arm accuracy.
- Paired trial IDs and bootstrap confidence intervals for speedup; report which statistic the CI describes (makespan vs P95 latency vs goodput), and ensure conversion to × uses original paired raw samples. Avoid mechanically transforming CI for a different metric.

### Stage 5 — Final paper-style tables, factorized contributions and verdict

**Mandatory Main Table 1: 5 rows, one concurrency at a time, default C64**. All speedups use **A0 1GPU naïve dynamic Ouro** for the same cohort and metric as baseline:

| Methods | GPUs | GSM8K Acc. ↑ (correct/N) | MATH-500 Acc. ↑ (correct/N) | Makespan (s) ↓ | **Speed-Up ↑ (vs A0 DP1)** | Peak HBM/GPU (GiB) |
|:--|--:|--:|--:|--:|--:|--:|
| Naive Ouro (w/ dynamic depth) GPU1-DP1 | 1 | TBD | TBD | TBD | **1.000×** | TBD |
| Naive DP4 Ouro (w/ dynamic depth) | 4 | TBD | TBD | TBD | TBD× | TBD |
| Naive DP4 Ouro (w/ dynamic depth) + Hindsight | 4 | TBD | TBD | TBD | TBD× | TBD |
| DP4 Ouro + CDB | 4 | TBD | TBD | TBD | TBD× | TBD |
| **DP4 Ouro + CDB + Hindsight** | **4** | TBD | TBD | TBD | **TBD×** | TBD |

Caption MUST say: **fixed C64 (or actual C), identical request cohort, same online adaptive gate and KV policy, speed-up computed as 1GPU A0 cohort makespan divided by method cohort makespan; this is multi-request throughput/makespan scaling, not single-request latency speedup.** Provide same five-row table for **C8/C16/C32/C64**, or one wide table of columns `Speedup@C8, Speedup@C16, Speedup@C32, Speedup@C64` PLUS C64 raw latency, Accuracy columns. If workloads differ across C, each C has its own corresponding 1GPU A0 baseline; never use C64's A0 as C32 denominator. Real values in every required cell, even <1.0×, whenever execution is valid.

**Mandatory Table 2: true factorial incremental contributions (speedups ×)**

| Local scheduling | Without Hindsight | With Hindsight | Incremental Hindsight gain |
|:--|:--|:--|:--|
| No-refill | A1 (1.000× within no-refill DP4) | A2 (A1 elapsed / A2 elapsed) | **A2/A1** elapsed-time speedup |
| CDB refill | A3 (1.000× within refill DP4) | A4 (A3 elapsed / A4 elapsed) | **A4/A3** elapsed-time speedup |

Additionally `CDB_no_hindsight = T(A1)/T(A3)` and `CDB_with_hindsight = T(A2)/T(A4)`. Compute `interaction = log[T(A1)/T(A2)] - log[T(A3)/T(A4)]` from **paired cohort times** (a multiplicative interaction; alternative ratio-of-ratios permitted); include CI and clarify direction. Avoid incorrect operations on arithmetic averages of ratios.

**Mandatory Table 3: operating-point / scaling diagnostics**, per-C mean/median real GPU time, idle-with-backlog, active core batch histogram, generated tokens, loop-step count, queue waiting, max HBM/GPU and Output Parity (%); use C8/C16/C32/C64 and C128/C256 if safely feasible.

**Mandatory visualizations** (if helpful): four C-vs-Speedup curves for the DP4 arms all normalized to corresponding A0; two incremental Hindsight curves `A2/A1` and `A4/A3`; per-rank GPU timelines for C32 and C64; Depth Work vs GPU milliseconds scatter; no-refill shrink / refill occupancy histograms.

**Mandatory final explanation** addressing explicitly:
1. Is CDB only optimizing GPU-local loop scheduling, or is the *actual used code* globally coordinating DP4? Show source locations or scheduler state logs.
2. Is the Hindsight action mathematically independent/orthogonal to CDB's **placement** action? Does it nonetheless have an interactive or diminishing effect on **measured performance**?
3. Exactly how much does Hindsight add on no-refill vs on refill, and what part comes from rank scheduling rather than local batching? Present **both absolute A0-normalized × and matched 4GPU incremental ×**.
4. How does C8/C16/C32/C64 change, and is any C128/C256 difference dominated by memory/queueing?
5. Is the old 1.046× improvement reproduced, and why does modeled improvement differ from live performance?
6. Do all five configurations give the same Adaptive-model output/accuracy? Describe any difference in EOS, exit distribution and accuracy.
7. What remains for a future RL predictor to exploit, **without prematurely claiming it will or will not succeed**?

**Suggested verdict taxonomy**: `ORTHOGONAL_AND_ADDITIVE`, `ORTHOGONAL_BUT_DIMINISHING`, `GENERIC_GAIN_ONLY`, `NO_GO_INCREMENTAL_HEADROOM`, `QUALITY_CONFOUNDED`, `INCOMPLETE_INFRASTRUCTURE`. The verdict must be based on results, not optimistic framing. **Even when NO-GO, finish and publish all five rows.**

## 5. Artifact and reproducibility contract

Save files beneath a new, isolated directory (e.g. `poc_depth_cdb_factorial/`) with the following paths; preserve existing branches, reports and dirty/staged work:

- `reports/spec.md` — link or snapshot of this English spec; additionally preserve the server root reports copy specified above.
- `reports/environment_and_reuse.md` — previously available code/trace, model hash, actual 4 GPU IDs, `CUDA_VISIBLE_DEVICES`, CDB mode switches, dataset revisions, exact commands.
- `reports/five_arm_functional_parity.md` — five-arm configuration matrix, gate/KV/refill decisions, output equivalence/mismatch samples.
- `reports/quality_five_arms.md` — per-arm GSM8K/MATH accuracy/CI, completions, truncations, parity.
- `reports/latency_five_arms.md` — all measured raw speeds, confidence intervals, five-row A0-normalized tables C8–C64.
- `reports/cdb_interaction_analysis.md` — factorial incremental speedups, interaction and local/global scope proof.
- `reports/concurrency_and_memory.md` — queue/active/resident GPU state, C128/C256 optional feasibility.
- `reports/final_report.md` — required tables, C-dependent interpretation, novel scheduling headroom, honest GO/NO-GO.
- `artifacts/arm_manifest.json` — five frozen configs; physical/visible GPU mapping, raw dataset IDs, hashes, run seeds.
- `artifacts/metrics_raw.csv`, `artifacts/quality_raw.jsonl`, `artifacts/placements.jsonl`, `artifacts/rank_timeline.jsonl`, `artifacts/trace_schema.json`, `artifacts/benchmark_commands.sh` — reproducible raw measurements.
- `figures/` — result plots, with raw generation scripts.

**GPU safety:** Use only physical 4/5/6/7; no killing other users' processes; do not reset drivers or destroy others' GPU sessions. Before each run verify GPU ownership and memory. Separate any 1GPU and 4GPU comparisons in time if resource contention would bias results. Preserve original git staged/unstaged changes. Commit/push only changes belonging to this PoC when safe, report branch and SHA. All estimated vs live values labeled unambiguously.

## 6. Minimal design checklist for an agent

- [ ] Confirm five arms, especially that A0 uses dynamic depth **and** no-refill.
- [ ] Confirm the same gate/q/R2–R4/KV/tokenizer/precision and sample IDs across all five.
- [ ] DP1 physical GPU4; DP4 physical 4,5,6,7.
- [ ] Distinguish C (all GPUs total) from resident batch capacity and decode recurrent batch size.
- [ ] Identify Hindsight algorithm in previous v3; implement identical rank assignment in both A2 and A4.
- [ ] Make fairness and no-migration semantics explicit, and preserve same local scheduling config within comparisons.
- [ ] Collect genuine live time, first and last request timestamps, GPU per-rank traces, not just a cost-model oracle.
- [ ] Execute C8/C16/C32/C64 × all 5 even when gains are low; no gain-based early exit.
- [ ] Report proper A0-normalized × and separate A2/A1 and A4/A3 ×; raw milliseconds/seconds.
- [ ] Report real GSM8K and MATH-500 scores for each arm and any nondeterministic token divergences.
- [ ] Explain whether CDB code *actually* coordinates across DP ranks rather than conjecturing.
- [ ] Evaluate effective local refill, rank idle-with-backlog, and modeled-to-live mismatch.
- [ ] Conclude with full paper-style tables and transparent statistical uncertainty.

**Sources and prior-art guardrail:** The paper-style table layout is inspired **only in presentation** by *Accelerating Diffusion via Hybrid Data-Pipeline Parallelism Based on Conditional Guidance Scheduling*, arXiv:2602.21760, Table 1 (GPU count, methods, raw latency, Speed-Up, quality). Do not copy its numbers or imply equivalent single-sample latency semantics. CDB implementation modes and scheduling are documented in *Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching*, arXiv:2608.09444 and the authors' repository `https://github.com/kschwethelm/continuous-depth-batching`. Review these and original code before final implementation.
