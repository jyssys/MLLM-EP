# PoC Specification — Decode-First Dynamic-Depth-Aware DP Scheduling Discovery

**Working title:** *Cross-GPU Decode-Depth Balancing for Looped LM Serving*  
**Document version:** 3.0 — dynamic-baseline recovery and continuation (supersedes v2 stop-policy, retains v2 scheduling protocol)  
**Date:** 2026-10-18  
**Target installation path:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/spec.md`  
**MUST READ on the server:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/decode_first_depth_aware_dp_spec.md` (the original decode-first plan). Back up existing `spec.md`; do not overwrite the original source specification.  
**Research platform:** 4 independent data-parallel (DP) replicas on **physical CUDA GPUs 4, 5, 6, 7 only**  
**Primary workload:** adaptive-depth *decode* with Ouro / Continuous Depth Batching (CDB)  
**Secondary workload:** focused prefill feasibility and comparison; not a second full project  
**Forbidden in this PoC:** reinforcement-learning training, new recurrent architecture, speculative decoding, presenting trace replay as live inference, fabricating unavailable upstream code or reported scores, silent protocol changes, destructive edits to existing repository state.

## 0. Executive decision, scope and continuity contract

**Research objective:** Measure *additional multi-GPU DP placement/scheduling gains* on top of a fixed, credible, pre-existing **dynamic-depth Ouro** implementation. The scheduling baseline and proposed scheduler **must use the same checkpoint, exit gate, q, KV policy, generation settings and physical work semantics**. The target is faster tail latency/goodput **at matching adaptive-model output quality**, not to match fixed-R4 quality exactly or train a new gate.

**What changes in v3:** The prior 32+32-request pilot terminated under v2's strict full-R4 quality guardrail (`QUALITY_FAIL`). Under this revised research question, that guardrail was misaligned: a literature-standard adaptive operating point may legitimately trade some task accuracy for fewer loops. Preserve the original result and its uncertainty. Add a **short, bounded baseline-recovery / upstream-reproduction track**; prefer a better-quality verified published implementation if it is runnable, otherwise reuse the incumbent adaptive runtime. Then continue v2 Stage 1–5 without retraining any RL policy.

**Frozen incoming pilot facts, NOT new measurements:** Base `ByteDance/Ouro-1.4B`, BF16, full-depth prefill, online decode q=0.5, min R2, max R4, `last_exited` KV; 32 naturally generated samples per domain, output cap 8192. Fixed/Adaptive: GSM8K 22/32 vs 21/32; MATH-500 24/32 vs 19/32. Executed decode loop work was lower by 22.86% and 27.97%, respectively. MATH paired absolute-accuracy-drop CI 3.125–28.125 percentage points (as previously reported), but evaluation-dependent/truncation sensitivities and partial exact-reference parity remain documented **warnings**. None of Stage 1 depth heterogeneity, DP4 baselines, scheduling oracle or predictors has been run in that campaign. Do not label those tasks as FAILED or complete.

**Mandatory continuation:** Before editing, read the server's existing `reports/decode_first_depth_aware_dp_spec.md`, `reports/final_report.md` and relevant Stage-0 reports / logs if present; list completed measurements, reusable scripts, exact branch/commit, missing stages and staged/dirty files. If no report is present, note it. Never reset prior work. Use new versioned filenames or safe append-only updates and cite the original evidence.

**Baseline selection rule:** Try upstream approaches in this order: (1) published CDB authors' reproducible code + Ouro original gate; (2) publicly available implementation of *Enabling Dynamic Computation* if and only if its actual repository/code and license can be verified; (3) existing validated `last_exited` runtime. `vllm-rlt` may serve as an **independent fourth practical candidate**, not as CDB authors' official implementation. Do not invent an official repository or call a hand-written imitation a paper's original code.

**Selection objective:** At the **same dataset samples and matched evaluation protocol**, prefer the adaptive implementation with better accuracy *subject to meaningful physical early-exit work reduction and the ability to collect traces and run four independent CDB replicas*. If differences are small/inconclusive, favor the already-validated, compatible, lower-engineering-cost candidate. If upstream quality is worse or integration is blocked, **continue with the incumbent q=.5 last_exited implementation**, explicitly preserving `QUALITY_WARNING` and original pilot numbers; do not claim fixed-R4 quality parity. Any new scheduler must then preserve **incumbent adaptive** outputs to the extent determinism/numeric behavior permits.

**Do not stop at the old 1–2 percentage-point full-R4 drop threshold.** A stop before scheduling now requires: invalid/degenerate outputs, critical KV/gate/correctness bugs, no actual variable-depth computation, an inoperable engine, or an explicit user-imposed quality floor. A moderate and honestly reported adaptive-vs-fixed gap alone is not the primary scheduling-study stop criterion. Never conceal a serious quality issue or represent a warning as proven quality equivalence.

## 1. Clarifications of modeling assumptions

### 1.1 Actual prefill and decode behavior

- Looped models can conceptually assign a depth to each prompt token position and each subsequently decoded token position. This does **not** imply every implementation physically omits loops in both phases.
- In the documented `vllm-rlt` Ouro serving path, prefill runs full depth to produce the first output token, while subsequent decode tokens can use adaptive early exit and loop-level CDB scheduling. Confirm the exact code path actually used; do not generalize this description to all models.
- MoR's token-choice and expert-choice designs select active tokens across recurrent depths and are **not inherently decode-only**. MoR is a separate model/training/routing architecture, not a drop-in Ouro gate or proof that Ouro prefill can be shortened safely.
- The first generated token belongs to the prefill-produced output in the cited runtime; do not label its depth/time as ordinary adaptive decode.
- Actual KV layout and early-exit policy may alter quality and the amount of physical work performed. Separate logical gate exit, actual recurrent operator execution, and quality-preserving cache semantics.

### 1.2 Units and workload counts

- System-wide concurrency `C` means total active/eligible requests across **all four GPUs**, not requests per GPU. For C=8, balanced placement is approximately 2 requests per GPU.
- Each active sequence advances by at most one autoregressive position per decode token generation, but a GPU processes a *batch* of active sequences with variable loop depths. Real loop cost is not proportional to the simple sum of depths because CDB refills/mixes loop tasks and kernel shapes vary.
- For request `i`, output token `t` (after full-depth prefill), let `d[i,t] in {1,...,Rmax}` be the **actual executed exit depth** under a specified gate/KV policy; let `M[i]` be the actual number of decode tokens before EOS/cap.
- `mean_depth[i] = sum_t d[i,t] / M[i]`. `total_loop_work[i] = sum_t d[i,t]`. The former alone cannot balance full request durations; the latter also ignores context and batching effects.
- Per-step heterogeneity compares active requests at a common scheduling epoch; request-level heterogeneity compares sequences over their full (variable) output lengths.
- `decode_only_time` excludes full-depth prefill and first-token generation; `TTFT`, `TPOT`, `ITL`, `P95 E2E`, and `goodput` retain their own rigorous definitions.
- `label`: measured online, measured replay, trace simulated, zero-cost oracle, cost-aware oracle, or speculative estimate. Every plot and table must identify its category.

### 1.3 What is and is not a feasible scheduling action

- **Allowed primary action:** choose a DP rank for an *arriving request before its prefill*; request remains pinned to that GPU for decode (KV locality preserved). Local CDB scheduler continues its native loop-level batching/refill.
- **Allowed secondary diagnostics:** choose *waiting/not-yet-prefilled* requests when rank capacity opens; change admission ordering; optionally tune microbatch/refill controls only if accurately supported.
- **Not allowed for primary oracle:** migrate a request already decoding without accounting for all KV/states and transfer timing; reshuffle decoded tokens independently across GPUs; use true future depths as if available to an online policy.
- **Separate unattainable bounds:** clairvoyant initial assignment (future depths/output lengths known); clairvoyant free migration with zero cost (labeled deliberately unrealistic); migration with measured KV costs (optional only, if the primary result warrants).

## 2. New Stage 0R — upstream adaptive quality recovery (MANDATORY, SHORT)

### 2.1 Sources, availability and provenance (as checked 2026-10-18)

| Source / exact URL | What is available | How it should be used |
|---|---|---|
| Ouro official checkpoint: https://huggingface.co/ByteDance/Ouro-1.4B | Official weights, model code, learned exit gate. Config default `total_ut_steps=4`, `early_exit_threshold=1.0`; official forward may still execute full loops before exit-state selection. | Reference gate and model, not a physical-skipping DP serving engine by itself. Pin revision. |
| CDB paper: https://arxiv.org/pdf/2608.09444 | Ouro q/exit experiments; Appendix E GSM8K quality and KV-policy ablations; serving often uses pre-recorded exit trajectories rather than a live online gate. | Quality guidance + scheduling method; match policy before comparison. |
| **CDB authors' repository:** https://github.com/kschwethelm/continuous-depth-batching | **Public official authors' research implementation**, greedy-only engines and `shells/paper/` reproduction scripts. Not a production serving engine. | First candidate for an upstream reproduction. Inspect actual Ouro loader, gate and KV policies; do not copy its Huginn `kv_policy="single"` example onto Ouro. |
| *Enabling Dynamic Computation in Looped LMs*: https://arxiv.org/pdf/2610.09013 | Paper defines `Dense-Ragged`, `ETE-Ragged`, best-available KV aliasing. **No publicly verifiable authors' code repository was found in this check**. | Re-check authors' official links / arXiv metadata / project page / repo availability. If absent, record `UPSTREAM_CODE_UNAVAILABLE` and do not spend the campaign re-engineering its serving runtime. |
| Independent vllm-rlt: https://github.com/ThinkFlowLab/vllm-rlt | Public, experimental Ouro serving implementation with original learned exit gate, loop-level scheduling, `LAST_EXITED` / `SHARED` paged KV, synchronous default. Some async modes have known correctness limitations in README. | Optional *independent* runtime candidate, never claim it is CDB authors' or the Ragged-KV paper's original code. |

**Conceptual alignment to verify in code:** The Ragged KV paper's best-available retrieval for token `i` and current depth `d` uses `min(d, exit_depth_i)`, whereas a runtime called `last_exited` may have a different layout or fallback policy. Compare **actual key/value lookup and write semantics**, per-depth masks, prefill/decode separation, and aliasing, not policy names. Do not assert equivalence before unit tests.

**Important boundary:** The Ragged KV paper also explores *modified training/exit objectives* to achieve heterogeneous exits. Do **not** accidentally compare or claim the quality of a re-trained gate when running the released **original Base Ouro-1.4B** checkpoint. Any gate/weight revision requires a separate named arm and pre-approval.

### 2.2 Reference numerical anchors — labeled, NOT parity targets

**Enabling Dynamic Computation, Table 1, Base Ouro-1.4B, Dense-Ragged, q=0.5:**

| Benchmark, paper's scoring | Fixed R4 | Dense-Ragged q=.5 | Mean decode depth |
|---|---:|---:|---:|
| GSM8K generated-answer **pass@1** | 76.6% | 75.3% | 3.05 overall for Dense-Ragged |
| MATH-500 generated-answer **pass@1** | 64.2% | 62.4% | same setting |
| BBH generated-answer **pass@1** | 69.0% | 67.5% | same setting |

This publication used **16 rollouts per sample, Qwen3.8-27B judgment** (pass@1 is reported across rollouts). Table 5 of its appendix specifies GSM8K **3-shot, 1,024 max output tokens, N=1,319** and MATH-500 **5-shot, 2,048 max output tokens, N=500**. Its results must **not** be declared numerically reproduced by our *32-question free-running greedy, answer-extraction, 8,192 cap* pilot. Different prompts/judge/rollout/sample definition change the metric. Capture exact decoding/randomness and judge instructions before claiming matched protocol.

**CDB Appendix E / Table 4:** Full depth-indexed KV, Ouro-1.4B, full depth, GSM8K flexible-match greedy = **78.09%**. CDB Figure 14 contains a plotted adaptive accuracy-vs-depth frontier, not a tabulated exact q=.5 MATH-500 reference. CDB's GSM8K protocol and caches differ from Dense-Ragged and this project's previous pilot. The CDB paper's *serving performance* traces often replay pre-recorded depths; separately validate **live online gate**.

**Internal incumbent pilot anchors** (different protocol): GSM8K Fixed 68.75%, Adaptive 65.625% (22/32→21/32), MATH Fixed 75.00%, Adaptive 59.375% (24/32→19/32). Do not subtract the paper's percentages from these values as if they came from a matched experiment.

### 2.3 Candidate investigation, quality micro-reproduction and bounded time budget

**0R-A. Snapshot first.**
- Inspect old server v2 spec, prior `QUALITY_FAIL` report, outputs/trace and tests; preserve old verdict. Record exact checkpoint/hash, tokenizer, prompts, answer parser, EOS/max tokens, q, min-depth, KV semantics, current exit histogram, executed vs logical depth and raw log locations.
- Inspect license and reproducibility scripts for all upstream options; pin **actual commit SHAs**, dependency versions, dataset revisions and GPU IDs. Never upgrade project dependencies globally or overwrite the project's current environment. Use separate venv/worktree/checkout if needed.

**0R-B. Choose at most TWO practical challenger variants** beyond the incumbent: CDB original authors' code first and Ragged-KV official code **only if truly published**; optionally independent vllm-rlt if the first options are incompatible/unavailable. No hours-long reimplementation of an unreleased method. If all challengers are unavailable or invalid, choose `INCUMBENT_CONTINUE` immediately.

**0R-C. Stage comparisons, identical on-device evaluation.**
1. `sanity`: a few shared prompts; check exact fixed-R4/official gate where feasible, tokenizer/chat template, online exit, minR2–maxR4, KV reads/writes, actual skipped loop kernels and EOS; report mismatches explicitly. Reference-output bitwise parity may differ across FlashAttention/batch kernels; classify strict/epsilon/token agreement.
2. `paired pilot`: use the **same frozen 32 GSM8K + 32 MATH-500 IDs**, generate free-running greedy with identical preprocessing/answer parser/max 8192 for incumbent and each compatible challenger; log correct/N, accuracy, paired deltas, paired bootstrap CI, EOS/truncation, outputs, per-request/token actual depth and physical loop reduction.
3. `confirmation`: if any challenger looks better on the small pilot, test on a **new held-out validation cohort** (e.g. >=128/domain where feasible, preferably full 1,319/500 for definitive broad quality claims). Avoid treating an improvement of 1 or 2 questions in 32 as established. Keep selection data and confirmation separate.
4. `paper-protocol cross-check`, if practical: independently reproduce a small paper-matched prompt/shot/max token variant; run available paper judge only when legitimately accessible. Full 16-rollout Qwen3.8 judge reproduction is **optional and may be expensive**, not a prerequisite for scheduling discovery. Mark `PARTIAL_PROTOCOL_MATCH` unless all defining conditions match. Preserve greedy results as the **shared operational metric**.
5. `scheduler readiness`: is there a runnable single-GPU CDB refill/microbatch engine, real online gate, trace hooks and a plausible 4 independent-GPU layout? Count implementation/porting cost. Quality alone cannot select a research engine without this prerequisite.

**0R-D. Freeze the implementation used for the REST of the PoC.**
- Better validated challenger accuracy on matched protocol **AND** meaningful *physical* loop savings **AND** CDB/DP readiness: `SELECT_UPSTREAM`; lock the chosen checkpoint, q, min/max depth, KV policy and commit. Preserve comparative metrics.
- Worse, statistically inconclusive, unavailable, or costly/nonfunctional challenger: `INCUMBENT_CONTINUE`; keep Base Ouro-1.4B q=.5 R2–R4 `last_exited`. Explicitly carry prior 32-question MATH **quality warning** into the final report.
- If the new upstream runtime is a reference-quality evaluation harness but lacks CDB/DP, it may be used **only to validate quality**. Do not mix its output traces/exit decisions with the incumbent CDB cost model and label such a hybrid "live measured DP".
- A candidate that obtains higher accuracy by silently running fixed R4 is **ineligible** as an adaptive replacement. Require actual depth histogram and physical skipped-work evidence.
- Only `INVALID_ADAPTIVE_IMPL` / `UNSAFE_QUALITY_COLLAPSE` / `NO_PHYSICAL_DEPTH_SAVINGS` / `NO_RUNTIME_CANDIDATE` blocks the scheduling track. A moderate Adaptive–Fixed R4 quality gap is a **warning**, not an automatic stop.

### 2.4 Stage-0R report and decisions

Write `reports/dynamic_quality_reference_recovery.md` and an immutable `artifacts/dynamic_baseline_selection.json` with: every repo/link/commit checked (including code-not-found evidence); matched and unmatched quality tables; exit/KV behavior; physical computation; 32-sample + confirmation outputs; selected runtime exact config; why this runtime is fit for Stage 1–4; and unresolved differences from paper/official Ouro.

Decision values: `SELECT_UPSTREAM`, `INCUMBENT_CONTINUE`, `REFERENCE_MATCH_PARTIAL`, `UPSTREAM_CODE_UNAVAILABLE`, `INVALID_ADAPTIVE_IMPL`, `UNSAFE_QUALITY_COLLAPSE`, `NO_PHYSICAL_DEPTH_SAVINGS`, `NO_RUNTIME_CANDIDATE`, `INCONCLUSIVE`.

**Then resume Stage 1, Stage 2, Stage 3 and conditional Stage 4/5 below exactly as in v2**, with the selected implementation pinned for all compared schedulers. Work must not stop at Stage 0R merely because an outside paper's absolute accuracy differs from our own scoring.

## 3. Research questions and hypotheses

- **RQ1 (Depth heterogeneity):** Does adaptive decode exhibit meaningful *between-request* and *time-varying within-request* depth variation beyond random sampling noise, at the pinned, quality-characterized adaptive operating point?
- **RQ2 (Cost relevance):** Given actual CDB loop/refill behavior, do depth differences explain latency variation after controlling for output length, prompt/context length, and batch state?
- **RQ3 (Cross-rank residual):** After queue-aware, length-aware, and measured-cost-aware placement across 4 independent CDB engines, are some GPUs idle/underutilized while other GPUs have runnable queued decode work?
- **RQ4 (Oracle headroom):** How much additional *P95 end-to-end*, *decode ITL/TPOT*, and *goodput* improvement can clairvoyant depth-aware **initial placement** deliver beyond the strongest general DP4 baseline?
- **RQ5 (Deployability):** If an oracle shows headroom, can admissible prompt-only features forecast sufficient future depth/work to realize a meaningful fraction of it without costly model computation?
- **RQ6 (Prefill cross-check):** Is the same effect present in true compute-skipping prefill? Otherwise is prefill essentially length-dominated and less compelling?

Null hypotheses must be tested, not assumed false. New multi-GPU recurrent-specific novelty depends on incremental gain **over strong generic length-aware/load-aware baselines**, not over round robin.

## 4. Experimental hardware and engineering rules

- Physical GPU IDs: **4, 5, 6, 7 only**. Export `CUDA_VISIBLE_DEVICES=4,5,6,7`. Process-local IDs become 0, 1, 2, 3; record both maps.
- Four independent replicas of the exact same approved checkpoint and KV policy. Do not assume TP4/EP4; this is DP4, one replica/rank. Verify fit and memory headroom on the actual allocated devices.
- Do not interfere with other users' jobs or GPUs; preserve unrelated git dirty/staged changes. Do not use sudo/kill on other processes. Record NVIDIA topology, available GPU memory, clocks/power constraints if permitted, software and repo revisions.
- Do not overwrite existing `/home/esjung/MLLM-EP-github/poc_flashvep/reports/spec.md` without preserving the previous specification as `spec_prefill_v1_backup.md` or a clearly versioned backup.
- Use `torch.cuda.Event` or profiler/Nsight traces with synchronization at correct boundaries and instrument prefill, first token, each recurrent loop, gating, decode, KV/cache and scheduler queue durations. Report timing definition; exclude one-time model loading from serving request latency.
- Re-run measurements (at least 3 independent measured trials when feasible); record warmup, sample size, seed, variance/CI, and profiler overhead. Log OOMs and unsupported conditions as such.

## 5. Data design and statistical discipline

### 5.1 Inputs

Use diverse authentic prompts from (subject to license/availability): ShareGPT, Alpaca, GSM8K, MATH-500, BBH, and optionally ARC-Challenge. No dataset is presumed to exhibit greater recurrence-depth variance. Include length-matched cohorts and domain-matched controls; report model-specific tokenizer lengths.

- Pilot: e.g., **128–256 distinct requests**, distributed across data domains and prompt-length strata; not enough to claim final significance.
- Scale up to >=512 diverse requests, preferably >=1000, only if pilot shows physically executed heterogeneity and the runtime is reliable.
- Predefine train/calibration, validation, held-out test splits; remove near duplicates. Do not tune gating thresholds or simulator on held-out test.
- Preserve raw prompt IDs, token lengths, max output token caps, actual EOS, and dataset provenance. Do not replace natural completion with forced tokens past EOS.

### 5.2 Two complementary output-length regimes

A. **Natural variable-length generation:** Free-running greedy decode until EOS or a generous common cap; measure actual output length. Crucial for request makespan, p95 E2E and genuine serving behavior.

B. **Output-length-controlled cohorts:** Group or match requests with comparable realized output lengths (e.g. quantiles/strata) and analyze first K naturally produced decode tokens **only for sequences that actually survive K**. Never force computation beyond EOS. Use this to isolate marginal depth signal from length effects; state selection bias from requiring survival.

Prefer K=32, 64, 128 if enough naturally long traces; adapt to observed lengths. Do not confuse equal *max_new_tokens* with equal *actual output length*.

### 5.3 Arrival patterns

- Total system-wide concurrency: `C = 1, 2, 4, 8, 16, 32, 64`. For C<=4 there may be almost no initial-placement balancing opportunity because four replicas can each hold one request; report this negative result explicitly.
- Arrival patterns: simultaneous bursts, rolling closed-loop replacement, and Poisson/open-loop at measured sub-saturation/near-saturation rates. Document arrival rate in req/s and exact definitions of active vs queued requests.
- Repeat multiple deterministic seeds and sample shuffles; report medians and tail CIs when sample sizes support them.

## 6. Stage 1 — Live decode-depth trace and heterogeneity discovery

Trace for each request `i` and autoregressive token position `t`:

`request_id`, domain, prompt_len, output_len_so_far, total_output_len (retrospective only), token_index, token_id, exit_depth_logical, exit_depth_executed, per-loop active-token counts, gate/threshold/Q-exit, KV/cache policy, scheduler epoch, assigned rank, per-loop timings, decode iteration wall-time, queue waiting time, batch membership/refill events, EOS/truncation, all runtime versions.

Compute:

- token-level depth histogram R1..R4, per-request mean/std depth, total executed loop-token work, depth-run autocorrelation and time-varying depth trends;
- across-request variability of mean depth **conditional on output length, prompt length and dataset**; within-request variability and burstiness across active requests;
- cross-rank measured idle vs other-rank queued/runnable recurrent work; load coefficient of variation; overlap and skew of per-rank completion times;
- regress/permutation-test incremental explanatory power of depth beyond prompt length and output length; do not conflate correlation with predictability at admission;
- check whether q and KV policy have turned almost all requests into the same effective R2/R4 behavior. No artificial quality-sacrificing threshold cherry-picking.

Publish fixed sample-size per domain, histograms, per-request distributions, same-length cohort scatter, depth vs per-token latency and CDB observed kernel batch sizes. If variance is negligible, record `NO_GO_HETEROGENEITY`.

## 7. Stage 2 — Measured CDB time/cost model

The cost to process a recurrent step depends on actual active token count, context distribution, GPU state and refill scheduling. Calibrate *on hardware*:

- CDB per-loop runtime vs active token/microbatch size (1,2,4,8,16,32 etc. within budget) and KV context bins (e.g. 1K,4K,8K where possible);
- gate overhead, sampling/output overhead, queue waiting, scheduling overhead, KV memory, GPU idle and stalls;
- cost of mixed-depth vs similar-depth batches using equivalent token/request sets, not arbitrary replacement workloads;
- contrast fixed R4, dynamic no-refill, dynamic CDB refill at matching quality/threshold.

Build a transparent event-driven or directly trace-replayed 4-replica simulator that respects:

1. online arrival and request-admission times;
2. per-rank CDB loop batching and refill;
3. request order, EOS/completion and active/pending queue states;
4. actual loop-level performance lookup/interpolation from measured GPU data;
5. one-request-one-rank pinned KV locality, no cross-rank token stealing;
6. shared resource capacity and per-rank KV/HBM constraints;
7. first-token full-depth prefill and prefill resource occupancy (even if measuring decode gains separately).

**Validate simulator first:** replay held-out workloads under the same plain round-robin and queue-aware DP4 policies and compare simulation predictions to **live four-GPU measurements** for makespan, P50/P95 TTFT, TPOT/ITL and E2E, throughput and imbalance. Report absolute/relative prediction error. Suggested acceptance: within ~10% on sufficiently sampled robust latency aggregations (and no systematic queueing error); if not achieved, stop claims of realistic oracle gains and label `SIMULATOR_UNVALIDATED`. This 10% is a research engineering heuristic, not a universal accuracy standard.

## 8. Stage 3 — Multi-GPU admission-only scheduling oracle (PRIMARY)

### 8.1 Baselines — same hardware, same traces, same quality

A. `RR-DP4+CDB` — round-robin initial assignment.  
B. `Queue-DP4+CDB` — queue depth / runnable work at admission.  
C. `Length-DP4+CDB` — known prompt length and separately calibrated *admissible* expected output length; distinguish oracle-true output length as a separate upper-bound control.  
D. `StrongCost-DP4+CDB` — calibrated rank service rate, active batches, queue, KV headroom and known prompt/context data, but **no true future depth**.  
E. `PerfectDepth-only` — replaces unknown depth with future true depth but uses a **matched output-length assumption** to isolate depth benefit.  
F. `PerfectDepth+Length` — clairvoyant true future depth **and** true future output length; optimistic ceiling, not deployable.  
G. `DepthHomogeneous` — same-depth grouping / locality, isolate local-batch effect and compare to CDB's native mixed-depth refill.  
H. `DepthBalance+Microbatch` — optional joint oracle; isolate delta beyond E/F after accounting for extra scheduling and fairness constraints.

**Strategy B, Cross-GPU Depth Balancing, is the primary test.** Compare `E` and `F` against the strongest of A–D, not RR alone. Compare `F` against a matched **true-length-only oracle** so a length effect cannot be misreported as a recurrent-depth benefit.

### 8.2 Bounds and operational realism

- **Lower bound / conservative:** CDB-aware measured-cost simulator plus dispatch, gate, cache and timing overheads, no model state migration, online arrivals and per-rank capacity.
- **Clairvoyant upper bound:** perfect future per-request executed depth trajectory and optionally actual EOS timing known before admission; mathematically infeasible online, useful solely for headroom.
- **Zero-cost ideal oracle:** optional sensitivity bound; never report as speedup or E2E observation.
- **Impossible schedule checker:** reject any proposal assigning already-prefilled request tokens to another rank without KV cost; all microbatch members must be on their resident rank.

Optimize the same resource budget and workload under one consistent objective (e.g. min p95 E2E subject to goodput/quality/fairness, with decode-tail secondary). Do not swap oracle objectives after seeing results. Use at least a fair heuristic search, e.g. exhaustive or branch-and-bound on tiny bursts, greedy balancing/local search for larger cases; do not call a weak heuristic 'perfect optimum'. Give both (i) actual solved-optimal small-instance bound and (ii) scalable best-known clairvoyant search when optimality is not proven.

### 8.3 Metrics and ablations

- Per-request P50/P95 E2E, TTFT, mean TPOT and decode ITL distribution; P95 decode latency; total-work makespan, requests/s, output tokens/s, goodput at predeclared SLO (if meaningful); queue delay, rank execution/idle windows, rank wall-clock skew and average KV memory.
- Report gain both absolute (ms, req/s) and relative, including paired request-level CI/seed variance. Keep throughput and latency tradeoffs visible; do not report only peak ideal speedup.
- Break down `depth-only incremental gain`, `length-only gain`, `combined gain`, `microbatch interaction` and `oracle-generic scheduling gap`.
- Critical comparison: concurrency <=4 vs >=8; fixed-output-length cohort vs natural EOS; shorter vs longer generation; mixed domain vs length-matched domain. If results only emerge at large bursts, say so.
- Run an explicit rank imbalance witness: a rank idle while a different rank has **eligible runnable work**, not merely different average GPU utilization.

**Stop before predictive training** if the oracle depth-specific incremental effect is negligible relative to strong baselines or disappears when accounting for real CDB batch costs.

## 9. Stage 4 — Causal prediction feasibility (conditional, no RL)

Only if Stage 3 is promising:

- Evaluate **admission-only** prompt length, domain-independent token features, cheap prompt embeddings/metadata, not future tokens or post-prefill hidden state. Include prediction overhead and placement decision cost.
- Predict (i) future average decode depth, (ii) total decode recurrent work and its uncertainty, (iii) expected future output length separately. Label the quantity and its information horizon.
- Compare constant-depth, prompt-length-only, calibrated linear/GBDT/MLP models (small and cheap); independent held-out tests, no leakage of gate trace into features.
- Predictive metrics: MAE/RMSE for future average depth, rank correlation for total work, calibrated interval coverage, quality-conditioned distribution; **primary metric is achieved held-out DP placement gain** compared with StrongCost. Depth prediction accuracy without scheduling gain is not a GO.
- Optional causal **history-aware** predictor after K produced tokens (K=8/16/32), but without cross-GPU migration it can only affect future local scheduling or waiting-request decisions; do not count impossible rank balancing.
- RL, policy-gradient tuning, online depth changes and PPO-style training **do not belong in this PoC**. If there is headroom but poor predictability, report `PREDICTION_BLOCKED` rather than manufacturing an RL story.

## 10. Stage 5 — Prefill secondary branch (strictly bounded)

Audit selected runtime for actual prefill early exit; record prompt-position logical depth distribution, executed recurrence count, quality/KV consequences and prefill component time. If prefill is full-depth:

- record `PREFILL_FULL_DEPTH`; a prefill-specific depth-aware placement advantage is **unsupported**;
- do not build a new token-selective prefill runtime in this PoC;
- compare prompt length / measured prefill cost only as a **generic admission** feature for decode work placement; include TTFT overhead in all E2E comparisons.

If **validated real prefill skipping** already exists at acceptable quality, analyze request-level prompt-depth heterogeneity and a *small parallel prefill oracle* using the same strong rank-placement baselines. This optional branch must not displace the decode experiment.

## 11. Revised accuracy, integrity and continuation gates

- **Research comparison of interest:** `DP4+CDB(adaptive, same selected checkpoint/gate/q/KV)` vs `DepthAware-DP4+CDB(adaptive, identical inference path)`. Output quality should be the **same under the same adaptive model**, allowing explicitly characterized nondeterministic BF16/batch-shape effects. Differences in request placement are not permission to change exit policy, model weights, KV semantics, or sampling.
- `Adaptive vs Fixed-R4` accuracy is a **contextual quality–compute tradeoff**, **not an automatic 1–2 pp fail gate**. The old Stage-0 MATH warning remains reported rather than discarded. Do not claim paper-equivalent quality merely because the paper reports a nearby average depth.
- Mandatory transparency: same-request GSM8K/MATH-500 correct/N, absolute accuracy, adaptive-vs-R4 drop, paired bootstrap 95% CI (clearly specifying direction), completion/EOS/truncation breakdown, cap, prompt template, parser/judge, output length and code revisions; include original pilot and any new comparative datasets separately.
- When confirming a selected dynamic implementation, evaluate enough held-out questions to make the quality warning actionable; full tests are preferable for paper-level claims but **are not mandatory before a clearly labeled exploratory DP oracle**. If 32 samples only, qualify all quality generalization.
- **Critical stop conditions:** wrong KV indexing/writes, uncontrolled numerical/gate bug, degenerate responses (e.g. mass EOS/truncation/zero scores), missing actual kernel loop savings, inability to run identical adaptive baselines on 4 GPUs, or explicit user quality constraint violation. Isolated MATH pilot drop with partial paper-protocol match is `QUALITY_WARNING`, not sufficient alone to prevent depth-trace/oracle discovery.
- If official/authoritative implementation differs from current pilot, record and explain differences. For two DP policies using the chosen runtime, compare greedy token traces under matched seeds/settings and report any divergences and impact. Do not falsify bitwise equality across variable batch shapes.
- Never tune q/Gate/KV using the held-out oracle/evaluation cohort, rerun until favorable results, overwrite previously poor results, force EOS continuation, or re-label replay as live performance.

## 12. Statistical and implementation rigor

- Separate intrinsic inference latency (CUDA events), request-level queueing delay, admission time, host scheduler overhead and entire service time.
- Provide source and checksum for prompt samples; document if a dataset cannot be accessed. Do not invent recorded depth traces or quality metrics.
- Distinguish evaluation policy's access to future information from the prediction model's causal features. Audit data leakage in every schedule and predictor.
- Before adopting an oracle, quantify how often it uses true future output length versus depth; list rank movement decisions and reject illegal ones.
- Check that all synthetic simulator transitions correspond to a realizable CDB execution order and actual KV locality.
- Sensitivity analysis for q/early-exit threshold, CDB refill mode, request arrival rate, output length, and C. Use only accuracy-approved q values.
- Present failed and negative findings as first-class results and STOP unproductive lines; do not expand into alternate models, MoE or dLLM work without evidence.

## 13. Deliverables and recommended repo layout

Root: `/home/esjung/MLLM-EP-github/poc_flashvep/`

- `reports/spec.md` — **this v3 specification**, if deployed; back up existing `spec.md` with a timestamp. **Do not alter** `reports/decode_first_depth_aware_dp_spec.md` (the prior v2 source to read and preserve).
- `reports/dynamic_quality_reference_recovery.md` — new Stage 0R implementation/quality comparison, decision and limitations.  
- `artifacts/dynamic_baseline_selection.json` — frozen chosen adaptive runtime checkpoint, exit/KV policy, paper-protocol comparability and refs.  
- `reports/environment.md` — physical GPU IDs, topology, env, model/CDB revisions and exact commands.
- `reports/stage0_model_gate_kv.md` — checkpoint support, full-depth baseline, online early-exit behavior and quality.
- `reports/decode_depth_traces.md` — reproducible depth/time histograms and dataset provenance.
- `reports/cdb_cost_model.md` — component timings, CDB refill, batch/context cost surfaces, calibration error.
- `reports/live_dp4_baselines.md` — actual four-rank tests incl. queue/length/cost policies.
- `reports/decode_oracle.md` — depth vs length incremental headroom; validity/prohibitions.
- `reports/prefill_secondary.md` — full-depth status and limited comparison.
- `reports/causal_predictor.md` — optional only if oracle GO.
- `reports/final_report.md` — English narrative with tables, explicit validity limits and recommendations.
- `artifacts/trace_manifest.json`, `artifacts/decode_depth_traces.jsonl` (or parquet), `artifacts/per_loop_costs.csv`, `artifacts/live_dp4_metrics.csv`, `artifacts/oracle_schedule_results.csv`, `artifacts/quality_predictions.csv`, plots and a manifest of seeds/splits.
- `scripts/` reproducible runner and analyses, protected from unrelated project files.

Report all milestones by true category: `RUN`, `NOT_RUN`, `BLOCKED`, `SIMULATED`, `MEASURED`, `FAILED`.

## 14. Revised scheduling decisions and stop/go gates

| Gate | GO evidence | Negative verdict / carry-on logic |
|---|---|---|
| G0R upstream recovery | Reproduced or quality-characterized *online dynamic* Ouro and physically saved loops; frozen config | `SELECT_UPSTREAM` or `INCUMBENT_CONTINUE` **both proceed**. Partial reference parity/accuracy warning documented. Stop only for clearly invalid/degenerate/unusable adaptive runtime. |
| G1 depth heterogeneity | Variation across requests beyond length/domain effects, backed by executed cost | `NO_GO_HETEROGENEITY` |
| G2 cost + actual DP4 | A 4-GPU measured cost model agrees with live baselines and shows residual rank imbalance | `SIMULATOR_UNVALIDATED`, `NO_GO_CROSS_GPU_IMBALANCE` |
| G3 scheduling headroom | Actual-depth oracle adds incremental latency/goodput gains beyond strong queue/length/cost baselines | `GENERIC_LENGTH_ONLY`, `NO_GO_SCHEDULING` |
| G4 predictor feasibility | Cheap features available before placement retain a useful fraction of oracle gain | `PREDICTION_BLOCKED` |

Suggested heuristics from v2 (not scientific universal thresholds): <5% incremental gain beyond the strongest non-depth baseline -> NO-GO; 5–10% -> conditional; >=10–15% with validated costs -> predictor follow-up. Distinguish C<=4 and C>=8. A free-migration oracle with zero KV transfer is deliberately unattainable and must never be reported as a deployable improvement.

Final verdict: `GO_DEPTH_AWARE_SCHEDULING`, `CONDITIONAL_HEADROOM`, `NO_GO_HETEROGENEITY`, `NO_GO_CROSS_GPU_IMBALANCE`, `SIMULATOR_UNVALIDATED`, `GENERIC_LENGTH_ONLY`, `NO_GO_SCHEDULING`, `PREDICTION_BLOCKED`, `INVALID_ADAPTIVE_IMPL`, `NO_RUNTIME_CANDIDATE`, `INCONCLUSIVE`. Record G0R source selection separately; do **not** use the old `QUALITY_FAIL` alone as the final scheduling verdict. Include persistent `QUALITY_WARNING` if warranted.

## 15. Minimum final presentation tables

### A. Gate and quality

| runtime / commit | checkpoint | q / min-max | KV policy | fixed R4 quality | adaptive quality | physical loops saved | protocol comparability / warning |
|---|---|---|---:|---:|---:|---:|---|
| measured | measured | measured | measured | correct/N | correct/N | % | MATCHED_GREEDY / PARTIAL_PAPER_MATCH |

### B. Workload heterogeneity controlling for length

| dataset | N req | prompt length | actual output length | mean depth variance (between req) | extra latency explanatory value | label |
|---|---:|---|---|---|---|---|

### C. Live vs oracle DP4

| C (total requests) | arrivals | policy | E2E P95 | ITL P95 | goodput | rank idle skew | evidence (live vs simulated) |
|---:|---|---|---:|---:|---:|---:|---|

### D. Depth-specific incremental headroom

| cohort | strongest generic baseline | length-only oracle | depth-only oracle | combined oracle | incremental benefit of depth | confidence / cost validation |
|---|---:|---:|---:|---:|---:|---|

### E. Research conclusion

What *physical* decode imbalance exists; why CDB cannot already erase it; why an ordinary queue/length-aware DP scheduler cannot resolve it; how much oracle gain survives measured runtime costs; whether a causal policy could access helpful information early enough. If any answer is unsupported, state so explicitly.

## 16. Source references to verify on the research server

1. Ouro official model and gate — https://huggingface.co/ByteDance/Ouro-1.4B ; config https://huggingface.co/ByteDance/Ouro-1.4B/blob/main/config.json ; project https://ouro-llm.github.io/ ; paper https://arxiv.org/abs/2510.25741
2. CDB — https://arxiv.org/abs/2608.09444 ; official code https://github.com/kschwethelm/continuous-depth-batching
3. `vllm-rlt` README and checkpoint support table — https://github.com/ThinkFlowLab/vllm-rlt/blob/main/README.md (full-depth prefill; adaptive decode for supported Ouro checkpoints; Thinking compatibility bounds).
4. MoR — https://arxiv.org/abs/2507.10524 ; official authors' code https://github.com/raymin0223/mixture_of_recursions (token-/expert-choice gating; not inherently decode-only).
5. Enabling Dynamic Computation in Looped LMs — https://arxiv.org/pdf/2610.09013 (Table 1, Appendix D evaluation details, `Dense-Ragged` and `ETE-Ragged`). **Official code unverified as of 2026-10-18; re-check instead of inventing.**
6. Learning Unmasking Policies for Diffusion Language Models — https://proceedings.mlr.press/v306/jazbec26a.html (inspiration only; unmasking policy is not precomputed Ouro decoding depth).

**End of v3 spec.** This document preserves the full v2 decode-first discovery route after Stage 0R. Do not initiate RL or a new 4GPU engine without demonstrated, quality-characterized, depth-specific headroom.
