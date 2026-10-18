# PoC Specification — Decode-First Dynamic-Depth-Aware DP Scheduling Discovery

**Working title:** *Cross-GPU Decode-Depth Balancing for Looped LM Serving*  
**Document version:** 2.0 (replaces the earlier prefill-first proposal)  
**Date:** 2026-10-18  
**Target installation path:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/spec.md`  
**Research platform:** 4 independent data-parallel (DP) replicas on **physical CUDA GPUs 4, 5, 6, 7 only**  
**Primary workload:** adaptive-depth *decode* with Ouro / Continuous Depth Batching (CDB)  
**Secondary workload:** focused prefill feasibility and comparison; not a second full project  
**Forbidden in this PoC:** reinforcement-learning training, new recurrent architecture, speculative decoding, pretending replay is live performance, accuracy-damaging KV shortcuts.

## 0. Executive decision and thesis

**Primary question:** When single-GPU CDB is already efficient, does cross-replica decode-depth heterogeneity leave *additional and actionable* multi-GPU DP scheduling headroom beyond strong queue-, length-, and load-aware placement?

**Method:** Verify a quality-preserving live adaptive-depth decode implementation; collect real causal exit trajectories and execution times; calibrate a live CDB cost model; compare measured DP4 baselines with trace-driven, cost-aware depth oracles; only then test tiny causal predictors. No RL policy is trained here. Cross-GPU depth balancing is the primary scheduling idea, with homogeneous microbatching and joint policies as controls.

**Critical distinction:** A loop-depth predictor is not automatically a usable DP admission feature. At request arrival, *future decoded tokens and their loop depths do not exist yet*. A perfect oracle may access future depths retrospectively to measure an **unattainable upper bound**. A deployable admission predictor must use prompt and request metadata available *before KV placement*. A causal online predictor can use past tokens/depths, but moving an active request between DP ranks requires KV transfer or a different architecture and is **out of scope** for the primary result.

**Hard stop:** If actual adaptive decode does not execute fewer loops while preserving acceptable quality, or if a validated oracle barely beats a strong DP4 scheduler, stop; do **not** fit an RL policy to a nonexistent optimization opportunity.

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

## 2. Model and runtime compatibility — mandatory Stage 0

**Start with:** `ByteDance/Ouro-1.4B-Thinking`, BF16, Rmax=4, because previous project experiments validated its fixed-R4 path. **Do not assume its learned adaptive gate is validated in the chosen CDB runtime.** Public `vllm-rlt` documentation describes bounded fixed-depth synchronous integration for Ouro-1.4B-Thinking; its adaptive-depth validation principally covers base Ouro-1.4B. See the references below.

Procedure:

1. Identify precise checkpoint IDs/revisions, model weights, exit-gate checkpoint availability, `q`/threshold semantics, recurrent loop bounds, KV policy, CDB branch/revision, and decode implementation. Verify that early exit *actually stops execution* and that per-token exit depths are observable.
2. Prefer an **actually validated gate-enabled** Ouro checkpoint/runtime pairing for the primary dynamic-depth study, potentially `ByteDance/Ouro-1.4B` if Thinking is unsupported; keep Thinking fixed-R4 as a diagnostic anchor. **Never silently substitute checkpoints or transfer accuracy numbers between them.** Document any switch in the executive summary.
3. Reproduce full-depth exact logits/token/KV consistency against the model's reference as previously done; then establish live adaptive-depth correctness with a quality-preserving KV layout (`depth-indexed` or validated `last-exited` / best-available policy as implemented). Avoid single shared KV when it breaks reasoning accuracy.
4. Run online gate, not just recorded-exit replay; inspect exit-depth distribution, number of executed recurrent calls, device time and actual KV writes. Exit depth labels without compute savings fail Stage 0.
5. For every threshold `q`, report achieved mean depth, exit histogram, gate policy, actual skipped work, quality, runtime, and KV semantics. Pre-register an acceptable quality operating point. Do not lower q solely to manufacture heterogeneous depths.
6. Maintain a separate fixed-R4 reference and compare against adaptive decode quality in free-running greedy generation. Gate correctness, first-token behavior, and output EOS/truncation must be logged.

**Stage-0 outcomes:** `ADAPTIVE_DECODE_VALID`, `MODEL_GATE_UNSUPPORTED`, `QUALITY_FAIL`, or `NO_PHYSICAL_DEPTH_SAVINGS`; unsupported/failed conditions block further primary scheduling claims.

## 3. Research questions and hypotheses

- **RQ1 (Depth heterogeneity):** Does adaptive decode exhibit meaningful *between-request* and *time-varying within-request* depth variation beyond random sampling noise, at a quality-preserving threshold?
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

## 11. Accuracy and safety gates

- Quality is non-negotiable. Validate fixed R4 vs actual online adaptive-depth *free-running greedy* generations on GSM8K and MATH-500: pilot 16–32 samples/domain, then preferably full GSM8K test (1,319) and MATH-500 (500) for any final claimed quality-preserving operating point. Record correct/N, accuracy %, percentage-point drop, paired bootstrap 95% confidence intervals, parse failures, EOS/truncation and output cap. Use identical answer extraction and generation configs. A 2/2 smoke pass is **not** enough to establish model quality.
- A user-selected quality guardrail may be established before inspecting results, e.g. target <=1–2 percentage points of absolute accuracy drop versus a relevant baseline; data uncertainty prevents overclaiming strict equivalence. Choose and report one policy before thresholds are tuned.
- Ensure KV layouts and early exits yield acceptable quality; never use a speed-friendly KV shortcut known to destroy Ouro math reasoning.
- For pure DP placement comparisons using the identical live gate and KV policies, verify deterministic-seed token output / token count agreement (allow documented BF16 batch-shape numeric tolerance where appropriate). Report any divergence and its effect. No mixing baseline accuracy of one checkpoint with another.
- Never suppress model output length changes or force generations past EOS simply to equalize the cost model.

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

- `reports/spec.md` — **this** revised specification; preserve previous version as `reports/spec_prefill_v1_backup.md` or equivalent.
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

## 14. Decisions and go/no-go gates

| Gate | GO evidence | Negative verdict |
|---|---|---|
| G0 checkpoint/quality/physical skipping | Actual, quality-acceptable adaptive decode with measured skipped loops | `MODEL_GATE_UNSUPPORTED`, `QUALITY_FAIL`, `NO_PHYSICAL_DEPTH_SAVINGS` |
| G1 request depth heterogeneity | Nontrivial variability beyond lengths, reflected in executed work/time | `NO_GO_HETEROGENEITY` |
| G2 cost + live DP evidence | Cost-model validation and residual rank imbalance under strong DP4+CDB | `SIMULATOR_UNVALIDATED`, `NO_GO_CROSS_GPU_IMBALANCE` |
| G3 depth oracle headroom | Measured-cost oracle shows meaningful **incremental** gain over strongest non-depth rank-placement baseline | `GENERIC_LENGTH_ONLY`, `NO_GO_SCHEDULING` |
| G4 causal deployability | Cheap prompt-only predictor retains enough of oracle gain without KV movement or quality loss | `PREDICTION_BLOCKED` |

Suggested **heuristics**, not established scientific thresholds: <5% incremental latency headroom => NO-GO; 5–10% => CONDITIONAL; >=10–15% in a validated and quality-safe setting => predictor/scheduler follow-up candidate. Require reasonable tails, robustness, actual service constraints and held-out evaluation; distinguish gain at high C from absent gain at C<=4. An unattainable oracle is *not* a deployable speedup and cannot alone establish architecture novelty.

Report overall verdict as one of `GO_DEPTH_AWARE_SCHEDULING`, `CONDITIONAL_HEADROOM`, `MODEL_GATE_UNSUPPORTED`, `QUALITY_FAIL`, `NO_PHYSICAL_DEPTH_SAVINGS`, `NO_GO_HETEROGENEITY`, `NO_GO_CROSS_GPU_IMBALANCE`, `SIMULATOR_UNVALIDATED`, `GENERIC_LENGTH_ONLY`, `NO_GO_SCHEDULING`, `PREDICTION_BLOCKED`, `INCONCLUSIVE`.

## 15. Minimum final presentation tables

### A. Gate and quality

| checkpoint | gate q | KV layout | fixed R4 accuracy | dynamic accuracy | mean actual depth | executed loops saved | label |
|---|---|---|---:|---:|---:|---:|---|
| measured | measured | measured | N/N | N/N | mean ± CI | % | LIVE |

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

1. Ouro project — https://ouro-llm.github.io/ ; paper https://arxiv.org/abs/2510.25741
2. CDB — https://arxiv.org/abs/2608.09444 ; official code https://github.com/kschwethelm/continuous-depth-batching
3. `vllm-rlt` README and checkpoint support table — https://github.com/ThinkFlowLab/vllm-rlt/blob/main/README.md (full-depth prefill; adaptive decode for supported Ouro checkpoints; Thinking compatibility bounds).
4. MoR — https://arxiv.org/abs/2507.10524 ; official authors' code https://github.com/raymin0223/mixture_of_recursions (token-/expert-choice gating; not inherently decode-only).
5. Enabling Dynamic Computation in Looped LMs — https://arxiv.org/abs/2610.09013 (practical skipped work and KV policy pitfalls).
6. Learning Unmasking Policies for Diffusion Language Models — https://proceedings.mlr.press/v306/jazbec26a.html (inspiration only; unmasking policy is not precomputed Ouro decoding depth).

**End of spec.** Do not initiate RL or another full 4GPU engine without demonstrated, quality-preserving, depth-specific headroom.
