# Ouro Dual-Objective Adaptive Chunk–Loop Scheduling: Four-GPU Characterization, Objective Search, and Workload Generalization PoC

**Version:** 1.0 — 2026-10-23  
**Status:** Prospective PoC specification; not a claim of measured improvement  
**Required server destination:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/ouro_dual_objective_cg_slo_workload_poc_spec.md`  
**Repository:** `/home/esjung/MLLM-EP-github`  
**New report directory:** `poc_flashvep/reports/ouro_dual_objective_cg_slo_workload_discovery/`  
**Authorized physical NVIDIA devices:** **4, 5, 6, 7 ONLY**. GPUs 0–3 are prohibited.  
**Serving execution:** Independent **single-GPU** Ouro workers, not a four-GPU TP/PP model.  
**Source model:** `ByteDance/Ouro-1.4B`, Base checkpoint revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`; BF16; fixed/static recurrence depth R=4; original weights.  
**Primary runtime:** pinned standalone vLLM-RLT `4a350344027960a48384626aa646f1765fda6d4f` with previously documented, removable G2/G3 adaptation and existing fixed-policy/complete-output-token-fairness infrastructure. Verify precise local patch hashes before use.  
**Starting evidence:** prior result commit `821b11653695d0a470bd88bfcc997ae1bbcf5bc9` (branch `flashvep/ouro-8gpu-budget-joint-cg-optimization-poc`).  
**Research question:** Does a deliberately two-objective **prefill-efficiency versus decode-interference/SLO-risk** decision rule produce **real, advantageous dynamic C choices beyond tuned adaptive G-only**, under multiple natural traffic regimes?

---

## 0. Executive execution contract

This is a **new, narrowly focused continuation** of the previous `NO_GO_INCREMENTAL` PoC, not a claim that its verdict was mistaken. The previous implementation chose **identical actual chunk/loop execution plans for Joint J1 and tuned G-only G1 on 144/144 paired serving requests**, with J1/G1 throughput ratios of **0.996–1.003×**. That negative result must be reproduced and explained before attempting any new Joint claim. The cost of C and G was nonseparable in a controlled P2048 endpoint test, but nonseparability alone did not imply an online gain.

Three research phases are mandatory:

- **Phase A — Decision-level diagnosis.** Determine whether C variation was blocked by legal frontier, cost/budget feasibility, existing fixed-C tuning, tie-breaking, the U2 utility, or the single-span execution model. Distinguish observed decisions from modeled alternatives.
- **Phase B — Broad, controlled dual-objective study.** Search multiple scientifically motivated objective families, varying both prefill efficiency and decode interference/SLO risk, while keeping a matched G-only control with **exactly the same** objective family, telemetry, compute budget, fairness, cost predictor, and optimizer effort. Characterize driver gradients and show whether C and G actually move.
- **Phase C — Generalization in realistic serving traffic.** Evaluate locked finalists under **Chat, Mixed-length, Long, Bursty** natural-generation workloads and an additional 8K stress lane. Promote to sustainable-rate tests only if there is genuine new executed C selection and a matched, useful performance advantage.

**Never force C to change merely to obtain a positive result.** Artificially changing C can serve as a causality diagnostic, but never as the headline online method. Likewise, do not use accuracy differences, differing output lengths, arbitrary fallback penalties, retrospective oracle knowledge, future arrivals, or future GPU timestamps to manufacture a Joint advantage. Preserve Full R4 and correctness throughout.

**Do not restart the suspended custom native-vLLM Ouro Loop PP program.** Do not build a new TP/PP/EP/disaggregated runtime, retrain the model, alter precision, alter early-exit depth, or change the checkpoint. Mixed+Joint inside a heterogeneous-loop batch is outside the required scope; optional future work only with independent evidence.

The task is **complete as an honest PoC** even if the final result remains negative, provided all mandatory diagnostic analyses and feasible tests are performed. A new serious Full-R4, KV, CUDA-event, or state-lifecycle error quarantines affected mechanisms pending investigation; an inaccurate SLO-risk estimator limits SLO claims but does **not automatically prohibit explicitly labeled exploratory policy comparisons**.

## 1. Canonical evidence and literature references

### 1.1 Prior internal evidence: do not overwrite or relabel

Read the following at the pinned commit before editing source code:

- `poc_flashvep/reports/ouro_8gpu_compute_budget_joint_cg_discovery/analysis.md`
- `online_policy.md`, `online_policy_contract.md`, `optimization.md`, `phase_gates.md`, `cost_interaction.md`
- `budget_action_trace.csv.gz`, `G1_J1_actual_plan_equivalence.csv`, `G1_J1_CPU_overhead.csv`
- `opportunity_frequency.csv`, `budget_overruns.csv`, `budget_stall_relationship.md`
- `fairness.md`, `predictor.md`, `quality.md`, `quality_baseline_correct_to_wrong.csv`
- `main_table.csv`, `main_paired_pooled_CI.csv`, `request_events.csv.gz`, `source_family_manifest.json`
- `registration.json`, `registration.md`, `runtime_patch.diff`, `reproduce.md`, `gpu_contention.md`

For preceding data, read the pinned `d6ba31c7275019867e2aca473818d42d9b1fcbb0` report `poc_flashvep/reports/ouro_8gpu_deadline_aware_joint_chunk_loop_discovery/`, particularly its raw quantum cost traces, stage-specific prediction evidence, and `joint_headroom.md`. Do not rebrand its modeled partial-token-loop envelope as a measured speedup.

Prior observations to preserve as **historical evidence**, not new measurements:

1. Original staged/G2 work supports loop-group scheduling but does not create a new model-parallel primitive.
2. Previous cost interaction at controlled P2048: categorical additive MAE **4.937 ms** versus interaction model **1.594 ms**; interaction contrast **31.725 ms** with process-block interval **[28.319, 38.280] ms**. P8192 had much noisier estimates.
3. Previous G1 (adaptive G-only; **not** fixed group size G=1) and J1 (adaptive Joint) produced identical chunk/group plans for **144/144** request pairs in quiet confirmation. The output counts were identical (16,623 per arm), with throughput ratios **0.996–1.003×**.
4. Previous U2 objective was based on eventual completed-prefill token efficiency over estimated remaining Full-R4 service; it did **not** directly charge observed decode ITL/SLO exposure and initially used zero additional decode-yield penalty. Its fixed budget constrained individual prefill units, not complete-token ITL.
5. Previous controller permitted one prompt span per adaptive batch, new-chunk C selection only, legal frontier G changes, and a 250-ms starvation/progress escape. It compared fixed tuned C, staged G4, adaptive C-only, adaptive G-only, adaptive Joint, and stock/fair RLT Mixed.
6. Full-answer quality was **configuration-sensitive**: stock reference GSM8K 47/64, staged adaptive C-only 43/64, G1/J1 47/64; pilot MATH extracted answer G1/J1 32/50, complete-correct 30/50, six capped. G1/J1 identical correctness labels on 114 pilot questions; do **not** claim universal quality equivalence.
7. GPU shared-host interference was nonzero in past eight-device runs. Recheck physical 4–7 for this campaign; do not assume fewer workers make all conditions neutral.
8. Prior sustainable-rate serving-capacity sweep was **NOT_RUN**. Prior finite drained-cohort tokens/s must not be called sustainable SLO goodput.

### 1.2 Public comparison anchors

- SLOWeave: *Deadline-Aware Adaptive Prefill Chunking for Efficient Large Language Model Serving* (September 2026), https://arxiv.org/abs/2609.07883 . Its deadline-constrained dynamic C is a strong conceptual C-only reference, **not** an official Ouro/RLT implementation. It evaluates chat, mixed-context, long-context, and bursty regimes.
- P-PAS: *Prefill-Pressure Adaptive Scheduling for Long-Context LLM Serving* (2026), https://arxiv.org/abs/2608.15171 . A strong pressure-adaptive token-budget concept; any local interpretation must be labeled **P-PAS-inspired**, not official reproduction.
- vLLM-RLT repository and its pinned scheduler source: https://github.com/InferMatrixLab/vllm-rlt . Native Split and Mixed are strong loop-aware controls, **not** inventions of this project.
- Sarathi-Serve, Layered Prefill, and standard SLO-goodput definitions as contextual references. Preserve differences among TTFT, pooled-token P99 TBT, **request-level every-gap TBT**, and complete-response joint SLO attainment.

Novelty is an *open hypothesis*. A weighted sum, a multiobjective formulation, or a binary selection variable is not novel by itself. An actionable contribution would be a reproducible, loop-specific difference between efficiency-favoring C and stall-favoring G that improves **quality-matched** serving beyond tuned single-axis and Mixed controls.

## 2. Definitions, invariants, and legality

- **C:** configured number of new prompt tokens in one chunk. Actual span may be clipped by prompt remainder, admission token budget, or KV capacity. Record **configured C**, **actual span size**, and **token budget** separately.
- **G:** consecutive recurrent iterations executed before a legal yield; Ouro static R4 means total depth exactly 4. Valid initial group sizes in `{1,2,3,4}`, clipped to the chunk's remaining loop frontier.
- **B:** optional predicted **single prefill service-quantum** budget in milliseconds; never equate B with a full-output-token TBT bound or a hard preemption deadline.
- **S:** causally observable scheduler state at a decision: ready/waiting decode requests, elapsed time since last **delivered** output, currently issued loop frontiers, verified completed-event readiness, residual prompt, context lengths, actual proposed batch packing, graph readiness, staged allocation, memory pressure, queue age, and past latency feedback. Log which fields were available and the exact moment they became available.
- **Atomic vs staged:** Native Atomic G4 is not the same physical path as staged G4. Compare staged G4 whenever isolating the G contribution. Mixed can pack prefill/decode rows; it is not a pure G1 variant.
- **Prefill completed progress:** A prompt token receives completion credit **only after all four recurrent iterations and the necessary prefix publication have occurred**, not after partial work R1/R2. Partial work is a separate proxy labeled `incomplete_token_loop_work` and never a headline speedup.
- **Decode completed token:** A recurrent batch execution is not a completed output token. Completion requires all remaining recurrent work, coda/logits/sampling, and the captured output-delivery event in the declared clock domain.
- **C changes:** only when opening a fresh chunk; never retroactively resize a staged active chunk. **G changes:** only at a legal chunk recurrent frontier, respecting remaining R4 work, event dependencies, and current chunk ownership. A prelude is separately recorded as executed **G=0**, with a distinct planned next group.
- Preserve original weights, BF16 operations, non-speculative greedy sampling and official tokenizer/stopping, original KV namespaces, Full-R4 execution, and supported CUDA stream/graph lifetimes. No full-answer quality claim based solely on kernel probes.

### 2.1 Fairness and packing controls

Every prospective C-only/G-only/Joint matched comparison uses the **same completed-output-token-aware fairness** with its effects measured in a separate stock-versus-fair ablation. Native Mixed remains available both stock and fairness-matched where the patch supports it. Do not attribute changes from fairness to Joint selection.

Retain the prior one-prefill-span packing rule for the **primary causal optimization experiment** so all adaptive policies use identical packing. Report separately that native Mixed has a different co-batching capability. An optional matched packing expansion requires its own evidence lane and cannot silently alter the primary results.

## 3. Hypotheses and falsifiable outcomes

**H1 — Objective response:** An efficiency-oriented score favors certain C choices under matched context, while a decode-interference-oriented score favors certain G choices. The effects may be cross-coupled and need empirical **sensitivity/gradient** evidence; not an assumed factorization.

**H2 — Objective-induced action diversity:** Compared with U2, one or more preregistered dual-objective policies select **different legal C values from tuned G-only** on a meaningful fraction of causally ready states, rather than merely outputting a different nominal configuration with the same actual clipped token span.

**H3 — Incremental advantage:** The distinct C choices generate a measured improvement in the TTFT–ITL Pareto frontier, request-level SLO attainment, or SLO goodput **at equal correctness and under the same monitoring/compute budget**. Improvements due only to bad fallback behavior in C-only do not count.

**H4 — Workload dependence:** The best trade-off changes with Chat/Mixed/Long/Bursty distributions and offered load. This is a potential source of real dynamic C behavior, not a guaranteed observation.

**H5 — Robustness:** The selected policy is not an artifact of post-hoc tuned objective weights, leakage from test traces, GPU/process effects, changed output lengths, or observer-induced batch geometry.

**Null hypotheses:** Optimal fixed C plus dynamic G covers the observed policy surface; SLO risk is predominantly controlled by G or by Mixed co-batching, leaving little incremental value to dynamic C. It is scientifically acceptable to retain `NO_GO_INCREMENTAL` if those hypotheses remain consistent with results.

## 4. Dual-objective mathematics and what drives each axis

### 4.1 Do not impose artificial separability

The user hypothesis suggests an **efficiency term that tends to promote C** and a **decode-risk term that tends to promote G**. Implement that as a testable design bias, **not** as the identity `E=E(C)` and `R=R(G)`. Actual full-chunk service, staging, attention, and arrival-dependent interference are C/G-coupled.

Let a legal action `a=(request_id, actual_C, G, phase)` be selected from `A(S)` and its predictor be frozen prior to evaluation.

Let `T_first(a|S)` be the predicted *immediate* service impact of one schedulable unit, with path-specific staged prelude accounted for.

Let `T_full(a|S)` be the predicted remaining **Full-R4 completion service** of the *same chunk*, including the prelude and every subsequent loop group necessary to finish its token span. It is a service-cost approximation, **not** a client TTFT simulation. Do not double-count stream overlap or future stock-policy waiting as if it were an isolated service cost.

Let `P_complete(a)` be actual chunk-completion credit at the current step (normally **zero** for a staged prelude and partial loop group) and `P_eventual(a)` be intended full-chunk token amount, whose denominator must include remaining service debt. Score immediate and eventual progress distinctly.

One grounded efficiency feature is:

`E(C,G,S) = eventual_completed_prompt_tokens / max(epsilon, estimated_full_R4_chunk_service_ms)`

Additional efficiency features may include batch GEMM efficiency proxies based on **measured** service per Full-R4 token, TTFT age/remaining-prompt urgency, number of future launch/yield operations, and bounded memory occupancy. Do not infer measured HBM bandwidth from parameter sizes.

A decode-interference risk feature can use *only past and current observables*:

`R(C,G,S) = r1 * normalized_predicted_nonpreemptible_exposure_ms + r2 * measured_or_calibrated_P(ITL_gap > threshold | state_bin, action_family) + r3 * predicted_repeated_yield_exposure + r4 * queue_age/TTFT_penalty`

This is an **empirical risk score**, not a proven future ITL value. Labels for conditional tail risk must be trained on eligible independent historical observations; never label the chosen alternative with the actual stock-policy outcome as if it were a counterfactual. Use separate predictor uncertainty and support flags; OOD action/state bins cannot receive fake confidence.

Normalize efficiency and risk features using frozen TRAIN statistics (robust quantiles or a documented unit conversion). Avoid misleading raw addition of `tokens/ms` to `milliseconds` or `probabilities`. Freeze clipping, epsilon, and aggregation rules before any held-out comparisons.

### 4.2 Axis-driver diagnostic — mandatory

For each representative **legal fresh-chunk state** where C and G can both vary, estimate:

- `Δ_C E = E(C_high,G,S) - E(C_low,G,S)` and `Δ_C R` at fixed G.
- `Δ_G E = E(C,G_high,S) - E(C,G_low,S)` and `Δ_G R` at fixed C.
- Interaction difference-in-differences in E, R, and total utility, including confidence/source support.
- Preferred action under E alone, R alone, and combined E/R; how often selected **actual** C and G differ.
- At continuation states (C already frozen), G-only legal choice, not fictional dynamic resizing.

Produce sensitivity maps indexed by Prompt Length, active decode count, arrival pressure, age, budget, stage readiness, and graph path. Report both supportive and contradictory cells. If prefill efficiency changes mainly with G and risk mainly with C in a state, **state that** instead of imposing the desired direction. Do not claim objective-gradient directions from a single optimizer output.

### 4.3 Primary utility family

For normalized components `E_n`, `R_n`, a transparent candidate is:

`U_dual(a,S) = w_E * E_n(a,S) - w_R(S) * R_n(a,S) - w_M * memory_cost_n(a,S) - w_H * switching_cost_n(a,S)`

subject to legal state transitions, compute-budget risk labels, and full-R4 completion debt. `w_R(S)` may depend on **observed** decode pressure and past ITL, but not on future timestamps. The utility must never reward uncompleted R1/R2 rows as completed prompt tokens. Explicitly compare with `w_R=0`, `w_E=0`, and shared-weight G-only/C-only controls.

A "two objective" design can also be evaluated as an **actual Pareto search** without collapsing objectives prematurely, or as an epsilon-constrained problem: maximize prefill efficiency subject to a supported risk limit. A 0–1 integer encoding is optional; selection from 32–70 legal actions can normally use direct enumeration. Integer programming is not an independent contribution unless a justified multi-step optimization genuinely needs it.

## 5. Phase A — Explain 144/144 plan equivalence before new policies

### A0. Ingest immutable prior records

- Validate hashes, schema, timestamps, source and GPU/process IDs of the pinned parent experiment.
- Parse `G1_J1_actual_plan_equivalence.csv`, `budget_action_trace.csv.gz`, `opportunity_frequency.csv`, `G1_J1_CPU_overhead.csv`, `pilot_training_selection.json`, per-request timing and `main_table.csv`.
- Derive fresh-chunk decisions separately from continuation decisions; only the former can change C.
- Reproduce the reported 144/144 request-plan equality by comparing actual `(request, span_start, actual_span_length, starting_loop, group_count, path)` sequences, not only policy labels or nominal C.
- Confirm why the candidate C set becomes small or collapses under actual clipping and RLT token-budget packing.

### A1. Per-decision opportunity funnel

For each scheduler decision with an active prompt, count and record the following stages, with denominators:

1. Decider was actually invoked; ready/eligible request count.
2. Fresh-chunk opportunity versus continuation-only.
3. Distinct legal **actual token spans** across nominal C choices, after clipping.
4. Distinct budget-feasible C choices under frozen predictor (and overbudget fallback cases).
5. Distinct utility ranks and top-two score margins.
6. The selected legal actual C/G differs from tuned G-only.
7. Predicted benefit exceeds predictor error, marginal CPU overhead and completion debt.
8. Independent live confirmation of beneficial difference (none may exist yet).

Classify no-change causes: `no_fresh_boundary`, `same_clipped_span`, `token_budget_limit`, `single_span_pack_limit`, `cost_infeasible`, `fixed_C_already_optimal`, `utility_prefers_same_C`, `tie_or_hysteresis`, `staging_or_memory_limit`, `fallback`, `unknown`. Allow multiple labels, report mutually exclusive primary reason separately. Do not incorrectly call a state with only one legal C a failed optimizer.

### A2. Counterfactual-score inspection without fake counterfactual outcomes

Replay only the **selection function** on frozen historical snapshots to compare E-only, risk-only and coupled score ranks; this provides action-diversity diagnostics but **not actual alternative-service latency**. For a small audited set of reproducible fresh-chunk source states, arrange controlled paired real-runtime interventions that create comparable initial states and force a legal alternative C for **diagnostic causality only**. Explicitly label failed state matching, changed graph packing and sources that diverge. No forced action may appear as a final "our adaptive method" win.

**Phase A output:** `phase_a_action_equivalence.md`, `decision_opportunity_funnel.csv`, `eligibility_collapse.csv`, `candidate_score_audit.parquet/csv`, `state_support.md`, `confirmed_alternative_states.md`.

**Phase A decision:** If no legal distinct C is available in realistic workloads, investigate the RLT span model and assess `NO_REACHABLE_C_HEADROOM`; do not spend GPU hours trying objective weights that cannot affect the actual schedule. Still complete offline objective sensitivity tests in supported states and honestly report the limitation.

## 6. Phase B — Broad objective-family ablation

**Purpose:** Find whether C-preferring prefill efficiency and G-preferring stall protection can jointly move *actual* actions, and whether those choices offer measurable benefit over equally informed single-axis controls.

### B0. Data splits and historical feature handling

Use existing cost records for an initial audit, but do not fit a new risk model on previously published **HELDOUT** errors and then reuse those errors for blind evaluation. Freeze new disjoint TRAIN/VALIDATION/CALIBRATION/FINAL families, process starts, arrival seeds, GPU rotation, and benchmark arms. A historical dataset can be exploratory/dev only if its original test role has already been consumed. Avoid deriving success thresholds from FINAL. Keep source-family grouping across multiple context variants. Version every objective family and optimizer implementation by source hash.

Calibrate event readiness and service labels by path (`atomic_core`, `staged_prelude`, `staged_recurrent`, `mixed`, `decode_core`, `coda`). An interval with queued waiting or overlapping GPU operations cannot simply be added to a service DAG. For the risk estimator, keep actual observed complete-token TBT, percentile/tail-CCDF counts, prefill exposure union, and request-level max-ITL distinct. Prefer empirical conditional risk bins or lightweight isotonic/quantile models with uncertainty and minimum support; unresolved strata should fallback to known controls.

### B1. Prespecified objective variants — evaluate ALL for action behavior and offline score geometry

| ID | Objective family | Mechanism to test | What it does NOT establish |
|---|---|---|---|
| O0 | Original U2 | Eventual full-R4 completion efficiency only | Existing incremental benefit was absent |
| O1 | Separable two-driver diagnostic | `E_C(C)` minus `λ R_G(G,S)` | Not a physically correct universal cost model |
| O2 | Fully coupled normalized weighted sum | `E(C,G,S) − λ R(C,G,S)` | May prefer the same fixed C |
| O3 | Efficiency with decode-risk epsilon cap | maximize E subject to `R≤τ` | Risk cap is estimated, not guaranteed SLO |
| O4 | Lexicographic decode safety then efficiency | minimum supported risk class; maximize E inside class | May harm TTFT if too conservative |
| O5 | Minimum efficiency then minimize risk | require `E≥η_min`, minimize R | Workload-sensitive thresholds |
| O6 | TTFT-pressure-aware dual weights | E weighted by prompt age/remaining work; risk weighted by live decode pressure | Historical response age is not future TTFT |
| O7 | Quantile/CVaR-style decode risk | penalize calibrated conditional upper-tail cost rather than mean risk | Tail estimate may be sparse/noisy |
| O8 | Pareto-archive selection | non-dominated E/R actions with deterministic state-based tie rule | Larger Pareto set alone is not gain |
| O9 | Throughput–latency product or geometric mean | normalized `E^a × (1−R)^b` with safe bounds | Dimensional consistency and calibration needed |
| O10 | Yield-aware full-chunk objective | E discounted by anticipated extra legal submissions, staging, state retention | Future co-batching remains unknown |
| O11 | Cost-model uncertainty penalty | coupled E/R minus confidence width or conservative unsupported-state penalty | Not deadline guarantee |
| O12 | Hysteresis/switching control | O2/O6 plus switching and staged continuation penalty | Less switching is not automatically better |
| O13 | Feedback dual-control variant | previous measured TBT/TTFT violation rate adjusts `λ` prospectively | Requires sufficient past samples and avoids future leakage |

O1 is the exact **"C tends to move for E, G tends to move for risk"** hypothesis in clean diagnostic form. O2 is the more realistic coupled counterpart. O3/O4/O5 test multiobjective **constraint** structures rather than arbitrary scalar weights. O6/O7/O13 specifically challenge the previous U2's absence of SLO/ITL pressure. O10 addresses deferred Full-R4 work and scheduler yielding.

For O0–O13, perform a full **offline legality/action-geometry screen**, not necessarily 14 expensive simultaneous online implementations. The online shortlist will be selected using TRAIN/VALIDATION and independent action-diversity and predicted-regret criteria, with a maximum of **three or four** objective families plus mandatory controls to avoid arbitrary multiple-comparison selection. Do not drop all families that fail to show C movement from the offline tables: their failure is part of the study. If promising objective functions require incompatible feature support, mark `UNSUPPORTED` instead of inventing values.

### B2. Weight and cap exploration (training-only)

- Fixed normalized `λ` ratios (example grid `{0,0.1,0.25,0.5,1,2,4,8}`) after verifying normalization; additional sensitivity grid allowed within TRAIN. Include both E-only and risk-only endpoints.
- Risk caps `τ` chosen prospectively from TRAIN risk percentiles, not to post-hoc favor a new policy.
- Prompt-age/remaining-work pressure weights chosen on TRAIN only; use simple monotone functions before training a complex selector.
- Decode pressure bins: none, sparse, moderate, dense; actual decode lifecycle stage, last-delivery age and backlog. Track cases where a budget decision is infeasible.
- Input/context strata including 256, 512, 1.8K, 2K, 4K, 8K and concurrency 1/4/8/16 when admission allows. Record actual context/packing rather than equating prompt length with KV attention length.
- Budget sensitivity around `{20,40,80,120,240}` ms, plus discovered feasible boundary values; include unconstrained native controls. A budget is not a complete-token SLO.
- Do not combine an unstable feedback controller and a new C/G objective in the same first live ablation. Freeze budget first, add feedback only after isolating fixed-budget incremental effects.

### B3. Essential factorial and ablation matrix

For each shortlisted objective `Oj`, run at **identical input states, telemetry and compute allowance**:

1. `E-only` Joint;
2. `Risk-only` Joint;
3. `Dual` Joint with locked weights;
4. `Oj` Adaptive C-only, fixed legal G4 staged and Atomic controls;
5. `Oj` Adaptive G-only with **identically tuned fixed C**;
6. `Oj` Best fixed (C,G) selected on TRAIN;
7. `Oj` Joint with no C change (a hard diagnostic identity check to the corresponding G-only);
8. Tuned stock and completed-token-fair RLT Mixed as serving controls;
9. SLOWeave-style adaptive C-only and P-PAS-style pressure policy when feasible in this RLT engine, clearly labeled **inspired adaptations**, not official paper code/performance.

Hold Full-R4 work, GPU kernels, sampling, graph settings, cache/reservation rules, fairness and prompt-span packing constant for matched primary contrasts. Explicitly record any control for which matched packing is impossible. Check the `same policy options except C dimension` invariant; if fails, repair comparability before claiming incremental C value.

### B4. Outputs and promotion thresholds

Mandatory **decision-science** metrics:

- Fresh-chunk decision fraction, multiple-legal-C fraction, distinct actual chosen-C rate, C-change rate relative to tuned G-only, distinct-G rate, state-to-action mutual information (descriptive), C/G conditional distributions, normalized score sensitivity, and legality-constrained Pareto width.
- `C chosen by efficiency term alone` versus `C chosen by dual` and `G chosen by risk alone` versus `G chosen by dual` agreement; raw cross-effects `Δ_C R`, `Δ_G E` to test whether the intended decomposition actually holds.
- Fraction of C changes that are **beneficial when confirmed by independent paired live interventions**, not merely model-estimated. Report negative alternative outcomes.
- Full-prompt completion time, prefill staged-live interval, total quantum count, accumulated service and observed scheduling overhead; never reward partial loop work as completed prompt progress.
- ITL and request-level SLO effects separately; P50/P95/P99 and max per request, pooled vs per-cohort labels, all-offered denominator, quality and token-match diagnostics.

**Pre-registered screening suggestion, not a fact:** Require a nontrivial legal-C opportunity rate (e.g., ≥5% among *fresh-chunk* opportunities), a reproducible actual-C plan difference in at least two independently held-out traffic strata, and a useful confidence-aware performance improvement without quality regression to justify the full Phase C capacity matrix. Thresholds must be frozen **before** reading final online outcomes, and `5%` action difference is not equivalent to `5%` speedup. If these fail, still produce completed Phase B characterization and Phase C small balanced characterization where safe, but avoid an unjustified large capacity run.

## 7. Phase C — SLOWeave-inspired workload families and serving generalization

### C0. Workload definitions (inspired distributions, not a reproduction claim)

| Workload | Prompt mix / target median | Requested output demand | Design reason |
|---|---|---|---|
| CHAT | chat-style, prompt median ~256 tokens | ~96-token target distribution | low prefill pressure, many decode tokens |
| MIXED | 70% chat-like (~256) + 30% long (~1,800 prompt median) | ~100-token target distribution | short/long coexistence tests C adaptation |
| LONG | long input median ~1,800 | ~180-token target distribution | sustained prefill and decode interference |
| BURSTY | the MIXED length mix, two-state low/high arrival process | natural outputs per same workload | rapidly changing scheduling pressure |
| LONG8K_STRESS | genuinely long natural context median ~8,000, with matched shorter active requests | documented natural output demand | tests prior P8177/N8 hypothesis and extrapolation |

These are **workload specification targets** based on SLOWeave's stated Chat, Mixed, Long, and Bursty intent, not a guarantee of actual generated lengths. Preserve real prompt-token counts and **actual natural EOS/stop lengths** per policy. Do not teacher-force, fake a 96-token completion by stopping all outputs exactly at 96, or pad meaningless prompt tokens in headline quality/serving lanes. A separate bounded equal-work diagnostic lane is permitted and clearly labeled.

Construct public, documented, non-private source-family manifests. Avoid excessive duplication of text snippets and split variants of the same article between TRAIN and FINAL. Record tokenizer revision, prompt templates, source hashes, length histograms (p10/p50/p90), generated output lengths (p10/p50/p90), EOS/cap fractions and any departures from target medians. Make benchmark requests reproducible with deterministic arrival seeds and model setup. If named workloads cannot be constructed with suitable sources, report the substitution explicitly.

**BURSTY:** Use a preregistered two-state modulated Poisson or alternating low/high arrival process. Record state switching/dwell distribution, high-to-low rate ratio, realized offered requests and per-state pressure. Do not assert parameters are exact copies of SLOWeave unless verified from its paper.

### C1. Arrival-rate and concurrency design

- Begin with a **per-GPU** decode-only reference and the stable request-rate envelope measured on the same runtime/hardware.
- Use paired Poisson arrivals, burst arrivals, and closed-loop C4/C8/C16 where memory permits. Include a low-load control and rates approaching saturation; adjust absolute req/s to actual model throughput rather than transplanting foreign-paper 10/25ms SLO values.
- Freeze **model-calibrated** TBT tiers (example multipliers 5×/10×/25× decode-only reference) and TTFT limits (2/5/10 seconds or workload-calibrated frozen alternatives) **before final**. Also report sensitivity across absolute TBT thresholds.
- Preserve a scenario with concurrent short decoded requests and a newly arriving long prefill request. Record new-chunk opportunity timing; heavy prefill may not exist in all-chat workload.
- Begin with balanced candidate screening on new VAL families; lock objective family, λ/caps, throughput saturation bracket, arrival seeds and strict/moderate/relaxed SLO definitions before FINAL.

### C2. Main controls and exact interpretation

Minimum primary arms: `stock Atomic`, `staged G4`, `best fixed C-only`, `adapt C-only using same Oj`, `adapt G-only using same Oj`, `adapt Joint using Oj`, `stock RLT Mixed`, `completion-fair RLT Mixed`. Include `SLOWeave-style dynamic C` on the same runtime when the adaptation can be implemented fairly; report if it is an incomplete proxy, not official reproduced results. `P-PAS-style` is optional but recommended for pressure-control comparison. All arms should use the same input family and realized arrival requests under paired seed/start, and, where token sequences differ, both same-output diagnostic and natural complete-response results should be shown.

At low-load CHAT a lack of Joint advantage is a valid result. At MIXED or BURSTY an apparent boost that disappears when compared to **tuned G-only** or **native Mixed** is not an incremental contribution. Per-model transfer/generalization to Ouro-2.6B or Huginn is optional and not required; do not extrapolate 1.4B findings to them.

### C3. Primary serving metrics and qualification

For every arrival rate/policy/workload report:

- TTFT P50/P95/P99 from **scheduled arrival to first delivered token**; E2E P50/P95/P99 to true stop, not 128-token truncation.
- Observed output TBT (ITL) P50/P95/P99; background versus all-request labels; full TBT CCDF at 50/100/150/200/250/500ms; request-level maximum TBT and every-gap pass.
- Request-level TTFT attainment, every-gap TBT attainment, **joint attainment with actual complete EOS/chat stop**; separate pooled-token P99-TBT test.
- SLO goodput in requests/s (both completed-in-arrival-window and declared-drain definitions); separate drained requests/s, generated tok/s, full-prompt completion throughput.
- Queue backlog trajectory, slope, arrival offered/realized counts, unfinished-at-window-end, caps, timeouts/errors; do not call an unstable queue sustainable.
- Actual per-request C/G actions, distinct-C decisions beyond G-only, group distributions, budget overruns, optimizer host cost, Stage/Graph memory, and percentage of actions falling back or using starvation escape.
- Correctness: Full-R4 accounting and KV/hidden/logit isolated probes; new dynamic boundary-cycle tests; natural greedy output comparison; GSM8K and MATH-500 frozen pilot first, then larger quality samples if required for publication-quality conclusions.

**Qualified SLO capacity:** preregister requirement **at least 1,000 offered requests per arm/rate** and **at least three independent model-process starts**, statistically meaningful 90%-attainment threshold (report uncertainty and 80%/95% sensitivity), stable queue/backlog and complete natural outputs. Bracket the highest sustainable supported rate rather than claiming one finite cohort's apparent success is capacity. Use blocked/run-state labels separately when no arm qualifies. Carry forward loss of output-token equivalence and quality differences as confounds, not ignored noise.

### C4. Statistical methodology

- Same-GPU process blocks for primary paired comparisons; rotate policy order and GPU assignment across physical 4/5/6/7.
- Keep new **source families** and arrival seeds disjoint from weight calibration and candidate selection. For workload lengths from a single source, keep all variants inside one partition.
- Bootstrap/cluster by process start, source family, and arrival cohort as appropriate; token gaps from one request are **not independent samples**. Report 95% paired intervals, interval method and n. Avoid claiming statistical certainty from 3 starts alone.
- Control multiplicity from the 14-family objective search: do not select the best by FINAL metrics. Use separate VALIDATION shortlist and pre-registered final primary contrast and threshold; clearly mark secondary exploratory results.
- Use predeclared adverse-quality threshold and question-level paired comparisons. Benchmark accuracy changes in prior C-only variants show quality cannot be waved away as benign floating-point rounding.

## 8. Hardware, concurrency, and deterministic orchestration

Only physical GPU **4, 5, 6, 7** may be used. Validate UUIDs and `CUDA_VISIBLE_DEVICES` mapping, including logical `cuda:0` remapping, on each launched process; reject any inadvertent physical 0–3 usage. Never kill another user's process. Stop only scope-owned burners or benchmark workers after PID/owner/manifest verification. Document processes already present and skip occupied GPU without forcible takeover. Confirm how/if project-owned burners should be restored after the study; restore only if previously owned and safe.

Use **four independent single-GPU workers** (one inference model/process per physical GPU) for characterization and objective dry-runs when contention is tolerable. Maintain isolated ports, caches, CUDA contexts, working dirs, GPU per-process env, logs, raw files, seeded arrivals, warm-up/JIT/graph metadata and run IDs. Build a resume-safe dispatcher: file/process lock, append-only run manifest, atomic completed outputs, checksums, job deduplication, ownership-aware retry and failure diagnostics. Idle GPU receives another independent job; paired policy block remains on one physical GPU.

Before **clean performance** comparisons, calibrate 1- vs 2- vs 4-worker interference on 2K/N4 and 8K/N8. Monitor GPU clocks/power/temp/utilization, CPU and RAM contention, disk I/O and scheduler load. If drift exceeds preregistered 3% review threshold or alters key conclusions, reduce simultaneous measurement workers (even to one for the decisive confirmation), while continuing offline analysis/correctness on other GPUs where safe. A clean lane must not run concurrent heavy Nsight on the shared host. Profiling data never replace clean wall-time serving results.

Set a bounded program with purposeful work rather than infinite sweeps: A analysis first, B offline broad + limited live finalists, C workload screening then only justified qualified-rate matrix. Long runs are permitted if authorized by the user, but no new run silently waits for future user feedback. If a job fails, record its artifact and continue independent safe tasks. Do not transform the work into a new serving engine.

## 9. Implementation touch points and staged execution

Read source before applying patches and identify the exact runtime checkout; do not assume latest upstream HEAD preserves the pinned behavior. Current RLT code includes `SchedulingPolicy`, `RefillPolicy`, `SplitPrefillPolicy`, `MixedPolicy`, `Scheduler._make_scheduled_request`, `Scheduler._apply_split_prefill_progress`, and `build_execution_batch` as relevant mechanisms. The previous PoC already contains legal action enumeration, budget selector, staged RLT adaptation, fairness patch, model data and event observers. **Reuse these** instead of rebuilding a mini-vLLM.

Recommended implementation modules (adapt to repository conventions):

1. `objective_features.py`: causal E/R/TTFT/interference features, calibration and support flags.
2. `objective_families.py`: pure, deterministic score functions O0–O13, normalized weights, state-dependent terms and unit tests.
3. `action_diagnostics.py`: decision opportunity funnel, legal alternatives, cause of identical C, score cross-effects.
4. `objective_search.py`: TRAIN/VALIDATION shortlist and frozen hyperparameters; no future leakage.
5. `adaptive_selector.py`: minimal switchable `policy_id` wiring into **existing** legal scheduler; avoid changing runner or kernels.
6. `workload_generator.py`: registered Chat/Mixed/Long/Bursty manifests, arrivals, source-family and frozen partitioning.
7. `serving_eval.py`: admission/arrival clocks, output timestamps, request-level SLO, queue stability and paired process statistics.
8. `plot_results.py`: paper-quality tables, plots, C/G action heatmaps and mechanism timeline.

A speculative offline action can be scored without issuing GPU work, but a claimed gain requires **actual** corresponding policy execution. Preserve staging and CUDA event semantics. Instrument policy overhead and sampling effects separately. Unit and live checks must cover: G1→G3, G1→G2→G1, G3→G1, C change only between complete chunks, actual span clipping, prelude G=0, no grouped decode leak, pending GPU event readiness, starvation escape, and interrupted/restarted requests. Software tests alone are not pretrained numerical validation.

## 10. Registration, search/promotion and truthfulness gates

### 10.1 Gates (no premature blanket BLOCKED)

- **A-GATE / mechanism:** `PASS` if the 144/144 identity is reproduced and the dominant causes of legal-C nonselection quantified with coverage; `CONDITIONAL` if logs are insufficient and missing columns documented. Diagnostic A should complete even if B later proves negative.
- **B-GATE / objective diversity:** `PASS` if multiple scientifically grounded utility families are evaluated offline, gradients/cross-effects reported, and locked finalists yield a genuine, reproducible **different actual C plan** without violating legal Full-R4 constraints; `NO_GO_INCREMENTAL` if they do not. Do not require SLO prediction upper-bound coverage as a prerequisite to exploratory tests since this controller makes no strict deadline guarantee.
- **B-GATE / performance:** independent on-GPU matched performance must show a useful incremental effect of C movement beyond equally informed G-only and C-only. A tiny difference in predictor score or predicted partial work does not qualify. Always compare to Mixed; effects must not rely on weak fallback.
- **C-GATE / applicability:** completed representative Chat, Mixed, Long, Bursty screening is valuable even if negative (within available resources). If no new C choice or performance headroom is seen, do not burn time on a massive capacity sweep; label qualified capacity `NOT_RUN` with explicit cause.
- **C-GATE / publication-quality:** quality-matched SLO goodput/attainment and reproducible Pareto improvement across independent held-out families, stable queue, correct denominators, adequate sample sizes. A finite-cohort gain cannot be upgraded to sustainable capacity.

A measured serving failure and a correctness failure are distinct. Correctness/fidelity violations may quarantine affected arms; poor risk calibration makes hard SLO-guarantee claims invalid; lack of incremental C value yields a real scientific NO-GO for this formulation, not an implementation BLOCKED. Where data or runtime support is unavailable, use `BLOCKED` precisely.

### 10.2 Final decision matrix

Record independently:

- Whether E primarily **steers C** and R primarily **steers G**, with measured cross-effects and state-conditioned exceptions.
- Whether the dual-objective changes C at legal boundaries and how often.
- Whether alternative C changes actually improve QoS after full completion debt and selector overhead.
- Whether Joint beats matched G-only, matched C-only, tuned fixed policies, and RLT Mixed on measured quality-matched workload slices.
- Whether measured improvement survives different λ/caps, load/length distributions, GPU rotations, and clean-room confirmation.
- Whether general policy is more efficient than the previous U2 or merely a workload-specific tuned rule.

## 11. Required artifacts and naming

Write new work under:
`/home/esjung/MLLM-EP-github/poc_flashvep/reports/ouro_dual_objective_cg_slo_workload_discovery/`

Mandatory files:

- `analysis.md`: executive verdict, settings, actual vs modeled evidence, caveats, next decision.
- `phase_a_action_equivalence.md`: reason for G-only/Joint 144/144 equality, with exhaustive denominators.
- `decision_opportunity_funnel.csv`: legal frontiers, actual distinct C and why no change.
- `objective_catalog.md`: O0–O13 definitions, units, normalized features, weights, hard legality.
- `objective_grid.csv` and `objective_shortlist.json`: every attempted variation, failure, score, selected finalists and frozen rationale.
- `objective_gradients.csv`: delta C/G E/risk response surfaces and interactions.
- `objective_action_transitions.csv`: state → nominal candidates → actual legal dispatched spans, phase/frontier/G, metadata.
- `phase_b_objective_ablation.md`: efficiency-only/risk-only/dual and single-axis comparisons.
- `predictor_scope.md`: path cost, risk predictor/calibration, OOD state support, blind error.
- `phase_c_workload_contract.md`: dataset construction, actual length distributions, arrivals, SLO choices and starts.
- `workload_manifest.csv`, `source_family_splits.json`, `arrival_manifest.csv` (or compressed equivalent).
- `serving_results.csv`: TTFT/TBT/E2E, request-level SLO, pooled tails, offered/completed/caps, throughput, queue stability, action diversity.
- `paired_comparisons.csv`: cluster-bootstrap CIs, source/start pairing, quality matching.
- `quality.md`: GSM8K/MATH measured correctness, completed-correct vs extracted-correct, cap and token mismatch detail.
- `correctness.md`: full R4, KV/event readiness, numerical and dynamic-boundary invariants.
- `gpu_contention.md`: device ID, worker scheduling, stable vs contended lanes, measured overhead.
- `optimization_overhead.md`: CPU selection and actual end-to-end impact, unsupported solver claims.
- `failure_log.md` and `decision_gates.md`: diagnostics, STOP/PASS/CONDITIONAL/NO_GO/BLOCKED/NOT_RUN scope.
- `reproduce.md`: source locks, environment, commands, data hashes, outputs, seeds and start/stop instructions.
- `figures/`: **(A)** C/G decision funnel; **(B)** E/risk gradient and interaction heatmaps; **(C)** objective-family action diversity; **(D)** TTFT–ITL Pareto; **(E)** SLO attainment vs request rate; **(F)** C/G state/time policy timeline; **(G)** workload transfer matrix; **(H)** tail CCDF; **(I)** overhead versus gain.

Retain append-only `run_manifest.jsonl`, immutable raw traces, observed-execution provenance tags (`LIVE_MEASURED`, `OFFLINE_ESTIMATE`, `SIMULATION_ONLY`, `NOT_RUN`, `QUALITY_DIAGNOSTIC`), and SHA256 manifests. Put huge captured tensors/profiler payloads in local paths with documented hashes if GitHub size limits require, but commit portable reports, code, and appropriately-sized raw summaries. Include an exact link or ID for each final claim's underlying data table.

## 12. Operational checklist and final Codex response

1. Confirm physical GPU4–7 access and ownership; freeze model/runtime/patch fingerprints; review past 144/144 evidence.
2. Build reproducible Phase A action-equivalence/eligibility-funnel report **before** creating a new reward function.
3. Construct normalized E/R features and support flags; run unit tests and all O0–O13 offline action-score surveys, including feature and gradient ablations.
4. Rank and freeze a small online shortlist using TRAIN/VALIDATION only; keep a locked G-only objective-matched control, C-only and Mixed.
5. Execute balanced low-risk online pilot in 4-GPU independent workers, then clean matched confirmation with rotation and contention review.
6. Build Chat/Mixed/Long/Bursty + optional Long8K distributions, measure action diversity and actual serving gains. Escalate to full SLO-capacity measurements only if justified by new executed C choices and consistent quality.
7. Review request completion, output/caps/accuracy and GPU/KV invariants; publish every negative or ambiguous finding. Do not manufacture improved C choices.
8. Commit and push code/report changes only under approved repository workflow; preserve past source and existing staged changes. Verify the remote commit and record the hash. Do not modify/run the paused native-vLLM Loop PP project.

The final human-readable response should answer in Korean: **why C previously never changed; which E/R objective families steered actual C/G, with cross-effects; which conditions produced different legal C choices; which of those choices improved actual quality-matched serving over tuned G-only, C-only and Mixed; whether Chat/Mixed/Long/Bursty changed this; and a final evidence-scoped GO/CONDITIONAL/NO_GO/BLOCKED decision**. Never turn "more objective variations" into a claim of novelty without incremental measured serving results.
