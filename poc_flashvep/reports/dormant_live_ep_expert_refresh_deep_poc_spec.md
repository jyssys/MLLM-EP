# Deep PoC Specification
# Dormant-Live Token Lifetime for Expert-Parallel MoE dLLM Inference

## 0. Mission

Investigate whether still-masked tokens contain a large **DORMANT-LIVE** subset that will not participate in decoding decisions for several future refinement steps and therefore may not need fresh routed-MoE execution every step.

The intended novelty is not generic token pruning or KV caching. It is:

> reduce the refresh frequency of the routed-MoE path for dormant-live tokens, thereby reducing router/dispatch/expert/combine work and remote EP traffic.

Use three token states:

- ACTIVE-LIVE: near the decision frontier -> fresh routed-MoE every step.
- DORMANT-LIVE: still masked/live but not expected to be selected soon -> periodically refresh or conditionally reuse routed-MoE output.
- DEAD/STABLE: no longer needs live diffusion work -> Epoch-like removable work, not our novelty.

Desired chain:

`Epoch removes DEAD -> large DORMANT routed-MoE region remains -> dormant routed work can be refreshed sparsely -> additional EP latency reduction`

Do not assume this is true.

## 1. Hardware / primary substrate

Use only physical GPUs 4,5,6,7:

`CUDA_VISIBLE_DEVICES=4,5,6,7`

Primary model:
`inclusionAI/LLaDA2.0-flash`

Primary path:
- dense TP4
- routed EP4
- DeepEP remote dispatch
- owner-rank fused expert execution
- reverse combine

Revalidate actual topology/config and record model revision, layers, hidden size, experts, top-k, shared experts, CUDA/NCCL/DeepEP, GPU UUID/topology.

All positive headline claims must return to LLaDA2.0-Flash 100B.

## 2. Window-Diffusion reference / positive control

Use the official Window-Diffusion repository as a reference/positive control:
`https://github.com/vhicrgit/Window-Diffusion`

Its official artifact supports LLaDA-8B-Base and Dream and exposes active/buffer/far-field token states with windowed selective computation and periodic KV refresh.

Use the reference only to:
1. reproduce a smoke test;
2. understand active/buffer/far-field token-state semantics;
3. validate token-state instrumentation.

Do not treat dense Window-Diffusion speedups as EP evidence.

The primary EP experiments must be on LLaDA2.0-Flash.

## 3. Prior-art boundary

Separate explicitly from:

- Window-Diffusion: token window pruning + KV caching.
- Elastic-Cache: sliding-window KV caching + attention-aware refresh.
- SureLock: skips computation for already-unmasked/converged tokens.
- REFLEX: changes expert budget/top-k allocation by refinement state.
- TEAM: changes expert activation/decoding using routing consistency.
- Epoch: removes dead/stable routed rows.

Our main distinction should be:

> still-live dormant MASK tokens keep normal routed-expert semantics on refresh steps, but the *frequency* of routed-MoE execution is reduced.

Do not make the main method “dormant token -> lower top-k”.

## 4. Common quality workloads

Primary:
- GSM8K
- HumanEval

Use identical bounded request sets/seeds across variants.

Record:
- task score
- final token sequence
- output length
- NFE
- per-token acceptance iteration
- accepted-token history

Use smaller fixed subsets for expensive causal sweeps, then revalidate promoted policies on full bounded sets.

## 5. Core definitions

For token position i:

`g(i) = baseline refinement iteration when token i becomes accepted/finalized`

At current iteration t:

`D(i,t) = g(i) - t`

This future distance is an oracle diagnostic only.

For horizon H:

- ACTIVE-H: still masked and D(i,t) <= H
- DORMANT-H: still masked and D(i,t) > H
- DEAD: already accepted / no longer needs live routed diffusion work

Test:
`H in {0,1,2,4,8}`

Only after oracle headroom is established should a future-free dormant proxy be built.

## 6. Stage 0 — strongest EP4 baseline

Reproduce the strongest valid LLaDA2.0-Flash EP4 baseline.

Measure clean:
- BCT
- throughput
- NFE
- peak HBM

Trace:
- attention
- router/prep
- dispatch
- expert
- combine
- shared expert if separable
- remote bytes
- physical rows
- active experts
- rows/expert
- rank fanout
- rank-load CV

Do not use a weak default baseline.

## 7. Stage 1 — Window-Diffusion positive control

Reproduce its active/buffer/far-field behavior on the supported dense model.

Required:
- fraction of active/buffer/far tokens over refinement
- mapping from those states to future acceptance distance
- quality baseline vs Window-Diffusion setting
- representative token-state traces

Goal:
establish a reproducible definition of “dormant-like” masked tokens.

Do not spend the majority of the study optimizing this dense reference.

## 8. Stage 2 — dormant-live census on LLaDA2.0-Flash EP4

Annotate baseline token rows with future acceptance time.

For each H, report:

`Task | H | Active-live % | Dormant-live % | Dead %`

Also stratify by:
- early/middle/late
- block
- request

First key question:

> After subtracting dead/stable work, is there still a large dormant-live region?

Token fraction alone is insufficient. Map it to routed-MoE cost.

Suggested discovery interpretation:
- post-Epoch dormant EP share <5% E2E: likely kill
- 5–10%: characterization
- 10–20%: promising
- >20%: strong discovery

## 9. Stage 3 — dormant EP cost attribution

For ACTIVE / DORMANT / DEAD states attribute:

- router rows
- dispatch rows
- remote assignments
- remote bytes
- expert rows
- combine rows
- active experts touched
- destination-rank fanout
- mapped routed-MoE E2E cost

Create:
`State | Token rows | Remote bytes | Dispatch | Expert | Combine | E2E share`

Report:
- raw dormant physical share
- dormant routed-MoE share
- post-Epoch dormant residual

A 40% dormant token fraction is only interesting if it corresponds to meaningful EP cost.

## 10. Stage 4 — dormant vs active routing/expert dynamics

Compare ACTIVE vs DORMANT:

- router entropy
- top-k set persistence
- top-k overlap
- destination-rank overlap
- fanout
- remote fraction
- hidden-state change
- routed-MoE output drift
- rows/expert contribution
- expert support

Key question:

> Are dormant tokens more temporally stable in routing or routed output than active tokens?

If yes, sparse refresh becomes more plausible.

## 11. Stage 5 — future-aware routed-MoE oracles

O1 — perfect dormant routed-MoE removal:
assume dormant routed work is free until activation. Map to E2E.

O2 — periodic dormant refresh:
for dormant tokens, refresh routed MoE every K steps, with:
`K in {2,4,8}`

Reuse prior routed-MoE contribution between refreshes.

O3 — event-triggered oracle:
force refresh if:
- token becomes active soon
- top-k route changes materially
- destination rank changes
- router-score drift exceeds threshold
- semantic confidence/margin changes sharply

O4 — post-Epoch dormant oracle:
remove dead work first, then report only additional dormant opportunity.

The post-Epoch number is the key novelty/economics result.

## 12. Stage 6 — causal dormant routed-MoE interventions

Do not skip the whole transformer initially.

Keep:
- attention fresh
- dense path fresh
- shared expert fresh

Target only routed MoE.

D1 — dormant routed-MoE stale reuse:
for dormant tokens, reuse previous cached routed-MoE contribution.

Diagnostic runs may execute original work first and replace its output; do not claim speed from such runs.

D2 — periodic routed refresh:
fresh every K steps, cached routed contribution otherwise.
Force refresh immediately when a token becomes ACTIVE.

D3 — route-change-gated refresh:
keep router fresh.
If top-k/destination/score changes materially, refresh routed experts.
Otherwise allow reuse.

D4 — EP-cost-aware dormant refresh:
among tokens already judged safe/dormant, prioritize deferring those with high remote EP cost:
- remote assignments
- destination fanout
- remote bytes
- critical expert rows

Safety gate comes before cost optimization.

## 13. Stage 7 — router freshness ablation

R1:
fresh router every step + stale routed expert output.

R2:
cached router + cached routed output.

R3:
lightweight route-stability check if feasible.

Prefer R1 initially because it offers a safety trigger and clearer prior-art separation.

Compare quality, NFE, oracle, and complexity.

## 14. Stage 8 — full-trajectory validation

Previous experiments showed:
`current-step agreement != future-trajectory agreement`

Therefore every promoted policy must run complete trajectories.

Measure:
- GSM8K
- HumanEval
- final-sequence exact rate
- NFE drift
- acceptance-iteration drift
- malformed generations

Do not certify safety from local logits or hidden cosine.

## 15. Stage 9 — dormant horizon / refresh Pareto

Diagnostic grid:
- H in {1,2,4,8}
- K in {2,4,8}

Use a small fixed subset first, then validate Pareto candidates.

Report:
`(H,K) -> quality, NFE, routed rows saved, remote bytes saved, E2E oracle`

Required Pareto:
- x: routed-MoE/EP cost saved
- y: quality / NFE stability

## 16. Stage 10 — deployable dormant classifier

Only if future-aware oracle is large.

Future-free semantic features:
- confidence
- entropy
- top1-top2 margin
- confidence slope
- current acceptance rank
- distance to active prefix/window

Expert/EP features:
- top-k persistence
- destination persistence
- router entropy
- router-score drift
- estimated remote EP cost

Compare:
1. semantic-only
2. expert-only
3. joint semantic + EP

Use simple thresholds / shallow tree / logistic model before complex policies.

Optimize for:
- very low active-token false-dormant rate
- saved routed-MoE cost

Overall accuracy is less important.

## 17. Stage 11 — EP-aware dormant refresh scheduler

Only if gates pass.

Concept:

ACTIVE:
- fresh attention
- fresh router
- fresh routed experts

DORMANT:
- fresh attention/shared path
- route stability check
- refresh routed experts only when due or unstable
- otherwise reuse cached routed contribution

DEAD:
- remove from routed live work analytically / via Epoch-like baseline

Objective:
`maximize safely deferred routed-MoE cost subject to quality/NFE guard`

## 18. Stage 12 — physical savings accounting

Measure separately:
- router rows saved
- dispatch rows saved
- remote assignments saved
- dispatch bytes saved
- expert rows saved
- combine rows saved
- active expert events saved
- DeepEP calls
- fanout
- expert GEMM shapes

Distinguish:
- payload reduction
- call elimination
- expert compute reduction

Do not infer speedup from row count alone.

## 19. Stage 13 — expert-aware variants

Only if the basic dormant policy is promising.

E1 — Dormant Expert Refresh Decimation
- primary recommended direction
- normal top-k semantics on refresh steps
- change when routed experts execute

E2 — Destination-Stability-Gated Refresh
- force refresh on destination-rank change

E3 — Cost-Aware Safe Dormant Budget
- among safe dormant tokens, skip highest EP-cost work first

E4 — Shared-Fresh / Routed-Dormant Split
- attention fresh
- shared expert fresh
- routed expert periodically refreshed

Do not claim any variant without trajectory-level quality evidence.

## 20. Stage 14 — Window-Diffusion composition

If feasible compare:
1. vanilla
2. Window-Diffusion-style token-window/KV optimization
3. dormant routed-MoE refresh only
4. both combined

Question:

> Does EP-aware dormant refresh remove routed expert work that remains after Window-Diffusion's window/KV optimization?

If Window-Diffusion removes the same rows entirely, report the residual as zero/limited.

## 21. Stage 15 — Epoch residual accounting

Analytically partition routed work:

- DEAD -> Epoch-removable
- DORMANT-LIVE -> our candidate residual
- ACTIVE-LIVE -> fresh

Headline economics must be:
`additional post-Epoch E2E headroom`

Do not double-count dead work.

## 22. Promotion gates

Discovery:
- post-Epoch dormant routed-MoE E2E share >=10%: promising
- >=20%: strong

Quality-safe oracle:
- <5%: NO-GO
- 5–8%: CHARACTERIZATION
- 8–12%: HOLD
- 12–20%: STRONG
- >20%: VERY STRONG

Prefer >=12% credible post-Epoch headroom before heavy runtime implementation.

## 23. Failure recovery

If dormant fraction is small:
- vary H
- compare Window-Diffusion states
- phase/position-conditioned dormancy
- stop if post-Epoch EP cost stays <5%

If dormant is large but stale routed MoE breaks quality:
- smaller K
- immediate refresh on active transition
- fresh router
- route-change trigger
- semantic-change trigger
- shared-fresh/routed-stale split

If rows saved but E2E small:
- inspect DeepEP fixed cost
- payload sensitivity
- call count
- GEMM packing changes
- stop if perfect removal <5%

If future-aware oracle is large but future-free classifier fails:
- characterize only; do not fake deployability.

## 24. Anti-cherry-picking

- log every H/K/threshold/task/model
- separate discovery and validation subsets
- report negative tasks
- final positive claim must include LLaDA2.0-Flash 100B
- Window-Diffusion is a reference/positive control only
- do not use answer correctness as an online feature

Create `ATTEMPT_LOG.csv`.

## 25. Required reports

- reports/00_environment_and_baseline.md
- reports/01_window_diffusion_positive_control.md
- reports/02_dormant_token_census.md
- reports/03_dormant_ep_cost_attribution.md
- reports/04_dormant_routing_dynamics.md
- reports/05_future_aware_oracles.md
- reports/06_causal_routed_moe_interventions.md
- reports/07_refresh_quality_pareto.md
- reports/08_deployable_dormant_classifier.md
- reports/09_ep_aware_refresh_scheduler.md
- reports/10_window_diffusion_composition.md
- reports/11_epoch_residual_analysis.md
- reports/12_prior_art_and_novelty.md
- reports/final_decision.md

Machine-readable:
- DORMANT_TOKEN_TRACE.csv
- DORMANT_EP_COST.csv
- DORMANT_ORACLES.csv
- REFRESH_POLICY_RESULTS.csv
- CLASSIFIER_RESULTS.csv
- ATTEMPT_LOG.csv

## 26. Final labels

- NO-DORMANT-SIGNAL
- CHARACTERIZATION-SIGNAL
- HOLD-CANDIDATE
- STRONG-CANDIDATE
- VERY-STRONG-CANDIDATE

## 27. Final questions

1. Does Window-Diffusion's active/buffer/far structure reproduce?
2. What fraction of live MASK tokens are future-dormant?
3. How much routed-MoE/EP cost belongs to dormant-live tokens?
4. How much remains after Epoch-like dead work removal?
5. Are dormant tokens more route-stable?
6. Are they more destination-rank-stable?
7. Can routed-MoE outputs be refreshed every K steps safely?
8. Does a fresh router improve safety?
9. Does route-change-triggered refresh improve the Pareto?
10. Does shared-fresh/routed-stale help?
11. What is the perfect dormant-removal E2E upper bound?
12. What is the quality-safe periodic-refresh oracle?
13. What is the independent post-Epoch oracle?
14. Can a future-free dormant proxy identify safe skips?
15. Do expert/EP features add value beyond semantic features?
16. How many remote bytes / expert rows are actually removed?
17. Does reduced routed work translate to latency?
18. Is the method complementary to Window-Diffusion / Elastic-Cache?
19. Is it clearly distinct from REFLEX / TEAM?
20. Is there paper-level headroom?

## 28. Final instruction

The desired discovery is not merely:
`some masked tokens are far away or low confidence`

Prior work already knows that.

The desired discovery is:

`a large subset of still-live MASK tokens is temporally dormant`
`+ dormant tokens account for substantial routed-MoE EP cost`
`+ their routed expert path can be refreshed less frequently`
`+ expert/routing state provides a safe refresh trigger`
`+ this removes dispatch + expert execution + combine beyond Epoch and KV/window caching`

The main method should change the **lifetime/frequency of routed expert execution**, not simply reduce top-k or perform generic token pruning.
