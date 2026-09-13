# PoC Specification
# Exact Overlap Opportunity Atlas + Refinement-Phase-Aware Overlap
# for LLaDA2.0-Flash 100B dLLM-MoE EP Inference

## 0. Core Goal

This PoC has two stages.

### Stage 1 — Exact Overlap Opportunity Atlas

First, do **not** assume any particular overlap method.

Decompose LLaDA2.0-Flash true-EP4 inference into fine-grained sub-operations and discover:

> Which operations are logically independent, use complementary hardware resources, and are currently serialized by the runtime?

For every candidate overlap pair, measure:
1. dependency legality;
2. serial cost;
3. ideal overlap ceiling;
4. real concurrent slowdown/contention;
5. net request-level gain.

If Stage 1 exposes a strong exact-overlap opportunity, prioritize it.

### Stage 2 — dLLM Refinement/Timestep-Aware Overlap

Only if Stage 1 does not produce a sufficiently strong general exact-overlap candidate, investigate whether the dLLM refinement process itself creates special overlap opportunities.

Core hypothesis:

> Early/middle/late refinement phases may have different communication-vs-compute resource profiles. If so, work from complementary refinement phases or requests may be overlapped without changing model semantics.

This is different from directly overlapping the same request's timestep `t` and `t+1` backbone computation.
Direct same-request `t -> t+1` overlap is dependency-constrained because `x_{t+1}` depends on timestep `t` output.
Do not use stale/speculative future activations as the primary method in this PoC.

---

## 1. Hardware

Use only:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
```

Record GPU model/UUID, NVLink topology, CUDA, NCCL, DeepEP, driver, runtime commit, and model revision.

---

## 2. Model / Runtime

Primary model: `inclusionAI/LLaDA2.0-flash`

Reuse the validated large-model sparse path:
- dense TP4
- routed EP4
- 64 full routed experts per rank
- DeepEP dispatch
- owner-rank fused expert execution
- reverse combine

Use production-like fused kernels. Do not use Python expert loops.
Reuse the strongest validated best-static runtime configuration from previous PoCs and reproduce it first.

---

## 3. Baseline Requirements

Before overlap analysis:
1. reproduce clean best-static EP4 baseline;
2. verify quality anchor;
3. measure stable BCT/request latency;
4. record throughput, NFE, peak HBM;
5. record stage breakdown.

Use the same controlled GSM8K/HumanEval workload throughout.
All speedup claims compare against the strongest baseline.

---

## 4. Exactness Principle

Preferred overlap is **exact**.
No change to model weights, router result, selected experts, attention semantics, denoising schedule, accepted tokens, logits, or token sequence should be required.

Staleness/speculation must be reported separately and cannot be mixed into the exact-overlap headline.

---

## 5. Stage 1 — Fine-Grained Execution Decomposition

Decompose one representative transformer/MoE layer into at least:
- attention input preparation
- Q/K/V projection
- Q/K normalization / RoPE if applicable
- attention kernel
- attention output projection
- TP synchronization/collective
- router GEMM
- router scoring / top-k
- route metadata / send-count / offset prep
- shared expert path
- local routed expert work
- remote routed work
- DeepEP dispatch issue/progress/completion
- remote receive
- expert gate/up GEMM
- activation
- expert down GEMM
- expert output packing
- combine issue/progress/completion
- residual accumulation
- layer finalization

Use actual runtime boundaries when different. The goal is real dependency boundaries, not high-level operator names.

---

## 6. Stage 1A — Dependency DAG

For every operation pair `(A,B)`, classify:
- STRICT_DEPENDENCY
- PARTIAL_DEPENDENCY
- INDEPENDENT
- UNKNOWN

Test carefully:
- remote dispatch vs local expert compute;
- routed dispatch vs shared expert compute;
- dispatch wave i+1 vs expert wave i;
- expert wave i vs combine wave i-1;
- combine-send of completed expert tiles vs remaining expert GEMM;
- request A DeepEP communication vs request B attention;
- request A DeepEP communication vs request B expert GEMM;
- next-step input-independent setup vs current-step compute.

Do not assume `Dispatch -> Expert -> Combine` is indivisible.

---

## 7. Stage 1B — Resource Profile Atlas

For each major sub-operation measure/estimate:
- wall duration
- SM utilization
- TensorCore utilization
- HBM bandwidth
- NVLink bandwidth
- NVSHMEM/communication activity
- copy-engine activity
- register/shared-memory pressure if available
- kernel count
- synchronization / idle time

Classify each as TensorCore-heavy, SM-heavy, HBM-heavy, NVLink-heavy, latency/startup-heavy, or mixed.

Build:

| Operation | Time | TensorCore | SM | HBM | NVLink | Main resource |
|---|---:|---:|---:|---:|---:|---|

Key question: are two independent serialized operations using different dominant resources?

---

## 8. Stage 1C — Exact Pairwise Concurrency Microbench

For promising independent pairs, build controlled concurrency microbenchmarks.

### Pair A — Remote dispatch + local expert compute
After routing is known, remote branches go to DeepEP while local branches execute local expert GEMM.
Measure standalone times, serial sum, concurrent wall, each-side slowdown, and net saving.

### Pair B — Routed dispatch + shared expert compute
If independent after common input, overlap shared-expert GEMM with routed DeepEP dispatch.

### Pair C — Dispatch wave i+1 + expert wave i
Split routed work into multiple waves/tiles and test true partial-arrival legality.

### Pair D — Expert wave i + combine wave i-1
Measure actual contention, not just ideal overlap.

### Pair E — Down GEMM + completed-tile combine/send
If partial outputs become available early, test streaming combine.

### Pair F — Request A communication + Request B TensorCore work
Cross-request exact overlap:
- A: DeepEP dispatch/combine
- B: attention or expert GEMM

This is structurally similar to prior generic/MLLM stock overlap. The new question is whether the LLaDA2 100B true-EP4 production substrate exposes a real resource-complementary window.

---

## 9. Stage 1D — Pairwise Opportunity Matrix

Create:

| Pair | Dependency | A time | B time | Ideal save | Concurrent time | Real save | E2E oracle |
|---|---|---:|---:|---:|---:|---:|---:|

Always separate ideal overlap `min(A,B)` from measured overlap after contention.
Use measured real saving for credible E2E projection.

---

## 10. Stage 1E — Multi-Stage Pipeline Search

If pairwise overlap is promising, test legal 3-stage pipelines such as:

```text
Dispatch wave i+1
||
Expert wave i
||
Combine wave i-1
```

or cross-request communication/compute pipelines.
Measure fill, steady-state, drain, contention, and full E2E.
Do not assume pairwise gains compose additively.

---

## 11. Stage 1 Gates

Map every candidate to full request E2E:
- <5% KILL
- 5–8% WEAK
- 8–12% PROMISING
- 12–20% STRONG
- >20% VERY STRONG

If any exact candidate has >=8% credible E2E headroom and no severe prior-art/implementation conflict, prioritize it before Stage 2.
If all remain weak, continue to Stage 2.

---

## 12. Stage 1 Prior-Art Boundary

Audit promoted candidates against:
- DeepEP communication/computation overlap
- DeepEP event-overlap APIs
- fine-grained EP overlap
- StreamEP-style streaming EP
- X-Stage
- MegaMoE
- TBO/double-batch overlap
- generic MoE pipelining
- TP/EP serving overlap

Do not claim novelty for "dispatch and GEMM overlap" alone.
Novelty must come from previously unused dependency/resource structure or dLLM/MoE-specific execution behavior.

---

## 13. Stage 2 Trigger

Run Stage 2 deeply only if:
1. Stage 1 has no credible exact candidate >=8%; or
2. Stage 1 exposes a building block whose strength may change with refinement phase.

Stage 2 asks:

> Does denoising/refinement timestep systematically change the physical resource profile enough to create dLLM-specific overlap opportunities?

---

## 14. Stage 2A — Refinement-Phase Resource Profiling

For every real refinement step record:
- normalized timestep
- early/middle/late
- decision-live ratio
- physical rows
- ready request count
- attention/router/route-prep/dispatch/expert/combine time
- whole MoE time
- TensorCore utilization
- SM utilization
- HBM bandwidth
- NVLink bandwidth
- remote bytes
- active experts
- rows/expert
- rank fanout
- rank-load CV

Key output is compute intensity, communication intensity, and bottleneck type vs refinement phase.

---

## 15. Stage 2B — Test for Resource Oscillation

Do not assume a pattern. Test whether phases systematically shift between compute-heavy and communication/startup-heavy regimes.

Required analyses:
- phase vs expert/dispatch ratio
- phase vs TensorCore utilization
- phase vs NVLink utilization
- phase vs comm/(comm+compute)
- phase vs idle/synchronization fraction

Require reproducibility across tasks/requests.

---

## 16. Stage 2C — Cross-Request Phase-Complementary Overlap

If phases have complementary resource profiles, test exact cross-request overlap.

Example:
- Request A: late / communication-heavy
- Request B: middle / TensorCore-heavy

Overlap A's DeepEP communication with B's attention/expert GEMM.

Compare phase pairings:
- early+early
- middle+middle
- late+late
- early+late
- middle+late

Measure standalone costs, concurrent wall, pairwise slowdown, net E2E, throughput, and per-request latency.

Question: are complementary phase pairings better than arbitrary pairings?

---

## 17. Relationship to Prior MLLM/Generic Stock Overlap

Explicitly compare against phase-agnostic cross-request overlap:

```text
Request A communication
||
Request B compute
```

Stage 2 is meaningfully dLLM-specific only if refinement phase predicts which pairings are resource-complementary and materially outperforms phase-agnostic pairing.

Required baselines:
- FIFO/random pairing
- resource-blind overlap
- phase-aware overlap

If phase-aware does not improve over resource-blind overlap, the dLLM-specific claim fails.

---

## 18. Stage 2D — Phase-Aware Intra-MoE Pipeline

If best overlap depth changes by phase, test phase-dependent policies over:
- inflight dispatch waves
- expert pipeline depth
- combine overlap depth
- communication pre-post distance

This is not RAWS: exact work amount stays unchanged; only concurrency/pipeline schedule changes.
Compare against best static overlap schedule.

---

## 19. Stage 2E — Next-Timestep Input-Independent Preparation

Same-request `t` and `t+1` backbone computation is dependency-constrained.
But test whether timestep `t` can overlap with input-independent preparation for `t+1`:
- buffer reuse/reset
- receive pre-post
- communication descriptors
- static workspace setup
- CUDA graph/persistent-kernel state
- topology metadata
- predicted coarse rank-geometry scheduling metadata

Use temporal geometry stability only for preparation, never to change routing/output semantics.
Measure total preparation cost first and kill immediately if perfect overlap is <5% E2E.

---

## 20. Same-Request t <-> t+1 Backbone Overlap

Treat as low-priority exploratory baseline only.
Direct overlap requires predicting/approximating `x_{t+1}` before timestep `t` finalizes the current refinement decision.
Previous causal evidence showed current-step agreement does not guarantee future-trajectory agreement.
Do not invest unless all exact alternatives fail and a strong oracle exists with a separate future-trajectory safety mechanism.

---

## 21. Stage 2 Oracle Requirements

Compare:
- best serial/static
- best phase-agnostic overlap
- perfect phase-aware overlap oracle
- real phase-aware concurrent execution

Report:
- p50/p95 latency
- BCT
- throughput
- per-request slowdown
- GPU/TensorCore/NVLink utilization

The method must improve total system/request performance, not merely move work between requests.

---

## 22. Contention Accounting

For every overlap candidate report:
- A standalone
- B standalone
- serial A+B
- ideal overlap
- actual concurrent
- A slowdown
- B slowdown
- net saving

Do not hide overlap-induced slowdown.

---

## 23. Quality / Correctness

Exact overlap candidates must preserve:
- request outputs
- NFE
- accepted-token behavior
- bounded benchmark quality

Report numerical differences separately if floating-point scheduling changes reductions, but semantic outputs must remain valid.

---

## 24. Strong Candidate Criteria

Especially valuable if all hold:
1. exact semantics;
2. >=8–10% E2E gain;
3. no training;
4. low quality risk;
5. production-compatible DeepEP/fused path;
6. not already implemented by existing EP streaming systems;
7. preferably tied to dLLM refinement/resource structure.

---

## 25. Required Figures

Stage 1:
1. fine-grained layer timeline
2. dependency DAG
3. resource-profile atlas
4. pairwise opportunity matrix
5. serial vs concurrent latency
6. contention matrix
7. E2E oracle ranking
8. best multi-stage pipeline timeline

Stage 2:
9. compute/communication ratio vs timestep
10. TensorCore utilization vs phase
11. NVLink utilization vs phase
12. phase-conditioned stage breakdown
13. phase-pair concurrency matrix
14. random/resource-blind vs phase-aware pairing
15. best-static vs phase-aware pipeline
16. next-step preparation oracle

---

## 26. Required Reports

Create:
- `reports/00_environment.md`
- `reports/01_best_static_baseline.md`
- `reports/02_fine_grained_dependency_dag.md`
- `reports/03_resource_profile_atlas.md`
- `reports/04_pairwise_overlap_matrix.md`
- `reports/05_multistage_pipeline_oracles.md`
- `reports/06_stage1_prior_art_audit.md`
- `reports/07_refinement_resource_profiles.md`
- `reports/08_phase_pair_overlap_matrix.md`
- `reports/09_phase_aware_pipeline.md`
- `reports/10_next_step_preparation.md`
- `reports/11_stage2_prior_art_audit.md`
- `reports/12_candidate_oracles.md`
- `reports/13_best_live_overlap_poc.md`
- `reports/final_decision.md`

---

## 27. Final Decision Labels

- `NO-OVERLAP-SIGNAL`: no exact candidate exceeds 5% credible E2E.
- `CHARACTERIZATION-SIGNAL`: interesting overlap exists but remains <8%.
- `HOLD-CANDIDATE`: >=8% credible candidate.
- `STRONG-CANDIDATE`: >=12% credible exact-overlap headroom + plausible novelty.
- `VERY-STRONG-CANDIDATE`: live prototype >=10–15% E2E or substantial throughput gain over strongest baseline with exact semantics and strong novelty.

---

## 28. Final Questions

1. Which sub-operations are actually independent?
2. Which independent operations use complementary hardware resources?
3. What is the strongest pairwise exact overlap?
4. Does contention destroy ideal overlap?
5. Is there a legal dispatch/expert/combine wave pipeline?
6. Can local expert work overlap remote dispatch?
7. Can shared expert compute overlap routed EP communication?
8. Does cross-request communication-compute overlap work on LLaDA2 true EP4?
9. Does refinement phase systematically change compute-vs-communication intensity?
10. Do complementary phase pairings outperform random/resource-blind pairing?
11. Is there a phase-dependent best overlap/pipeline policy?
12. Can next-timestep input-independent preparation be overlapped meaningfully?
13. What is the strongest credible request-level E2E oracle?
14. Is the strongest candidate generic MoE overlap or truly dLLM-specific?
15. How does it differ from DeepEP/StreamEP/X-Stage/TBO and diffusion-system overlap prior art?

---

## 29. Final Instruction

Follow this order:

```text
Stage 1:
Find exact overlap opportunities without assuming dLLM phase behavior.

If strong:
    prioritize strongest exact candidate.

If weak:
    Stage 2:
    test whether dLLM refinement/timestep creates resource complementarity
    that makes phase-aware exact overlap materially better.
```

Do not force a timestep-based method if the exact atlas already exposes a stronger opportunity.
Do not force overlap if realistic concurrent E2E oracles are small.

Desired discovery:

> A large region of exact required work is serialized today even though its true data dependency and hardware-resource dependency allow concurrency.
