# PoC Specification
# Refinement-Aware Dynamic Wave Sizing for dLLM-MoE Expert-Parallel Inference

## 0. Objective

This PoC tests a focused hypothesis:

> In a diffusion language model, refinement workload changes over time. If those changes alter the physical sparse-MoE execution shape, the optimal EP microbatch / wave size may also change. Therefore a fixed `mini_batch_size` may be suboptimal.

Target method, only if the hypothesis is supported:

> **Refinement-Aware Dynamic Wave Sizing (RAWS)**

RAWS dynamically chooses the physical EP wave size / `mini_batch_size` using the current refinement-state workload shape.

This is not the same as changing the submitted request batch size.
It is also not automatically the same as continuous batching.

---

## 1. Terminology

### 1.1 32 rows

For the current LLaDA2.0-Flash setup:

`block_length = 32`

One request contributes up to 32 token-position rows to one refinement forward.

Examples:

- mini=1 -> 1 request × 32 rows -> M=32
- mini=4 -> 4 requests × 32 rows -> M=128
- mini=8 -> 8 requests × 32 rows -> M=256
- mini=16 -> 16 requests × 32 rows -> M=512

`32 rows` means 32 token positions are processed in that refinement forward.
It does not mean all 32 tokens are fully denoised/finalized in one step.

### 1.2 Submitted batch

`submitted batch` = number of requests in the controlled request pool.

Example:
`submitted batch = 16`

means 16 requests exist.

### 1.3 mini_batch_size

`mini_batch_size` = how many ready requests are grouped into one physical model/EP wave.

With submitted batch 16:

- mini=1 -> 16 small waves
- mini=4 -> 4 waves
- mini=8 -> 2 waves
- mini=16 -> 1 wave

The user-visible request count does not change.
Only physical execution granularity changes.

---

## 2. Existing Motivation

Previous EP4 measurements showed:

- mini=1  -> BCT 83.608 s
- mini=2  -> BCT 26.604 s
- mini=4  -> BCT 13.115 s
- mini=8  -> BCT 6.904 s
- mini=16 -> BCT 8.040 s

Therefore:

> increasing wave size first amortizes startup / launch / fragmentation cost, but oversized waves eventually regress.

A physical wave-size sweet spot exists.

TP4 showed a different observed optimum, with mini=16 best among tested points.

Thus optimal granularity depends on topology.

---

## 3. dLLM Refinement Motivation

Previous phase analysis showed:

- Early decision-live / physical rows ≈ 82.88%
- Middle ≈ 49.30%
- Late ≈ 13.31%

However the current runtime still processes dense physical 32-position blocks.

Therefore logical work changes strongly, but physical row count may remain unchanged.

This PoC must not assume:

> late refinement automatically requires a different mini size.

It must test whether refinement changes the physical execution features that actually determine optimal wave size.

---

## 4. Main Research Question

Does the optimal `mini_batch_size` change as a function of denoising/refinement state?

Potential explanatory variables:

- normalized refinement iteration
- decision-live fraction
- accepted positions
- masked positions
- physical rows M
- token-expert assignments
- active experts
- per-expert row histogram
- tiny-expert fraction
- mean/p50/p90 expert rows
- destination-rank fanout
- remote assignment fraction
- rank-load CV
- critical-rank load
- dispatch bytes
- combine bytes

---

## 5. Hardware / Runtime

Use only physical GPUs 0,1,2,3.

Primary topology:
`true EP4`

Reuse the already validated LLaDA2.0-Flash true sparse EP4 path:
- dense TP4
- routed EP4
- DeepEP remote dispatch
- owner-rank expert execution
- reverse combine

Use fused kernels.
Do not use Python expert loops.

Record:
- model revision
- runtime commit
- patch
- DeepEP
- CUDA/NCCL
- GPU topology

---

## 6. Stage 0 — Reproduce Static Sweet Spot

Use a submitted request pool large enough to test the mini-size set.

Preferred:
`submitted batch = 32`

Test:
`mini = 1,2,4,8,16,32`

If mini32 is OOM or unsupported, stop at the largest feasible point.

Use:
- identical prompts
- identical generation budget
- identical decoder
- multiple restarts
- randomized order

Measure:
- BCT / request latency
- throughput
- HBM
- dispatch
- expert
- combine
- GPU utilization
- rank-load CV
- message sizes

Establish:
`best static EP4 mini size`

This is the baseline.

---

## 7. Stage 1 — Per-Iteration Workload Trace

For each refinement step save:

Logical:
- request/block ID
- iteration index
- normalized phase
- early/middle/late
- decision-live rows
- masked rows
- accepted positions
- confidence summary

Physical:
- ready request count
- physical rows
- token-expert pairs
- remote pairs
- active experts
- per-expert row histogram
- tiny expert fraction
- fanout
- rank-wise assignment counts
- rank-load CV
- critical rank
- dispatch/combine bytes

Timing:
- attention
- router/prep
- dispatch
- expert
- combine
- whole MLP
- whole block

Use low-overhead tracing for shapes.
Do not use observer-heavy absolute times as request-level latency claims.

---

## 8. Stage 2 — Phase-to-Shape Correlation

Test:

- phase vs physical rows
- phase vs expert rows/expert
- phase vs tiny-expert fraction
- phase vs active experts
- phase vs rank-load CV
- phase vs fanout
- phase vs dispatch/combine bytes
- phase vs stage latency

Use:
- phase-conditioned medians
- Spearman/Pearson when appropriate

Central gate:

> not merely whether logical liveness changes, but whether physical features that control wave efficiency change enough to move the optimal mini size.

If physical shape is nearly phase-invariant, a phase-only policy is likely dead.

---

## 9. Stage 3 — Per-Shape Mini-Size Replay Matrix

Capture representative real workload states from early/middle/late.

Replay equivalent ready work under:

`mini = 1,2,4,8,16,32`

subject to available ready requests.

For each state measure:
- latency
- throughput
- dispatch
- expert
- combine
- rank imbalance
- HBM

Build:

| Shape/phase | mini1 | mini2 | mini4 | mini8 | mini16 | mini32 | Winner |
|---|---:|---:|---:|---:|---:|---:|---|
| Early A | | | | | | | |
| Early B | | | | | | | |
| Middle A | | | | | | | |
| Late A | | | | | | | |

The key question:
> does the winner actually change?

---

## 10. Stage 4 — Perfect Dynamic Oracle

### Best static baseline

Choose one mini size for the whole request.

### Perfect dynamic oracle

At each refinement step choose the fastest measured feasible mini size.

Compare:
`Best Static vs Perfect Per-Step Dynamic`

Request-level gain:

`(T_static - T_dynamic_oracle) / T_static`

This is the decisive gate.

---

## 11. Oracle Gates

- <3% -> NO-GO
- 3-5% -> weak / likely tuning-only
- 5-8% -> interesting
- 8-12% -> promising
- >=12% -> strong
- >=15% -> very strong

Do not implement a controller if the perfect dynamic oracle is below 5%.

Prefer >=8-10% before live implementation.

---

## 12. Stage 5 — Explain the Winner

If oracle is meaningful, find the smallest interpretable feature set that predicts the winner.

Candidate groups:

### Phase-only
- normalized denoising iteration

### Logical liveness
- decision-live fraction
- masked count
- accepted count

### Physical shape
- physical rows
- expert row histogram
- tiny-expert fraction
- active experts

### EP geometry
- fanout
- rank-load CV
- critical-rank work
- remote bytes

Prefer simple threshold / lookup-table policies before learned models.

---

## 13. Stage 6 — Live Dynamic Prototype

Only if oracle passes.

At each scheduling/refinement point:

1. inspect current physical workload shape;
2. choose mini size;
3. execute wave;
4. preserve model semantics.

No model retraining.

Compare:
- best static EP4
- dynamic EP4

Measure:
- p50/p95 latency
- BCT
- throughput
- HBM
- quality
- scheduler overhead

Include controller overhead.

---

## 14. Important Caveat

Do not claim:
> late refinement automatically needs a different mini size.

Current runtime may still expose 32 physical rows/request at all phases.

Therefore the useful predictor may be:
- expert fragmentation
- active expert count
- rank load
- fanout

rather than phase itself.

---

## 15. Optional Epoch-Like Compaction Sensitivity

Do not implement Epoch.

But create an offline hypothetical compacted-shape analysis.

Use decision-live rows to estimate compacted physical work and ask:

> if physical M shrank with liveness, would optimal mini size become more phase-dependent?

Report separately:
- current dense-block runtime
- hypothetical compacted runtime

This tests complementarity with future Epoch-like runtimes.

---

## 16. Online Serving Relevance — Analysis Only

Do not implement full online serving.

Analyze whether RAWS extends to online serving.

In online serving:
- requests arrive continuously;
- different requests occupy different refinement phases;
- ready work shapes are heterogeneous.

Distinguish:

Continuous batching:
`which requests are ready/admitted now?`

RAWS:
`how many ready requests should be grouped into one physical EP wave?`

Possible future stack:

`continuous batching -> ready pool -> refinement-aware wave sizing -> EP execution`

Measure offline trace statistics:
- ready-pool size
- phase heterogeneity
- shape heterogeneity
- coalescing opportunity

Do not claim serving speedup without live serving results.

---

## 17. Online Serving Hypothesis

Online serving may make the idea more useful because:

- Request A: early/heavy
- Request B: late/light
- Request C: middle
- Request D: late/light

A scheduler could choose physical wave size using aggregate shape.

But queueing delay must be included.

Future serving objective:

`execution efficiency gain - extra waiting/batching delay`

not throughput alone.

---

## 18. TP4 Control

If EP4 oracle is promising, repeat a smaller oracle study on TP4.

Reason:
- EP4 static optimum ≈ mini8
- TP4 static optimum ≈ mini16

Determine whether dynamic granularity is:
- strongly EP-specific, or
- generic adaptive microbatching.

If equally useful in TP4 and unrelated to sparse EP shape, novelty as an EP method is weaker.

---

## 19. Prior-Art Audit

If oracle is strong, audit:
- continuous batching
- dynamic batching
- adaptive microbatching
- SGLang scheduling
- TBO / Two-Batch Overlap
- DeepEP microbatching
- generic MoE batching
- dLLM serving schedulers
- Epoch

Novelty cannot simply be:
> dynamically choose batch size.

Target novelty:
> dLLM refinement-dependent sparse workload shape + topology-specific EP wave efficiency.

---

## 20. Required Plots

1. Static EP4 mini-size latency curve
2. Static EP4 throughput curve
3. HBM vs mini size
4. Decision-live fraction vs iteration
5. Physical M vs iteration
6. Active experts vs phase
7. Tiny-expert fraction vs phase
8. Rank-load CV vs phase
9. Expert-row histogram early/middle/late
10. Winning mini size by state
11. Best mini vs physical M
12. Best mini vs tiny-expert fraction
13. Best mini vs rank-load CV
14. Best-static vs perfect-dynamic latency
15. Static vs dynamic live result
16. Hypothetical compacted-runtime oracle
17. TP4 vs EP4 oracle comparison

---

## 21. Required Reports

Create:
- `reports/00_environment.md`
- `reports/01_static_wave_sweep.md`
- `reports/02_refinement_shape_trace.md`
- `reports/03_phase_shape_correlation.md`
- `reports/04_shape_replay_matrix.md`
- `reports/05_dynamic_wave_oracle.md`
- `reports/06_predictor_analysis.md`
- `reports/07_dynamic_live_poc.md`
- `reports/08_compaction_sensitivity.md`
- `reports/09_online_serving_analysis.md`
- `reports/10_prior_art_audit.md`
- `reports/final_decision.md`

---

## 22. Final Labels

- `NO-GO`
- `STATIC-SUFFICIENT`
- `HOLD`
- `PROMISING`
- `STRONG-CANDIDATE`

Definitions:
- NO-GO: perfect dynamic oracle <3%
- STATIC-SUFFICIENT: dynamic oracle <5%
- HOLD: 5-8%
- PROMISING: >=8% with clear shape dependence
- STRONG-CANDIDATE: live dynamic controller >=8% over best static with preserved quality

---

## 23. Final Questions

1. Is static sweet spot reproducible?
2. Does optimal mini size differ across refinement states?
3. Which physical shape feature best predicts winner?
4. Is phase itself predictive, or only correlated?
5. How much better is perfect dynamic than best static?
6. Does gain survive controller overhead?
7. Is effect stronger in EP4 than TP4?
8. Would Epoch-like live-row compaction strengthen the opportunity?
9. Is this meaningfully different from continuous batching?
10. Is online serving likely to increase the opportunity due to heterogeneous refinement phases?

---

## 24. Final Instruction

The falsifiable chain is:

`refinement state changes -> physical sparse shape changes -> optimal EP wave size changes -> dynamic selection beats best static`

Every arrow must be measured.

If the last arrow is small, stop.
