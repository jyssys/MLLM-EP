# Discovery PoC Specification
# Finding Hidden Physical Inefficiencies in Large dLLM-MoE EP Inference

## 0. Core Goal

This PoC is intentionally **discovery-first**, not method-first.

Primary objective:

> Discover strong, reproducible mismatches between logical dLLM refinement progress and physical multi-GPU MoE execution cost, then estimate how much E2E latency could be removed if each mismatch were exploited perfectly.

Do not begin with a fixed optimization idea.

Use:
- LLaDA2.0-Flash 100B
- true EP4
- physical GPUs 0,1,2,3

The desired outcome is a new systems observation analogous in spirit to:
- Epoch: logical liveness collapses but physical work remains dense.
- MLLM: vision tokens dominate, but token count alone does not determine latency.

The method should follow from the data.

---

## 1. Hardware

Use only:
`CUDA_VISIBLE_DEVICES=0,1,2,3`

No other GPUs.

Record:
- GPU model / UUID
- topology
- CUDA / NCCL
- DeepEP
- runtime commit
- model revision

---

## 2. Model / Runtime

Primary model:
`inclusionAI/LLaDA2.0-flash`

Reuse the already validated 100B true-EP4 path:
- dense TP4 + routed EP4
- DeepEP dispatch
- 64 complete routed experts per rank
- reverse combine

Use the strongest validated static execution configuration from prior PoCs.
Do not benchmark against a weak default if a stronger existing knob is already known.

Use production-like fused kernels.

---

## 3. Baseline Principle

Before discovery measurements, reproduce a stable best-static EP4 baseline.

At minimum record:
- mini_batch_size
- submitted batch / concurrency
- generation budget
- block size
- NFE
- BCT / latency
- throughput
- peak HBM
- quality anchor

If prior mini32 remains best, use it only after reproducing the result.

All candidate oracles must compare against the **best static baseline**.

---

## 4. Discovery Philosophy

Search for places where a logical metric says:
- less work,
- easier refinement,
- more stable state,

but physical runtime cost:
- fails to fall proportionally,
- remains flat,
- or moves in the opposite direction.

Main axes:

- refinement phase
- logical liveness
- token confidence
- router entropy / confidence
- expert support size
- rows per expert
- tiny-expert fraction
- destination-rank geometry
- rank-load balance
- kernel shape
- communication payload
- kernel latency
- whole-MoE latency
- whole-request latency

Do not over-prioritize stragglers.
The goal is total latency reduction.

---

## 5. Stage 0 — High-Fidelity Trace Collection

Collect low-overhead traces across multiple real requests.

Recommended:
- GSM8K
- HumanEval
- optional MBPP

Use enough requests to cover different NFE trajectories.

For every request / block / iteration / layer record:

### 5.1 Logical refinement state
- iteration index
- normalized phase [0,1]
- early/middle/late
- decision-live rows
- masked rows
- accepted rows
- acceptance rate
- token confidence mean/p50/p90
- confidence margin

### 5.2 Routing state
- router entropy
- top-1 router probability
- top-k mass
- top-k score gap
- unique experts
- active experts
- top-k expert set
- effective expert support
- shared-expert contribution if observable

### 5.3 Fragmentation state
- rows-per-expert histogram
- mean rows/expert
- p50 rows/expert
- p90 rows/expert
- fraction of active experts with <=1 row
- <=2 rows
- <=4 rows
- <=8 rows
- number of non-empty experts

### 5.4 EP geometry
- remote fraction
- remote assignments
- remote bytes
- destination-rank set
- mean rank fanout
- rank-wise assignment vector
- rank-load CV
- critical rank
- critical-rank persistence

### 5.5 Timing / efficiency
- attention
- router
- route preparation
- dispatch
- expert-kernel time
- combine
- whole MLP
- whole block
- kernel launch count
- expert-kernel count
- GPU utilization
- SM utilization if available
- GEMM-efficiency proxy
- idle gaps

Use observer-light traces for shape/trend analysis.
Use clean runs for E2E claims.

---

## 6. Stage 1 — Reproduce Known Signals

First reproduce these on fresh traces.

### 6.1 Liveness collapse
Expected:
early live ratio high -> late live ratio low.

### 6.2 Expert-support broadening
Check whether active expert count rises early -> late.

### 6.3 Tiny-expert growth
Check whether the <=4-row expert fraction rises early -> late.

### 6.4 Coarse-stable / fine-volatile routing
Check whether:
- rank-load cosine and destination overlap stay high,
- exact top-k identity is much lower.

These are starting observations, not final contributions.

---

## 7. Stage 2 — Logical-to-Physical Mismatch Search

### Mismatch A — Liveness down, physical MoE latency flat
Measure:
- live-row ratio
- physical rows
- whole MLP latency
- expert latency

Strong example:
logical live work falls 5x but latency falls <1.5x.

This may overlap Epoch, so treat it as baseline context.

### Mismatch B — Liveness down, expert support up
Measure:
- live rows vs active expert count
- live rows vs effective expert support

This is the main "late-refinement fragmentation paradox" hypothesis.

### Mismatch C — Liveness down, rows/expert collapse
Measure:
- live ratio
- p50 rows/expert
- tiny-expert fraction

Strong signal example:
- live rows -80%
- active experts +30%
- p50 rows/expert -70%
- tiny-expert fraction +20pp

### Mismatch D — Work/FLOP proxy down, expert latency plateaus
Build a work proxy:
`sum(rows_per_expert × expert_compute_per_row)`

Compare to actual expert-kernel latency.

Look for sharp work reduction without proportional latency reduction.

### Mismatch E — Token confidence up, router entropy/support up
Measure:
- token confidence
- router entropy
- top-k mass
- expert support

Question:
Are token predictions becoming more certain while MoE routing becomes more diffuse?

### Mismatch F — Route overlap high, exact reuse value low
Measure:
- top-k overlap
- exact set match
- hidden change
- expert-output change

Mostly negative characterization unless it exposes a new compressed representation.

### Mismatch G — More work can be faster
Measure:
- physical rows
- token-expert pairs
- expert-row distribution
- latency

Identify regimes where larger, less fragmented expert batches are faster than smaller work.

### Mismatch H — Same total work, different latency
Find matched waves with similar:
- total rows
- expert pairs
- remote bytes

but very different latency.

Explain differences using:
- fragmentation
- fanout
- rank imbalance
- kernel count
- largest expert batch
- tiny-expert ratio

This may identify a better cost metric than token count.

---

## 8. Stage 3 — Joint Predictor / Matched Analysis

Targets:
- expert latency
- whole MoE latency
- whole block latency
- request-level contribution

Candidate predictors:
- logical live rows
- physical rows
- expert pairs
- active experts
- p50 rows/expert
- tiny-expert fraction
- router entropy
- top-k mass
- fanout
- rank-load CV
- remote bytes
- kernel count

Use:
- Spearman
- Pearson where appropriate
- matched-pair comparisons
- stratification by physical M / ready-pool size

Do not claim causality from raw correlation.

Goal:
identify which physical-shape variables explain latency after controlling for total work.

---

## 9. Stage 4 — Post-Compaction Sensitivity

Do not implement Epoch.

Construct a hypothetical live-row compaction model.

For each real wave:
1. retain only decision-live rows;
2. preserve the original expert assignments of those rows;
3. recompute:
   - live rows
   - active expert support
   - rows/expert
   - tiny-expert fraction
   - rank fanout
   - predicted expert kernel shape.

Question:

> Does liveness compaction remove work but make expert fragmentation more severe?

Compare before vs hypothetical after compaction:
- work removed
- active expert support retained
- p50 rows/expert
- tiny-expert fraction
- kernel count
- estimated kernel efficiency

This is sensitivity analysis, not a measured Epoch speedup.

---

## 10. Stage 5 — Strong-Signal Gates

Promote a discovery if at least one holds.

### S1 — Large physical mismatch
Logical work decreases >=50% but target latency decreases <=20%.

### S2 — Opposite-direction behavior
For example:
live rows down, active experts up, tiny-expert fraction up.

### S3 — Shape predicts latency better than work count
For example:
tiny-expert fraction or p50 rows/expert explains latency materially better than token/expert-pair count.

### S4 — Post-compaction second-order bottleneck
Hypothetical live-row compaction removes large work but leaves/creates a residual physical inefficiency with >=10% E2E oracle.

Only strong signals proceed to method generation.

---

## 11. Stage 6 — Generate Methods from the Data

Do not preselect a method.

For each promoted signal, generate at least three method families and calculate oracle headroom first.

### If fragmentation is strong

F1 — Expert-major live-work coalescing
- collect live rows for the same expert across independent requests/waves;
- execute larger grouped expert GEMMs.

F2 — Fragmentation-aware wave composition
- group ready requests whose live expert supports overlap;
- maximize rows per active expert, not just total rows.

F3 — Bounded-delay expert queues
- allow a tiny wait window to accumulate same-expert live rows;
- include queueing delay in E2E.

F4 — Tiny-expert kernel specialization
- if 1–4-row kernels are disproportionately inefficient, evaluate specialized kernels / kernel dispatch.

### If confidence-routing mismatch is strong

C1 — Confidence-conditioned expert budget
- high token confidence + diffuse routing may not justify full top-k.
- audit REFLEX/TEAM/DES prior art.

C2 — confidence-triggered reduced expert scope/shared-expert use
- only if architecture and quality support it.

### If shape, not work count, predicts latency

P1 — Shape-aware EP scheduler
- predict cost from fragmentation/fanout/load instead of token count.

P2 — Expert-support-aware request grouping
- batch requests whose live expert support overlaps.

P3 — Kernel-shape-aware microbatch construction
- choose groups to maximize useful rows per active expert.

---

## 12. Stage 7 — Oracle Tournament

For every generated method candidate estimate against the best static baseline:

- perfect removable E2E latency
- realistic feasible E2E
- quality risk
- waiting/queueing cost
- runtime overhead
- prior-art risk

Create:

| Candidate | Signal | Perfect E2E | Feasible E2E | Quality risk | Prior-art risk | Decision |
|---|---|---:|---:|---|---|---|

Gates:
- <5% -> KILL
- 5–8% -> WEAK
- 8–12% -> PROMISING
- 12–20% -> STRONG
- >20% -> VERY STRONG

Do not implement a complex candidate below 8% credible headroom.

---

## 13. Special Oracle — Perfect Fragmentation Removal

If fragmentation is confirmed:

For each wave:
1. preserve exactly the same expert assignments;
2. assume all rows for each expert can be executed in one perfectly packed GEMM;
3. remove per-fragment launch/packing overhead;
4. keep required expert FLOPs and required communication.

Compare:
- actual expert latency
- ideal expert-major packed latency

Map to request E2E.

Question:
> How much latency is caused by fragmentation rather than useful expert compute?

---

## 14. Special Oracle — Cross-Request Same-Expert Coalescing

Using independent requests and scheduling windows:
- 0 ms
- 0.1 ms
- 0.5 ms
- 1 ms
- 2 ms

Estimate:
- rows accumulated per expert
- expert GEMM size growth
- kernel-count reduction
- added wait
- net latency/throughput gain

Only run deeply if fragmentation is strong.

---

## 15. Special Oracle — Post-Epoch Residual

Using hypothetical live-row compaction:

Estimate:
- liveness-work reduction
- residual fragmentation cost

Question:

> If Epoch-like compaction were already solved, how much additional E2E could a fragmentation-aware method recover?

Preferred conceptual composition:

`Epoch removes dead work -> ours makes the remaining live sparse work efficient`

Do not claim measured Epoch numbers.

---

## 16. Online Serving

Do not implement full online serving unless a strong coalescing/fragmentation signal appears.

If it does, analyze whether heterogeneous request phases naturally improve:
- same-expert row accumulation
- expert-major coalescing
- fragmentation smoothing

Every serving oracle must include queueing delay.

---

## 17. TP4 Control

If a strong fragmentation phenomenon appears under EP4, collect a smaller TP4 control trace.

Question:
> Is the phenomenon specifically amplified by EP sparse ownership, or is it generic MoE tiny-GEMM fragmentation?

If equally strong in TP4, EP-specific novelty is weaker.

---

## 18. Prior-Art Audit

For promoted methods audit:
- Epoch
- TEAM
- REFLEX
- DES
- DICE
- generic MoE batching
- expert-major batching
- grouped GEMM / fused MoE
- SGLang / DeepEP scheduling
- continuous batching
- expert queues / coalescing
- tiny-GEMM specialized kernels

Do not claim novelty for generic batching unless dLLM refinement creates a distinct physical problem.

---

## 19. Required Figures

1. decision-live fraction vs phase
2. active expert count vs phase
3. p50 rows/expert vs phase
4. <=4-row expert fraction vs phase
5. router entropy vs token confidence
6. top-k mass vs phase
7. expert latency vs expert-pair count
8. expert latency vs tiny-expert fraction
9. matched waves: same total work, different latency
10. route overlap vs exact-set match
11. hypothetical post-compaction rows/expert
12. post-compaction tiny-expert fraction
13. fragmentation-removal oracle
14. candidate oracle comparison
15. if strong: coalescing benefit vs wait window

---

## 20. Required Reports

Create:
- `reports/00_environment.md`
- `reports/01_trace_summary.md`
- `reports/02_known_signal_reproduction.md`
- `reports/03_logical_physical_mismatches.md`
- `reports/04_latency_predictor_analysis.md`
- `reports/05_post_compaction_sensitivity.md`
- `reports/06_discovery_ranked_findings.md`
- `reports/07_method_candidates.md`
- `reports/08_candidate_oracles.md`
- `reports/09_prior_art_audit.md`
- `reports/10_best_candidate_live_poc.md`
- `reports/final_decision.md`

---

## 21. Final Labels

- `NO-NOVEL-SIGNAL`
- `CHARACTERIZATION-SIGNAL`
- `HOLD-CANDIDATE`
- `STRONG-CANDIDATE`
- `VERY-STRONG-CANDIDATE`

Definitions:
- NO-NOVEL-SIGNAL: no strong mismatch beyond known effects.
- CHARACTERIZATION-SIGNAL: interesting mismatch, but novel oracle <8%.
- HOLD-CANDIDATE: >=8% credible E2E headroom.
- STRONG-CANDIDATE: >=12% credible oracle and plausible novelty.
- VERY-STRONG-CANDIDATE: live prototype >=10–15% E2E/throughput gain over strongest baseline, bounded quality, complementary to Epoch.

---

## 22. Final Questions

1. Which logical refinement metrics change most strongly early->late?
2. Which physical metrics move unexpectedly weakly or oppositely?
3. Does active expert support broaden as liveness falls?
4. Does rows/expert collapse?
5. Does expert latency plateau despite reduced useful work?
6. Which metric predicts expert/MLP latency best after controlling for total rows?
7. Is late-refinement fragmentation real and reproducible?
8. Would live-row compaction worsen fragmentation?
9. Is there a post-Epoch residual >=8–10%?
10. Which mismatch yields the strongest novel oracle?
11. Does any candidate reduce total latency, not only straggler time?
12. Is the signal EP-specific or generic MoE execution?

---

## 23. Final Instruction

Do not search for a predetermined optimization.

Search for:

> a surprising, reproducible mismatch between logical dLLM progress and physical EP execution cost.

The most interesting current hypothesis is:

`refinement progresses -> live rows collapse -> expert support remains broad or broadens -> rows/expert shrink -> tiny-expert fragmentation increases -> useful work falls faster than physical latency`

But this is only a hypothesis.

If unsupported, discard it and continue discovery.
