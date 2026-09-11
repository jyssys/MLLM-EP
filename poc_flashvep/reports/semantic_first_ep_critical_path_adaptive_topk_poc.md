# Semantic-First + EP-Critical-Path Adaptive Top-K PoC

## Executive decision

**Original EP4 final status: NO-GO for a distinct EP-critical-path paper
direction. Post-hoc recovery status: HOLD for one bounded real-EP8 validation;
see the recovery addendum linked at the end.**

Semantic-first visual expert reduction is real: a calibration-aggregate full
K=1..8 schedule removed 27.15% of held-out visual expert assignments while
preserving all 32 short greedy outputs and the 24/32 aggregate GQA/ChartQA
score.  Real four-rank DeepEP HT replay translated a 29.46% schedule into a
16.96% median MoE reduction and a 10.55% optimistic Amdahl TTFT projection.

The narrow new claim does not survive.  Same-budget EP refinement raises the
layer-24 MoE reduction to 21.13% and projected TTFT to 13.14%, but adds only
2.59 percentage points of projected TTFT to the deployable semantic schedule.
Across camera layers 4/24/44, that EP-specific increment is 0.53/2.51/0.57
points.  Fixed K=6 already recovers 82.9% of the strict quality-safe assignment
reduction, and MACS already makes visual semantic load and physical EP
stragglers a central joint objective.  A production ragged-K runtime cannot
create the missing novelty or headroom and was therefore not implemented.

## Provenance and isolation

- Local `jyssys/MLLM-EP` baseline inspected first:
  `f0dc8372ec0d1d2e10347c82e630b84430c5dc5f`.
- GitHub remote HEAD observed on 2026-09-11:
  `e228b44cec1f5ffa32953e52540086423f84f33f`.
- Committed predecessor tail-first PoC:
  `209354e8a58301493b6e262e0d4ef2045baec661`.
- This work uses an isolated worktree/branch and does not modify the dirty
  original checkout.
- Contract SHA-256:
  `a1f7f33657f789fd07d8ffc29029813b1328dd903af8a90a043b8825801e5a80`.

## Hardware and runtime

- Model: Qwen3-VL-30B-A3B-Instruct, BF16 snapshot
  `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`.
- Physical GPUs: only 4,5,6,7; four H100 80GB HBM3; all-to-all NV18.
- Replay: four NCCL ranks, EP4, DeepEP high-throughput, `num_sms=20`,
  FusedMoE/Triton experts, one large invocation (primary exact-budget M=8177;
  selector/layer controls M=8192).
- Quality: full Qwen model with eager experts; the mask changes outputs but
  still computes all eight experts, so timing is never used as speed evidence.
- Clean predecessor regime used for Amdahl mapping: TTFT 384.937 ms and
  critical MoE span 239.397 ms (62.191%).
- Meaningful fresh GPU experiment wall: 5,084.5 s (1.412 h on four GPUs,
  5.649 GPU-hours), excluding burn and CPU-only capture analysis.

The installed vLLM does not contain a tuned H100 configuration for local
E=32/N=768 experts and warns that it uses a default MoE config.  All paired
policies share this path.  Absolute numbers are not generalized; randomized
relative results are the evidence.

## Evidence hierarchy

1. **Captured logical oracle:** nine real images x six early/middle/late
   layers, including actual hidden states, top-8 routes/weights and individual
   expert outputs (54 cases).
2. **Task-conditioned oracle:** eight GQA + eight ChartQA calibration requests,
   all 48 MoE layers, 16 spatial groups, every K=1..7 perturbation.
3. **Held-out quality:** 16 GQA + 16 ChartQA requests, answer-token metrics,
   16-token greedy output, and task answer score.
4. **Actual operator replay:** three image routes at layer 24 plus camera
   layers 4/44, five warmups, 30 randomized repetitions, rank-critical
   same-device CUDA events.
5. **Amdahl projection:** measured replay reduction x 62.191% prior clean
   request share.  It is not observed request E2E.

## Stage 1 — semantic global compression

### Full integer K matters locally

The full K grid gives a better quality frontier than `{1,2,4,6,8}`.  On
ChartQA calibration, at exactly matched assignment budgets, its additive
answer-KL is lower at every 10–50% target.  In the held-out full model, full
20/30% schedules preserve 32/32 short outputs; both coarse schedules preserve
31/32 and lose one benchmark answer.

Error is not monotone in K: only 41.4% of GQA spatial-group cases obey
`error(K=1) <= error(K=7)`.  An exact multiple-choice dynamic program was
needed; a greedy K decrement is not a valid quality oracle.

### Held-out quality frontier

| Policy | Realized visual assignment drop | Short greedy exact | Score |
|---|---:|---:|---:|
| Stock | 0.00% | 32/32 | 24/32 |
| Full semantic 20% | 18.61% | 32/32 | 24/32 |
| Full semantic 30% | **27.15%** | **32/32** | **24/32** |
| Coarse semantic 30% | 26.19% | 31/32 | 23/32 |
| Fixed visual K=6 | 22.52% | **32/32** | **24/32** |
| Full semantic 40% | 35.81% | 31/32 | 24/32 |
| Full semantic 50% | 45.04% | 30/32 | 24/32 |

The unchanged aggregate score at 40/50% is not called quality-safe: generated
answers changed and 32 requests cannot rule out compensating flips.  The
strict output-preserving ceiling is 27.15%; the observed score-only ceiling is
about 36%.

### Selector diagnosis

Actual weighted contribution `||g_ie E_e(h_i)||` is not the missing signal.
It explicitly computes every branch, yet at 30% changes one held-out output
and does not establish a score advantage.  Router-mass selection behaves
similarly.  The semantic task oracle is better at preserving exact output, but
fixed K=6 recovers `22.52/27.15 = 82.9%` of its work reduction at the same
score and exact-output rate.

Keeping retained weights unchanged is safer than renormalizing.  In the
captured output atlas, 10% router-risk removal has 16.54% median combined
relative L2 without renormalization versus 29.03% with renormalization.

**Stage-1 gate:** G0/G1 pass empirically, but Stage 1 fails non-triviality and
is already covered by adaptive expert-allocation/skipping work.

## Stage 2 — same-budget EP refinement

The refinement only swaps omissions: current critical-rank suffix assignments
are removed and equally many off-critical omitted suffix assignments are
restored under a 1.05x router-risk slack.  Total assignments remain exact.

### Logical count effect

Across 54 captured cases, the full-K router schedule's max-rank reduction is
8.12/16.19/25.10% at 10/20/30%.  Refinement increases it to
14.71/25.25/33.28%, or 1.81x/1.56x/1.33x.  Thus the requested 1.5x gate holds
at low and medium budgets but not at the quality-safe 30% point.

### Actual DeepEP effect

Primary three-image layer-24 medians:

| Target | Semantic MoE / TTFT | Refined MoE / TTFT | EP-only TTFT increment |
|---:|---:|---:|---:|
| 10% | 6.89% / 4.28% | 9.34% / 5.81% | +1.53 pp |
| 20% | 12.21% / 7.60% | 15.94% / 9.92% | +2.32 pp |
| 30% | 16.96% / 10.55% | **21.13% / 13.14%** | **+2.59 pp** |

These primary values use M=8177 (37 exact copies of a 221-token route), so
semantic and refined assignment totals are exactly identical.  At M=8192,
route-specific router/contribution policies see +3.69/+2.78 points
from refinement, still far below an independent 8–12% request opportunity.

Layer generality is weak.  At camera layers 4/24/44, the global 30% schedule
projects 11.18/10.61/12.25% TTFT reduction, and refinement projects
11.71/13.12/12.82%; the EP-specific increment is only
0.53/2.51/0.57 points.  The strong middle-layer value is not a uniform
all-layer execution principle.

## Decision gates

| Gate | Result | Decision |
|---|---|---|
| G0 contribution/semantic oracle safety | >=27.15% strict work reduction | PASS |
| G1 Stage-1 work reduction | 27.15%, score and output preserved | PASS |
| G2 Stage-2 value | Count ratio >=1.5x only at <=20%; 30% ratio 1.33x; exact-budget TTFT increment 2.59 pp | **FAIL as material EP contribution** |
| G3 replay | Combined candidate 13.14% projected TTFT at layer 24 | PASS for combined operator candidate |
| G4 real quality | Semantic 30% is 32/32; refined 30% is 31/32; refined 20% is 32/32 | **FAIL jointly with speed gate** |
| G5 prior-art | MACS already couples visual semantic load and EP straggler control | **FAIL** |
| Trivial-fix | Fixed K=6 recovers 82.9% of strict Stage-1 work reduction | **FAIL** |

The held-out refinement makes the quality/speed intersection decisive: its
20% point preserves 32/32 outputs but projects 9.92%, whereas its 30% point
projects 13.14% but preserves 31/32.  Semantic-only 30% remains 32/32.

The total candidate crossing 12% does not override G2/G5: roughly 10.6 points
come from generic semantic skipping, while the distinct EP refinement adds
about 2.5 points in its best representative layer and much less in early/late
layers.

## Prior-art conclusion

AnyExperts and MoDES already establish token/modality-adaptive real expert
allocation.  ACE is directly adjacent to the response/contribution proxy.
Most importantly, MACS uses entropy-weighted visual semantic load,
modality-adaptive capacity and local semantic rerouting/drop specifically for
MLLM EP straggler mitigation on Qwen/Kimi.  Preserving the exact assignment
count and original prefix routes is a different primitive, but the empirical
residual is too small to establish a new problem or superior frontier.

ReaLB is adjacent rather than identical: it reduces overloaded-rank work by
precision instead of omission.  That does not restore novelty to the tested
two-stage policy.

## Why no production variable-K integration

The contract permits minimal runtime work only after the quality, replay, and
novelty gates.  G2 and G5 fail before integration.  The operator projection is
already optimistic because it omits semantic scoring, rank aggregation,
selection, mask construction, and production ragged-K overhead.  Implementing
those costs cannot turn a 0.5–2.5 point EP-specific opportunity into the
required material headroom.

## Answers to the research questions

1. **Can semantic-first allocation remove substantial work?** Yes: 27.15%
   strict held-out visual assignment reduction on this small cohort.
2. **Does full K=1..8 matter?** Yes for output stability versus coarse K, but a
   fixed-K baseline recovers most work reduction.
3. **Is the selector the fundamental limitation?** No evidence: the expensive
   actual-contribution oracle is not materially better than router risk.
4. **Does EP refinement help at the same budget?** Yes causally in max-rank
   count and DeepEP time, but only +2.59 projected TTFT points at the primary
   30% policy, with one held-out output change, and less in early/late layers.
5. **Is the narrow direction paper-worthy?** No: insufficient incremental
   headroom, weak layer generality, trivial Stage-1 baseline, and a close MACS
   collision.

## Final recommendation

Do not build the semantic-first + EP-critical-path ragged-K runtime as a new
paper direction.  The reusable system facts are: full integer K has a better
quality frontier than a coarse grid; max-rank row count overstates latency
benefit; and task outputs can be robust despite large local hidden error.
Those facts may be useful as controls in future work, but do not justify this
method family without a new causal axis beyond semantic capacity and current
rank load.

Detailed tables and all 14 required figures are under
`poc_semantic_first_topk/results/final_analysis/`.

## Post-hoc recovery addendum (2026-09-11)

The earlier `31/32` refinement result was token-sequence identity, not task
accuracy. Official-style benchmark rescoring gives stock 25/32 and refined
30% 26/32, with zero negative accuracy flips. An exact offline load oracle
also finds that EP-aware incremental max-rank reduction grows substantially at
virtual EP8, rather than remaining near 3%. These results do not overturn the
prior-art/triviality concerns or establish EP8 latency, but they do justify one
bounded real EP8 validation before permanently closing the scale-dependent
variant. Full definitions and evidence boundaries are in
`poc_flashvep/reports/semantic_first_ep_recovery_checks.md`.
