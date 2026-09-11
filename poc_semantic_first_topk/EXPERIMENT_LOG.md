# Experiment log

## 2026-09-11 — provenance and prior evidence

- GitHub `jyssys/MLLM-EP` remote HEAD: `e228b44cec1f5ffa32953e52540086423f84f33f`.
- Local project baseline inspected first: `f0dc8372ec0d1d2e10347c82e630b84430c5dc5f`.
- Committed predecessor Top-K PoC: `209354e8a58301493b6e262e0d4ef2045baec661`.
- New isolated branch/worktree avoids the dirty original checkout.
- Physical GPUs 4–7: H100 80GB HBM3, all-to-all NV18; no task launch exposed GPUs 0–3.
- Primary checkpoint: Qwen3-VL-30B-A3B-Instruct snapshot
  `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`.

The predecessor established a real 8K/16K MoE regime but killed tail-first
Top-K: the 10% tail policy projected only 6.94% TTFT and harmed four-token
agreement.  Its 54 real image/layer captures are reused only as offline input;
all new semantic sensitivity, full-model quality, and DeepEP timing results
are separately labeled.

## E01 — real-output allocation pilot

- Input: three real Qwen3-VL image captures, layers 4/24/44.
- Conditions: full integer K, coarse K, router risk, actual weighted expert
  contribution risk, retained-weight and renormalized output.
- Observation: contribution risk did not dominate router risk; renormalizing
  retained top-K weights substantially increased local representation error.
- Child: determine whether task-level sensitivity contradicts local rel-L2.

## E02 — GQA task-conditioned sensitivity

- Eight GQA calibration requests, all 48 language MoE layers, 16 spatial
  groups, K=1..7 perturbations, eager experts.
- Stock repeat noise was near numerical zero.
- A multiple-choice dynamic program was required because error versus K was
  non-monotonic; a greedy K descent is not a valid oracle.
- Child: validate composed schedules in the full model and on OCR/chart data.

## E03 — GQA composed full-model quality

- Eight requests, request-conditioned full/coarse schedules at 10/20/30%,
  16-token greedy generation, benchmark answer check.
- All six schedules preserved the short greedy output and aggregate benchmark
  score on this small in-sample cohort.
- Boundary: this is a label-conditioned offline oracle, not a deployable
  selector and not a performance run.

## E04 — ChartQA sensitivity and composed quality

- Eight ChartQA calibration requests with the same full-K perturbation and
  full-model validation protocol.
- Full grid improved the isolated additive-KL oracle at every tested budget,
  but full-model interactions made the ordering less stable.
- Up to roughly 27.5% realized vision-assignment reduction preserved the
  aggregate benchmark score in this calibration cohort.

## E05 — semantic-first same-budget EP refinement

- The refinement decrements a suffix branch on the current critical rank and
  restores one omitted suffix branch off the critical rank, preserving the
  exact assignment count.  Final router-risk is constrained to 1.05x the
  semantic schedule.
- At 10%, max-rank reduction exceeded semantic-only by about 1.65x on GQA and
  1.85x on ChartQA.  At 20/30%, the ratio fell to roughly 1.4x/1.27x.
- Child: test whether the low-budget count advantage becomes real DeepEP
  latency; do not extrapolate count to TTFT.

## E06 — held-out selector pilot

- Eight unseen GQA/ChartQA requests.
- Compared calibration-aggregate semantic schedules, fixed K, router-mass,
  and a true expert-contribution oracle that first computes every expert.
- The global semantic schedule preserved all eight short greedy outputs at
  ~26% realized assignment reduction.  Several much simpler policies were
  similarly stable, so the cohort was expanded before drawing a selector
conclusion.

## E07 — held-out 32-request selector/trivial attack

- Sixteen unseen GQA plus sixteen unseen ChartQA requests; stock, global
  full/coarse 20/30%, fixed K=6, router-mass and true contribution oracles.
- Global full 30% realizes 27.15% visual assignment removal with 32/32 short
  output agreement and unchanged 24/32 score.  Fixed K=6 realizes 22.52% with
  identical quality, recovering 82.9% of that reduction.
- Contribution and router policies each change one output and their apparent
  +1 benchmark answer is treated as sampling noise, not improvement.
- Failed assumption: a substantially better semantic selector is the missing
  systems opportunity.

## E08 — actual DeepEP HT replay

- Actual Qwen hidden/routes/weights/expert weights; EP4; M=8192; one large
  invocation; five warmups; 30 randomized/interleaved reps per policy.
- Three-image layer-24 median global semantic 30%: MoE -17.06%, projected TTFT
  -10.61%.  EP refinement: MoE -21.09%, TTFT -13.12%.
- The new Stage-2 increment is only 2.51 points.  At camera layers 4/24/44 it
  is 0.53/2.51/0.57 points, showing a middle-layer concentration.
- Child: remove the tiny assignment-count mismatch caused by truncating the
  last repeated route, and directly validate refined quality on unseen data.

## E09 — high-removal quality ceiling

- Full semantic 40/50%, coarse, fixed K=4 and actual contribution oracle on
  the same 32 held-out requests.
- Full 40/50% realizes 35.81/45.04% reduction and retains the aggregate 24/32
  score, but changes one/two short outputs.  Fixed K=4 reaches the same 45%
  work regime and changes two outputs.
- The conservative output-preserving ceiling remains 27.15%; approximately
  36% is an observed score-only ceiling, not a statistically established safe
  policy.

## E10 — held-out EP-refinement quality

- Same-budget global refinement at 20/30% on all 32 held-out requests.
- 20% is 32/32 exact and 24/32, but its total replay projection is only 9.92%.
- 30% is 31/32 exact and 25/32, while semantic-only is 32/32.  Its apparent
  repaired baseline error is not evidence of quality improvement.
- The strict quality and >=12% projected-speed gates do not intersect.

## E11 — exact-assignment replay

- Repeated the primary three-image layer-24 replay at M=8177, exactly 37
  copies of each 221-token captured route.  Every semantic/refined pair has
  identical visual and total assignment counts.
- Exact-budget 10/20/30% semantic TTFT projections are 4.28/7.60/10.55%;
  refined values are 5.81/9.92/13.14%.
- This confirms the effect and the negative decision: the 30% EP increment is
  2.59 points, while the 20% strict-quality point is below the 12% gate.

## Evidence rules

- Full-model masking still computes all experts: quality evidence only.
- Expert-contribution selection sees computed outputs: offline oracle only.
- Captured-layer DeepEP replay is operator timing, not observed request E2E.
- TTFT uses an explicitly labeled Amdahl projection from the prior clean
  16K request and observer-assisted stage attribution.
