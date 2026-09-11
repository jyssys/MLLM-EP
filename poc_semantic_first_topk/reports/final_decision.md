# Final decision

## Status

**NO-GO** for semantic-first + EP-critical-path adaptive Top-K as a distinct
paper-level EP systems direction.

## Gate summary

- Semantic work-removal gate: **PASS**.  Strict held-out ceiling is 27.15%
  visual assignments at unchanged 32/32 short outputs and 24/32 score.
- Full integer-K gate: **PASS but incremental**.  It is safer than coarse K,
  but fixed K=6 recovers 82.9% of the assignment reduction.
- Contribution-oracle gate: **NO new selector signal**.  Computing actual
  expert outputs does not materially improve the task frontier.
- Same-budget EP count gate: **PARTIAL PASS**.  Max-rank reduction is 1.81x
  and 1.56x at 10/20%, but only 1.33x at 30%.
- Real DeepEP gate: **combined candidate passes, EP increment fails**.  At
  layer 24, 30% semantic+EP projects 13.14% TTFT versus 10.55% semantic-only.
- Joint quality/speed gate: **FAIL**.  The held-out 20% refined policy is
  32/32 exact but projects 9.92%; the 30% policy projects 13.14% but is 31/32.
- Generality: **FAIL**.  EP-specific projected increments across camera
  layers 4/24/44 are 0.53/2.51/0.57 percentage points.
- Prior-art differentiation: **FAIL**.  MACS already makes visual semantic
  load and EP straggler mitigation a central coupled problem; AnyExperts and
  MoDES cover Stage 1.

## Exact kill reason

The new physical-rank refinement produces a causal load/latency effect, but
not a material independent request-level opportunity.  Almost all headline
gain belongs to generic semantic expert skipping, which has both a trivial
fixed-K control and direct prior-art collisions.  The remaining EP-specific
2.5-point best representative projection is optimistic and layer-local.

## Recommendation

Do not implement production ragged-K, dynamic rank-load aggregation, or a
custom kernel for this direction.  Retain full-K versus coarse-K and
assignment-count-versus-latency as useful negative controls for future MoE
work.  A successor should require a new causal variable not already expressed
as semantic token capacity plus current EP load.
