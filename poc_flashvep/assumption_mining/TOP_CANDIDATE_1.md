# Candidate 1: token-scoped combine release

## Counterfactual

Current DeepEP exposes one `EventOverlap` and the receiver waits before the
expert input is visible. The counterfactual is to expose rank/token-scoped
readiness so independent token groups can enter downstream work while late
remote contributions continue.

## Why it was considered

This attacks a genuine structural assumption (one invocation-wide readiness
boundary), rather than a knob. It could support token-group dependency graphs,
deadline-aware partial release, or a representation that carries unresolved
remote contributions.

## Adversarial result

The only measured large dependency anomaly is the DeepEP dispatch wait. Exact
route/input replay removes it, and the direct request-level removable mass is
1.09%. The earlier 11--24% numbers were rank/wave or optimistic projections,
not a request join. Partial/speculative paths additionally require nonlinear
downstream verification and are adjacent to SpecMoE, ScMoE and FarSkip.

**Status: CLOSED before GPU.** A future semantic study is justified only if a
new trace demonstrates repeated token groups with >=20% request-level ready
slack.
