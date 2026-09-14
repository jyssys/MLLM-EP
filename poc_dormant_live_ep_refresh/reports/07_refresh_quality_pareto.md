# Dormancy × refresh quality/economics Pareto

No tested point is both trajectory-safe and economically material.

- Larger K defers more work but rapidly reduces exact sequences and perturbs
  acceptance order.
- H=2 is semantically more conservative than H=1, but H2/K8 provides only
  3.98%/4.21% feasible E2E and still changes 23/32 GSM and 15/32 HumanEval
  sequences, with NFE 59/106 versus 66/86.
- Route-change gating improves exact-sequence counts to 17/32 and 20/32 but
  leaves only 0.63%/0.54% E2E before implementation overhead.
- Full H1 future-aware removal has a 6.95%/7.27% fixed-floor ceiling and cannot
  itself cross the 8% HOLD gate, even if quality and implementation were free.

The benchmark score sometimes improves under perturbation.  This is sampling
noise/trajectory change, not proof that stale reuse is safe: the per-sample
outputs, NFE, and acceptance times demonstrate substantial intervention.

Consequently there is no live Pareto point to promote.  A full variable-row
DeepEP runtime or custom kernel would be evidence-inappropriate.
