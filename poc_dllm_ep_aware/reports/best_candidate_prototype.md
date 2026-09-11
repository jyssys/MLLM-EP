# Best candidate prototype decision

## Candidate

The largest nontrivial bound is Candidate F, perfect-future early commitment of
TEAM speculative branches: **6.28% request E2E**.  Candidate E, EP-aware
speculation width, is the cleanest implementable variant but has only a 4.01%
generous physical-cost bound.

## Why no prototype was implemented

The spec requires method implementation only after a material oracle.  Both
bounds fail:

- F is below the 8% promising threshold before prediction/verification cost;
- E is below the 5% kill gate even while assuming unchanged NFE;
- width selection is an obvious static/runtime knob, so a small gain would be
  incremental rather than a new method space;
- communication becoming 2× faster cuts E's bound to about 2.01%;
- REFLEX/DES cannot provide clean cross-family generalization because their
  official execution contracts were unavailable and the ports failed quality.

The plot named `15_original_vs_ep_aware.png` intentionally shows no live bar.
Zero means “not implemented due to failed gate,” not a measured 0% speedup.

## Cheapest future falsification, if revisited

Only a faithful public REFLEX/DES implementation plus production ragged DeepEP
could change this decision.  The decisive test would be a quality-matched
EP4 action trace showing at least 10% additional request headroom over the
native method.  Parameterizing TEAM width alone is not recommended.
