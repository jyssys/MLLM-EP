# 02 — Phase-specific acceptance aggression

The stock threshold decoder already varies accepted token count `k_t`.
The opt-in intervention changes only its confidence threshold, never model
weights, top-k routed experts, MoE outputs, EP ownership, or routing.
`phase:early,middle,late` uses remaining-mask progress within a ready block:
early `<0.25`, middle `<0.75`, late otherwise. It is a deployable generic
semantic-state feature, **not** an EP-cost signal. Each candidate was actually
rolled out through final output; no one-step savings were added.

| n=32 phase thresholds | Correct | NFE | Clean BCT (s) |
|---|---:|---:|---:|
| 0.9/0.9/0.9 hook identity pilot | 13 | 102 | 7.704 |
| 0.9/0.8/0.9 | 12 | 96 | 7.133 |
| 0.9/0.825/0.9 | 12 | 99 | 7.351 |
| 0.825/0.9/0.9 | 12 | 103 | 7.612 |
| 0.9/0.9/0.825 | 12 | 102 | 7.558 |
| **0.9/0.825/0.825** | **13** | **95** | **7.115** |
| 0.825/0.8/0.9 | 12 | 92 | 7.050 |
| 0.8/0.9/0.9 | 12 | 102 | 7.965 |

Only the bold schedule preserved both 13/32 bounded score and all 32 paired
correct/incorrect states. Because one bounded example equals 3.125 pp, it was
promoted rather than declared quality-safe from this screen. Early aggression
was not clearly economical here: 0.825/0.9/0.9 increased NFE from 102 to 103.

| Matched promoted task | Feasible mini | Correct baseline→phase | NFE baseline→phase | Direct clean BCT baseline→phase | Paired 95% quality interval |
|---|---:|---:|---:|---:|---:|
| GSM8K n=512 | 32 | 68→67 | 1,300→1,190 | 100.336→87.361 s, three-restart medians (-12.93%) | [-1.367,+0.977] pp |
| GSM8K full n=1,319 | 32 | 144→146 | 3,226→2,965 | 227.782→212.089 s, one restart (-6.89%) | [-0.455,+0.758] pp |
| HumanEval full n=164 | 16 | 13→14 | 672→613 | 47.155→45.294 s, one restart (-3.95%) | [0,+1.829] pp |

The n=512 speed ranges overlap across independent restarts; the full-set
results are each one policy pair. Observer-heavy decision traces confirm
accepted-token counts and phase progression but inflated BCT by +57% for
baseline and about +112% for this phase schedule. The panels
[baseline](../figures/observer_state_ep_observer_base_gsm32.png) and
[phase](../figures/observer_state_ep_observer_phase_gsm32.png) are structural
plots, not clean speedup evidence. Full-trajectory strings changed on about
half the samples despite similar benchmark scores, so phase-specific semantic
risk is not established by n=32 current-step matching.
