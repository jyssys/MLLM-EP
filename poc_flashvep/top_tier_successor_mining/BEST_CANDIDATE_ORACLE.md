# SERE — measured recovery, not an invented perfect oracle

Primary actual EP requests:128 questions/task,three randomized within-cohort
policy repetitions,cold images,natural EOS,official unchanged task scores.
Pair each treatment with its own vanilla; do not mix HF and EP quality numerators.

| B1 comparison | ChartQA | GQA |
|---|---:|---:|
| Aggressive S2/rho.5 quality delta vs vanilla,pp |−19.53125|−7.03125|
| Existing S4/rho.5 quality delta,pp |0|−.78125|
| Observed loss recovered by S4 |100%|88.8889%|
| S4 paired E2E reduction vs aggressive S2 |1.30936%|.21316%|
| Paired cohort95%CI for that E2E reduction |[.87149,1.77927]%|[−.16757,1.03343]%|

This is an **executed trivial-fix comparator**,not an unmeasured ideal successor.
At approximately the same port speed it recovers19.53pp/6.25pp. Quality point
estimates do not establish noninferiority to vanilla: use image-cluster intervals.
S4 itself remains13.51%/8.85% slower than the corresponding stock EP vanilla.

We do not infer a10% additional quality-matched E2E successor opportunity from
the slowdown. The algorithmic no-op cost floor shows that much of this port
overhead exists without intended rerouting. Removing instrumentation/port costs
is not a new research contribution.

PERFECT_NONTRIVIAL_SUCCESSOR_E2E: NOT_IDENTIFIED/NOT_ESTIMATED.
FEASIBLE_NONTRIVIAL_SUCCESSOR_E2E: NOT_ESTABLISHED.
Unknown is not0%; it cannot satisfy a positive headroom gate. There is no basis
to spend Kimi/prototype budget on a candidate whose observed loss is already
mostly fixed by an existing S/rho setting.
