# Checkpoint H1 (~1 hour)

- Strongest positive: fixed text c8→c16 changes layer-local MoE p50 by 19.8% and request p50 by 17.8%.
- Strongest negative: fanout/load hierarchy still has zero incremental held-out signal; prior fixed-tail direct E2E cap is 1.09%.
- Open anomaly: event-wait share rises to 14.6% at text c16, but this may be scheduler load rather than a new EP mechanism.
- Fresh live traces: mixed c8, high c16, text c8/c16, vision-hi c8, mixed c2 (text c2 still completing).
- Next: finish low-concurrency fixed-text control, compute c2/c8/c16 matched table, then decide whether C1/K1 has enough direct headroom for a bounded causal probe.
