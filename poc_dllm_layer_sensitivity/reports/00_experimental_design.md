# Experimental design

The first live pass will revalidate clean best-static latency and quality. A
separate observer-heavy pass will capture layer update and component cost maps.
Causal policy sweeps keep the model loaded and execute matched dataset states
sequentially, avoiding model-load noise while preserving one exact generation
trajectory per intervention. The first affected wave in a phase has the same
pre-state as baseline; later divergence is treated as trajectory effect rather
than paired current-step evidence.

Full-layer bypass is a diagnostic upper bound. Routed-MoE bypass and stale
routed output are the primary MoE-specific probes. None of these intervention
runs is a performance result because the original work is still executed; cost
removal is estimated only by joining causal safety with separately measured
clean component latency.
