# Dependency graph

- Captured top-k → per-token fanout → DeepEP layout/dispatch/combine: **CONDITIONAL** (routing geometry is upstream, but the online trace is observational).
- Cross-request invocations: **CROSS_REQUEST_INDEPENDENT** at the model boundary; scheduler timing and worker resource contention remain confounders.
- Fanout → T_MoE: **CONDITIONAL**, tested by hierarchical models and histogram-preserving replay.
