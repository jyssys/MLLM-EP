
# Q1 rank latency model

Natural trace correlations are weak because pair count, active set, layer, cache state, and launch behavior co-vary:

| Topology | token-pairs Spearman | active-experts Spearman | active-bytes Spearman | token-only R² | joint R² |
|---|---:|---:|---:|---:|---:|
| EP2 | 0.0652 | 0.0596 | 0.0596 | 0.1041 | 0.1041 |
| EP4 | 0.1229 | 0.1094 | 0.1094 | 0.0314 | 0.0435 |

The causal evidence is the controlled Sweep A, not the raw correlation: constant paths plus more distinct weights increased execution by 55.60%. Hardware-counter collection was not run; therefore the conclusion is **PHYSICAL_ACTIVE_EXPERT_BOTTLENECK_SUPPORTED (bounded kernel/stage evidence)**, not a complete roofline proof.
