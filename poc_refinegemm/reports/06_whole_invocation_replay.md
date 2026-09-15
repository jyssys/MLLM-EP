# Whole-invocation replay and matched controls

The corpus contains 18 actual dense owner vectors and 16 non-empty compacted
owner vectors, sampled across both tasks and all phases. For each real vector we
also construct two controls with identical total assignments and identical
number of active experts:

- low heterogeneity: rows spread as evenly as possible;
- high heterogeneity: half the active experts receive one row, with the
  remainder concentrated on the other half.

Three independent processes execute every case/backend after five warmups, with
30 randomized repetitions. The high/low latency ratio under the strongest whole
kernel is only **1.0205** median for dense and **1.0139** for compacted. Dense
restart-unit medians are 1.0211/1.0194/1.0204; compacted medians are
1.0131/1.0188/1.0137. Individual directions are not uniform.

Therefore M_e shape is a real secondary cost variable, but after controlling
assignment count and active support its median direct effect is about 1--2% at
the owner-local operator. Figures [11](../figures/11_invocation_latency_vs_tiny_fraction.png)
and [12](../figures/12_invocation_latency_vs_me_cv.png) show the real cases.
Machine controls are in [MATCHED_HETEROGENEITY_CONTROLS.csv](../MATCHED_HETEROGENEITY_CONTROLS.csv).

