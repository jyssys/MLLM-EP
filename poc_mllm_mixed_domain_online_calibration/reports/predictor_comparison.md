# Predictor comparison

All policies use the same simple cosine kNN/prototype family. Metrics use checkpoint-sampled actual decode routes through step 128 and the frozen 10% calibration ratio. `P_full` is an ELDR-style whole-prefill control, not an exact ELDR reproduction.

| Predictor | top-8 recall | expert cosine | expert JSD | active-set recall | rank cosine | rank JSD | rank-load MAE | hottest-rank acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P_domain | 0.9997 | 0.9526 | 0.1263 | 1.0000 | 0.9970 | 0.0036 | 0.0216 | 100.00% |
| P_text | 0.9997 | 0.9574 | 0.1154 | 1.0000 | 0.9973 | 0.0033 | 0.0207 | 100.00% |
| P_full | 0.9997 | 0.9563 | 0.1179 | 1.0000 | 0.9972 | 0.0035 | 0.0212 | 100.00% |
| P_vision | 0.9997 | 0.9546 | 0.1214 | 1.0000 | 0.9971 | 0.0036 | 0.0215 | 100.00% |
| P_separated | 0.9997 | 0.9560 | 0.1183 | 1.0000 | 0.9972 | 0.0035 | 0.0212 | 100.00% |

Positive bootstrap delta means P_separated is better; request is the bootstrap unit.

| Comparison | metric | beneficial delta | paired 95% CI | N |
|---|---|---:|---:|---:|
| P_separated − P_full | expert_histogram_cosine | -0.0003 | [-0.0013, 0.0007] | 80 |
| P_separated − P_full | rank_distribution_cosine | -0.0000 | [-0.0001, 0.0001] | 80 |
| P_separated − P_full | per_rank_load_mae | -0.0000 | [-0.0004, 0.0003] | 80 |
| P_separated − P_text | expert_histogram_cosine | -0.0014 | [-0.0027, -0.0002] | 80 |
| P_separated − P_text | rank_distribution_cosine | -0.0001 | [-0.0002, -0.0000] | 80 |
| P_separated − P_text | per_rank_load_mae | -0.0005 | [-0.0009, -0.0001] | 80 |
| P_separated − P_domain | expert_histogram_cosine | 0.0034 | [0.0010, 0.0060] | 80 |
| P_separated − P_domain | rank_distribution_cosine | 0.0002 | [-0.0000, 0.0003] | 80 |
| P_separated − P_domain | per_rank_load_mae | 0.0004 | [-0.0002, 0.0010] | 80 |

Window-specific values for 1–32, 33–64, 65–96, and 97–128 are preserved in `artifacts/predictor_comparison.json`. Point estimates whose CI includes zero are not called confirmed increments.
