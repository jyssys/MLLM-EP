# Frozen versus continual online calibration

Every prediction follows `prefill signature → prediction record → actual decode → optional DB update`. The current request ID is rejected if present in memory. The complete causal ledger is `artifacts/no_future_leakage_events.jsonl.gz`.

| stream | mode | expert cosine | rank cosine | rank-load MAE |
|---|---|---:|---:|---:|
| IID heldout | M1_frozen | 0.9560 | 0.9972 | 0.0212 |
| IID heldout | M2_continual | 0.9598 | 0.9975 | 0.0200 |
| domain shift | M1_frozen | 0.9560 | 0.9972 | 0.0212 |
| domain shift | M2_continual | 0.9597 | 0.9975 | 0.0200 |

Positive deltas below favor continual calibration:

| stream | metric | beneficial delta | paired 95% CI |
|---|---|---:|---:|
| domain_shift | expert_histogram_cosine | 0.0037 | [0.0017, 0.0055] |
| domain_shift | per_rank_load_mae | 0.0012 | [0.0006, 0.0017] |
| domain_shift | rank_distribution_cosine | 0.0003 | [0.0001, 0.0005] |
| iid_heldout | expert_histogram_cosine | 0.0038 | [0.0018, 0.0057] |
| iid_heldout | per_rank_load_mae | 0.0012 | [0.0007, 0.0018] |
| iid_heldout | rank_distribution_cosine | 0.0003 | [0.0002, 0.0005] |

Frozen versus continual behavior after each frozen stream boundary:

| phase | frozen expert cosine | continual expert cosine | frozen rank cosine | continual rank cosine |
|---|---:|---:|---:|---:|
| chart_structured_heavy | 0.9635 | 0.9652 | 0.9977 | 0.9978 |
| general_heavy | 0.9506 | 0.9543 | 0.9968 | 0.9970 |
| math_heavy | 0.9531 | 0.9555 | 0.9970 | 0.9975 |
| tail_balanced | 0.9564 | 0.9617 | 0.9971 | 0.9976 |

Predictor CPU overhead on already-collected compact histograms: signature construction 39.71 µs/request, kNN lookup 337.54 µs, database update 0.05 µs. No GPU synchronization is performed by the offline predictor. A deployable asynchronous GPU histogram-export path was not implemented, so router-trace extraction overhead is `UNAVAILABLE` rather than silently treated as zero.
