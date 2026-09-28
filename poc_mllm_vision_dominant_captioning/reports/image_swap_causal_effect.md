
# Causal identical-prompt image swap

Every comparison fixes the prompt, decode token IDs, decode positions, and replay length. Only the image changes. Forced replay is off-distribution for the replacement image and is interpreted as causal route sensitivity, not natural-response quality.

`expert change` and `rank change` below use the same histogram/multiset-displacement definition as the earlier reasoning PoC, so the regime comparison is like-for-like. Ordered top-k-slot churn is preserved separately in `artifacts/causal_effect.json` and is not substituted for this metric.

| sampled decode window | repeat expert change | same-domain expert change | cross-domain expert change | EP4 cross-domain rank change |
|---|---:|---:|---:|---:|
| 1-32 | 0.000% | 5.315% | 7.085% | 2.633% |
| 33-64 | 0.000% | 5.239% | 5.815% | 2.840% |
| 65-96 | 0.000% | 4.572% | 5.129% | 2.482% |
| 97-128 | 0.000% | 4.970% | 5.399% | 2.812% |
| 129-192 | N/A | N/A | N/A | N/A |
| 193-256 | N/A | N/A | N/A | N/A |
| 1-256 | 0.000% | 4.472% | 5.359% | 1.631% |

Aggregate cross-domain bootstrap vs repeat noise: `{"changed_rank_fraction": {"ci95_high": 0.017372142650462965, "ci95_low": 0.015284921151620369, "effect_pairs": 20, "mean_difference_vs_repeat_noise": 0.01631221064814815, "repeat_pairs": 5}, "expert_changed_fraction": {"ci95_high": 0.05982123480902777, "ci95_low": 0.04802110460069445, "effect_pairs": 20, "mean_difference_vs_repeat_noise": 0.05358796296296296, "repeat_pairs": 5}, "expert_jsd": {"ci95_high": 0.02725109942866924, "ci95_low": 0.02025973252076755, "effect_pairs": 20, "mean_difference_vs_repeat_noise": 0.023573013949107338, "repeat_pairs": 5}, "rank_jsd": {"ci95_high": 0.0006461058585074351, "ci95_low": 0.0005103649834873432, "effect_pairs": 20, "mean_difference_vs_repeat_noise": 0.0005777865604988183, "repeat_pairs": 5}}`.
