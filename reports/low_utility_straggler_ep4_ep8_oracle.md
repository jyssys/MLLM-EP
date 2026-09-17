# EP4/EP8 counterfactual pruning oracle

These are route-trace counterfactuals, not quality-validated policies. Only current-block exact routes are removable; prefix/prior physical work remains unchanged. Costs are rebuilt from the remaining expert histogram and unique source-destination activation matrix and passed through grouped-mm/EP2-calibrated models—no route-count latency scaling is used. P3 recomputes calibrated pressure in four safe batches; P4 uses assignment pressure once. All rows enforce min_k=4.

| target | policy | nominal mass budget | actual removed mass | full physical route reduction | stage gain | mean max/mean | mean wait | remote-byte reduction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| EP4 | P2_utility | 0.5% | 0.410% | 0.018% | 0.002% | 1.0903 | 8.082% | 0.002% |
| EP4 | P3_calibrated_oracle | 0.5% | 0.410% | 0.016% | 0.007% | 1.0903 | 8.077% | 0.002% |
| EP4 | P4_rank_pressure | 0.5% | 0.413% | 0.016% | 0.006% | 1.0903 | 8.077% | 0.002% |
| EP4 | C4_random_tail | 0.5% | 0.454% | 0.013% | 0.002% | 1.0903 | 8.081% | 0.002% |
| EP8 | P2_utility | 0.5% | 0.410% | 0.018% | 0.003% | 1.2093 | 16.745% | 0.003% |
| EP8 | P3_calibrated_oracle | 0.5% | 0.417% | 0.016% | 0.009% | 1.2092 | 16.738% | 0.003% |
| EP8 | P4_rank_pressure | 0.5% | 0.423% | 0.016% | 0.008% | 1.2092 | 16.738% | 0.003% |
| EP8 | C4_random_tail | 0.5% | 0.454% | 0.013% | 0.003% | 1.2092 | 16.744% | 0.003% |
| EP4 | P2_utility | 1.0% | 0.903% | 0.036% | 0.004% | 1.0904 | 8.082% | 0.006% |
| EP4 | P3_calibrated_oracle | 1.0% | 0.906% | 0.032% | 0.015% | 1.0902 | 8.072% | 0.006% |
| EP4 | P4_rank_pressure | 1.0% | 0.910% | 0.032% | 0.012% | 1.0902 | 8.073% | 0.006% |
| EP4 | C4_random_tail | 1.0% | 0.951% | 0.023% | 0.003% | 1.0904 | 8.082% | 0.007% |
| EP8 | P2_utility | 1.0% | 0.903% | 0.036% | 0.005% | 1.2093 | 16.746% | 0.008% |
| EP8 | P3_calibrated_oracle | 1.0% | 0.918% | 0.032% | 0.020% | 1.2090 | 16.730% | 0.007% |
| EP8 | P4_rank_pressure | 1.0% | 0.924% | 0.031% | 0.015% | 1.2091 | 16.733% | 0.007% |
| EP8 | C4_random_tail | 1.0% | 0.951% | 0.023% | 0.004% | 1.2093 | 16.745% | 0.009% |
| EP4 | P2_utility | 2.5% | 2.390% | 0.086% | 0.009% | 1.0904 | 8.083% | 0.016% |
| EP4 | P3_calibrated_oracle | 2.5% | 2.395% | 0.075% | 0.034% | 1.0901 | 8.060% | 0.015% |
| EP4 | P4_rank_pressure | 2.5% | 2.402% | 0.074% | 0.026% | 1.0901 | 8.063% | 0.015% |
| EP4 | C4_random_tail | 2.5% | 2.446% | 0.056% | 0.008% | 1.0904 | 8.082% | 0.020% |
| EP8 | P2_utility | 2.5% | 2.390% | 0.086% | 0.012% | 1.2093 | 16.749% | 0.021% |
| EP8 | P3_calibrated_oracle | 2.5% | 2.410% | 0.074% | 0.043% | 1.2088 | 16.716% | 0.019% |
| EP8 | P4_rank_pressure | 2.5% | 2.422% | 0.072% | 0.034% | 1.2089 | 16.720% | 0.019% |
| EP8 | C4_random_tail | 2.5% | 2.446% | 0.056% | 0.008% | 1.2093 | 16.749% | 0.025% |
| EP4 | P2_utility | 5.0% | 4.878% | 0.159% | 0.015% | 1.0904 | 8.087% | 0.033% |
| EP4 | P3_calibrated_oracle | 5.0% | 4.865% | 0.136% | 0.062% | 1.0899 | 8.044% | 0.032% |
| EP4 | P4_rank_pressure | 5.0% | 4.877% | 0.135% | 0.047% | 1.0900 | 8.050% | 0.032% |
| EP4 | C4_random_tail | 5.0% | 4.947% | 0.111% | 0.015% | 1.0904 | 8.083% | 0.040% |
| EP8 | P2_utility | 5.0% | 4.878% | 0.159% | 0.022% | 1.2094 | 16.754% | 0.044% |
| EP8 | P3_calibrated_oracle | 5.0% | 4.895% | 0.134% | 0.072% | 1.2086 | 16.701% | 0.039% |
| EP8 | P4_rank_pressure | 5.0% | 4.914% | 0.132% | 0.060% | 1.2087 | 16.706% | 0.041% |
| EP8 | C4_random_tail | 5.0% | 4.947% | 0.111% | 0.014% | 1.2094 | 16.755% | 0.051% |
| EP4 | P2_utility | 10.0% | 9.863% | 0.289% | 0.027% | 1.0905 | 8.093% | 0.069% |
| EP4 | P3_calibrated_oracle | 10.0% | 9.624% | 0.242% | 0.105% | 1.0896 | 8.023% | 0.066% |
| EP4 | P4_rank_pressure | 10.0% | 9.595% | 0.240% | 0.084% | 1.0897 | 8.028% | 0.064% |
| EP4 | C4_random_tail | 10.0% | 9.946% | 0.222% | 0.027% | 1.0904 | 8.088% | 0.081% |
| EP8 | P2_utility | 10.0% | 9.863% | 0.289% | 0.039% | 1.2095 | 16.761% | 0.092% |
| EP8 | P3_calibrated_oracle | 10.0% | 9.801% | 0.244% | 0.103% | 1.2086 | 16.698% | 0.084% |
| EP8 | P4_rank_pressure | 10.0% | 9.851% | 0.243% | 0.092% | 1.2085 | 16.694% | 0.085% |
| EP8 | C4_random_tail | 10.0% | 9.946% | 0.222% | 0.028% | 1.2095 | 16.764% | 0.103% |

Canonical P1/P2 and matched route-count controls are retained in `low_utility_straggler_summary.json`. No model rollout or output-weight renormalization was performed.

## Supplemental full-row counterfactual

The systems screen uses 2,048 evenly spaced full physical invocations (27,016,192 expert slots) from the existing GSM8K-32 heavy trace. It is a sampled systems oracle, not a request-level quality result.

| target | policy | mass budget | stage gain | mean max/mean | mean wait | remote-byte reduction |
|---|---|---:|---:|---:|---:|---:|
| EP4 | P2_utility | 0.5% | 0.279% | 1.0973 | 8.66% | 0.24% |
| EP4 | P3_calibrated_oracle | 0.5% | 0.814% | 1.0914 | 8.17% | 0.13% |
| EP4 | P4_rank_pressure | 0.5% | 0.586% | 1.0927 | 8.28% | 0.12% |
| EP8 | P2_utility | 0.5% | 0.251% | 1.2250 | 17.77% | 0.32% |
| EP8 | P3_calibrated_oracle | 0.5% | 0.826% | 1.2128 | 16.94% | 0.15% |
| EP8 | P4_rank_pressure | 0.5% | 0.579% | 1.2147 | 17.09% | 0.15% |
| EP4 | P2_utility | 1.0% | 0.427% | 1.0976 | 8.69% | 0.43% |
| EP4 | P3_calibrated_oracle | 1.0% | 1.442% | 1.0862 | 7.72% | 0.25% |
| EP4 | P4_rank_pressure | 1.0% | 1.016% | 1.0887 | 7.95% | 0.23% |
| EP8 | P2_utility | 1.0% | 0.448% | 1.2261 | 17.82% | 0.61% |
| EP8 | P3_calibrated_oracle | 1.0% | 1.390% | 1.2068 | 16.49% | 0.32% |
| EP8 | P4_rank_pressure | 1.0% | 0.937% | 1.2109 | 16.83% | 0.32% |
| EP4 | P2_utility | 2.5% | 0.806% | 1.0985 | 8.75% | 1.09% |
| EP4 | P3_calibrated_oracle | 2.5% | 3.050% | 1.0729 | 6.61% | 0.64% |
| EP4 | P4_rank_pressure | 2.5% | 2.360% | 1.0769 | 6.97% | 0.62% |
| EP8 | P2_utility | 2.5% | 0.801% | 1.2296 | 18.03% | 1.52% |
| EP8 | P3_calibrated_oracle | 2.5% | 2.847% | 1.1911 | 15.38% | 0.86% |
| EP8 | P4_rank_pressure | 2.5% | 2.187% | 1.1976 | 15.93% | 0.86% |
| EP4 | P2_utility | 5.0% | 1.857% | 1.0968 | 8.60% | 2.30% |
| EP4 | P3_calibrated_oracle | 5.0% | 5.088% | 1.0573 | 5.28% | 1.40% |
| EP4 | P4_rank_pressure | 5.0% | 3.818% | 1.0675 | 6.18% | 1.32% |
| EP8 | P2_utility | 5.0% | 1.636% | 1.2312 | 18.07% | 3.07% |
| EP8 | P3_calibrated_oracle | 5.0% | 6.238% | 1.1522 | 12.67% | 1.88% |
| EP8 | P4_rank_pressure | 5.0% | 4.912% | 1.1688 | 13.99% | 1.87% |
| EP4 | P2_utility | 10.0% | 4.388% | 1.0888 | 7.94% | 4.67% |
| EP4 | P3_calibrated_oracle | 10.0% | 6.925% | 1.0516 | 4.78% | 3.29% |
| EP4 | P4_rank_pressure | 10.0% | 5.420% | 1.0662 | 6.12% | 3.06% |
| EP8 | P2_utility | 10.0% | 3.365% | 1.2342 | 18.19% | 6.20% |
| EP8 | P3_calibrated_oracle | 10.0% | 11.052% | 1.1051 | 9.28% | 4.64% |
| EP8 | P4_rank_pressure | 10.0% | 9.199% | 1.1295 | 11.16% | 4.53% |

At 5% mass, P3 exceeds P2 by 3.23pp EP4 and 4.60pp EP8; P4 exceeds P2 by 1.96pp EP4 and 3.28pp EP8. At 10%, P3's EP8 increment is 7.69pp. These gains reduce max/mean and wait, but the approximation budget is not quality-validated.
