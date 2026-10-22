# Frozen SLO registration

The reference is the median of three warmed GPU4-alone ITL medians at 2,033 prompt tokens and batch width4: 27.189323 ms.
This differs from Sarathi's 4K/B32 reference. These are exploratory Ouro thresholds, not product requirements.

| TBT tier | Multiplier | Limit ms |
| --- | ---: | ---: |
| Strict | 5 | 135.946614 |
| Moderate | 10 | 271.893227 |
| Relaxed | 25 | 679.733069 |

The absolute sensitivity grid is 75/100/125/150/200/250/500 ms.
The fixed TTFT grid is 2/5/10 s. The primary grid uses 2/5 s and all three relative TBT tiers.
The 1.25×/1.5×/2× no-interference TTFT thresholds are also frozen per prompt bin in the JSON.
All offered requests count. Request-max and pooled-p99 contracts remain separate.
An unsafe or diverging queue cannot establish sustainable capacity.
A rate with fewer than 1,000 requests per arm remains exploratory for p99/capacity claims.
No SLO value depends on candidate measurements. The legacy decision remains unchanged.
