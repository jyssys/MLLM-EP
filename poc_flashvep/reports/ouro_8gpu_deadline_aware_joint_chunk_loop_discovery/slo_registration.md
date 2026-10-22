# Frozen SLO registration

`slo_registration.json` froze numeric values before main held-out costs or any adaptive outcome.
The reference is GPU4 solo, warmed P2033/N4.
It uses the median of six cohort-median decode ITLs across two independent starts.
The reference is 27.187321800738573 ms.

| Tier | Factor | TBT limit |
| --- | --- | --- |
| Strict | 5 | 135.93660900369287 ms |
| Moderate | 10 | 271.87321800738573 ms |
| Relaxed | 25 | 679.6830450184643 ms |

Absolute TBT sensitivity limits are 75/100/125/150/200/250/500 ms.
TTFT sensitivity limits are 2/5/10 seconds.
The primary combination is TTFT≤5 seconds and Moderate TBT.

Request-level TBT attainment requires every observed gap to meet the limit.
Primary joint attainment also requires TTFT compliance and natural completion.
Caps, errors, and timeouts remain in the offered denominator.
Population p99 TBT is a different endpoint.

No new adaptive capacity curve uses these thresholds yet.
The historical 271.893227-ms threshold remains a historical result.
It is not silently substituted for this new registration.
