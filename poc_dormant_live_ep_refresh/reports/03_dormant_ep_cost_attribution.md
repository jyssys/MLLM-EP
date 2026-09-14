# Dormant EP cost attribution

The attribution joins baseline acceptance time to actual per-layer top-8 route
rows, expert owners, and same-GPU CUDA-event durations.  Observer-heavy event
totals are normalized to the identical-substrate clean component shares.

## H=1 all-phase cost

| task/state | physical rows | remote assignments | router E2E % | dispatch % | expert % | combine % | routed total % |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K ACTIVE | 553,815 | 3,328,978 | 1.34 | 2.62 | 6.63 | 1.61 | 10.85 |
| GSM8K DORMANT | 385,919 | 2,309,036 | 0.97 | 2.01 | 4.97 | 1.11 | 8.09 |
| GSM8K DEAD | 1,112,714 | 6,676,657 | 4.11 | 7.44 | 17.87 | 3.69 | 29.00 |
| HumanEval ACTIVE | 591,914 | 3,562,187 | 1.57 | 3.01 | 7.09 | 1.88 | 11.98 |
| HumanEval DORMANT | 285,572 | 1,721,361 | 1.28 | 2.37 | 5.17 | 1.07 | 8.61 |
| HumanEval DEAD | 1,010,290 | 6,058,236 | 4.37 | 7.94 | 17.29 | 3.71 | 28.95 |

H=1 DORMANT accounts for 18.80%/15.13% of timing-aligned physical MoE rows and
18.75%/15.14% of remote assignments on GSM8K/HumanEval.  The candidate keeps
the router fresh, so the removable optimistic mass is dispatch+expert+combine:
**8.09%/8.61% of request E2E**, not the full routed total plus router.

By phase, that removable mass is 1.78/4.75/1.56% on GSM8K and
0.25/7.44/0.92% on HumanEval for early/middle/late.  The strong concentration
in the middle phase suggests where an intervention would have to operate, but
not that it is safe.

## Economic caveat

Removing rows does not remove DeepEP startup.  Identical-substrate payload
sweeps measured approximate dispatch/combine floors of 0.0998/0.1529 ms even
at tiny M.  Retaining those floors reduces perfect H=1 removal to
**6.95% GSM8K and 7.27% HumanEval**.  These are analytical, post-Epoch residual
upper bounds; no live physical row removal was implemented.
