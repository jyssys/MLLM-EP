# 01 — Strongest static frontier

All entries are clean, full-generation trajectories with block size 32,
temperature 0, threshold decoder, BF16, dense TP4/routed EP4 and DeepEP
normal. The fixed threshold controls were run *before* dynamic schedules.
Raw attempts: [STATIC_THRESHOLD_SWEEP.csv](../STATIC_THRESHOLD_SWEEP.csv).

| GSM8K discovery n=32 threshold | Correct | NFE | BCT (s) | Repetitions |
|---:|---:|---:|---:|---:|
| 0.950 | 12 | 118 | 13.129 | 1 |
| 0.925 | 12 | 115 | 12.976 | 1 |
| 0.900 | 13 | 102 | 7.653 median of three later clean runs | 3 plus one 11.251-s outlier |
| 0.875 | 12 | 98 | 12.871 | 1 |
| 0.850 | 12 | 100 | 10.442 | 1 |
| 0.825 | 13 | 88 | 9.286 median | 3 |
| 0.800 | 12 | 91 | 6.876 | 1 |

One n=32 answer is 3.125 pp; the equal 13/32 score at 0.825 is not a
sub-1-pp safety certificate. Promotion used official GSM8K n=512 and then
the full n=1,319 set at matched sorting, submitted=32, mini=32, gen-request=128.

| Full GSM8K n=1,319 static threshold | Correct | Quality vs 0.900 | Paired 95% quality interval (pp) | NFE | BCT restart values (s) | Descriptive median gain vs 0.900 |
|---:|---:|---:|---:|---:|---|---:|
| 0.900 | 144 | reference | — | 3,226 | 227.782 | 0% |
| 0.825 | 140 | -0.303 pp | [-0.986,+0.379] | 2,863 | 215.670, 201.954 | 8.33% |
| 0.800 | 142 | -0.152 pp | [-0.834,+0.531] | 2,758 | 265.438, 229.017 | -8.54% |

Static 0.825 is non-inferiority certified by the paired bootstrap only at a
1.0-pp budget, not at 0.3 or 0.5 pp. Static 0.800 also passes only the 1.0-pp
budget and has no stable latency benefit. The 0.825 BCT spread and the 0.800
36.4-s restart swing make a single threshold run unsuitable for GO. At a
statistically supported 0.5-pp budget, 0.900 is the strongest evaluated static
configuration; at 1.0 pp, the descriptive median static frontier is 0.825.
This is not a statistical speedup certificate because the full baseline had
one independent restart.

For the matched n=512 pool, 0.900 was 68/512 and NFE 1,300 in three restarts
with median BCT 100.336 s. Static 0.825 was 69/512 and NFE 1,125, yet its
single BCT was 113.888 s. The paired 95% quality interval was
[-0.977,+1.367] pp: observed score gain does not certify 0.5-pp safety.
This is a particularly clear warning that logical NFE removal alone does not
calibrate direct request latency in this runtime.
