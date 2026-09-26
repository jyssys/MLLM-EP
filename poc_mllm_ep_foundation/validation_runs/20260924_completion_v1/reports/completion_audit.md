# Completion audit

This versioned run resumed from atomic per-request and per-DP checkpoints. No
completed sample was silently recomputed as a new observation.

## Before / after

| Gate | Before | Current |
|---|---:|---:|
| Public MMStar | NOT_RUN | n=1500, score=0.7198 |
| Public ChartQA | NOT_RUN | n=2500, score=0.8560 |
| Actual EP2 | NOT_RUN | PASS: TP1/DP2/EP2, 64 experts/rank, DeepEP |
| EP2 simulator median/P90 APE | NOT_RUN | 12.962% / 22.728% |
| EP4 simulator median/P90 APE | 15.705% / 46.817% | 14.510% / 41.084% |
| Profiler overhead | 13.550% | 5.616% median |

The simulator miss is `SIM_UNVALIDATED`, not a research-method NO-GO. The v4
untouched final test was created image-disjoint after each earlier test was
promoted to development during feature repair. No parameter was changed after
the v4 final scores were observed.
