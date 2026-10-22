# Actual task-serving measurements

These are actual 400-question task runs. They are not observer-free hardware headlines.
Custom T4, TD, and native TP4 remain NUMERICALLY_UNQUALIFIED_EXPLORATORY. These rows do not establish a qualified TP architecture baseline.
Cohort sum covers 101 measured service windows. It is not whole-command wall time or model-cold latency.
The custom clock excludes request-bound graph preparation. Earlier Static pilots lack full preparation-inclusive coverage.
Use extension286_serving.csv for the fully instrumented 286-question preparation control. Native graphs use a different lifecycle.
Generated work differs between engines. Native TP1/TP4 use packed FA3, paged KV, and optimized graphs.
The custom family uses request-local B1 stages. Custom TP also uses FP32 partial-output reductions and separate projections.
Use replay_summary.csv for identical-work five-repeat results and request-level paired intervals.
Use task_cohort_paired_ratios.csv for paired cohort-composition intervals. These intervals do not measure repeat-to-repeat noise.

| Arm | GPUs | Cohort sum s | Mean E2E s | Mean TTFT ms | Mean TPOT ms | P95 E2E s | Generated tokens | Output tok/s | Output tok/s/GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S4 | 1 | 6074.192 | 38.685 | 295.872 | 116.729 | 186.181 | 179246 | 29.509 | 29.509 |
| T4 | 4 | 6219.233 | 39.294 | 264.441 | 122.359 | 163.979 | 174090 | 27.992 | 6.998 |
| L4 | 4 | 3707.747 | 16.832 | 151.036 | 36.000 | 83.596 | 179246 | 48.344 | 12.086 |
| O4 | 4 | 3311.532 | 15.267 | 177.744 | 33.067 | 75.691 | 179246 | 54.128 | 13.532 |
| SD | 1 | 4373.494 | 27.525 | 295.938 | 91.609 | 115.487 | 170182 | 38.912 | 38.912 |
| TD | 4 | 4267.465 | 27.544 | 242.954 | 93.252 | 115.043 | 165651 | 38.817 | 9.704 |
| LD | 4 | 3326.628 | 15.615 | 175.527 | 39.269 | 81.320 | 170182 | 51.158 | 12.789 |
| OD | 4 | 3777.943 | 20.155 | 195.046 | 58.052 | 94.025 | 170182 | 45.046 | 11.262 |
| native_TP1 | 1 | 948.674 | 4.651 | 103.638 | 11.052 | 22.693 | 162994 | 171.812 | 171.812 |
| native_TP4 | 4 | 780.024 | 3.832 | 112.677 | 8.328 | 22.932 | 178672 | 229.060 | 57.265 |
