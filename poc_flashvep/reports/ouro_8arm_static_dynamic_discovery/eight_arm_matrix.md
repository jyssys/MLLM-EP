# Eight-arm status

Updated UTC: 2026-10-21T08:36:29.235050+00:00.

| Arm | Parallelism | Mode | Physical GPUs | Evaluated target | Execution status | Fidelity |
|---|---|---|---|---|---|---|
| S4 | single | static | 4 | 64/300 + 50/100 | IN_PROGRESS | See correctness.md |
| T4 | tp4 | static | 4,5,6,7 | 64/300 + 39/100 | IN_PROGRESS | NUMERICALLY_UNQUALIFIED_EXPLORATORY |
| L4 | layer_pp4 | static | 4,5,6,7 | 64/300 + 50/100 | IN_PROGRESS | See correctness.md |
| O4 | loop_pp4 | static | 4,5,6,7 | 64/300 + 50/100 | IN_PROGRESS | See correctness.md |
| SD | single | dynamic | 4 | 0/300 + 0/100 | IN_PROGRESS | See correctness.md |
| TD | tp4 | dynamic | 4,5,6,7 | 0/300 + 0/100 | IN_PROGRESS | NUMERICALLY_UNQUALIFIED_EXPLORATORY |
| LD | layer_pp4 | dynamic | 4,5,6,7 | 0/300 + 0/100 | IN_PROGRESS | See correctness.md |
| OD | loop_pp4 | dynamic | 4,5,6,7 | 0/300 + 0/100 | IN_PROGRESS | See correctness.md |

The target is 400 questions per arm. The complete 1819-question suites are USER_REDUCED_NOT_RUN.
A completed task target does not pass the historical numerical gate.

| Dataset | Arm | Correct/N | Accuracy % | Delta vs S4 pp [95% CI] | Delta vs mode Single pp [95% CI] | Mean output tokens | Token-weighted decode depth | Truncated | Fidelity |
|---|---|---|---|---|---|---|---|---|---|
| gsm8k | S4 | 46/64 | 71.875 | NOT_RUN | NOT_RUN | 106.484 | 4.000 | 0 | EXPLORATORY_BACKEND_QUALIFICATION_PENDING |
| math500 | S4 | 31/50 | 62.000 | NOT_RUN | NOT_RUN | 1412.760 | 4.000 | 9 | EXPLORATORY_BACKEND_QUALIFICATION_PENDING |
| gsm8k | T4 | 47/64 | 73.438 | 1.562 [0.000, 4.688] | 1.562 [0.000, 4.688] | 104.391 | 4.000 | 0 | NUMERICALLY_UNQUALIFIED_EXPLORATORY |
| math500 | T4 | 23/39 | 58.974 | 0.000 [-10.256, 10.256] | 0.000 [-10.256, 10.256] | 1110.359 | 4.000 | 4 | NUMERICALLY_UNQUALIFIED_EXPLORATORY |
| gsm8k | L4 | 46/64 | 71.875 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 106.484 | 4.000 | 0 | EXPLORATORY_BACKEND_QUALIFICATION_PENDING |
| math500 | L4 | 31/50 | 62.000 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1412.760 | 4.000 | 9 | EXPLORATORY_BACKEND_QUALIFICATION_PENDING |
| gsm8k | O4 | 46/64 | 71.875 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 106.484 | 4.000 | 0 | EXPLORATORY_BACKEND_QUALIFICATION_PENDING |
| math500 | O4 | 31/50 | 62.000 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1412.760 | 4.000 | 9 | EXPLORATORY_BACKEND_QUALIFICATION_PENDING |
| gsm8k | SD | 0/300 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| math500 | SD | 0/100 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| gsm8k | TD | 0/300 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| math500 | TD | 0/100 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| gsm8k | LD | 0/300 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| math500 | LD | 0/100 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| gsm8k | OD | 0/300 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| math500 | OD | 0/100 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
