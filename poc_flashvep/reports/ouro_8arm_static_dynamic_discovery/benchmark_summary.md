# Benchmark summary

Updated UTC: 2026-10-21T08:36:29.235050+00:00.
This is an evidence snapshot. Missing experiments remain NOT_RUN.

## Actual answer accuracy

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

The same 300 GSM8K and 100 MATH question IDs define each final target.
The current denominator is the completed, canonical-input subset, not the intended target.
The sample preserves the original pilot and adds metadata-only random questions.
All accuracy uses actual greedy generation, EOS, and a 4096-token cap.
Token-weighted decode depth excludes the first token, which comes from full R4 prefill.
Answer accuracy includes correctly parsed truncated answers. The CSV reports completed-correct answers separately.
Paired intervals use question-level bootstrap with 5000 draws and seed 20261021.
Small intervals from identical observed answers do not prove population equivalence.

## Native production-oriented Static controls

| Dataset | Native engine | Correct/N | Accuracy % | Truncated | Fidelity |
|---|---|---|---|---|---|
| gsm8k | native_TP1 | 46/64 | 71.875 | 0 | BACKEND_CONTROL |
| math500 | native_TP1 | 32/50 | 64.000 | 6 | BACKEND_CONTROL |
| gsm8k | native_TP4 | 45/64 | 70.312 | 0 | NUMERICALLY_UNQUALIFIED_EXPLORATORY |
| math500 | native_TP4 | 34/50 | 68.000 | 8 | NUMERICALLY_UNQUALIFIED_EXPLORATORY |

The frozen arm manifest prefers native vLLM TP4 as the production-oriented T4 reference.
The custom eight-arm table provides the compatibility-controlled diagnostic family.
These native controls use the same 400-question target. They are not additional depth policies.
Do not mix native batching/kernel gains with a loop-placement gain.

## Task-observer timing

| Run | N | Cohort sum s | Mean request E2E s | Mean TTFT ms | Mean TPOT ms | Output tok/s | Status |
|---|---|---|---|---|---|---|---|
| quality400_S4 | 114 | 2596.538 | 57.527 | 273.255 | 113.174 | 29.829 | EXPLORATORY_BACKEND_QUALIFICATION_PENDING |
| quality400_T4 | 92 | 1532.364 | 44.662 | 292.943 | 122.787 | 28.028 | NUMERICALLY_UNQUALIFIED_EXPLORATORY |
| quality400_L4 | 114 | 1553.787 | 25.406 | 172.719 | 36.289 | 49.848 | EXPLORATORY_BACKEND_QUALIFICATION_PENDING |
| quality400_O4 | 114 | 1388.841 | 23.021 | 193.777 | 33.080 | 55.768 | EXPLORATORY_BACKEND_QUALIFICATION_PENDING |
| quality400_native_TP1 | 114 | 389.648 | 6.591 | 109.791 | 11.029 | 170.510 | BACKEND_CONTROL |
| quality400_native_TP4 | 114 | 314.052 | 5.200 | 151.208 | 8.295 | 227.886 | NUMERICALLY_UNQUALIFIED_EXPLORATORY |

These times include task observers and controller work. They are not observer-free hardware headlines.
The custom service clock excludes request-bound graph preparation. It is not cold-cohort E2E.
Later raw records separately measure resident-model preparation-inclusive cohort wall time.
Earlier custom TP pilot records lack that field. Preparation-inclusive aggregate coverage remains PARTIAL until all source cohorts expose it.
The cohort sum excludes setup between cohorts. It is not end-to-end wall time for the entire experiment command.
Static custom PP uses asynchronous request FIFO. Custom TP uses token waves.
Those Static task timings are not fully scheduler-matched.
Native vLLM has different batching, attention, graph, and KV implementations.
Keep the custom eight-arm comparison separate from the two native production-oriented controls.
Generated token counts can differ between backends. They remain visible in `subset400_serving.csv`.

## Identical-work clean replay

NOT_RUN. No completed clean replay comparison is available.

Replay uses frozen output IDs, not benchmark answers. Dynamic replay also locks observed Single-Dynamic exit depths.
Every replay excludes setup and warm-up repeat zero. The plan requests five measured repeats.
Numerically unresolved TP timing remains exploratory.
`replay_summary.csv` separates request-paired E2E intervals from weak five-repeat cohort intervals.
