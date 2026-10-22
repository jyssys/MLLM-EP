# Eight-arm final evidence matrix

All eight primary arms completed 400 actual free-running answers.
Task ratios use the sum of 101 service-cohort durations. They exclude model load and request-bound graph preparation.
Static task scheduling differs between PP FIFO and Single/TP waves. Outputs can differ across backends.
Clean C4 uses identical frozen prompt/output IDs and five measured repetitions. Dynamic depth is also trace-locked.
A four-GPU ratio against one GPU does not establish four-GPU resource efficiency. Historical TP numerical failure remains FAIL.

| Arm | GPUs | GSM8K correct/N | MATH correct/N | Task cohort speedup vs mode Single | Clean C4 replay speedup vs mode Single | Qualification |
| --- | --- | --- | --- | --- | --- | --- |
| S4 | 1 | 232/300 | 61/100 | 1.000x | 1.000x | BOUNDED_NUMERICAL_AND_400_OUTPUT_PARITY |
| T4 | 4 | 233/300 | 61/100 | 0.977x | 0.931x | NUMERICALLY_UNQUALIFIED_EXPLORATORY |
| L4 | 4 | 232/300 | 61/100 | 1.638x | 2.824x | BOUNDED_NUMERICAL_AND_400_OUTPUT_PARITY |
| O4 | 4 | 232/300 | 61/100 | 1.834x | 2.245x | BOUNDED_NUMERICAL_AND_400_OUTPUT_PARITY |
| SD | 1 | 232/300 | 59/100 | 1.000x | 1.000x | CUSTOM_POLICY_VALIDATED_WITH_NATIVE_BACKEND_DRIFT |
| TD | 4 | 231/300 | 61/100 | 1.025x | 0.941x | NUMERICALLY_UNQUALIFIED_EXPLORATORY |
| LD | 4 | 232/300 | 59/100 | 1.315x | 2.298x | CUSTOM_POLICY_VALIDATED_WITH_NATIVE_BACKEND_DRIFT |
| OD | 4 | 232/300 | 59/100 | 1.158x | 1.482x | CUSTOM_POLICY_VALIDATED_WITH_NATIVE_BACKEND_DRIFT |
