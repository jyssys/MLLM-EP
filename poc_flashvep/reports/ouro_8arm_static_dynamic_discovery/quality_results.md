# Actual free-running quality

Each primary arm used the same 300 GSM8K and 100 MATH questions. The cap was 4096 tokens.
Accuracy counts a correct extracted answer even if the generation reaches its cap. Completed-correct excludes those outputs.
Intervals use 5000 question-paired bootstrap draws. Identical observed scores do not establish population equivalence.
Native controls are separate engine comparisons. Their mode-Single reference is native TP1. Raw source flags remain unchanged.

| Arm | Dataset | Correct/N | Accuracy % | Delta vs S4 pp [95% CI] | Delta vs mode Single pp [95% CI] | Completed-correct/N | Parse failures | Truncated | Reviewed scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S4 | gsm8k | 232/300 | 77.333 | 0.000 (reference) | 0.000 (reference) | 232/300 | 20 | 2 | BOUNDED_NUMERICAL_AND_400_OUTPUT_PARITY |
| S4 | math500 | 61/100 | 61.000 | 0.000 (reference) | 0.000 (reference) | 58/100 | 8 | 16 | BOUNDED_NUMERICAL_AND_400_OUTPUT_PARITY |
| T4 | gsm8k | 233/300 | 77.667 | +0.333 [-1.667, 2.333] | +0.333 [-1.667, 2.333] | 233/300 | 21 | 3 | NUMERICALLY_UNQUALIFIED_EXPLORATORY |
| T4 | math500 | 61/100 | 61.000 | +0.000 [-5.000, 6.000] | +0.000 [-5.000, 6.000] | 58/100 | 7 | 13 | NUMERICALLY_UNQUALIFIED_EXPLORATORY |
| L4 | gsm8k | 232/300 | 77.333 | +0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] | 232/300 | 20 | 2 | BOUNDED_NUMERICAL_AND_400_OUTPUT_PARITY |
| L4 | math500 | 61/100 | 61.000 | +0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] | 58/100 | 8 | 16 | BOUNDED_NUMERICAL_AND_400_OUTPUT_PARITY |
| O4 | gsm8k | 232/300 | 77.333 | +0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] | 232/300 | 20 | 2 | BOUNDED_NUMERICAL_AND_400_OUTPUT_PARITY |
| O4 | math500 | 61/100 | 61.000 | +0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] | 58/100 | 8 | 16 | BOUNDED_NUMERICAL_AND_400_OUTPUT_PARITY |
| SD | gsm8k | 232/300 | 77.333 | +0.000 [-2.667, 2.667] | 0.000 (reference) | 232/300 | 18 | 3 | CUSTOM_POLICY_VALIDATED_WITH_NATIVE_BACKEND_DRIFT |
| SD | math500 | 59/100 | 59.000 | -2.000 [-10.000, 6.000] | 0.000 (reference) | 58/100 | 7 | 11 | CUSTOM_POLICY_VALIDATED_WITH_NATIVE_BACKEND_DRIFT |
| TD | gsm8k | 231/300 | 77.000 | -0.333 [-3.333, 3.000] | -0.333 [-2.667, 2.000] | 231/300 | 19 | 0 | NUMERICALLY_UNQUALIFIED_EXPLORATORY |
| TD | math500 | 61/100 | 61.000 | +0.000 [-10.000, 9.000] | +2.000 [-5.000, 9.000] | 59/100 | 10 | 14 | NUMERICALLY_UNQUALIFIED_EXPLORATORY |
| LD | gsm8k | 232/300 | 77.333 | +0.000 [-2.667, 2.667] | +0.000 [0.000, 0.000] | 232/300 | 18 | 3 | CUSTOM_POLICY_VALIDATED_WITH_NATIVE_BACKEND_DRIFT |
| LD | math500 | 59/100 | 59.000 | -2.000 [-10.000, 6.000] | +0.000 [0.000, 0.000] | 58/100 | 7 | 11 | CUSTOM_POLICY_VALIDATED_WITH_NATIVE_BACKEND_DRIFT |
| OD | gsm8k | 232/300 | 77.333 | +0.000 [-2.667, 2.667] | +0.000 [0.000, 0.000] | 232/300 | 18 | 3 | CUSTOM_POLICY_VALIDATED_WITH_NATIVE_BACKEND_DRIFT |
| OD | math500 | 59/100 | 59.000 | -2.000 [-10.000, 6.000] | +0.000 [0.000, 0.000] | 58/100 | 7 | 11 | CUSTOM_POLICY_VALIDATED_WITH_NATIVE_BACKEND_DRIFT |
| native_TP1 | gsm8k | 232/300 | 77.333 | +0.000 [-2.333, 2.333] | 0.000 (reference) | 232/300 | 21 | 1 | PRODUCTION_BACKEND_CONTROL |
| native_TP1 | math500 | 61/100 | 61.000 | +0.000 [-6.000, 6.000] | 0.000 (reference) | 59/100 | 9 | 13 | PRODUCTION_BACKEND_CONTROL |
| native_TP4 | gsm8k | 232/300 | 77.333 | +0.000 [-2.000, 1.667] | +0.000 [-2.000, 2.000] | 232/300 | 20 | 1 | NUMERICALLY_UNQUALIFIED_EXPLORATORY |
| native_TP4 | math500 | 60/100 | 60.000 | -1.000 [-7.000, 5.000] | -1.000 [-7.000, 5.000] | 56/100 | 9 | 15 | NUMERICALLY_UNQUALIFIED_EXPLORATORY |
