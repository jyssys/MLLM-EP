# Full-answer quality protocol

Status: MEASURED_PUBLIC_PILOT. The full 1,819-question benchmark is NOT_RUN.
All 798 registered generations completed across seven arms.
Do not interpret the pilot as a universal accuracy or output-parity PASS.

The measured target is the preserved public pilot: 64 GSM8K and 50 MATH-500 questions per arm.
These are the same historical questions, not a new blind accuracy set.
The benchmark uses the preserved Base-model few-shot prompts and stop strings.
It does not substitute the serving workload's chat-summary prompts.

All arms use pretrained Base Ouro-1.4B, BF16, Static R4, and greedy free-running generation.
Natural EOS and the preserved evaluator stops remain enabled.
The cap is 4,096 new tokens.
A capped output remains truncated even if the extractor finds an answer.
No teacher forcing or frozen output tokens determine accuracy.

| Arm | GSM8K correct/64 | GSM8K accuracy, % | MATH extracted correct/50 | MATH extracted accuracy, % | MATH completed-correct/50 | MATH capped outputs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F0A, stock Atomic | 47 | 73.4375 | 28 | 56.0 | 26 | 8 |
| M0, stock Mixed | 47 | 73.4375 | 28 | 56.0 | 26 | 8 |
| M1, fair Mixed | 47 | 73.4375 | 28 | 56.0 | 26 | 8 |
| C1A | 45 | 70.3125 | 32 | 64.0 | 29 | 7 |
| C1S | 43 | 67.1875 | 33 | 66.0 | 30 | 7 |
| G1, adaptive G-only | 47 | 73.4375 | 32 | 64.0 | 30 | 6 |
| J1, adaptive Joint | 47 | 73.4375 | 32 | 64.0 | 30 | 6 |

All GSM8K responses completed without the token cap.
MATH extracted correctness and complete-response correctness are different metrics.
For J1, the latter is 30/50=60%, not 64%.
The six capped J1 answers remain in the denominator.

## Paired quality interpretation

C1A lost 3.125 GSM8K percentage points versus F0A, with paired 95% interval [-7.8125, 0].
C1S lost 6.25 points, with interval [-12.5, -1.5625].
The C1S pilot therefore has an observed GSM8K regression.
Neither C-only arm receives a blanket quality-equivalence PASS.
Their performance contrasts remain exploratory quality-variant comparisons.

J1 had zero observed GSM8K accuracy difference versus F0A.
Its extracted MATH difference was +8 points, with paired interval [2, 16].
This is an exploratory numerical/batching-sensitive result, not a claimed reasoning improvement from scheduling.
Its precise numerical cause remains unresolved.
No weights or precision changed.

J1 and G1 had identical correctness labels on all 114 questions.
Their paired observed accuracy difference was 0 points on both domains.
The resampled empirical intervals were [0,0].
Such intervals reflect this finite matched sample, not proof of population equivalence.
They matched token IDs on 64/64 GSM8K and 49/50 MATH answers.
Stock Atomic, stock Mixed, and fair Mixed matched all 114 token sequences exactly.

The retained baseline-correct-to-candidate-wrong table contains six arm/question losses, all in GSM8K C-only variants.
It identifies the exact raw output files and extracted answers.
No manual answer correction changes the frozen evaluator result.

The B40 comparison includes C1A, C1S, G1, J1, M1, M0, and F0A.
F0A is stock Atomic C1024 and supplies the native accuracy reference.
M0 is stock Mixed. M1 separates the fairness change from that native path.
All policies cross on each GPU/model-process block with rotated order.
Input question groups and their ordering remain matched.

The evaluator is the preserved `parpp-final-v2` extractor and exact-answer checker.
The actual imported extractor must match its frozen source hash.
The tables retain correct/N, extracted accuracy, completed-correct/N, parsing failures, and truncation.
Paired confidence intervals resample questions, not generated tokens.
Output-token agreement remains separate from answer accuracy.

This small pilot cannot prove population-level equivalence at a one- or two-percentage-point tolerance.
A zero observed paired difference is not proof of zero unseen quality loss.
No result from the previous 114-question fixed-policy pilot is relabeled as a new adaptive measurement.
Quality timing includes text decoding and stop checks, so it is not a clean serving benchmark.
See `quality_summary.csv`, `quality_paired_differences.csv`, `quality_J1_vs_G1.csv`, and `quality_baseline_correct_to_wrong.csv`.
