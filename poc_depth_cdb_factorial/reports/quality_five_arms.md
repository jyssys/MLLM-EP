# Five-arm natural-generation quality

Every configuration uses the same 128 GSM8K and 128 MATH-500 questions.
The corpus covers 128/1319 and 128/500 questions, respectively.
The pre-registered subset can overlap old pilot questions. It is not an untouched held-out confirmation.
All generations are greedy with native EOS, frozen stop sequences, and an 8192-token cap.

GSM8K uses the CDB three-shot chain-of-thought prompt.
MATH-500 uses the frozen zero-shot boxed-answer prompt.
The Base checkpoint does not receive a chat template in this protocol.
The answer parser is `parpp-final-v2`.

The intervals describe paired accuracy differences in percentage points.
An interval that includes zero does not establish quality equivalence.

These descriptive intervals do not correct for multiple arm/domain comparisons.

| Arm | Domain | Correct/N | Accuracy | Correct completed | Vs A0, pp [95% CI] | A4 vs A3, pp [95% CI] | EOS | Configured stop | Cap | Parse fail | Output tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | gsm8k | 101/128 | 78.906% | 101 | +0.000 [+0.000, +0.000] | — | 0 | 128 | 0 | 7 | 13498 |
| A0 | math500 | 79/128 | 61.719% | 75 | +0.000 [+0.000, +0.000] | — | 111 | 0 | 17 | 9 | 246010 |
| A1 | gsm8k | 97/128 | 75.781% | 97 | -3.125 [-6.250, -0.781] | — | 0 | 128 | 0 | 9 | 13313 |
| A1 | math500 | 80/128 | 62.500% | 79 | +0.781 [-4.688, +6.250] | — | 117 | 0 | 11 | 7 | 197564 |
| A2 | gsm8k | 99/128 | 77.344% | 99 | -1.562 [-3.906, +0.000] | — | 0 | 127 | 1 | 9 | 21437 |
| A2 | math500 | 74/128 | 57.812% | 72 | -3.906 [-9.375, +0.801] | — | 113 | 0 | 15 | 8 | 212371 |
| A3 | gsm8k | 100/128 | 78.125% | 100 | -0.781 [-3.906, +2.344] | — | 0 | 126 | 2 | 10 | 29430 |
| A3 | math500 | 81/128 | 63.281% | 78 | +1.562 [-3.906, +7.031] | — | 111 | 0 | 17 | 11 | 227350 |
| A4 | gsm8k | 97/128 | 75.781% | 97 | -3.125 [-6.250, -0.781] | -2.344 [-5.469, +0.781] | 0 | 126 | 2 | 9 | 29567 |
| A4 | math500 | 86/128 | 67.188% | 82 | +5.469 [-0.781, +11.719] | +3.906 [-2.344, +9.375] | 113 | 0 | 15 | 8 | 222389 |

A configured stop is a valid completion, not an output truncation.
The native API exposes completion status, not a separate stop-reason enumeration.
The wrapper classifies terminal EOS first, then a reached cap, then an earlier configured stop.
The primary accuracy counts extracted answers that match the gold answer.
`correct_completed` separately excludes cap-truncated outputs in the raw score file.

The quality corpus is different from the 256-token bounded speed corpus.
This report does not infer speed-cohort accuracy from this quality table.

Quality runs use four C64 question cohorts per arm.
Accuracy under the separate C8/C16/C32 speed operating points is `NOT_MEASURED`.
The study does not assume that batch-dependent accuracy is identical across those operating points.

## Generated work in the quality runs

| Arm | Questions | Output tokens | Scheduled loop work | Sum of four separate C64 serving intervals, s |
|---|---|---|---|---|
| A0 | 256 | 259508 | 730822 | 1786.503 |
| A1 | 256 | 210877 | 599878 | 824.013 |
| A2 | 256 | 233808 | 655857 | 639.830 |
| A3 | 256 | 256780 | 719761 | 2046.111 |
| A4 | 256 | 251956 | 708237 | 1302.688 |

These are actual generated lengths, not fixed token counts or forced continuations.
The summed serving intervals exclude intervening trials of other arms.
They are not a C256 burst makespan or the main table latency denominator.

| Exploratory natural-workload comparison | Ratio of summed serving intervals [95% CI] |
|---|---|
| no_refill_hindsight | 1.288× [0.922, 1.505] |
| refill_hindsight | 1.571× [1.178, 2.006] |

The bootstrap resamples four matched but different 64-question cohorts, with one timing measurement each.
It is not the primary five-repeat makespan bootstrap.
These positive timing diagnostics do not establish quality-preserving or depth-only scheduling gains.
Generated lengths, answers, and exit traces change between arms.
A strong length-aware natural-workload control is also absent.

## Full-answer output parity

| Pair | Token-exact requests | Depth-exact requests | Free-running positional token agreement |
|---|---|---|---|
| A0 → A1 | 134/256 | 60/256 | 31.737% |
| A0 → A2 | 145/256 | 74/256 | 34.728% |
| A0 → A3 | 128/256 | 54/256 | 32.438% |
| A0 → A4 | 130/256 | 44/256 | 25.564% |
| A1 → A2 | 140/256 | 69/256 | 35.151% |
| A3 → A4 | 134/256 | 46/256 | 30.252% |

Positional agreement compares two free-running outputs.
It is not teacher-aligned next-token agreement or a logit-equivalence test.
The raw parity file records first divergent tokens and changed output lengths.

The short functional probe passed gate, work, and terminal-KV checks.
Those checks do not isolate the cause of every long-answer divergence.
Batch-dependent numerical changes and runtime-state effects remain possible.
This experiment does not prove that BF16 rounding alone caused the differences.

## Paired answer changes

| Pair | Domain | Correct → wrong | Wrong → correct |
|---|---|---|---|
| A0 → A1 | gsm8k | 4 | 0 |
| A0 → A1 | math500 | 6 | 7 |
| A0 → A2 | gsm8k | 2 | 0 |
| A0 → A2 | math500 | 9 | 4 |
| A0 → A3 | gsm8k | 3 | 2 |
| A0 → A3 | math500 | 5 | 7 |
| A0 → A4 | gsm8k | 4 | 0 |
| A0 → A4 | math500 | 6 | 13 |
| A1 → A2 | gsm8k | 1 | 3 |
| A1 → A2 | math500 | 9 | 3 |
| A3 → A4 | gsm8k | 4 | 1 |
| A3 → A4 | math500 | 5 | 10 |

`quality_answer_flips.json` records every changed answer, its parsed value, and completion status.
The parity artifact also records first differing token and depth positions.
Common-history greedy comparisons stop at the first differing token.
Later positional comparisons do not share an identical token history.

## Inherited quality warning

The old 32-question pilot reported adaptive GSM8K 21/32 and MATH-500 19/32.
Its fixed-R4 references were 22/32 and 24/32.
The old fixed-minus-adaptive GSM8K difference was 3.125 pp, with 95% CI [−6.25, 12.5] pp.
The old MATH-500 difference was 15.625 pp, with 95% CI [3.125, 28.125] pp.

Those historical values are not the new five-arm scores.
The experiment preserves the adaptive setting despite that warning.
It does not claim fixed-R4 quality equivalence.
