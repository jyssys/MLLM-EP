# Frozen prompt diagnostic

This diagnostic uses AIME2024 IDs60–67.
The same questions also occur in the final public benchmark.
These eight overlapping problems are not an untouched confirmation cohort.
The diagnostic uses one seed and an 8,192-token cap.
P0 and P1 use the same TP4 engine, precision, sampling policy, and batch size.
The arms execute sequentially, not simultaneously.

| Arm | Completed correct / all | EOS / all | Capped / all | Parsed / all |
|---|---:|---:|---:|---:|
| P0, default template | 3/8 | 3/8 | 5/8 | 3/8 |
| P1, explicit thinking | 2/8 | 2/8 | 6/8 | 3/8 |

One capped P1 output contains a matching extracted answer.
It does not count as a completed correct answer.
Both arms have median output length 8,192 tokens.
The problem-bootstrap 95% intervals for completed-correct fractions are 12.5–75.0% and 0–62.5%, respectively.
The small overlapping cohort cannot establish a general prompt benefit or harm.

The thinking prefix is a concrete tokenizer difference.
All eight P0 outputs generate the opening thinking ID3 themselves.
Seven begin with IDs `[3, 198]`, which encode the marker and a newline.
P1 already contains those two IDs in its frozen prompt.

None of the eight P1 outputs generates another opening ID3 as its first output token.
The default template therefore did not suppress the opening thinking marker in these eight cases.
The model's behavior on this small cohort does not establish universal prompt equivalence.
This experiment does not show that this difference repairs the previous accuracy gap.

P1 remains the source-selected primary regardless of these scores.
No question, seed, prompt, or parser changes after this comparison.
The complete frozen benchmark and longer-cap rule remain unchanged.

The reused scorer marks the diagnostic collection incomplete against its full-benchmark denominator.
That generic field does not mean the eight-question diagnostic is still collecting.
The diagnostic itself completed all eight requests per arm.
Its counts must not become a 30-problem AIME score.

Evidence: `results/prompt_diagnostic_v1/` and `results/prompt_diagnostic_score/`.
