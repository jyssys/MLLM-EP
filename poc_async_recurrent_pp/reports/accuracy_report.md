# PARPP free-running accuracy report

## Decision and scope

Verdict: `QUALITY_FAIL`.
The result rejects the three tested predictors under the frozen pilot protocol.
It does not establish full-benchmark scores or rule out every possible causal predictor.

The primary evaluation contains 16 GSM8K test questions and 16 MATH-500 questions.
Each of four arms generates an independent response to all 32 questions.
The primary wave therefore contains 128 generations but only 32 independent questions.
The complete GSM8K-1,319 and MATH-500-500 evaluations are `NOT_RUN` because no candidate survives screening.
The unused frozen pilot questions do not enter these scores.

## Generation and evaluator

The model is `ByteDance/Ouro-1.4B-Thinking`, revision `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`.
All arms use BF16, fixed R4, SDPA, exact full-depth prefill, and independent depth-indexed KV.
The official tokenizer supplies the chat template with `add_generation_prompt=True`.
Each question receives the same suffix:

```text
Please reason step by step and put your final answer in \boxed{}.
```

The official `model.generate()` performs free-running greedy decoding with EOS enabled.
Every arm uses the same fixed batches of eight questions.
Candidate-generated tokens and approximate KV persist into later decoding positions.
No teacher token or reference hidden state enters this task evaluation.

The primary output cap is 8,192 tokens after the complete 2,048-token wave triggered the frozen expansion rule.
All arms repeat the same questions and batch groups at the higher cap.
All 128 shared output prefixes match between the two caps over their common generated lengths.
The shorter-cap evidence remains available.

The final parser version is `parpp-final-v2`.
GSM8K uses normalized numeric answers. MATH-500 uses strict Math-Verify equivalence.
The primary endpoint is final-answer accuracy.
A matching answer in a capped response would count as correct under this endpoint.
The separate completed-answer endpoint also requires EOS.
No primary response has a correct capped answer, so both endpoint scores are identical here.
Independent CPU rescoring changes zero answer grades and zero completed-answer grades.

## Primary measured accuracy

All intervals below are marginal 95% bootstrap intervals.
The bootstrap uses 10,000 paired question resamples with seed 20261017.
Accuracy differences use candidate minus exact, in percentage points.
The absolute accuracy intervals use the same question resamples.
No token, layer, recurrent round, or repeat becomes an independent sample.

| Dataset | Arm | Correct / n | Accuracy % [95% CI] | Difference pp [paired 95% CI] |
|---|---|---:|---|---|
| GSM8K pilot | Exact R4 | 15/16 | 93.75 [81.25, 100.00] | Reference |
| GSM8K pilot | W1 stale | 8/16 | 50.00 [25.00, 75.00] | -43.75 [-68.75, -18.75] |
| GSM8K pilot | W1 delta | 7/16 | 43.75 [18.75, 68.75] | -50.00 [-75.00, -25.00] |
| GSM8K pilot | W0 initializer + delta | 6/16 | 37.50 [12.50, 62.50] | -56.25 [-81.25, -31.25] |
| MATH-500 pilot | Exact R4 | 14/16 | 87.50 [68.75, 100.00] | Reference |
| MATH-500 pilot | W1 stale | 3/16 | 18.75 [0.00, 37.50] | -68.75 [-87.50, -43.75] |
| MATH-500 pilot | W1 delta | 3/16 | 18.75 [0.00, 37.50] | -68.75 [-87.50, -43.75] |
| MATH-500 pilot | W0 initializer + delta | 1/16 | 6.25 [0.00, 18.75] | -81.25 [-100.00, -62.50] |

Every candidate's CI upper bound is below -10 pp on both endpoints and both datasets.
The automatic controller and independent final adjudicator both return `QUALITY_FAIL`.
The result exceeds the specification's allowable losses by a large margin.
These small-sample intervals cannot establish 1–2 pp noninferiority.
They do support rejection of the large regressions observed in this pilot.
The intervals do not include a familywise multiple-comparison correction.

## Completion, parsing, and length

All rows have 16 requested generations and zero generation errors.
Parsing failure is `16 - parsed` and remains separate from truncation.
These categories can overlap.

| Dataset | Arm | EOS completed | Parsed | Parse failures | Truncated | Mean tokens | Median tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| GSM8K | Exact | 16 | 16 | 0 | 0 | 687.44 | 547.5 |
| GSM8K | Stale | 10 | 11 | 5 | 6 | 3,618.56 | 1,130.5 |
| GSM8K | Delta | 10 | 10 | 6 | 6 | 3,668.44 | 1,593.0 |
| GSM8K | W0 | 9 | 10 | 6 | 7 | 4,023.81 | 1,853.5 |
| MATH-500 | Exact | 14 | 14 | 2 | 2 | 2,541.25 | 1,787.5 |
| MATH-500 | Stale | 3 | 4 | 12 | 13 | 7,030.00 | 8,192.0 |
| MATH-500 | Delta | 3 | 8 | 8 | 13 | 6,947.25 | 8,192.0 |
| MATH-500 | W0 | 3 | 4 | 12 | 13 | 6,851.75 | 8,192.0 |

The exact GSM8K maximum is 2,298 generated tokens.
Every other dataset/arm row reaches the common 8,192-token cap in at least one response.
The candidates often continue without a usable final answer.
However, truncation does not explain every failure.
Some candidates reach EOS with a clear, parsed, incorrect answer.
The score remains a bounded-generation result, not a claim about unlimited decoding.

## Paired correctness transitions

The counts below use final-answer accuracy.
The completed-answer transitions are identical in this primary wave.

| Dataset | Candidate | Both right | Exact right → candidate wrong | Exact wrong → candidate right | Both wrong |
|---|---|---:|---:|---:|---:|
| GSM8K | Stale | 8 | 7 | 0 | 1 |
| GSM8K | Delta | 7 | 8 | 0 | 1 |
| GSM8K | W0 | 6 | 9 | 0 | 1 |
| MATH-500 | Stale | 3 | 11 | 0 | 2 |
| MATH-500 | Delta | 3 | 11 | 0 | 2 |
| MATH-500 | W0 | 1 | 13 | 0 | 2 |

There are 59 right-to-wrong candidate/question pairs across the three candidates.
These are not 59 independent questions.
No exact-baseline error becomes a matching candidate answer in this pilot.
Median first-divergence indices are 8.0 for stale, 8.5 for delta, and 8.0 for W0.
The indices are zero-based generated-token positions over 32 paired requests per candidate.
The exact-prefill first token matches in every pair.

### EOS-completed counterexamples

| Candidate | Question | Exact and reference answer | Candidate answer | Candidate output tokens |
|---|---|---:|---:|---:|
| Stale | `gsm8k/test/589` | 15 | 2.5 | 910 |
| Delta | `gsm8k/test/206` | 860 | 460 | 591 |
| W0 | `gsm8k/test/1058` | 250 | 450 | 645 |

Each example reaches EOS and contains an explicit boxed answer.
The raw prompt, output, token IDs, and grade appear in the corresponding primary-wave JSON file.
The complete transition list and all 59 counterexamples remain machine-readable.

## Diagnostic agreement is not task accuracy

The separate teacher-forced diagnostic uses two calibration questions and 64 positions per arm.
Mean KL(exact ∥ candidate) is 0.266986 nats for stale and 0.258074 nats for delta.
Both agree with the exact top-1 token on 55/64 positions, or 85.94%.
W0 has mean KL 0.284372 nats and agreement 56/64, or 87.50%.
All observed diagnostic logits are finite.
These calibration observations do not predict the severe free-running accuracy losses reliably.
They do not establish a safe approximation.

## Evidence and limits

- `outputs/final_analysis/final_verdict.json`: final decision, endpoint definitions, and summary.
- `outputs/final_analysis/accuracy_summary.csv`: per-dataset statistics and intervals.
- `outputs/final_analysis/request_scores.csv`: independent canonical per-response grades.
- `outputs/final_analysis/scorer_audit.json`: unchanged original grades and rescoring comparison.
- `outputs/final_analysis/correctness_transitions.json`: all paired transitions and first divergence.
- `outputs/final_analysis/baseline_right_candidate_wrong.json`: complete counterexample list.
- `outputs/final_analysis/cross_cap_prefix_audit.json`: 128/128 common-prefix matches.
- `outputs/validation/evidence_final.json`: source, device, KV, EOS, and manifest checks.
- `outputs/final_plots/`: primary accuracy, paired differences, and diagnostic boundary errors.

Eight exact task-sanity questions overlap the pilot. They are not fresh confirmation samples.
Predictor fitting instead uses eight disjoint GSM8K training questions.
No predictor or quality threshold changes after this evaluation.
The full canonical evaluations and all distributed architecture stages remain `NOT_RUN`.
