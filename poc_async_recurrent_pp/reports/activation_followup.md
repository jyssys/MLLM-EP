# Logit-margin follow-up

## Measurement scope

This diagnostic uses two GSM8K training questions from the existing calibration set.
Each arm processes 16 forced continuation positions per question.
The total is 32 positions per arm, not 32 independent questions.
These questions also supplied predictor calibration data.
The diagnostic does not estimate held-out task accuracy.

The test starts each arm with exact prefill and an independent copy of its historical KV.
Candidate KV then persists across the forced continuation.
The exact and candidate arms receive the same forced token sequence.
The test injects the actual predicted stage inputs before it measures logits.
It uses no offline exact state as a predictor input.

## Measured margins

The margin is the largest logit minus the second-largest logit.
The table reports arithmetic means over the 32 positions.

| Arm | Positions | Exact mean margin | Candidate mean margin | Finite logits |
|---|---:|---:|---:|---|
| W1 stale | 32 | 4.154297 | 3.332031 | 32/32 |
| W1 delta | 32 | 4.154297 | 3.480469 | 32/32 |
| W0 initializer + delta | 32 | 4.154297 | 3.363281 | 32/32 |

All three candidates reduce the mean margin in this small diagnostic.
This result does not prove that margin reduction causes the task failures.
The teacher-forced observations do not replace free-running generation.

The probe ran after the pilot screening began.
It changed no coefficient, prompt, sample selection, decoding rule, or quality threshold.
The frozen predictor SHA-256 is `ca283b0b79fb811b2265e1478948e3190cb3183e605f51ed7709c2e519bc352d`.

## Evidence

- `outputs/margins_v1/margin_samples.json`: per-position margins, KL, agreement, and finite checks.
- `outputs/margins_v1/summary.json`: aggregate values and scope.
- `outputs/margins_v1/source_snapshot.zip`: executed source.
- `outputs/margins_v1/guard_status.json`: successful execution and scoped burner restoration.

The earlier activation report contains the separate 64-position teacher-forced ablations.
The sample counts in these two diagnostic lanes must not be combined as independent evidence.
