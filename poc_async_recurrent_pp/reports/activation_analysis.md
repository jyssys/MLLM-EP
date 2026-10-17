# Activation and prediction diagnostics

## Scope

The exact activation study uses eight calibration requests and 136 positions.
Each position contributes three later recurrent rounds and three stage-input boundaries.
These 1,224 boundary observations are not independent requests.
The mean adjacent-loop cosine is 0.915495. The mean relative L2 change is 0.340335.
These values describe exact activation dynamics. They do not establish safe substitution.

The diagnostic intervention study uses two calibration requests and 32 teacher-forced positions per request.
Each arm therefore has 64 next-token comparisons, not 64 independent math problems.
Every arm starts with exact prefill. Approximate KV then persists during its forced continuation.
The evaluator compares distributions under the same token prefix and the arm's own KV history.
The final quality report uses separate free-running held-out generations.
These diagnostic requests also occur in predictor fitting. Their results do not estimate held-out prediction performance.

## Injected-input results

| Diagnostic arm | Mean KL(exact ∥ candidate), nats | Top-1 agreement |
|---|---:|---:|
| W1 stale, all three boundaries in R2–R4 | 0.266986 | 55/64 = 85.94% |
| W1 delta, all three boundaries in R2–R4 | 0.258074 | 55/64 = 85.94% |
| W0 initializer + delta, all boundaries in R1–R4 | 0.284372 | 56/64 = 87.50% |
| Delta, all boundaries in R2 only | 0.005341 | 62/64 = 96.88% |
| Delta, stage 2 boundary only in R2–R4 | 0.031626 | 59/64 = 92.19% |
| Delta, stage 3 boundary only in R2–R4 | 0.028526 | 58/64 = 90.63% |
| Delta, stage 4 boundary only in R2–R4 | 0.033313 | 59/64 = 92.19% |
| Delta, stages 2–3 boundaries in R2–R4 | 0.033358 | 59/64 = 92.19% |
| Delta, stages 3–4 boundaries in R2–R4 | 0.046190 | 58/64 = 90.63% |

The main W1 delta diagnostic has less KL than stale prediction, but both have the same top-1 agreement here.
One approximate round produces less divergence than repeated approximation in this small calibration sample.
This observation does not establish a free-running quality result or a useful physical speedup.
No correction mechanism or predictor retuning follows these diagnostic outcomes.

## Effective boundary errors

Across later rounds and stage boundaries, mean relative L2 errors are 0.549860 for stale and 0.545706 for delta.
The corresponding mean cosines are 0.820797 and 0.823433.
Warm-up 0 has mean relative L2 0.892347 and mean cosine 0.449378 over these later-round boundaries.
These errors include the accumulated candidate-state and KV effects along the forced continuation.
They are not isolated one-step forecasts from untouched exact histories.

Relative L2 is `norm(prediction - target) / norm(target)`.
An RMSE normalized by the target RMS gives the same ratio for equal-size tensors.
The raw files retain boundary, round, position, request ID, and arm.
No layer or token bootstrap supports an accuracy claim.

## Causal and reference checks

The corrected emulator retains computed preceding-stage outputs from the previous round.
It freezes every relaxed input before current-round stage execution.
CPU tests reject current-round leakage and recycled-prediction bookkeeping.
The GPU event audit checks source readiness before prediction and prediction readiness before consumption.
All event times use a single-GPU clock. They do not show distributed overlap.

The corrected diagnostic repeats exact logits and all 96 key/value tensors against the unmodified official forward.
Both calibration continuations pass bitwise parity over 32 positions each.
Every observed candidate position writes each depth/layer cache slot exactly once.
The first diagnostic version has invalid retained-state bookkeeping and remains excluded.

## Figures and raw evidence

- `outputs/plots/loop_cosine.png`
- `outputs/plots/loop_relative_L2.png`
- `outputs/plots/teacher_KL.png`
- `outputs/baseline_v1/activation_observations.json`
- `outputs/diagnostics_v2/teacher_metrics.json`
- `outputs/diagnostics_v2/boundary_errors.json`
- `outputs/diagnostics_v2/cache_audit.json`
- `outputs/diagnostics_v2/*GPU_availability.json`
