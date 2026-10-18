# Stage 4: bounded causal predictor probe

The C64 refined oracle passed the exploratory headroom threshold.
This allowed a small supervised probe. No RL or exit-gate training occurs.
The predictor uses known prompt length, prompt text statistics, and declared domain metadata.
It does not use future output tokens, true exit traces, or post-prefill hidden states as inputs.

Thirty-two calibration requests fit separate depth and output-length ridge models.
Calibration leave-one-out error selects the ridge coefficient.
The 96 held-out requests do not select model parameters.

| Target | Calibration N | Held-out N | Ridge MAE | Constant MAE |
| --- | --- | --- | --- | --- |
| mean_depth | 32 | 96 | 0.157745 | 0.147477 |
| output_length | 32 | 96 | 7.539452 | 7.516927 |

| C | Prompt-only policy | Modeled gain vs generic | Oracle recovery |
| --- | --- | --- | --- |
| 8 | ridge_depth_length | -3.002% | -308.57% |
| 8 | ridge_depth_matched_length | -3.002% | -308.57% |
| 16 | ridge_depth_length | 1.360% | 15.13% |
| 16 | ridge_depth_matched_length | 1.360% | 15.13% |
| 32 | ridge_depth_length | -1.469% | -17.22% |
| 32 | ridge_depth_matched_length | -1.469% | -17.22% |
| 64 | ridge_depth_length | -7.960% | -62.34% |
| 64 | ridge_depth_matched_length | -5.563% | -43.57% |

These scheduling values are cost-model estimates, not measured predictor-runtime speedups.
The matched-length arm keeps the output-length prior fixed to isolate predicted depth.
The simple predictor does not recover useful broad oracle headroom.
Its held-out depth error is worse than the constant-depth control.
The status is `PREDICTION_BLOCKED` for this probe.
This is not proof that every possible predictor must fail.

The derived work forecast multiplies predicted depth by predicted decode-token count.
Its held-out MAE is 56.850 loop-token operations.
The constant control's MAE is 55.276 loop-token operations.

We do not train a more complex predictor to rescue this result.
We do not start RL. We do not implement a new serving framework.
