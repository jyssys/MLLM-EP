# FlashVEP DBO final numerical correctness gate

## Final status

**FINAL STATUS: HOLD**

The original `498` versus `697` split is consistent with numerical sensitivity:
the compared executions have the identical generated prefix
`2132,4977,1075`, their full-vocabulary logits are close, and a 0.25-logit
top-1/top-2 margin is reversed by a 0.5 change in token 697. However, DBO-on
generation is not fully repeatable: one of six rank/runs diverged one step
earlier. This violates the explicit GO repeatability criterion. There is no new
evidence of Attention or DeepStack metadata corruption, but bit-exact parity is
not established.

## Configuration

Qwen3-VL-30B-A3B-Instruct, BF16, TP2/DP2/EP4/PP1, DeepEP high-throughput,
vLLM 0.20, enforce eager, and the existing Attention plus DeepStack source
fixes were retained. Only physical GPUs 4,5,6,7 were exposed. Each DP rank ran
the original two-request `[blue, red]` wave three times. vLLM's existing
`raw_logits` output was converted to FP32; no sampler or model math was patched.

## Generated tokens and divergence

DBO-off was identical in all six rank/runs:

`2132, 4977, 1075, 498`

Five of six DBO-on rank/runs were identical:

`2132, 4977, 1075, 697`

Their exact first divergence is zero-based step 3, the fourth generated token:
`498` versus `697`. The prefix `2132,4977,1075` is exactly equal, so the logits
below are evaluated before either autoregressive path has diverged.

One DBO-on run (DP0 repetition 0) instead produced:

`2132, 4977, 498, 3003`

Its first divergence is zero-based step 2. At that step tokens 1075 and 498
were tied at 23.0; other DBO-on repetitions had 23.25 versus 23.125. This is
additional evidence of numerical sensitivity, but also the reason the gate is
HOLD rather than GO.

## Raw logits at the primary divergence

| Rank | DBO-off token/logit | DBO-on token/logit |
|---:|---:|---:|
| 1 | 498: 27.000 | 697: 27.250 |
| 2 | 697: 26.750 | 498: 27.000 |
| 3 | 1052: 23.125 | 1052: 23.250 |
| 4 | 279: 22.625 | 279: 23.000 |
| 5 | 358: 20.375 | 358: 20.500 |
| 6 | 582: 18.750 | 582: 18.875 |
| 7 | 847: 18.625 | 847: 18.500 |
| 8 | 419: 18.125 | 419: 18.000 |
| 9 | 429: 16.875 | 429: 17.125 |
| 10 | 264: 16.750 | 264: 17.000 |

- Token 498: off 27.000, on 27.000
- Token 697: off 26.750, on 27.250
- Top-1/top-2 margin: off 0.250, on 0.250
- Max absolute error: 0.65625
- Mean absolute error: 0.1013994
- RMSE: 0.1258357
- Cosine similarity: 0.9995824
- Relative L2 error: 0.0388658

Thus token 498 itself is unchanged while token 697 moves by 0.5 and reverses a
small 0.25 ordering. The vocabulary-wide direction remains highly similar.

## Repeatability

- DBO-off: generated tokens and step-3 logits are exactly identical across all
  three repetitions and both ranks (max difference 0).
- DBO-on: generated tokens are not fully deterministic. Among the five runs
  reaching the primary common prefix, step-3 logits differ by at most 0.0625.
  One DP0 repetition encounters an exact tie a step earlier and selects 498.
- The natural generation already supplies the requested teacher-forced sanity
  condition for five runs: the complete prefix before step 3 is identical.
  A separate forced-token path was therefore unnecessary.

## Decision

The evidence favors BF16/kernel execution-order sensitivity, most plausibly in
concurrent DBO MoE/DeepEP reduction order, rather than remaining Attention or
DeepStack state corruption. Nevertheless, the explicit GO rule also requires
stable within-mode behavior. Since DBO-on selected a different token in one
repetition, the defensible final gate is **HOLD**.

## Limitations

- Only the bounded red-prompt case and first four generated tokens were tested.
- Raw logits were observed at the vLLM sampler boundary, not intermediate
  hidden states or per-layer MoE outputs.
- An external process occupied about 4 GiB on GPUs 4-7; the already fixed 1 GiB
  KV cache was retained and startup utilization guard set to 0.90. Model math
  and software versions were unchanged.
- Two failed startup/validation attempts are retained in the raw directory.

## One recommended next action

Capture the pre-lm-head hidden state and per-layer checksum for the identical
prefix in the one nondeterministic DBO-on step to localize whether the 0.25
variation first appears inside MoE/DeepEP reduction or after it.

Raw directory:
`poc_flashvep/deepep_revalidation/results/dbo_final_numerical_correctness_20260807_175000/`
