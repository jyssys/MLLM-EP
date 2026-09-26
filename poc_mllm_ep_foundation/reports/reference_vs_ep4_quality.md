# Stock reference versus physical EP4 quality

Status: **UNRESOLVED (sanity passes; benchmark parity not established)**

## Paired deterministic sanity

| Metric | Stock reference TP4 | Physical TP2/DP2/EP4 |
|---|---:|---:|
| Requests | 12 | 12 |
| Empty outputs | 0 | 0 |
| Prompt leaks | 0 | 0 |
| Mean generated tokens | 85.50 | 84.92 |
| EOS stop / length cap | 5 / 7 | 5 / 7 |
| Mean request wall | 5,016.01 ms | 13,025.95 ms |

Exact generated token sequence agreement was 2/12. This low exact-string rate
does not by itself prove semantic failure: the two distributed paths use
different collective/reduction orders, and seven requests were capped at 96
tokens. Conversely, sensible-looking output is not a parity proof.

The blank-image control described an empty white image and the wrong-image
control described the replacement rocket image. This is direct image plumbing
evidence. Manual inspection found complete, image-dependent text in both
runtimes.

## Missing evidence

There is no task accuracy for this manual cohort, no paired correctness delta,
and therefore no paired confidence interval or -1 pp non-inferiority test.
Teacher-forced per-layer logits/MoE tensor parity was not captured at the
precision required for a PASS verdict. The wall times above use bounded debug
generation and are not a performance comparison between equivalent production
backends.

Evidence: `artifacts/quality/sanity_reference_tp4/`,
`artifacts/quality/sanity_physical_ep4/`, and
`artifacts/analysis/quality_bridge.json`.
