# Track B2 — pipeline-boundary temporal similarity

Lag-1 state was measured after layers 8/16/24/32, keyed by logical request,
block, and position.  Values below are medians across refinement waves.

| Task | Boundary | Matched rows | Cosine p50 | rel-L2 p50 | rel-L2 p90 |
|---|---:|---:|---:|---:|---:|
| GSM8K | 8 | 96.62% | 0.99695 | 0.0824 | 0.3466 |
| GSM8K | 16 | 96.62% | 0.98632 | 0.1715 | 0.5219 |
| GSM8K | 24 | 96.62% | 0.96606 | 0.2659 | 0.7821 |
| GSM8K | 32 | 96.62% | 0.98615 | 0.1783 | 0.5493 |
| HumanEval | 8 | 92.32% | 0.99673 | 0.0867 | 0.3315 |
| HumanEval | 16 | 92.32% | 0.98029 | 0.2089 | 0.5616 |
| HumanEval | 24 | 92.32% | 0.94891 | 0.3276 | 0.8385 |
| HumanEval | 32 | 92.32% | 0.97974 | 0.2164 | 0.5952 |

Cosine alone is misleading: boundary 8 looks almost identical while its
relative error is already 8--9%; boundary 24 is substantially less stable.
Late phases are usually more similar than middle phases, but the live-row
relative errors do not vanish.  For example HumanEval boundary-24 p50 rel-L2
improves from 0.3568 in middle to 0.1681 in late, yet mean error on still-live
rows remains about 0.50.

Conclusion: temporal similarity is boundary- and phase-dependent, strong
enough to motivate trajectory intervention but not a safety certificate.
