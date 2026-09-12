# Hypothetical live-row compaction and post-Epoch residual

## Counterfactual

For every real wave and layer, retain only decision-live rows while preserving their exact recorded top-8 routes. Recompute expert histograms and use the held-out-validated shape model to estimate owner-rank expert time. This is an offline sensitivity, not an Epoch implementation or measured speedup.

## Shape after compaction

Late-phase compacted live work is extremely fragmented:

| dataset | compact active experts early→late | compact p50 rows/expert | compact <=4-row fraction |
|---|---:|---:|---:|
| GSM8K | 173→71.5 | 14.25→2 | 27.0%→68.5% |
| HumanEval | 166→45 | 13→2 | 29.3%→79.4% |

So the qualitative composition is real: removing dead work exposes a much smaller, highly fragmented live set.

## Economic separation

| dataset | hypothetical live-row compaction, expert-only | perfect residual fragmentation removal | p99-trim residual | 50%-capture feasible proxy |
|---|---:|---:|---:|---:|
| GSM8K | 7.10% E2E | 2.13% | **2.12%** | **1.06%** |
| HumanEval | 6.48% E2E | 1.14% | **1.13%** | **0.57%** |

The first column is existing-work-removal context and directly adjacent to Epoch. The residual columns answer the new question. Even a zero-cost perfect residual optimizer remains below 2.2% E2E, far below the 8--10% continuation gate.

## Robustness and limitations

- P99 winsorization changes the residual by only 0.01 percentage point, so outlier tails do not create the result.
- The lower-envelope oracle is optimistic: it assumes every layer-wave can reach the local 10th-percentile count-matched expert time without communication, queueing or packing cost.
- The compaction time itself is model-predicted. It must not be presented as measured Epoch performance.
- A dedicated live-row kernel could have a different cost surface, but it would need to exceed this measured-substrate oracle by roughly 4--7× merely to reach the gate, which is not supported by the data.

Conclusion: `Epoch removes dead work → ours packs remaining live work` is conceptually clean but economically too small on this substrate.

