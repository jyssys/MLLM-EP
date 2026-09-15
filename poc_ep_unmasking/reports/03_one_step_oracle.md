# 03 — One-step future-route screen and its boundary

`ONE_STEP_ORACLES.csv` preserves the stock transfer count and considers one
near-cutoff selected/unselected position swap. It computes the resulting
next-step max-rank assignment vector from the *baseline's* subsequent routes
in five representative layers. Future route data are used, but the swapped
trajectory was not executed. In particular, the selected token's baseline
next hidden state is finalized whereas it would remain MASK after the swap.
Therefore this is a **frozen-baseline-route sensitivity screen**, not a
causal perfect oracle, measured EP latency, or request E2E speedup.

| Delta | GSM8K mean/median max-rank reduction | HumanEval partial mean/median |
| ---: | ---: | ---: |
| 0.010 | 0.124% / 0% | 0.081% / 0% |
| 0.020 | 0.244% / 0% | 0.191% / 0% |
| 0.050 diagnostic | 0.637% / 0% | 0.617% / 0% |
| 0.100 diagnostic | 1.291% / 0% | 1.446% / 0% |

At delta 0.02, only 5.70% of 474 matched GSM8K next steps and 7.98% of
213 partial HumanEval next steps had any positive max-rank reduction. The
combined mean is 0.228% of max-rank *assignments*, before any load→latency
conversion. This is not a valid direct-E2E bound, but it provides no credible
signal for a costly counterfactual campaign.

There is also a structural constraint: for a live-only compacted runtime,
if both policies finalize the same `k_t` positions in step `t`, they have the
same number of fresh positions in `t+1`. With fixed top-k=8 across the same
routed layers, their total expert rows are equal. A preferred expensive
position is removed, but a different position remains live in its place.
Only within-invocation rank/expert geometry can improve; this is exactly the
weak physical effect found in the previous matched-composition study at the
strongest EP4 scale. In the current vanilla runtime, even the fresh-row
reduction does not occur because every ready request executes all 32 block
positions.

The gate was stopped here. No measured one-step alternative GPU replay or
causal full-trajectory oracle was performed, so no request-level gain is
claimed.
