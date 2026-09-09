# Quality propagation diagnostic

## Scope and correctness boundary

This is a Hugging Face single-GPU quality diagnostic on physical GPUs 5--7,
not EP serving performance evidence.  It loads the exact Qwen3-VL checkpoint,
keeps all tokens and original top-k routes/weights at the intervention layer,
and applies the counterfactual only to visual branches.  Four-request one-layer
and eight-request six-layer GQA controls use identical inputs and greedy decode.

The “output oracle” recomputes every visual routed branch and selects the
lowest-error same-expert substitution for 5% of visual rows.  It is deliberately
non-deployable and more favorable than any hidden/spatial proxy.  Contribution
skipping and adjacent-position sharing use the same 5% row budget.

## One-layer intervention (layer 44, 4 requests)

| Policy | logit rel-L2 median | logit cosine median | route agreement median | greedy 8-token exact | answer matches baseline/method |
|---|---:|---:|---:|---:|---:|
| Captured-output oracle share | 6.85% | 0.99765 | 97.83% | 4/4 | 3/4, 3/4 |
| Lowest-contribution skip | 7.54% | 0.99715 | 97.76% | 4/4 | 3/4, 3/4 |
| Spatial contiguous share | 7.46% | 0.99721 | 97.46% | 4/4 | 3/4, 3/4 |

The output oracle is slightly better than skipping at the same row count, but
it already fails the preregistered >=99% next-route agreement.  Greedy equality
on four short answers is a coarse pass, not benchmark-quality proof.

## Six-layer accumulation (layers 4/12/24/36/44/47, 8 requests)

| Policy | logit rel-L2 median | logit cosine median | route agreement median | greedy 8-token exact | answer matches baseline/method |
|---|---:|---:|---:|---:|---:|
| Captured-output oracle share | 23.00% | 0.97338 | 41.93% | 8/8 | 6/8, 6/8 |
| Lowest-contribution skip | 23.41% | 0.97242 | 41.18% | 8/8 | 6/8, 6/8 |
| Spatial contiguous share | 23.66% | 0.97195 | 40.65% | 8/8 | 6/8, 6/8 |

The accumulated hidden/routing trajectory is not preserved.  The exact short
answers happen to be unchanged for these eight examples, but the 23% logit
drift and roughly 42% route agreement fail the local and propagation screens by
large margins.  Expanding this unsafe setting into a benchmark campaign would
not repair the failed geometry gate and is not used as positive evidence.

## Causal interpretation

1. Sharing is only modestly better than zeroing the cheapest contributions;
   the proposed operation does not open a distinct quality frontier.
2. Spatial adjacency is worse than the impossible output oracle and no better
   than skipping.
3. Later nonlinear layers can preserve a few short greedy answers while the
   internal route and logit distributions have already diverged substantially.
   Greedy exactness therefore cannot override the stricter mechanistic gate.
4. A full benchmark was not run because no policy reached the prerequisite
   strict-safe branch reduction or >=99% route-agreement gate.  This is a gate
   outcome, not an unreported accuracy claim.
