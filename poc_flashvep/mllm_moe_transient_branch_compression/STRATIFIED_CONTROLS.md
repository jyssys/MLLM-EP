# Stratified route and router-weight controls

## Question

Could route overlap or router weight identify safe visual branch-sharing pairs
after controlling the hidden-state distance that a deployable proxy already
observes?

## Data and split

- Source: 358,291 same-expert pair observations from the fresh nine-image,
  six-layer Qwen3-VL capture.
- Primary population: 338,490 visual pairs.
- Held-out unit: complete request/image (`leave-one-sample-out`, nine folds).
- Baseline covariates: input relative-L2 plus layer, expert, pair policy, and
  whether a spatial distance is defined.
- Additions were tested in order: whole-top-k route Jaccard, the minimum and
  maximum pair router weights, and spatial distance.

## Held-out results

| Model | Macro median MAE | Macro median RMSE | Incremental median RMSE reduction |
|---|---:|---:|---:|
| Hidden-distance controls | 0.081236 | 0.109482 | -- |
| + route Jaccard | 0.081163 | 0.109406 | 0.034% |
| + router weights | 0.080534 | 0.108674 | 0.419% |
| + spatial distance | 0.080490 | 0.108645 | 0.035% |

The route addition ranged from -0.159% to +0.100% RMSE reduction across held
out requests. It therefore has neither material magnitude nor stable sign.

As a non-parametric control, pairs were matched within the same request,
layer, expert, and pair policy, within 0.02 input relative-L2 and 0.01 minimum
router weight. Only 12 strata contained at least three low-route-overlap and
three high-route-overlap pairs. High minus low route-overlap output relative-L2
had median -0.00354, mean -0.01593, and p10/p90 [-0.03797, +0.02974]; only
58.3% of strata favored high overlap. This is small and inconsistent.

Most decisively, among 14,727 visual pairs with route Jaccard >=0.75, median
expert-output relative-L2 remained 0.6059 (p10 0.2967), and **zero pairs** met
the 1% strict branch-local tolerance.

## Decision

- H09 (route overlap adds a usable safety signal): **CLOSED**.
- H10 (low router weight enables branch sharing): **CLOSED as sharing**. At a
  fixed row budget, contribution-weighted sharing is only marginally better
  than skipping and does not pass the 1% combined-update gate; this collapses
  into the already crowded expert-skipping direction rather than establishing
  branch-output reuse.
- Spatial metadata adds only 0.035% held-out RMSE reduction after the other
  controls and cannot rescue the failed geometric premise.
