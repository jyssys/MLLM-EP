# Contribution-weighted and adjacent-successor results

## Contribution-weighted sharing versus skipping

This oracle uses true captured branch outputs to rank substitutions by their
weighted change to the exact combined MoE update.  The matched control ranks
branches by the norm of the contribution and replaces them with zero.  Thus it
directly asks whether same-expert sharing adds a new Pareto point beyond expert
skipping.

| Visual row budget | Output-oracle share affected rel-L2 | Skip affected rel-L2 | Spatial-oracle affected rel-L2 | strict affected-token pass (share/skip) |
|---:|---:|---:|---:|---:|
| 1% | 6.07% | 7.16% | 9.00% | 0% / 0% |
| 2% | 8.87% | 9.68% | 11.30% | 0% / 0% |
| 5% | 11.75% | 12.92% | 15.93% | 0% / 0% |
| 10% | 16.39% | 17.60% | 20.35% | 0% / 0% |
| 25% | 26.15% | 28.44% | 36.39% | 0% / 0% |

Sharing is only 6--15% better in local error than skipping, not an order-of-
magnitude frontier shift.  At 5%, layer 47 is the best stratum (3.02% affected
rel-L2; 82.35% pass a relaxed 5% screen), but it still has 0% strict 1% pass,
and the other five layers are 9.76--17.12% even under the impossible oracle.
No 8--16-layer safe subset exists.

The economic upper bound is decisive: removing 5% of all visual expert work
can save at most `5% * 3.26% = 0.16%` request E2E in the measured high-resolution
vision regime, before map/packing/expansion overhead.  The S2 child is therefore
closed on quality, economics, and novelty (MoDES, AnyExperts, and ACE directly
occupy contribution-aware skipping).

## S1: sub-expert-stage compression

For Qwen's expert,

```text
z_i = SiLU(W_gate h_i) * (W_up h_i)
E_e(h_i) = W_down z_i.
```

Sharing `z_j` for token `i` and running the down projection gives exactly
`E_e(h_j)`, the full-output substitution already measured.  It cannot produce
a smaller final error than the output-aware upper bound.  Sharing only the down
stage can remove at most one third of expert FLOPs (1.09% perfect request
upper bound); sharing only gate/up at most two thirds (2.17%).  A new
intermediate capture cannot cross the 8% implementation gate and is not run.

## S3: eligible layers

Layers 4--44 expand nearest visual differences.  Layer 47 contracts them, but
its output-nearest rel-L2 remains 40.79%, strict pair safety is 0.971%, and the
strict group oracle's maximum safe row reduction is 0.0283%.  A one-layer
island is neither the requested multi-stratum phenomenon nor economically
material.

## S4: exact route-run packing

Route runs are common, but the previous full-serving Nsight trace caps all
router plus DeepEP-layout kernel work at about 1.16% of observed trace wall even
if both categories disappeared entirely.  Exact run packing removes only a
subset and leaves expert computation unchanged.  The 8% direct-E2E gate fails;
no packing prototype is justified.

## S5/S6: codebook and cheap repair

An expert-output codebook that magically eliminated every expert calculation
is capped at the 3.26% expert-compute share.  Scalar norm repair cannot change
cosine, while the median output-nearest cosine is far below the approximately
0.995 cosine needed for <=10% equal-norm rel-L2.  Centroid and router-weighted
centroid are stronger repairs and were tested exactly; their best cap-2
affected rel-L2 remains about 32%.  Learned auxiliary models are outside scope.

## Adjacent-domain decision

The main full-output oracle, sub-expert arithmetic, contribution weighting,
layer subsets, exact packing, codebooks, and cheap repair all fail either the
strict quality gate or an independent <8% direct E2E bound.  No adjacent child
has a >=10% direct oracle.
