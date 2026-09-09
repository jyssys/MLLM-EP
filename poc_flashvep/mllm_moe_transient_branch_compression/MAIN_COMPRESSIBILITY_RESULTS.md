# Main compressibility and grouping results

## Capture population

- Model/runtime: real Qwen3-VL-30B-A3B-Instruct, BF16,
  TP2/DP2/EP4, DeepEP HT, TritonExperts, DBO off.
- Nine real images: natural, fine-grained texture/object, and chart/document;
  336/448/672-pixel inputs.
- Layers 4/12/24/36/44/47; every prompt token captured.
- 2,211 visual tokens, 106,128 visual routed branch assignments, 358,291
  evaluated same-expert pair observations.
- Reconstructed stock combined output: minimum cosine 0.9999943 and worst
  rel-L2 0.3412%.  Exact checkpoint replay is stronger: median branch rel-L2 0,
  worst p99 `1.56e-8`, minimum cosine 0.9999983.

## Pair geometry

| Modality/pair | input rel-L2 median | expert-output rel-L2 median | contraction ratio median | strict-safe fraction |
|---|---:|---:|---:|---:|
| Vision arbitrary | 1.0186 | 1.1681 | 1.1304 | 0.029% |
| Vision hidden-nearest | 0.6749 | 0.8309 | 1.2173 | 0.150% |
| Vision output-nearest | 0.6893 | 0.8162 | 1.1750 | 0.162% |
| Vision contiguous 1D | 0.8072 | 0.9672 | 1.1844 | 0.063% |
| Vision true 2D neighbor | 0.7783 | 0.9347 | 1.1889 | 0.091% |
| Text arbitrary | 1.1038 | 1.2455 | 1.1171 | 0% |
| Text hidden-nearest | 0.9462 | 1.0987 | 1.1464 | 0% |
| Text output-nearest | 0.9613 | 1.0830 | 1.1177 | 0% |

Vision outputs are absolutely closer than Text outputs, but Vision inputs are
also much closer.  On common support within the same layer/expert/pair type and
input rel-L2 within 0.02, Vision-minus-Text output rel-L2 is about -0.033 for
hidden-nearest and -0.072 for output-nearest pairs.  This is a small residual
Vision advantage, not an operationally compressible regime: output-nearest
Vision rel-L2 remains 0.816 and only 0.162% of directed pairs pass the loose
branch-local cosine>=0.99/rel-L2<=0.10 screen.

Experts generally **expand**, rather than contract, the difference between
nearby visual inputs.  Layer 47 is the sole systematic exception (output-
nearest contraction ratio median 0.739), yet its output rel-L2 is still 0.408
and its pair-safe rate remains below 1%.  No 8--16-layer eligible subset exists.

Hidden distance is already highly informative: the Vision input/output
distance Spearman correlation is 0.881.  An impossible output-nearest oracle
improves median output rel-L2 only from 0.831 to 0.816 and strict pair safety
from 0.150% to 0.162% relative to hidden-nearest selection.

## Strict group oracle

Under the preregistered branch-local gate and the combined-update
rel-L2<=1%/cosine>=0.9999 gate:

- output-oracle median safe branch reduction: **0%** for every cap 2/3/4/6/8;
- maximum single sample/layer safe reduction: **0.0283%**;
- all layers 4--44: exactly 0%; only layer 47 contains any output-oracle group;
- text output-oracle reduction: exactly 0%.

The primary >=15% screen is therefore missed by more than 500x even when
grouping sees the true expert outputs.  Per-rank safe row reductions are also
zero in the median; there is no hidden critical-rank saving.

## RLE and spatial structure

Route structure does exist:

- 1D same-expert run fraction at length>=2: median 22.22%, p90 58.82%;
- 1D run fraction at length>=4: median 0%, p90 17.50%;
- 2D connected-component fraction at size>=2: median 42.86%, p90 79.55%;
- 2D component fraction at size>=4: median 0%, p90 57.55%.

But this is packing locality, not functional redundancy.  Contiguous and 2D
neighbor outputs have rel-L2 0.967 and 0.935; pair-safe rates are 0.063% and
0.091%.  The deployable contiguous/neighbor/window policies produce median
safe row reduction 0%, including at cap 2.  They recover none of the already
tiny group-level output oracle in a stable multi-layer sense.

## Representative choice

Exact centroid evaluation was validated against the checkpoint rather than
approximated from captured outputs.  For cap-2 contiguous groups, affected-
token combined-update rel-L2 medians are approximately:

- anchor 0.481;
- output medoid 0.461;
- centroid 0.335;
- router-weighted centroid 0.323.

The weighted centroid is the best tested representative, but still has median
strict-pass fraction 0% and error 32x above the 1% gate.  The result therefore
rejects “a better representative will rescue spatial sharing.”

## Main gate

`PRIMARY_COMPRESSIBILITY_GATE: FAIL` at geometry and local quality.  No compact
DeepEP execution prototype is allowed because the output-aware upper bound is
orders of magnitude below the 15% row-reduction prerequisite.
