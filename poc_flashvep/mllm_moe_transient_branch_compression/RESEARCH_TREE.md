# Research tree

Each node is promoted only by measured branch geometry **and** an economic
mapping. The following table preserves the preregistered initial status; the
definitive disposition of every node appears in the final evidence ledger.

| ID | Causal hypothesis | Intervention/control | Gate | Status |
|---|---|---|---|---|
| H01 | Same-image, same-expert visual branches contain output-level redundancy. | Output-oracle grouping vs no sharing. | >=15% strict-safe reduction. | CHEAP_PROBE |
| H02 | Experts contract visual input differences. | Paired input vs output relative distance; text control. | contraction ratio <0.8 broadly. | CHEAP_PROBE |
| H03 | Contraction is specifically stronger for Vision than Text. | Match layer/expert and compare modalities. | robust positive modality gap. | CHEAP_PROBE |
| H04 | Original-order 1D RLE recovers the output oracle. | Contiguous same-expert runs vs output oracle. | >=80% oracle recovery. | CHEAP_PROBE |
| H05 | True 2D adjacency predicts safe sharing better than 1D order. | Same pair budget, 2D vs 1D. | >=5 pp strict-safe gain. | CHEAP_PROBE |
| H06 | 2D connected components expose longer safe groups. | CC vs adjacent pairs. | >=25% rows in safe runs >=2. | CHEAP_PROBE |
| H07 | Fixed 2x2 windows are a deployable spatial proxy. | 2x2 vs output oracle/random. | >=80% oracle recovery. | CHEAP_PROBE |
| H08 | Hidden similarity predicts expert-output similarity. | Hidden-nearest vs output-nearest. | strong rank correlation and Pareto. | CHEAP_PROBE |
| H09 | Whole top-k route overlap adds signal beyond hidden similarity. | Matched hidden threshold, route-overlap on/off. | >=5 pp recovery. | CHEAP_PROBE |
| H10 | Low-router-weight branches tolerate sharing more safely. | Weight-stratified perturbation, same row budget. | materially better quality/reduction. | UNTESTED |
| H11 | Output medoids outperform anchors at useful group sizes. | Same groups, medoid vs first. | >=5 pp gate recovery. | CHEAP_PROBE |
| H12 | Hidden medoids are an adequate deployable substitute. | Same groups, output vs hidden medoid. | >=80% output-medoid recovery. | CHEAP_PROBE |
| H13 | Centroid evaluation is safer than evaluating an existing token. | Exact centroid expert forward vs medoid. | quality improvement exceeds creation cost. | UNTESTED |
| H14 | Router-weighted centroids improve weighted combined updates. | Plain vs weighted centroid. | higher combined gate pass. | UNTESTED |
| H15 | A useful group-size boundary exists. | Caps 2/3/4/6/8, same tolerance. | nontrivial cap >=3 remains safe. | CHEAP_PROBE |
| H16 | Compressibility persists across early/middle/late layers. | Identical policy across layer strata. | >=2 strata, not one layer. | CHEAP_PROBE |
| H17 | Higher-resolution images create more local redundancy. | Matched category at multiple grids. | monotone safe-reduction gain. | FRESH_CAPTURE |
| H18 | Compressibility is content-dependent but predictable. | Natural/OCR/chart/document strata. | stable proxy-controlled differences. | FRESH_CAPTURE |
| H19 | Errors cancel in the weighted multi-branch update. | Branch-local gate vs exact combined-update gate. | useful extra safe rows. | CHEAP_PROBE |
| H20 | One-layer-safe sharing preserves the next router. | Causal HF intervention, router top-k/KL. | >=99% top-k agreement. | GPU_QUALITY |
| H21 | Gated intermediate or down-input is more compressible than full expert output. | Capture gate/up, activation, down output. | >=25% rows plus >=10% direct oracle. | GPU_CHILD |
| H22 | Contribution-weighted sharing survives where raw outputs do not. | Rank by `g_ie E_e(h_i)` with matched row budget. | new quality-efficiency frontier. | GPU_CHILD |
| H23 | A large eligible layer subset yields economic headroom. | Best static 8–16 layers vs all layers. | direct E2E >=10%. | ANALYTICAL_CHILD |
| H24 | Exact spatial route-run packing saves indexing/layout work without approximation. | Same output, packed vs stock timing. | overhead mass >=8% E2E. | TIMING_CHILD |
| H25 | Contractive experts admit an expert-specific low-rank/codebook oracle. | Per-expert PCA/codebook vs identity. | direct oracle >=10%. | CPU_CHILD |
| H26 | Scalar norm or group residual correction repairs anchor reuse cheaply. | No repair vs two allowed corrections. | material gate recovery at low cost. | GPU_CHILD |

## Forced rethink checkpoints

- RETHINK_05: after H01–H05, check whether spatial locality is being mistaken for output redundancy.
- RETHINK_10: after H06–H10, check whether the only surviving effect is low-weight expert skipping.
- RETHINK_15: after H11–H15, check whether representative creation erases any arithmetic saving.
- RETHINK_20: after H16–H20, check whether one-layer error compounds and whether any direct request headroom remains.

## Final evidence ledger

Each row below is a distinct causal question, not a knob setting. `CLOSED`
means the preregistered gate failed; it does not mean the measurement was
missing.

| ID | EXPECTED | OBSERVED | FAILED ASSUMPTION / NEW SYSTEM FACT | DIRECT E2E IMPLICATION | NEXT CHILDREN | Final status |
|---|---|---|---|---|---|---|
| H01 | Same-expert visual branches share outputs. | Output-oracle safe reduction median 0%, maximum 0.0283%. | Same route owner does not imply interchangeable expert functions. | Safe compute saving is effectively zero. | H08, H13 | CLOSED |
| H02 | Expert nonlinearity contracts nearby visual states. | Median output/input distance ratio was 1.18–1.22 for deployable visual pairings. | Qwen experts usually amplify, rather than erase, token differences. | Compression cannot rely on a contraction basin. | H16, H21 | CLOSED |
| H03 | Vision contracts materially more than Text. | Matched Vision output error was 0.033–0.072 lower, but absolute errors remained 0.60–0.83. | A real relative modality effect is not an operational tolerance regime. | No MLLM-specific saving. | none: low headroom | CLOSED |
| H04 | 1D contiguous route runs recover the oracle. | 22.2% of rows occur in runs >=2, but pair-safe rate was 0.063%. | Route continuity is packing structure, not output redundancy. | Approximate RLE cannot remove meaningful work. | H24 | CLOSED |
| H05 | True 2D neighbors are substantially safer. | Pair-safe rate 0.091% versus 0.063% for 1D; output error 0.935. | Spatial adjacency adds only a tiny relative signal. | Far below the 15% geometry gate. | H06, H07 | CLOSED |
| H06 | 2D components expose long safe groups. | 42.9% of branches lie in route components >=2, yet safe group reduction is 0%. | Large same-expert components are functionally heterogeneous. | No exact spatial-sharing headroom. | H24 | CLOSED |
| H07 | Fixed local windows are a deployable sharing proxy. | Spatial grouping failed every median strict gate. | Locality cannot replace output-oracle safety. | No prototype. | none | CLOSED |
| H08 | Hidden nearest neighbors predict output neighbors. | Spearman(input, output)=0.881, yet output-nearest lowers median error only 0.831→0.816 and safety 0.150→0.162%. | Predictive ranking exists but the attainable error floor is enormous. | Proxy improvement cannot rescue the oracle. | H09 | CLOSED |
| H09 | Top-k route overlap adds independent safety signal. | Held-out RMSE reduction 0.034%; 14,727 high-overlap pairs had zero strict-safe cases. | Route affinity is not a branch-output equality certificate. | Zero deployable headroom. | none | CLOSED |
| H10 | Low-weight branches tolerate sharing. | Weight features add 0.419% held-out RMSE; at 5% rows, sharing rel-L2 11.75% versus skipping 12.92%. | The small advantage is not safe and becomes low-contribution skipping. | Ideal 5%-row request benefit <=0.16%. | H22 | CLOSED |
| H11 | Output medoids make grouping safe. | Oracle medoids improve error slightly but median strict pass remains 0. | Representative choice is not the dominant failure. | No method headroom. | H13 | CLOSED |
| H12 | Hidden medoids recover output-medoid quality. | Both fail strict safety; their difference is irrelevant at the gate. | The problem is not merely proxy selection. | No deployable selector. | none | CLOSED |
| H13 | Evaluating an exact centroid is much safer. | Router-weighted centroid is best, but cap-2 affected rel-L2 is 32.3%; strict pass median 0. | Even newly evaluated representatives cannot summarize the group. | Creation overhead is moot because quality fails. | H14, H26 | CLOSED |
| H14 | Router-weighted centroids repair weighted updates. | 32.3% versus 33.5% affected rel-L2 for plain centroid; strict pass still 0. | Router weighting is a minor numerical adjustment. | No economic mapping. | none | CLOSED |
| H15 | A useful group-size boundary exists. | Caps 2/3/4/6/8 all have median safe reduction 0%. | Failure begins at the smallest possible group. | No smaller nontrivial compression unit exists. | H21 | CLOSED |
| H16 | Compressibility persists across layer strata. | Layers 4–44 have exactly 0 safe reduction; layer 47 alone reaches only 0.0283%. | A late-layer relative contraction is isolated and still unsafe. | No 8–16-layer eligible subset. | H23 | CLOSED |
| H17 | Resolution increases local redundancy. | 336/448/672 captures all remain at median 0 safe reduction. | More patches add route runs, not interchangeable outputs. | No resolution-triggered method. | none | CLOSED |
| H18 | Content type exposes predictable safe strata. | Natural, fine-grained, and chart/document samples all have median 0 safe reduction. | No tested content family crosses the gate. | No selective request policy. | none | CLOSED |
| H19 | Branch errors cancel in the combined update. | Rare random cases reach 1.339% safe row reduction, median 0 and branch-local unsafe. | Accidental cancellation is not a stable causal mechanism. | Maximum is still below even the negative gate. | H20 | CLOSED |
| H20 | One-layer sharing preserves downstream routing. | A 5% output-oracle intervention gives 97.83% next-route agreement and 6.85% logit rel-L2. | Local output ranking does not imply trajectory safety. | Fails the 99%/1% propagation screen. | multi-layer control | CLOSED |
| H21 | Gate/up or down-input can be shared when full output cannot. | Sharing the down input from token j produces exactly the already-tested full output E_e(h_j). | This child is algebraically dominated by H01 unless a learned correction is added. | Perfect all-expert compute is only 3.26% request E2E. | H25 | CLOSED |
| H22 | Contribution-weighted sharing creates a new Pareto point. | At 1/2/5/10% rows, affected rel-L2 is 6.07/8.87/11.75/16.39%; strict pass 0. | It is only slightly better than skipping and shares its prior-art space. | 5% ideal expert saving <=0.16% E2E. | none | CLOSED |
| H23 | A useful static layer subset exists. | Only layer 47 is relatively favorable; no 8–16-layer subset passes safety. | Layer selectivity cannot accumulate material work. | Far below 10% direct oracle. | none | CLOSED |
| H24 | Exact route-run packing alone removes material overhead. | Route/layout work is at most 49.9 ms of a ~4.3 s trace (<=1.16%). | Route runs may help implementation, but their entire cost mass is small. | Below the 8% packing gate. | none | CLOSED |
| H25 | Expert-specific codebooks exploit a narrow output manifold. | Even eliminating all expert compute is <=3.26% request E2E in the measured high-resolution regime. | The systems ceiling kills codebook research before quality modeling. | Below the 10% child gate. | none | CLOSED |
| H26 | Cheap scalar/norm correction repairs anchor reuse. | Centroid evaluation is strictly more expressive and still misses by ~32x; scaling cannot repair direction error. | Cheap norm repair cannot change cosine/direction. | No reason to prototype. | none | CLOSED |
| H27 | Short greedy agreement proves end-task safety. | 8/8 short outputs matched, but six-layer route agreement fell to 41.93% and logit rel-L2 rose to 23.0%. | Tiny answer sets are an insensitive quality test. | Benchmark promotion is disallowed. | none | CLOSED |

## Forced rethink outcomes

### RETHINK_05 — locality is not redundancy

The first five tests separated route/spatial locality from expert-function
similarity. Same-expert adjacency is common, but the expert output is farther
apart than the input. New non-cosmetic questions were representative choice
(H11/H13), contribution weighting (H10/H22), and exact packing (H24).

### RETHINK_10 — do not rename skipping as sharing

Route overlap and low router weight failed as safety certificates. The only
remaining weight-conditioned behavior was to perturb branches that matter
less, which is expert skipping under another name. New questions were exact
centroid evaluation (H13/H14), downstream route stability (H20), and
sub-expert algebra (H21).

### RETHINK_15 — the group is not representable by one point

Medoids and exact centroids both failed already at group size two. Therefore
map-building or representative-creation tuning cannot fix the main failure.
New questions were eligible-layer subsets (H23), codebook dimensionality
(H25), and cheap repair (H26).

### RETHINK_20 — local error compounds before it pays

One-layer quality failed the registered route/logit screen, and six-layer
rollout amplified the disturbance. Economic analysis then showed that even
perfect removal of every expert computation is only 3.26% of the measured
request E2E. This jointly closes full-output sharing, sub-expert sharing,
packing, and learned-codebook successors without spending GPU time on methods
whose ceiling is below the spec.
