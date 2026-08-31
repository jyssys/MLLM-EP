# Multimodal Routing × EP Discovery (offline PoC)

## Executive conclusion

The strongest reproducible signals are (i) a modality-boundary expert-routing shock and (ii) 2-D spatial locality in visual routing. Cross-layer persistence and visual traffic bursts were not supported. This is a discovery result, not an optimization implementation, and no new GPU run was required.

**Primary recommendation:** run one bounded live DeepEP instrumentation study that tags dispatch/combine timing by the already identified boundary and spatial-window regimes; do not implement a scheduler until that attribution exists.

## Candidate scorecard

Scores are ordinal 0–3 (effect, consistency, MLLM-specificity, MoE-specificity, EP relevance, control robustness, optimization potential, novelty potential), used to rank candidates rather than as a statistical gate.


| Candidate phenomenon | Effect size | Consistency | MLLM-specific | MoE-specific | EP relevance | Verdict |
|---|---:|---:|---:|---:|---:|---|

| Modality-boundary expert-routing shock (A) | 3 | 3 | 3 | 3 | 2 | STRONG |

| Spatial locality × expert/EP routing (C) | 3 | 3 | 3 | 3 | 3 | STRONG |

| Image-conditioned visual working-set expansion / cross-image inconsistency (F/G) | 2 | 3 | 3 | 3 | 2 | PROMISING |

| Local visual EP traffic burst (D) | 0 | 1 | 2 | 2 | 1 | REJECT |

| Cross-layer visual persistence (B) | 0 | 1 | 2 | 2 | 1 | REJECT |

| Directional EP migration (H) | 1 | 1 | 2 | 2 | 1 | WEAK |


Top 3: boundary shock; spatially local but globally image-conditioned routing; modality-dependent expert working-set expansion.

## Configuration and provenance

* Analysis branch/base at start: `flashvep/multimodal-routing-ep-discovery`, HEAD `51152d5c5c4b179bf190b2d7c1e5b9cee4649631`.
* Route source: `poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609/`, 24 image requests plus paired text controls, 48 layers, top-8, 128 experts, EP4 linear map `expert_id // 32`, image token ID `151655`.
* Historical capture provenance was TP2/DP2/EP4, BF16, DeepEP high-throughput, DBO off, eager; that artifact was captured on physical GPUs 4–7. The requested mapping for any live follow-up is `CUDA_VISIBLE_DEVICES=1,2,3,4`; this run was offline CPU-only and used no GPU.
* Spatial metadata: `tile_slack_mechanism_20260820_150852/stage_a/sample_manifest.json`, using `token_span`, `post_merge_grid_hw`, and image boundaries; no hard-coded 784/grid shape.
* Exact analysis command: `python poc_flashvep/multimodal_routing_ep_discovery/analyze.py --run-id 20260831_173000`. Fixed seed `20260831`; fixed spatial pair cap 128/relation/image/layer; fixed window sizes 32/64/128; bootstrap seed and policy are recorded in code.

## Workload and controls

| category       |   requests |   median_tokens |   median_vision_ratio |
|:---------------|-----------:|----------------:|----------------------:|
| chart_document |          8 |            1255 |                0.9817 |
| fine_grained   |          8 |             277 |                0.9242 |
| multi_image    |          1 |             511 |                0.955  |
| natural        |          7 |             276 |                0.9275 |


Across 24 image requests, visual tokens have median ratio 0.936 (range 0.837–0.990); the internal Text comparison always uses non-image tokens from the same image-containing request. Paired text-only routes are diagnostic controls for arbitrary boundaries/working sets, not the primary modality label.


## PoC A — modality-boundary routing transition

For adjacent-token expert-set distance (`1 − Jaccard`), medians are: **TT 0.7692, VV 0.8571, TV 0.9333, VT 1.0000**. EP-rank JSD medians are TT 0.0778, VV 0.0982, TV 0.0939, VT 0.1405.


Request/layer paired expert-distance difference vs TT: TV mean +0.0882 (bootstrap 95% CI [0.0799, 0.0959], positive in 76.2%), VT mean +0.1798 (CI [0.1745, 0.1854], positive in 95.3%).


The paired text-only arbitrary-boundary control has median distance 0.7692 and median destination JSD 0.0778, below the image boundary values. The label-shuffle control has median distance 0.8571; it removes modality alignment while preserving counts.


Interpretation: a boundary shock is repeated at every image boundary and layer, with a larger and asymmetric vision-entry/exit expert-set change. Destination-rank changes are smaller than expert-ID changes, so this is strongest as a routing-state transition, not proof of a communication latency penalty. See `figures/plot1_modality_boundary_transition.png` and `plot7_transition_directionality.png`.


## PoC B — cross-layer persistence

Equal-cap samples give mean expert-set Jaccard Vision 0.0352 vs Text 0.0329, destination-set Jaccard Vision 0.8340 vs Text 0.8292. Vision−Text expert Jaccard difference is 0.0023 (CI [0.0014, 0.0033]); although the CI excludes zero, the absolute effect is only about 0.0023. Top-1 persistence is near zero for expert IDs and ~0.25 for rank IDs.


Verdict: **REJECT** as a useful Vision-specific persistence phenomenon; statistical detectability here does not imply a practically meaningful depth-temporal effect. `figures/plot2_cross_layer_persistence.png`.


## PoC C — 2-D spatial locality × expert routing

Median metrics: adjacent expert Jaccard 0.1429, expert JSD 0.7500, destination JSD 0.1022; random expert Jaccard 0.0667, expert JSD 0.8750, destination JSD 0.1188.


At the request/image/layer level, random-minus-adjacent destination-JSD mean is 0.0150 (bootstrap CI [0.0141, 0.0158], positive in 85.5%). Random-minus-adjacent expert-JSD mean is 0.1101 (positive in 98.9%). The direction is present in early, middle, and late layer strata, while far-vs-random is weaker.


Verdict: **STRONG** routing-level candidate. It is a genuine 2-D visual structure coupled to expert and EP-rank routing, but live dispatch/combine attribution remains unmeasured. `figures/plot3_spatial_routing_locality.png` and `plot8_representative_ep_heatmap.png`.


## PoC D — spatial-region EP traffic burst

|                 |   max_dest_fraction |   dest_hhi |   dest_entropy |
|:----------------|--------------------:|-----------:|---------------:|
| ('text', 32)    |           0.09375   |  0.038559  |        2.5816  |
| ('text', 64)    |           0.0898438 |  0.0345154 |        2.68619 |
| ('text', 128)   |           0.0839844 |  0.0316982 |        2.76588 |
| ('vision', 32)  |           0.0625    |  0.0247498 |        2.86969 |
| ('vision', 64)  |           0.0585938 |  0.0223312 |        2.9669  |
| ('vision', 128) |           0.0566406 |  0.020771  |        3.02812 |


Visual windows are *less* destination-concentrated than text windows (for 32 tokens, max-rank fraction 0.0625 vs 0.0938; HHI 0.0247 vs 0.0386). Thus the proposed visual burst/whole-window concentration does not hold in these traces. Verdict: **REJECT**. `figures/plot4_spatial_region_ep_burst.png`.


## PoC F/G — working set and cross-image consistency

| modality   |   unique_experts |   effective_experts |   expert_entropy |   top4_fraction |   top8_fraction |   ep_coverage |
|:-----------|-----------------:|--------------------:|-----------------:|----------------:|----------------:|--------------:|
| text       |               58 |             41.9769 |         0.770217 |        0.244565 |        0.396739 |            58 |
| vision     |               70 |             54.7223 |         0.824865 |        0.18125  |        0.30625  |            70 |


With equal per-request token subsampling, Vision has median 70 unique experts vs Text 59, effective experts 54.72 vs 41.98, and lower top-4/top-8 concentration (Vision 0.181/0.306 vs Text 0.245/0.397). This expansion repeats across natural, chart/document, fine-grained, and multi-image categories; it is not only a token-count effect.


Across requests at equal 64-token samples, visual expert-histogram cosine is 0.5556 vs text 0.7485, and JSD is 0.3061 vs 0.2022. This means visual routing is local/structured but strongly image-content-conditioned rather than a single shared visual expert vocabulary. Verdict: **PROMISING** combined F/G phenomenon. `figures/plot5_working_set.png` and `plot6_cross_image_consistency.png`.


## PoC H — transition directionality

TV mean centered rank-migration norm/mean step 0.3416/2.6891; VT 0.2670/3.0428. Mean-vector norms are small relative to per-transition norms and cosine consistency is not stable enough to claim a repeated rank migration direction. Verdict: **WEAK**.


## PoC E — live EP execution latency

Not run. Existing artifacts have token-level routes and EP destination labels but not boundary/spatial-window-attributed dispatch/expert/combine timing. A full live run was intentionally avoided in this discovery pass because it would require new instrumentation and could not be interpreted without changing the bounded analysis protocol. Therefore this report does not claim a latency or makespan effect.


## Negative controls and limitations

* Label-shuffle and arbitrary text-boundary controls are included. Spatial pairs use a fixed random seed and equal relation caps; coordinate permutation control is represented by the random-pair baseline rather than a second coordinate permutation file.
* No dense MLLM execution was run; Q1 (absence in Dense MLLM) is a theoretical boundary, not an empirical negative-control result. The text-only paired routes are MoE controls, not a dense-model control.
* Expert IDs are mapped to EP ranks using the historical validated linear placement `expert_id//32`; alternative placement could change rank-level effects.
* Route artifacts were historically captured on GPUs 4–7. This branch performed CPU-only analysis; any follow-up live command must use only physical GPUs 1–4.
* Router outputs are top-k IDs without probabilities, so entropy is assignment/set entropy, not router-logit confidence.
* CSVs are compressed where large; all exact top-k IDs are retained in `raw/per_token_layer.csv.gz`.

## Direct answers

1. **Beyond histogram?** Yes: modality-boundary position and 2-D adjacency predict expert-set/rank similarity; global visual working-set expansion is also repeatable.
2. **Token/spatial/temporal structure?** Strong position/spatial signals; no useful Vision-specific cross-layer persistence.
3. **EP destination?** Spatial adjacency changes destination-rank JSD modestly but consistently; visual burst concentration is refuted.
4. **Best MLLM-specific candidate?** Spatially local yet image-conditioned expert/EP routing, with a repeatable modality-boundary shock. Dense absence is not directly tested.
5. **Different optimization opportunity?** A future boundary/spatial-window-aware EP communication/phase mechanism is plausible, but no implementation is justified yet because live latency attribution is missing.
6. **One next PoC:** bounded live Qwen3-VL run with `CUDA_VISIBLE_DEVICES=1,2,3,4`, tagging DeepEP dispatch/combine events by modality boundary and spatial window, with no routing or placement change.

## Artifact index

* Result directory: `poc_flashvep/deepep_revalidation/results/multimodal_routing_ep_discovery_20260831_173000/`
* Figures: `plot1_modality_boundary_transition.png`, `plot2_cross_layer_persistence.png`, `plot3_spatial_routing_locality.png`, `plot4_spatial_region_ep_burst.png`, `plot5_working_set.png`, `plot6_cross_image_consistency.png`, `plot7_transition_directionality.png`, `plot8_representative_ep_heatmap.png`.
* Analysis code: `poc_flashvep/multimodal_routing_ep_discovery/analyze.py`; report builder: `make_report.py`.
