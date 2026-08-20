# FlashVEP Vision-Tile Motivation Profiling

## 1. Experiment configuration

Read-only prefill routing capture on Qwen3-VL-30B-A3B-Instruct, BF16,
TP2/DP2/EP4/PP1, DeepEP high-throughput, vLLM 0.20, eager execution, and
physical GPUs 4–7. DBO was disabled to avoid ubatch segmentation; the validated
Attention/DeepStack source fixes remained installed. The router and expert
placement were not changed. vLLM 0.20's public routed-expert buffer does not
handle DeepEP's sequence-parallel shape, so a read-only wrapper saved the exact
`topk_ids` passed to that buffer. TP0/TP1 contiguous sequence chunks were
concatenated per model call, one padding token was removed where needed, and
calls were reassembled in submission order. Every recovered request has shape
`[prompt token, 48 layers, top-k 8]`; IDs span the valid 0–127 expert range.

Spatial coordinates use processor `image_grid_thw` and merge size 2. Qwen3-VL's
post-merge token order is the row-major `(H/2, W/2)` logical grid documented by
the model's encoder metadata path; no fixed 784-token assumption is used.

## 2. Workload/sample manifest

The bounded local suite contains 34 requests across
4 categories. It includes natural scenes/objects,
fine-grained scientific/texture imagery, charts/documents/interfaces, varied
resolutions, and one two-image diagnostic. No dataset was downloaded.

| sample | category | DP rank | prompt tokens | vision tokens | image metadata |
| --- | --- | ---: | ---: | ---: | --- |
| astronaut | natural | 1 | 276 | 256 | astronaut.png [512, 512] grid=[1, 32, 32] |
| bit_allocate | chart_document | 1 | 671 | 648 | bit-allocate.png [1161, 572] grid=[1, 36, 72] |
| brick | fine_grained | 0 | 277 | 256 | brick.png [512, 512] grid=[1, 32, 32] |
| camera | natural | 1 | 276 | 256 | camera.png [512, 512] grid=[1, 32, 32] |
| cat | natural | 0 | 146 | 126 | chelsea.png [451, 300] grid=[1, 18, 28] |
| cell | fine_grained | 0 | 378 | 357 | cell.png [550, 660] grid=[1, 42, 34] |
| chessboard_gray | chart_document | 1 | 87 | 64 | chessboard_GRAY.png [200, 200] grid=[1, 16, 16] |
| chessboard_rgb | chart_document | 0 | 87 | 64 | chessboard_RGB.png [200, 200] grid=[1, 16, 16] |
| clock_motion | fine_grained | 0 | 129 | 108 | clock_motion.png [400, 300] grid=[1, 18, 24] |
| coffee | natural | 0 | 248 | 228 | coffee.png [600, 400] grid=[1, 24, 38] |
| coffee_rocket | multi_image | 0 | 511 | 488 | coffee.png [600, 400] grid=[1, 24, 38]; rocket.jpg [640, 427] grid=[1, 26, 40] |
| coins | natural | 1 | 128 | 108 | coins.png [384, 303] grid=[1, 18, 24] |
| color_wheel | fine_grained | 1 | 165 | 144 | color.png [371, 370] grid=[1, 24, 24] |
| deep_field | natural | 1 | 857 | 837 | hubble_deep_field.jpg [1000, 872] grid=[1, 54, 62] |
| fast_gptq | chart_document | 1 | 1589 | 1566 | fast_gptq.png [1717, 916] grid=[1, 58, 108] |
| grass | fine_grained | 0 | 277 | 256 | grass.png [512, 512] grid=[1, 32, 32] |
| gravel | fine_grained | 1 | 277 | 256 | gravel.png [512, 512] grid=[1, 32, 32] |
| histology | fine_grained | 1 | 277 | 256 | ihc.png [512, 512] grid=[1, 32, 32] |
| horse | natural | 1 | 140 | 120 | horse.png [400, 328] grid=[1, 20, 24] |
| logo | chart_document | 0 | 279 | 256 | logo.png [500, 500] grid=[1, 32, 32] |
| method | chart_document | 0 | 2363 | 2340 | method.png [2481, 960] grid=[1, 60, 156] |
| microaneurysms | fine_grained | 1 | 85 | 64 | microaneurysms.png [102, 102] grid=[1, 16, 16] |
| model_card | chart_document | 0 | 807 | 784 | card_3.png [1575, 519] grid=[1, 32, 98] |
| moon | natural | 0 | 276 | 256 | moon.png [512, 512] grid=[1, 32, 32] |
| motorcycle | natural | 0 | 388 | 368 | motorcycle_left.png [741, 500] grid=[1, 32, 46] |
| motorcycle_right | natural | 1 | 388 | 368 | motorcycle_right.png [741, 500] grid=[1, 32, 46] |
| phantom | fine_grained | 1 | 165 | 144 | phantom.png [400, 400] grid=[1, 24, 24] |
| retina | fine_grained | 1 | 1957 | 1936 | retina.jpg [1411, 1411] grid=[1, 88, 88] |
| rocket | natural | 1 | 280 | 260 | rocket.jpg [640, 427] grid=[1, 26, 40] |
| scanned_page | chart_document | 0 | 95 | 72 | page.png [384, 191] grid=[1, 12, 24] |
| text_page | chart_document | 0 | 93 | 70 | text.png [448, 172] grid=[1, 10, 28] |
| tui_log | chart_document | 1 | 1255 | 1232 | tui-log-streaming.png [1400, 900] grid=[1, 56, 88] |
| tui_main | chart_document | 0 | 1255 | 1232 | tui-main.png [1400, 900] grid=[1, 56, 88] |
| tui_model_selection | chart_document | 0 | 1255 | 1232 | tui-model-selection.png [1400, 900] grid=[1, 56, 88] |

Full paths and SHA-256 values are in `poc_flashvep/deepep_revalidation/results/tile_slack_mechanism_20260820_150852/stage_a/sample_manifest.json`.

## 3. Plot 1 — Visual-token dominance

**PLOT1_STATUS: GO**

Median/mean vision ratios are 0.9259/0.9043; p25/p75 are
0.8654/0.9630. Fractions above 0.5, 0.7, and 0.8 are
1.000, 1.000, and
0.853. Token and top-k assignment ratios agree within
0.000e+00.

![Figure 1A](../deepep_revalidation/results/stage_a/figures/plot1_visual_token_ratio.png)

*Figure 1A.* Each bar is one real-image request; the dashed line marks vision
majority. Interpret broad category-wide height above the line as visual-token
dominance, not as evidence of spatial routing structure.

## 4. Plot 2 — Vision-dominated critical-rank excess

**PLOT2_STATUS: GO**

For 1627 meaningful request-layers, median visual share of
critical excess is 0.9789; vision exceeds
non-vision in 0.980, and explains >70% in
0.924. Median vision-only/nonvision-only
imbalances are 104.00/
12.00 assignments. The decomposition
identity error is at most 0.000e+00. The filter is:
delta_total >= max(8 assignments, 1% of mean rank load).

| category | meaningful layers | median visual contribution | fraction vision > non-vision |
| --- | ---: | ---: | ---: |
| chart_document | 574 | 0.9844 | 0.970 |
| fine_grained | 477 | 0.9603 | 0.979 |
| multi_image | 48 | 0.9929 | 1.000 |
| natural | 528 | 0.9822 | 0.991 |

![Figure 2A](../deepep_revalidation/results/stage_a/figures/plot2_critical_rank_excess.png)

*Figure 2A.* Layer-wise medians decompose the selected total-load critical
rank's excess into signed vision and non-vision terms. Negative contributions
are retained. This is assignment evidence; per-rank expert CUDA latency was not
captured, so it does not prove a latency-critical rank match.

## 5. Plot 3 — Spatial tile routing signatures

**PLOT3_STATUS: GO**

The preregistered primary metric is mean pairwise JSD of EP-rank routing
distributions. Controls preserve each spatial tile's token count; random uses
10 fixed seeds. The gate requires both 2x2 and 4x4: spatial/random >=1.20,
paired bootstrap 95% CI above zero, and >=70% request-image-layer pairs above
their random mean.

| grid | spatial JSD | sequential JSD | random JSD | spatial/random | 95% CI of difference | fraction spatial > random |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 2x2 | 0.004709 | 0.005178 | 0.002363 | 1.993 | [0.002214, 0.002483] | 0.861 |
| 4x4 | 0.015120 | 0.014406 | 0.010029 | 1.508 | [0.004852, 0.005318] | 0.920 |

Secondary expert-JSD spatial/random ratios are
1.646 (2x2) and
1.319 (4x4).

| category | 2x2 ratio | 2x2 fraction > random | 4x4 ratio | 4x4 fraction > random |
| --- | ---: | ---: | ---: | ---: |
| chart_document | 2.127 | 0.898 | 1.525 | 0.927 |
| fine_grained | 1.688 | 0.787 | 1.359 | 0.879 |
| multi_image | 2.008 | 0.844 | 1.699 | 0.958 |
| natural | 2.161 | 0.890 | 1.618 | 0.943 |

![Figure 3A](../deepep_revalidation/results/stage_a/figures/plot3_tile_rank_heatmap.png)

*Figure 3A.* The highest-JSD 4x4 request/layer is shown as a diagnostic, not a
cherry-picked aggregate claim; rows are tiles and columns are EP ranks.

![Figure 3B](../deepep_revalidation/results/stage_a/figures/plot3_spatial_vs_controls.png)

*Figure 3B.* Distributions over all request-image-layer observations compare
spatial grouping with same-size sequential and random controls. Spatial values
must exceed random, rather than merely be nonzero, to support novelty.

## 6. ReBA source-image diagnostic

[ReBA](https://arxiv.org/abs/2608.00574) reports source-image routing
correlation and motivates image-level balancing. The two-image request here
yields source-image rank-JSD median
0.002704 over layers. This small diagnostic
is consistent with testing image-level boundaries but is not a ReBA replication.
The spatial gate is deliberately conditioned on beating within-image random
grouping, so image correlation alone cannot make Plot 3 pass. Here, both 2x2
and 4x4 spatial groups beat random controls, adding within-image spatial
structure without contradicting ReBA's coarser image boundary.

## 7. Overall motivation gate

**FINAL MOTIVATION STATUS: GO**

Plot 1, Plot 2, and Plot 3 are independently gated. The overall tile-specific
motivation is GO only when all three pass; a Plot 3 NO-GO makes the tile novelty
story NO-GO even if visual tokens and visual excess dominate.

The strongest positive evidence is Plot 3's spatial/random rank-JSD ratio:
1.993 at 2x2 and 1.508 at
4x4, with paired 95% intervals wholly above zero. The strongest counter-
evidence is that absolute rank-JSD remains small (0.004709
and 0.015120 bits), and the 2x2 sequential control is
slightly stronger than spatial grouping; locality is statistically structured,
but its practical scheduling value is not established.

## 8. Threats and limitations

- The suite is bounded and locally available rather than a random benchmark
  sample; chart/document assets are research/UI figures, not full ChartQA/MMMU.
- Routing IDs, not router weights, were captured; the public vLLM buffer could
  not represent the DeepEP sequence-parallel shape, requiring a read-only hook.
- EP-rank criticality is inferred from assignment counts; actual per-rank expert
  CUDA time was not instrumented.
- Only one multi-image request is available, so the ReBA-style diagnostic is
  descriptive.
- Results cover one model, expert placement, precision, and hardware topology.

## 9. Next single recommended action

Run the identical preregistered spatial/random analysis on a small, locally
cached benchmark subset with source-image IDs (at least 16 samples per category)
before designing any tile scheduler.
