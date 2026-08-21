# FlashVEP Pre-Router Visual Signal PoC

## Environment and workload

Qwen3-VL-30B-A3B-Instruct, BF16, TP2/DP2/EP4/PP1, DeepEP high-throughput, DBO off, GPUs 4–7. The suite contains 48 unique local source images and 80 requests (48 canonical plus preregistered resolution/prompt variants). All five CV folds are grouped by source-image SHA; no variant crosses folds. No routing, weights, or runtime scheduling were changed.

## Routing target — Stage A: GO

Real visual tokens are selected from exact processor token spans; idle-DP padding and text are excluded. Across three live repeats, profile cosine was 1.000000, JSD 0, and critical-rank agreement 100.00%. All 240 source requests returned one output token and all 80 requests were repeat-exact. Whole-batch padding routing was deliberately not retained as a label; source-DP visual rows and idle-DP execution are separated by construction.

## Pre-router features

C0 is the training-fold layer prior; C1 uses only visual-token count; C2 uses processor-known image count, pixel/area, aspect, grid THW/post-merge geometry, per-image token counts and fixed aggregates. F2 adds 20 fixed summary values from the actual Qwen3-VL vision output (four output blocks × mean/std/max-absolute/mean-token-norm/std-token-norm). No routing, expert ID, downstream latency, prompt text, or category is a predictor.

Feature construction completed as preregistered. Processor metadata is available before the vision encoder; the encoder summary is available after the encoder and before layer-0 LLM MoE routing.

## Metadata prediction — Stage C: HOLD

| Model | cosine | JSD | critical | top-2 | imbalance MAE |
|---|---:|---:|---:|---:|---:|
| prior | 0.986169 | 0.005309 | 47.89% | 74.84% | 0.12608 |
| token_count | 0.986308 | 0.005254 | 48.28% | 75.62% | 0.12251 |
| metadata | 0.985658 | 0.005512 | 48.78% | 76.22% | 0.11740 |
| metadata_encoder | 0.985107 | 0.005848 | 51.30% | 76.72% | 0.11014 |

Metadata minus token-count critical accuracy is +0.49%; source-clustered 95% CI [-0.43%, +4.18%].

## Resolution — Stage D: HOLD

Same-source resolution variants: cosine 0.993999, JSD 0.002328, critical agreement 73.44%. Different images at nearest token load: cosine 0.981755, JSD 0.006883, critical agreement 52.08%. Repeat noise JSD is 0.

## Prompt robustness — Stage E: GO

Within-source prompt cosine 1.000000, JSD 0.000000, critical agreement 100.00%; the across-image control is shown above.

## Encoder signal — Stage F: HOLD

Adding the already-computed encoder summary changes critical accuracy by +2.53% (95% CI [+0.75%, +8.75%]) and JSD from 0.005512 to 0.005848.

## Availability and absolute pressure

Median processor construction time was 77.255 ms and live encoder CUDA time 19.262 ms. Metadata exists before the encoder; F2 exists at encoder completion, both before layer-0 MoE routing. A device-accurate encoder-to-router gap was not instrumented, so no unsupported lookahead duration is claimed.

Absolute-pressure errors (assignments):

- token_count: per-rank MAE 122.54, max-rank MAE 108.70, imbalance MAE 0.11735
- metadata: per-rank MAE 131.82, max-rank MAE 110.40, imbalance MAE 0.13567
- metadata_encoder: per-rank MAE 163.72, max-rank MAE 156.42, imbalance MAE 0.31763

## FINAL NOVELTY STATUS: HOLD

The strongest positive evidence is exact prompt robustness and an encoder-summary critical-rank gain of +2.53% over metadata with a positive clustered CI. The strongest counter-evidence controls the gate: metadata improves critical-rank accuracy over token count by only +0.49%, while normalized-profile JSD worsens (0.005254 → 0.005512 → 0.005848); all feature models also trail the layer prior on profile JSD. Absolute-pressure MAE likewise worsens beyond token count. Useful information before routing is therefore modest, held-out source-image generalization does not support a proactive-EP claim, and `Visual-Foresight Expert Parallelism` is not recommended from this PoC.

## Limitations

The 48 sources are bounded local assets rather than a full benchmark, encoder summaries are intentionally small and fixed, and only EP4/Qwen3-VL is tested. Resolution variants are resampled inputs under the stock processor. The representative heatmap uses the first manifest entries and fixed layers, not outcome-selected examples.

## Single recommended action

Run one preregistered external-image replication using a frozen spatially pooled encoder summary and the same source-grouped gates before designing any proactive EP scheduler.
