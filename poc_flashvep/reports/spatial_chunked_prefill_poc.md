# Spatial / Modality-Aware Chunked-Prefill Feasibility PoC

**Status: `HOLD` (offline oracle headroom is scale-limited; metadata-only cuts are weak).**

## 1. Scope and provenance

This is an offline, route-preserving analysis.  No model forward, vLLM scheduler,
DeepEP collective, or CUDA kernel was run or modified.  The requested GPU mapping
(`CUDA_VISIBLE_DEVICES=1,2,3,4`) was reserved but not used.  Every strategy uses
the same token order, prompt token IDs, top-8 expert IDs, expert-to-rank map
(`expert_id // 32`), and total assignment count; only chunk endpoints differ.

* route source: `poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609/`
* spatial source: `poc_flashvep/deepep_revalidation/results/tile_slack_mechanism_20260820_150852/stage_a/sample_manifest.json`
* route workload: 24 real-image requests, 48 layers, 128 experts, top-k=8
* route artifact branch/commit: `flashvep/multimodal-routing-ep-discovery` /
  `228378e5c792437d18bf139646c7f361e0b72ed6`
* source route capture configuration: Qwen3-VL-30B-A3B-Instruct, BF16,
  TP2/DP2/EP4/PP1, DeepEP high-throughput, eager, prefix cache off
* analysis run: `spatial_chunked_prefill_20260831_200000`
* code: `poc_flashvep/spatial_chunked_prefill/analyze.py`

Strategies:

* **Fixed**: contiguous endpoints at `b` (last remainder is retained).
* **Modality**: nearest text/vision boundary within a bounded `[0.75b, 1.25b]`
  endpoint interval.
* **Spatial**: nearest true image row boundary reconstructed from
  `token_span` and `post_merge_grid_hw`, in the same interval.
* **Modality + spatial**: nearest endpoint from the union of both boundary sets.
* **Spatial-shuffled control**: per-image 2-D coordinates are randomly permuted
  before row-boundary candidates are derived; token order/routes are unchanged.
* **Random-shifted control**: ten deterministic random endpoints, each in the
  same bounded interval, without metadata.
* **Routing oracle (upper bound only)**: exact dynamic programming over all
  bounded endpoints, minimizing the sum of visual expert tile counts.  It is not
  an implementable pre-router method.

All reported primary shape metrics are visual-token attribution metrics.  The
`BLOCK_M` proxy is selected from the complete invocation chunk M, as in the
source-audited vLLM BF16 TritonExperts default: M<=32→16, M<=96→32,
M<=512→64, otherwise 128.  This is a shape/cost proxy, not a measured kernel
latency.

## 2. Practical chunking relevance

The 24 prompts contain 128–2,363 decoder input tokens (median 328.5, p25 269.0,
p75 956.5, p90 1,488.8, max 2,363).  Visual tokens are 83.7–99.0% of each
prompt (median 93.6%, mean 93.4%).  Request categories are natural (7),
chart/document (8), fine-grained/scientific (8), and multi-image (1).

The validated vLLM 0.20 source reports chunked prefill enabled by default and
uses `max_num_scheduled_tokens` when set, otherwise `max_num_batched_tokens`
(`vllm/v1/core/sched/scheduler.py:105-112`).  At each scheduling step it starts
with that token budget (`:369-372`), caps a request by the remaining budget
(`:677-692`), schedules it, and decrements the budget (`:517-522`).  With
`max_num_batched_tokens=16384` in the captured serving configuration, these
24-request prompts would normally fit in one scheduler iteration; the budgets
128/256/512/1024 below are controlled offline chunking stress points, not a
claim that the captured production command used those values.  The scheduler
also explicitly gates request chunking on `enable_chunked_prefill` (`:682-692`).

Expected fixed contiguous chunks (all prompt tokens):

| budget | median chunks | range | requests with >1 chunk |
|---:|---:|---:|---:|
| 128 | 3 | 1–19 | 23/24 |
| 256 | 2 | 1–10 | 18/24 |
| 512 | 1 | 1–5 | 9/24 |
| 1024 | 1 | 1–3 | 6/24 |

Thus the mechanism has meaningful coverage at 128/256, while 512/1024 are
mostly one-chunk controls and cannot create a boundary effect for short prompts.

## 3. Shape metrics and oracle result

`tile_ratio_vs_fixed = fixed visual tile proxy / strategy visual tile proxy`;
values above 1 are reductions relative to fixed.  `padding_ratio_vs_fixed` is
the analogous fixed/strategy ratio for padded rows.  The medians below are
paired across the same 24 requests and all 48 layers.

| budget | strategy | tile ratio | tile reduction | padding ratio | p10 expert batch | fraction groups <=4 |
|---:|---|---:|---:|---:|---:|---:|
| 128 | Fixed | 1.000 | 0.00% | 1.000 | 1.00 | 44.85% |
| 128 | Modality | 1.019 | 1.90% | 1.011 | 1.00 | 34.89% |
| 128 | Spatial | 1.014 | 1.34% | 1.029 | 1.00 | 39.90% |
| 128 | Modality+spatial | 1.014 | 1.34% | 1.029 | 1.00 | 39.90% |
| 128 | Spatial-shuffled | 1.000 | 0.00% | 1.000 | 1.00 | 42.69% |
| 128 | Route oracle | **1.145** | **12.68%** | 1.212 | 1.00 | 34.67% |
| 256 | Fixed | 1.000 | 0.00% | 1.000 | 1.55 | 34.15% |
| 256 | Modality | 1.000 | 0.00% | 1.000 | 1.55 | 27.61% |
| 256 | Spatial | 1.001 | 0.07% | 1.001 | 1.50 | 29.42% |
| 256 | Modality+spatial | 1.001 | 0.07% | 1.001 | 1.50 | 29.42% |
| 256 | Spatial-shuffled | 1.000 | 0.05% | 1.001 | 1.50 | 31.90% |
| 256 | Route oracle | **1.175** | **14.93%** | 1.075 | 2.00 | 24.03% |
| 512 | Fixed | 1.000 | 0.00% | 1.000 | 2.00 | 23.12% |
| 512 | Modality | 1.000 | 0.00% | 1.000 | 2.00 | 22.85% |
| 512 | Spatial | 1.000 | 0.00% | 1.000 | 2.00 | 23.06% |
| 512 | Route oracle | 1.000 | 0.00% | 1.000 | 2.00 | 22.99% |
| 1024 | Fixed | 1.000 | 0.00% | 1.000 | 2.53 | 20.63% |
| 1024 | Modality | 1.000 | 0.00% | 1.000 | 2.00 | 18.72% |
| 1024 | Spatial | 1.000 | 0.00% | 1.000 | 2.53 | 20.29% |
| 1024 | Route oracle | 1.000 | 0.00% | 1.000 | 2.88 | 18.72% |

Request-level mean tile reductions (fixed versus strategy) were 7.08%, 5.66%,
0.78%, and 4.91% for the modality cut at budgets 128/256/512/1024; for the
true spatial cut they were 4.26%, 4.44%, 2.16%, and approximately 0%.  The
median reductions are much smaller (modality 1.90% at 128 and 0% at 256+;
spatial 1.34% at 128 and 0.07% at 256).  The modality+spatial policy was
identical to spatial on this manifest, so no additive gain was observed.

The route oracle reaches 12.68% median at budget 128 and 14.93% at 256, but
0% median at 512/1024 because the majority of requests are single chunks.
The oracle's lower padded-row count and improved p10 density are upper-bound
signals; they do not establish a kernel speedup.

## 4. Controls and robustness

The spatial-shuffled coordinate control removes 2-D locality while preserving
the per-image grid boundary cardinality.  Its paired median tile reduction was
0% at 128 and 0.05% at 256, versus 1.34% and 0.07% for true spatial cuts;
there is therefore no robust spatial-only advantage over the control.  The
ten-seed random-shifted control had broad chunk-count-dependent variation:
pooled tile-sum medians were 14,449.5 (128), 10,373.5 (256), 5,938.5 (512),
and 5,938.5 (1024), with ranges reaching 4,316–96,416 at 128.  This confirms
that arbitrary endpoint choices can dominate the proxy; it is not evidence for
a metadata effect.  Fixed/metadata boundaries were all constrained to the
same 0.75–1.25 budget range.

Layer-local reductions were directionally stable for the oracle: early/middle/
late mean reductions were 13.37/13.61/14.07% (128) and 10.51/10.68/11.36%
(256).  Spatial means were only 4.16/4.28/4.33% (128) and 4.19/4.47/4.70%
(256), with medians near zero at 256.  The weak metadata trend is not confined
to one decoder-depth region, but it is also not large or consistently positive
per request.  Category-level results vary (e.g. spatial median improvement at
128 is about 5.1% for chart/document and 2.1% for fine-grained, while natural
is near zero); the single multi-image sample is insufficient for a category
claim.

All scopes preserve exact assignment totals: visual scope counts the same
visual token/top-k rows for every strategy, and the all-token scope is retained
in `chunk_summary.csv` as an invariant check.  No token reorder, routing edit,
expert reassignment, or GPU execution was performed.

## 5. Figures and artifacts

Figures are generated from the final corrected run:

* `figures/plot1_chunk_tile_headroom.png` — tile proxy ratio by budget.
* `figures/plot2_padding_fragmentation.png` — padded-row ratio by budget.
* `figures/plot3_expert_batch_density.png` — p10 expert-batch density.
* `figures/plot4_oracle_headroom.png` — reduction relative to fixed.

Result directory:
`poc_flashvep/deepep_revalidation/results/spatial_chunked_prefill_20260831_200000/`

Key files:

* `analysis/chunk_layer_metrics.csv` (request × budget × strategy × chunk × layer × scope)
* `analysis/chunk_summary.csv`
* `analysis/strategy_aggregate.csv`
* `analysis/random_shifted_control.csv`
* `provenance.json`, `summary.json`

## 6. Gate and interpretation

`ORACLE_HEADROOM_GATE: HOLD` and overall `SPATIAL_CHUNKED_PREFILL: HOLD`.

The exact-route oracle clears the 10% guide only at the two smallest budgets,
but does not clear the preregistered 15% requirement at multiple scales.  More
importantly, the metadata-only modality/spatial policies recover only about
0–2% median paired headroom, and the spatial-shuffled control is comparable.
The result supports a bounded *upper-bound* opportunity when chunks are very
small, not a demonstrated spatial/modality-aware kernel or scheduler benefit.

No live grouped-MoE replay was run: the offline gate is HOLD and this PoC's
scope forbids full scheduler implementation.  The shape proxy also omits
communication, launch overhead, inter-chunk state effects, and the exact
runtime kernel configuration for mixed-modality M; these are limitations, not
positive evidence.

**Next single recommended action:** if the direction remains important, run one
bounded GPU grouped-MoE replay at budgets 128 and 256 using the exact route
histograms and the same fixed/spatial/oracle cuts, measuring real kernel time
before considering any scheduler work.  Do not generalize from the current
offline oracle to production performance.
