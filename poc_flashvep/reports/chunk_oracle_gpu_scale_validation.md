# Chunk-oracle GPU and scale validation

**Bounded conclusion:** `STAGE1_STATUS: GO`; `STAGE2_OFFLINE_STATUS: GO` for
the measured route-shape upper bound; `CHUNK_ORACLE_GPU_SCALE_VALIDATION: GO`
with an important serving-validation limitation.  The exact-route cuts reduced
the measured grouped-MoE expert time by 37.36% (budget 128) and 38.90%
(budget 256), and the longer route traces retain at least 10% median offline
oracle headroom at budgets 128/256/512/1024.  The GPU replay uses the validated
layer-24 activation capture cycled to each route length, so it is GPU truth for
the operator/cut mechanism, not yet a live layer-specific serving result.

## 1. Scope and configuration

This validation changes only contiguous cut points.  Token order, top-8 expert
IDs, expert placement, dtype, and assignment totals are invariant.  No
scheduler, router, placement, or model code was changed.

| item | value |
|---|---|
| model | Qwen3-VL-30B-A3B-Instruct (local snapshot) |
| dtype | BF16 |
| vLLM | 0.20, enforce eager |
| parallelism | TP2 / DP2 / EP4 / PP1 |
| communication | DeepEP high-throughput |
| MoE backend | TritonExperts; `DeepEPHTPrepareAndFinalize` |
| placement | linear, 32 of 128 experts per EP rank |
| visible physical GPUs | **1,2,3,4 only** (`CUDA_VISIBLE_DEVICES=1,2,3,4`) |
| DBO / prefix cache | off / off |
| Stage 1 layers | 0, 12, 24, 36, 47 |
| Stage 1 requests | coins, cat, logo, coffee, coffee_rocket, model_card, retina, method |
| Stage 1 timing | 5 warmups, 20 measured iterations per rank/configuration |
| base commit | `212e93872078f23ad763607af2a1de547ba4a78e` |

The previous spatial-chunk analysis used the same bounded endpoint rule:
every oracle chunk is in `[0.75b, 1.25b]`, and the exact route-aware DP
minimizes the sum of visual expert tile counts.  Fixed cuts use strict
contiguous `b`-token chunks.  Thus an oracle can also avoid a very small final
chunk; this is part of the permitted cut-point intervention and is reported as
a limitation because it combines tile-shape and invocation-count effects.

## 2. Stage 1 — GPU truth at budgets 128 and 256

The replay ran all 8 requests × 5 layers × 2 strategies × 4 EP ranks.  All
rank files completed with `status=ok`; all 160 fixed/oracle observations had
`route_identity=true`, `token_partition_identity=true`, and passing output
correctness.  The replay invokes the real TritonExperts and DeepEP path for
each chunk; it does not synthesize routing.

### 2.1 Primary expert result

Reduction is `1 - oracle/fixed`; values are paired medians over the 40
request/layer observations at each budget.

| budget | fixed max-rank expert (ms) | oracle (ms) | median reduction | p25–p75 reduction | positive / ≥5% |
|---:|---:|---:|---:|---:|---:|
| 128 | 1.9755 | 1.4406 | **37.36%** | 27.06–47.30% | 40/40, 40/40 |
| 256 | 1.4069 | 0.8685 | **38.90%** | 30.98–51.50% | 40/40, 40/40 |

The smallest paired reduction was 25.17% (budget 128) and 16.89% (budget
256); the largest was 67.78% and 77.97%, respectively.  Per-layer medians
were positive at every measured layer and every request median was positive.

### 2.2 Wall/communication breakdown

| budget | wall reduction | dispatch reduction | combine reduction |
|---:|---:|---:|---:|
| 128 | **40.92%** | 18.24% | 26.03% |
| 256 | **43.40%** | 8.38% | 20.09% |

The primary gate therefore passes:

`STAGE1_STATUS: GO` (measured expert-kernel reduction is well above the 10%
guide at both budgets, with the same direction in every paired observation).

### 2.3 Offline proxy versus GPU

The prior offline visual-tile proxy and GPU reduction have the same direction,
but not the same magnitude:

| budget | offline all-token proxy | offline visual proxy | GPU expert reduction |
|---:|---:|---:|---:|
| 128 | 10.44% | 12.15% | **37.36%** |
| 256 | 4.73% | 6.30% | **38.90%** |

This confirms that the proxy identified a real opportunity, while the measured
kernel path adds launch/shape effects not represented by the scalar tile proxy.
It is not evidence that every production prefill receives a 37–39% speedup:
the replay uses a validated layer-24 activation tensor cycled to each route
length and serially sums chunks.

## 3. Stage 2 — bounded long multimodal route scale

Because Stage 1 passed, a four-request local-only multi-image capture was run
with the same Qwen3-VL TP2/DP2/EP4 DeepEP path, DBO off, max length 16,384,
and no new data download.  Routes contain 48 layers × 8 experts/token and
were saved exactly as returned by the live model.

| request | images | decoder tokens | visual tokens |
|---|---:|---:|---:|
| long_6img_natural_fine | 6 | 3,203 | 3,170 |
| long_8img_mixed | 8 | 6,984 | 6,947 |
| long_10img_chart_mixed | 10 | 5,234 | 5,193 |
| long_12img_broad | 12 | 9,065 | 9,020 |

Three of four requests are above 4K tokens and one is above 8K.  No 16K
request was available without expanding beyond the bounded local sample
suite, so 16K coverage is a limitation rather than an extrapolation.

### 3.1 Exact route-oracle headroom by scale

The DP and endpoint constraint are identical to the prior offline oracle.  The
table reports all-token shape proxy (visual-only values are within 0.1–1.0
percentage points of these medians).

| budget | fixed median chunks | oracle median chunks | all-token tile reduction | visual tile reduction | positive pairs |
|---:|---:|---:|---:|---:|---:|
| 128 | 48.0 | 38.5 | **16.39%** | **16.35%** | 192/192 |
| 256 | 24.5 | 19.5 | **16.26%** | **16.13%** | 192/192 |
| 512 | 12.5 | 10.0 | **24.41%** | **24.25%** | 192/192 |
| 1024 | 6.5 | 5.5 | **11.62%** | **10.77%** | 192/192 |

Per-request medians at 512 were 17.20%, 28.48%, 23.96%, and 24.98% (in
6/8/10/12-image order); at 1024 they were 18.71%, 10.16%, 12.13%, and
8.24%.  Thus the >=10% result at 512 is not caused by one request; the 1024
median is above 10% but has one request below that line.  The long-route
analysis is an offline upper bound, not a measured 512/1024 CUDA latency.

`STAGE2_OFFLINE_STATUS: GO` (repeated >=10% route-shape headroom at all four
budgets in this bounded long suite).  The result is materially stronger than
the previous short-prompt 512/1024 result, where most requests had only one
chunk.

### 3.2 Interpretation of chunk-count confounding

At long scale the oracle commonly uses fewer, larger chunks (for example,
budget 512 fixed median 12.5 versus oracle 10.0).  This is allowed by the
pre-registered `[0.75b,1.25b]` boundary constraint, but means the observed
proxy headroom is a combination of expert tile alignment and avoiding tiny
tail invocations.  No token reorder or route edit was used.  A future serving
experiment must hold scheduler invocation policy and measure layer-specific
activation/kernel time to separate these components.

## 4. Artifacts and figures

Final Stage 1 GPU result:
`poc_flashvep/deepep_revalidation/results/chunk_oracle_gpu_scale_validation_20260831_221000/`

Stage 2 live route capture and scale analysis:
`poc_flashvep/deepep_revalidation/results/chunk_oracle_gpu_scale_validation_20260831_223000/`

The latter contains `sample_manifest.json`, `routing.*.npz`, both DP profile
files, `capture_summary.json`, `stage2_summary.json`, and analysis CSVs.

Figures:

* `figures/plot1_fixed_vs_oracle_gpu_latency.png`
* `figures/plot2_gpu_speedup_distribution.png`
* `figures/plot3_offline_proxy_vs_gpu.png`
* `figures/plot4_stage2_oracle_headroom_by_budget.png`
* `figures/plot5_stage2_token_length_distribution.png`

Code is under `poc_flashvep/chunk_oracle_gpu_scale_validation/`.  The exact
commands and environment are in `run_stage1.sh` and `run_stage2.sh`; all
execution used `CUDA_VISIBLE_DEVICES=1,2,3,4`.

## 5. Final gate and next action

`CHUNK_ORACLE_GPU_SCALE_VALIDATION: GO (bounded)`.

Strongest positive evidence is the actual DeepEP/TritonExperts replay: median
max-rank expert latency fell 37–39% at both tested small budgets, with 100% of
paired request/layer observations improving and exact route/output invariants
passing.  The long live route capture also shows repeated >=10% route-oracle
headroom through budget 1024.

Strongest counter-evidence/limitation is that the GPU truth is based on a
validated layer-24 activation template cycled to route length, and the long
512/1024 values are shape-proxy values rather than CUDA timings.  The oracle's
larger chunks also reduce invocation count, so the result does not isolate a
ragged kernel benefit by itself.

**Next single recommended action:** run one bounded real-serving trace with
layer-specific Qwen3-VL activations at budgets 128/256/512/1024, keeping the
same route and scheduler policy, and measure grouped-MoE CUDA time before any
production scheduler or kernel work.  This is the required Real Serving Trace
step; do not generalize the current oracle numbers to production latency.
