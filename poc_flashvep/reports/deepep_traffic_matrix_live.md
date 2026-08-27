# Live DeepEP traffic-matrix validation

`DEEPEP_TRAFFIC_MATRIX_LIVE: HOLD`

## Scope and controls

This bounded capture reuses the validated Qwen3-VL-30B-A3B-Instruct real-image
workload, DBO off, BF16, linear expert placement, and DeepEP high-throughput.
The fixed request-pair subset is `[0, 1, 4, 8, 9, 12, 16, 20]`;
there are two warmups and two measured repetitions per request. Only physical
GPUs 1,2,3,4 were exposed. No route, placement, model, or communication policy
was modified.

The live capture code was fixed and committed at `95626aa` before execution.
Commands were:

```bash
RUN_ID=20260827_baseline2 WARMUPS=2 ITERATIONS=2 \
  ./poc_flashvep/scripts/run_live_traffic_matrix_validation.sh \
  poc_flashvep/deepep_revalidation/results/live_traffic_matrix_20260827_baseline2 baseline
RUN_ID=20260827_instrumented WARMUPS=2 ITERATIONS=2 \
  ./poc_flashvep/scripts/run_live_traffic_matrix_validation.sh \
  poc_flashvep/deepep_revalidation/results/live_traffic_matrix_20260827_instrumented instrumented
```

The launcher hard-sets `CUDA_VISIBLE_DEVICES=1,2,3,4`; logical EP ranks 0–3
therefore map to physical GPUs 1–4. Analysis was run with
`./poc_flashvep/scripts/analyze_live_traffic_matrix_validation.sh`.

The wrapper measures the existing vLLM `_prepare` (DeepEP dispatch/receiver)
and `_finalize` (DeepEP combine/receiver) calls with CUDA events, and records
expert GEMM separately. No extra collective and no per-layer synchronize were
introduced; events are resolved by one final bounded synchronization.

## Scale distribution

Complete measured four-rank invocations: **768**. Fixed buckets use
real route tokens/source: `<256`, `256–512`, `512–1024`, `≥1024`.

| bucket | invocations |
|---|---:|
| <256 | 288 |
| 256-512 | 288 |
| 512-1024 | 96 |
| >=1024 | 96 |

The synthetic N≈1024-like bucket contains **96** invocations. Most
observations therefore do not probe the high-scale regime where the synthetic
penalty was strongest.

## Matrix and feature definition

For each invocation, `M[source_dp_rank, ep_rank]` is populated from the actual
per-EP local expert histogram. Local expert index `e` maps to global expert
`ep_rank*32+e`; histograms and matrices are retained in the analysis CSV/JSON.
Features include volume, active pairs, max-pair load/fraction, normalized pair
entropy, HHI, source-row and destination-column imbalance. Route-derived S is
computed from the exact prior route artifacts (`expert_id//32`), separately from
the observed matrix.

Example highest-scale matrix (`tui_model_selection`, layer 20,
real tokens/source 1255, HHI 0.2887):

```text
[[1668, 3658, 1406, 3332], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
```

This runner intentionally used one active request per DP wave, so the matrices
have one nonzero source row. It is still a real source→destination matrix, but
cross-source pair concentration cannot be fully separated from destination
distribution. This is a material limitation, not replaced by synthetic data.

## Scale-conditioned and matched results

| scale_bucket   |   count |   dispatch_ms_hhi_spearman |   dispatch_ms_hhi_pvalue |   dispatch_ms_base_r2 |   dispatch_ms_shape_r2 |   dispatch_ms_delta_r2 |   combine_ms_hhi_spearman |   combine_ms_hhi_pvalue |   combine_ms_base_r2 |   combine_ms_shape_r2 |   combine_ms_delta_r2 |   comm_total_ms_hhi_spearman |   comm_total_ms_hhi_pvalue |   comm_total_ms_base_r2 |   comm_total_ms_shape_r2 |   comm_total_ms_delta_r2 |
|:---------------|--------:|---------------------------:|-------------------------:|----------------------:|-----------------------:|-----------------------:|--------------------------:|------------------------:|---------------------:|----------------------:|----------------------:|-----------------------------:|---------------------------:|------------------------:|-------------------------:|-------------------------:|
| <256           |     288 |                     0.0559 |                   0.3443 |                0.1257 |                 0.1271 |                 0.0014 |                    0.0220 |                  0.7101 |               0.0552 |                0.0552 |                0.0000 |                       0.0356 |                     0.5472 |                  0.0990 |                   0.0996 |                   0.0007 |
| 256-512        |     288 |                     0.0575 |                   0.3305 |                0.0204 |                 0.0205 |                 0.0000 |                   -0.0560 |                  0.3433 |               0.0199 |                0.0232 |                0.0033 |                      -0.0178 |                     0.7637 |                  0.0280 |                   0.0281 |                   0.0001 |
| 512-1024       |      96 |                    -0.0583 |                   0.5728 |                0.0004 |                 0.0051 |                 0.0047 |                   -0.0357 |                  0.7300 |               0.0150 |                0.0155 |                0.0005 |                      -0.0580 |                     0.5743 |                  0.0047 |                   0.0084 |                   0.0037 |
| >=1024         |      96 |                    -0.0558 |                   0.5895 |                0.0682 |                 0.1006 |                 0.0323 |                    0.2257 |                  0.0270 |               0.0050 |                0.0209 |                0.0159 |                       0.0931 |                     0.3672 |                  0.0511 |                   0.0969 |                   0.0459 |

The fixed matched rule was same layer, real assignment volume within 5%, route
S within 0.03, destination-column imbalance within 5%, and HHI contrast ≥0.01.
The resulting matched rows are in `analysis/matched_comparisons.csv`.

Matched higher-HHI rows: **2**; median dispatch change:
**-30.81%**; combine change: **18.05%**.

Figures: `analysis/plot1_concentration_vs_dispatch.png`,
`analysis/plot2_concentration_vs_combine.png`.

## Instrumentation overhead and validity

`{"available": true, "pairs": 16, "median_relative": 0.028490257084015203, "p95_relative": 0.0750202045224318, "baseline_median_wall_ms": 2731.292333, "instrumented_median_wall_ms": 2812.6935985}`

Backend proof records verify `DeepEPHTPrepareAndFinalize`,
`DeepEPHTAll2AllManager`, EP world size 4, and `CUDA_VISIBLE_DEVICES=1,2,3,4`.
All measured waves completed without CUDA/DeepEP errors; driver output tokens
are retained for a correctness audit. The coarse baseline is a separate run and
therefore includes run-to-run noise.

The common measured greedy output tokens matched **16/16**
between baseline and instrumented runs (`all_exact=True`).

## Gate / interpretation

The conservative status is **HOLD**. This real suite enters N≈1024 only in
the largest request(s); it does not provide broad high-scale coverage. Any
positive HHI/latency association is bounded evidence, and one active source row
means the full four-source synthetic Family-A analogue is not yet established.
The result therefore does **not** justify implementing a dynamic communication
scheduler. The next useful experiment is a bounded two-source-per-wave capture
with the same routes and fixed analysis rules.

Result directory: `/home/esjung/MLLM-EP-github/poc_flashvep/deepep_revalidation/results/live_traffic_matrix_20260827_instrumented`

Raw trace: `/home/esjung/MLLM-EP-github/poc_flashvep/deepep_revalidation/results/live_traffic_matrix_20260827_instrumented/raw_live/rank0..3.jsonl`

Analysis: `/home/esjung/MLLM-EP-github/poc_flashvep/deepep_revalidation/results/live_traffic_matrix_20260827_instrumented/analysis`
