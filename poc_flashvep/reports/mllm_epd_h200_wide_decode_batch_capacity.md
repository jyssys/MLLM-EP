# H200 measured decode capacity by admission cap

| Cap | Start / repetition | Evidence | Requests / tokens | Requests/s | Tokens/s | Service p50 ms | Mean TBT ms | ITL p99 ms | TPOT p95 ms | HTTP max | HTTP binding | Scheduled / prepared max | Padded rows max | Target attained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | capacity_wide r0 | ISOLATED_GPU_MEASUREMENT | 100 / 5269 | 3.31 | 174.40 | 177.56 | 4.68 | 5.41 | 4.82 | 1 | 95.67% | 1 / 1 | 1 | yes |
| 2 | capacity_wide r0 | ISOLATED_GPU_MEASUREMENT | 100 / 5269 | 5.98 | 315.23 | 192.67 | 4.97 | 12.95 | 5.37 | 2 | 92.02% | 2 / 2 | 2 | yes |
| 4 | capacity_wide r0 | ISOLATED_GPU_MEASUREMENT | 100 / 5269 | 9.70 | 511.27 | 246.13 | 5.53 | 19.97 | 6.57 | 4 | 83.44% | 4 / 4 | 4 | yes |
| 8 | capacity_wide r0 | ISOLATED_GPU_MEASUREMENT | 100 / 5269 | 13.67 | 720.16 | 303.99 | 6.71 | 27.04 | 8.33 | 8 | 66.74% | 8 / 8 | 8 | yes |
| 16 | capacity_wide r0 | ISOLATED_GPU_MEASUREMENT | 100 / 5269 | 13.65 | 719.41 | 537.13 | 8.60 | 43.53 | 12.80 | 16 | 45.57% | 16 / 16 | 16 | yes |
| 32 | capacity_wide r0 | ISOLATED_GPU_MEASUREMENT | 100 / 5269 | 13.91 | 732.83 | 930.17 | 8.37 | 62.90 | 19.11 | 32 | 24.58% | 32 / 32 | 32 | yes |
| 64 | capacity_wide r0 | ISOLATED_GPU_MEASUREMENT | 100 / 5269 | 13.60 | 716.55 | 1634.89 | 7.87 | 60.10 | 20.19 | 64 | 20.50% | 64 / 63 | 64 | no |
| free | capacity_wide r0 | ISOLATED_GPU_MEASUREMENT | 100 / 5269 | 13.36 | 703.78 | 1672.15 | 7.43 | 75.65 | 26.58 | 68 | — | 66 / 65 | 72 | — |

The timed D drain begins after actual E/P generation of KV in each cohort. It includes NIXL import, decoding and local SSE delivery. All arms use the frozen replay output targets. GPU capacity means isolated D throughput under this ordered cohort and scheduler. HTTP concurrency, prepared decode requests, query rows and CUDA graph padding are separate quantities. A cap that was never attained is an upper bound on admission for this cohort.

The aggregate and individual arm receipts are `analysis/capacity_decode_wide.json` and `analysis/capacity_decode_wide_*_r*.json` inside their run. They retain KV input lengths, prepared request-count and graph-mode distributions, original request IDs and KV block bounds.

## E and P controls

| Role | Start | Requests/s | Tokens/s | Evidence |
| --- | --- | --- | --- | --- |
| encoder | — | — | — | NOT_RUN |
| prefill | capacity_prefill | 3.18 | 35053.67 | ISOLATED_GPU_MEASUREMENT |

The prior same-group H200 controls and their differences in observer and cohort context are available separately. [Previous independent-stage capacity report](mllm_epd_h200_independent_stage_capacity.md)
