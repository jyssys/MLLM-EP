# RQ2 — Chunked-Prefill communication vs MoE efficiency

Each arm uses the same 2,029-token high-resolution visual request and resident fixed64 B8 victim. `ExpertReloadAmp` is structural. Mixed totals sum critical-rank CUDA-event time over every affected chunk/layer; expert excess subtracts aligned resident-decode execution. Estimated weight bytes use the exact loaded BF16 tensor shapes (9,437,184 bytes/expert) and are explicitly not measured HBM traffic.

| Baseline | physical injection chunks | MNCD dispatch ms | CIDD ms/layer | response peak | logical dispatch MB | ExpertReloadAmp | estimated expert-weight GB/layer | mixed expert total ms | expert excess ms | mixed MoE total ms | prefill wall ms | paths/active expert |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FULL | 1.0 | 34.472 | 6.323 | 40.237% | 1445.331 | 1.000 | 1.189 | 28.499 | 4.903 | 357.663 | 718.678 | 129.856 |
| CP2048 | 1.0 | 34.093 | 5.958 | 35.204% | 1445.331 | 1.000 | 1.189 | 28.567 | 4.801 | 344.395 | 697.151 | 129.856 |
| CP1024 | 2.0 | 40.571 | 5.939 | 29.984% | 1445.331 | 1.976 | 2.340 | 61.013 | 2.143 | 757.436 | 1314.586 | 65.844 |
| CP512 | 4.0 | 29.232 | 0.409 | 24.247% | 1445.581 | 3.821 | 4.473 | 121.927 | 0.737 | 473.035 | 1250.733 | 34.221 |
| CP256 | 8.0 | 37.511 | 1.712 | UNRESOLVED_ASYNC_LOW_BASELINE_DENOMINATOR | 1445.634 | 7.161 | 8.399 | 230.637 | 27.580 | 1226.499 | 2714.032 | 18.202 |

A clean caller-wall response peak is marked `UNRESOLVED_ASYNC_SCHEDULER_DRIFT` when the first physical mixed iteration moves by more than four caller iterations across repeats. CUDA-event MNCD/CIDD and internal physical chunk counts remain available and are not replaced by the unstable caller-wall value.

RQ2 verdict: **CHUNKED_COMM_VS_MOE_TRADEOFF_CONFIRMED**.
