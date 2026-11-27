# Native and generic online serving controls

## Contract

New A–D operator, placement and hybrid candidates remain `NOT_RUN` online. They failed physical or incremental promotion screening. This report records actual open-loop HTTP streaming inference for the native and generic controls. It gives the observed serving baseline for this study.

The two-GPU comparisons use GPUs 4 and 5 on the same H200 NVSwitch domain, Qwen3-8B BF16, FA3, paged KV, CUDA Graph, greedy decoding, APC off, matching traces and a common 48 GiB KV reservation per GPU. vLLM's native PD uses NIXL GPU transfer. Colocated controls route each arrival to the replica with the least outstanding work. The standard and local whole-request-deflection arms share the same client scheduling boundary. Every fixed study request completes 16 or more generation tokens. The P-only service reference is a separate one-token stage test.

Primary joint SLO is scheduled-time TTFT ≤5,000 ms, TPOT ≤33 ms and each observed ITL ≤34 ms. Failures count against joint SLO and goodput. Latencies are from the client SSE stream. Completion and goodput rates use cohort wall time including drain. These are N1 control points under a fixed Poisson seed, and the selected near and overloaded rates reflect actual H200 observations. Sustainable-load claims need an independent long-arrival confirmation. Candidate task accuracy, N3 and confidence intervals are `NOT_RUN`.

## Matched native PD and colocated chunked DP

This comparison uses the threaded GPM collector for both arms. Each row completed all 64 requests.

| Workload and offered load | Arm | TTFT p95, s | TPOT p95, ms | ITL p99, ms | Joint goodput, req/s |
|---|---|---:|---:|---:|---:|
| 8K/O16, 4.5 req/s | Native PD 8192 | 1.962 | 4.89 | 6.85 | 4.052 |
| 8K/O16, 4.5 req/s | Chunked DP2 8192 | 0.532 | 46.81 | 191.73 | 3.120 |
| 16K/O16, 1.9 req/s | Native PD 8192 | 3.893 | 5.02 | 6.54 | 1.762 |
| 16K/O16, 1.9 req/s | Chunked DP2 8192 | 1.009 | 72.16 | 294.19 | 1.352 |

At these fixed points the long prefill chunks give colocated replicas faster first tokens along with substantial decode tail interference. Native PD isolates D. Exact greedy sequence agreement was 62/64 at 8K and 60/64 at 16K. Uncontrolled differences between the numerical batch paths are part of the baseline observations. A systems/quality comparison should use a separate matched-token and task metric evaluation before claiming quality protection.

At 16K/O16 and 2.6 req/s, DP2 with 2,048-token chunks, 8,192-token chunks and unchunked execution completed 64/64 each. Their ITL p99 values were 84.74, 296.93 and 973.18 ms, respectively. Their TTFT p95 values were 1.397, 1.334 and 1.474 s. Joint goodput was 1.200, 1.348 and 1.430 req/s. Smaller chunks reduced the worst decode delays in this pilot and still exceeded the absolute decode protection target. All DP derivations use the same two GPUs and 48 GiB KV reservation per GPU.

## Native NIXL option and whole-request deflection

The native enable_cross_layers_blocks arm records a connector option request. Its P/D copy receipts preserve 144 GiB at 16K, and the average number of descriptors per transfer was 29,973 versus 29,979 for the default at 1.9 req/s. Compatibility hashes show the same effective connector transport layout. This provides a flag-requested control. The physical operator result comparing direct KV and hidden forwarding stands separately.

Both endpoints in the forced-HND confirmation used the V2 Model Runner. Its installed connector path in vllm/v1/worker/gpu/kv_connector.py registers per-layer cache tensors and has a pending cross-layer allocation implementation. The first layout inspection assumed a V1 runner field and returned HTTP 500 on V2. The 24 serving requests still completed. The corrected inspection returned HTTP 200 on both HND V2 endpoints in the final native observation, with the cross-layer field and allocation absent.

| Load | Arm | TTFT p95, s | TPOT p95, ms | ITL p99, ms | Joint goodput, req/s | Completed D-local prompt tokens |
|---|---|---:|---:|---:|---:|---:|
| 8K/O16, 4.5 req/s | Native NIXL option | 1.919 | 4.90 | 8.95 | 4.003 | 0 |
| 8K/O16, 4.5 req/s | Whole request deflection | 1.416 | 9.15 | 20.82 | 3.803 | 49,152 |
| 16K/O16, 1.9 req/s | Native NIXL option | 3.868 | 4.91 | 9.19 | 1.768 | 0 |
| 16K/O16, 1.9 req/s | Whole request deflection | 2.101 | 5.42 | 20.79 | 1.819 | 180,224 |
| 16K/O16, 2.6 req/s | Native NIXL option | 9.963 | 4.59 | 8.17 | 0.757 | 0 |
| 16K/O16, 2.6 req/s | Whole request deflection | 4.498 | 13.64 | 25.56 | 1.964 | 196,608 |

The control deflects 6, 11 and 12 of 64 requests. All rows completed 64/64. The high-load point shows why a whole request is a strong unit for absorbing P backlog and avoiding an entire KV handoff. The observed TTFT and goodput improvements coexist with decode tail regressions above the frozen relative 5% limit.

## Equal three-GPU static allocations

All layouts reuse H200 GPUs 4–6, a 48 GiB KV reservation per GPU and the batch token budget of 8,192. Each row is 64/64 complete with no failed measured KV reads. P2/D1 uses two independent prefill replicas. TP2-P/D1 runs one tensor-parallel P replica. The GPU-normalized goodput values divide the finite cohort goodput by three.

| Load | Allocation | TTFT p95, s | TPOT p95, ms | ITL p99, ms | Goodput, req/s | Goodput per GPU |
|---|---|---:|---:|---:|---:|---:|
| 8K/O16, 4.5 req/s | P2/D1 | 0.687 | 8.87 | 62.64 | 3.723 | 1.241 |
| 8K/O16, 4.5 req/s | P1/D2 | 1.943 | 4.85 | 8.93 | 4.066 | 1.355 |
| 8K/O16, 4.5 req/s | TP2-P/D1 | 1.122 | 40.42 | 134.19 | 3.008 | 1.003 |
| 16K/O16, 1.9 req/s | P2/D1 | 1.252 | 11.29 | 20.34 | 1.761 | 0.587 |
| 16K/O16, 1.9 req/s | P1/D2 | 3.882 | 4.17 | 8.44 | 1.773 | 0.591 |
| 16K/O16, 1.9 req/s | TP2-P/D1 | 1.530 | 40.39 | 265.18 | 0.921 | 0.307 |
| 16K/O16, 2.6 req/s | P2/D1 | 1.379 | 11.59 | 23.29 | 2.344 | 0.781 |
| 16K/O16, 2.6 req/s | P1/D2 | 9.876 | 4.38 | 6.78 | 0.758 | 0.253 |
| 16K/O16, 2.6 req/s | TP2-P/D1 | 1.892 | 51.78 | 262.28 | 0.844 | 0.281 |

The 16K/2.6 point gains most when the second P independently handles queued work. P1/D2 protects the decode tails while leaving the P backlog. The 8K point shows the different trade-off when two P replicas present work to one D. The goodput-favoring allocation depends on the offered load.

The first TP2 launch failed a local FlashInfer CUDA driver link. A launch-local linker path recovered the native MNNVL communication and CUDA Graph with the original runtime. The 16K TP2 handoff still split each request across two source ranks. At 2.6 req/s it issued 128 reads versus 64 for P2/D1, with the same 144 GiB overall. Its mean transfer receipt was 201.89 ms versus 137.93 ms and its D-submit-to-first p95 was 631 ms versus 328 ms. This locates extra observed cost in the native transport path. The statistical comparison groups multiple runtime differences, including P execution order and D batching.

P1/D2 and matching Native 1P1D agree on all 192 greedy sequences. P2/D1 agrees on 182/192 and differs on the same two requests in the repeated 16K cohort. TP2-P/D1 agrees on 179/192. The placement changes preserve the model definition. Task-quality noninferiority remains untested.

## Mixed arrivals and context lengths

The 1P/1D N64 mixed-heavy cohort at 1.45 req/s uses 8K/16K/32K contexts and 16/32/64 output tokens. TTFT p95 is 4.283 s, TPOT p95 6.34 ms, ITL p99 15.13 ms and joint goodput 1.148 req/s (53/64). The N24 1.25 req/s control spans 2K–32K and 16–256 output tokens, with TTFT p95 1.965 s, TPOT p95 6.87 ms, ITL p99 12.05 ms and joint goodput 0.763 req/s (17/24). All requests completed. These are controlled LongBench-derived workloads with APC disabled.

## Control inventory and interpretation

Generic whole-request deflection here is an independent local policy. It does not reproduce the paper-specific scheduling of Crossflow, FluidPD or Kairos. Token/layer offload online, POD-Attention, PP, and published elastic artifacts without a compatible verified run remain `NOT_RUN`. A and D physical kernel and layout experiments are separate from these full server measurements. [Prior work and implementation audit](pd_resource_complementary_prior_work.md)

[Per-case quantiles and counters](../pd_resource_complementary_parallelism/results/native/native_summary.json) · [Matched pairs](../pd_resource_complementary_parallelism/results/native/native_matched_controls.csv) · [Actual deflected work](../pd_resource_complementary_parallelism/figures/native_deflection_useful_work.png) · [Same-budget goodput](../pd_resource_complementary_parallelism/figures/native_same_budget_goodput.png) · [Reproduction and raw archive](../pd_resource_complementary_parallelism/README.md)
