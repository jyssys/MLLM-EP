# Resident memory in native H200 EPD

Qwen2.5-VL-7B uses 1.260 GiB of vision parameters on E. P and D each hold 14.185 GiB of language parameters plus 1.260 GiB of vision parameters in this vLLM configuration. Each P/D worker preallocates 42 GiB of KV. E has no language KV allocation. After native Main100, E reported 98.835 GiB CUDA free, P 77.653 GiB, and D 77.655 GiB. E's allocator had reserved 40.221 GiB and recorded 18.437 GiB peak allocated. Reserved, allocated and free are different measurements.

## Touched E arena at 2.5 req/s

| Arena GiB | Completed | TTFT p95 s | TPOT p95 ms | ITL p99 ms | Joint SLO | Goodput req/s |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 100/100 | 9.761 | 5.117 | 11.506 | 60/100 | 1.223 |
| 8 | 100/100 | 7.356 | 5.185 | 11.643 | 70/100 | 1.481 |
| 16 | 100/100 | 7.038 | 5.156 | 11.845 | 67/100 | 1.453 |
| 32 | 100/100 | 7.195 | 5.101 | 9.868 | 77/100 | 1.682 |
| 64 | 100/100 | 6.611 | 5.089 | 9.748 | 75/100 | 1.623 |
| 80 | 100/100 | 7.447 | 5.055 | 11.085 | 68/100 | 1.459 |

`MEASURED_ONLINE` for request outcomes and latencies. A separate allocator on E touched each capacity and retained its allocations through drain. At 80 GiB it reported 18.232 GiB CUDA free just after touching. The zero-capacity control preceded the five successive sizes within one server start. All six arms used Main100 and one-second GPM observation. The N1 order confounds comparisons with drift and thermal or runtime warmup. Use these results to bound feasible residency. The experiment does not estimate the cost of actual insertions or loads, or certify a latency-neutral capacity. The earlier EADA residency result at 80 GiB had ITL p99 9.04→10.91 ms.

[Compact memory and timing receipt](../mllm_epd_memory_asymmetry_discovery/results/native_residency_sweep.json) · [memory figure](../mllm_epd_memory_asymmetry_discovery/figures/native_memory_breakdown.svg) · [residency figure](../mllm_epd_memory_asymmetry_discovery/figures/native_residency_sweep.svg)

## Replication boundary

The previous EADA online candidate invoked E on every decoder layer and every step. This campaign places complete active decode KV on D. A KV tier contributes bytes through cache maintenance and prefill admission. The EADA 11.3% decode Attention fraction remains diagnostic. [EADA final report](./mllm_encoder_assisted_decode_attention_final_summary.md)

The zero-arena GPM capture had 49 E samples. SM activity averaged 46.1%, DRAM utilization averaged 14.8% and reached 31.3% at p95. This capture has no clean observer A/A pair. Use these values as `DIAGNOSTIC`. HBM average utilization does not establish latency-safe copy windows. The physical transfer campaign separately measures explicit contention. [Transfer and interference report](./mllm_epd_representation_cache_economics.md)
