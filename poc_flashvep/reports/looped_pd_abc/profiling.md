# Profiling and communication evidence

Status: clean serving is in progress. Short GPU profiles follow the main comparisons.

## NIXL operator measurement

The persistent two-process probe completed 40 verified GPU copies on physical GPUs 0 and 1.
Each process owns one CUDA device and one NIXL agent.
Registration persists across repetitions. Each copied payload changes and receives a destination byte check.
The probe discards three warmup copies per size and records ten copies per size.

| Payload bytes | Median host-observed transfer ms | p95 ms | Median payload rate, GB/s |
|---:|---:|---:|---:|
| 1,024 | 0.096810 | 0.156010 | 0.010577 |
| 1,048,576 | 0.088396 | 0.108364 | 11.862254 |
| 16,777,216 | 0.099687 | 0.104470 | 168.299095 |
| 67,108,864 | 0.227632 | 0.242202 | 294.812960 |

Evidence: `MEASURED_PERSISTENT_TWO_PROCESS_NIXL_OPERATOR_NOT_SERVING`.
Raw data: `raw/nixl_two_process_pair01/samples.jsonl`.

The timer includes submission and completion polling.
It excludes registration, descriptor preparation, and transfer-handle initialization.
The rate is not measured end-to-end KV handoff bandwidth.
The measurement uses contiguous buffers, not every model's scattered cache descriptors.
One process start does not satisfy the repeated serving acceptance gate.

The earlier single-process probe produced a 188.238ms median for 64MiB.
It does not match the serving process topology or its observed transport rate.
The experiment therefore rejects that result as a hardware bandwidth floor.
It preserves the raw result instead of replacing or deleting it.

## Short profile scope

`--profile` enables the existing RLT profiler and separate operator annotations.
`ABC_NIXL_submit` includes metadata preparation and issue on the CPU.
`ABC_NIXL_poll` records CPU progress work.
Neither range is GPU DMA duration.
The normal serving path does not install those profiler wrappers.

RLT packages traces, operator tables, metadata, and checksums in compressed archives.
`profile_analysis.py` reads those archives without extracting arbitrary paths.
It writes `profile_device_summary.csv` and `profile_operators.csv` when archives exist.
Missing or failed profiles remain explicit. The analysis does not synthesize kernel events.

The original short-profile matrix includes conventional PD, A, B, and C.
Its conventional PD profile is not automatically the matched staged B control.
A separate matched profile is required before attributing a difference solely to frontier transfer.
Clean B-versus-staged-PD request timings remain the primary comparison.

## Memory and overhead

`worker_memory_transport.csv` separates original weights, derived weight packs, reserved KV, and allocator peaks.
Dory's derived QKV and gate/up packs consume additional memory.
They contain unchanged BF16 weights and apply to every adapter-v2 arm.
This generic packing cost or benefit is not an A/B/C contribution.

Post-drain used blocks do not measure peak active KV occupancy.
Graph workspace is not measured separately from the allocator peak.
The NVML trace supplies a separate sampled physical-device memory peak.

`evidence_IO_cost.csv` reports available logging counter deltas.
Early runs without that hook have unavailable logging cost.
The final report must evaluate whether instrumentation could explain a small observed difference.
It must not silently label such a difference architectural.
