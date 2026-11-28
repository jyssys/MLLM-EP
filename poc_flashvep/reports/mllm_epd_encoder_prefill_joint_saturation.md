# Encoder–Prefill joint saturation

E and P work overlap in a measured interval only when their clock-aligned kernel unions intersect. The three-way overlap additionally intersects the complement of D kernels. All fractions use the same central scheduled-arrival duration. E capacity proximity uses completed encoder HTTP requests and the corresponding independent E witness. The M2 screen uses an explicit 85% comparison with that witness and an E kernel threshold of 80%. P compute-ready sampling and D idle scope follow the operating-region contract.

## Same-window intervals

| Epoch and cohort | ROI s | E | P | D | E ∩ P | E ∩ P ∩ D idle | E ∩ D idle | E capacity proximity | M2 | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `late-native/v25_primed_n24_profile` | 3.81 | 31.54% | 97.48% | 18.41% | 30.53% | 23.12% | 24.11% | 44.84% | Open | DIAGNOSTIC |
| `natural-confirmation/long_confirm24_profile` | 6.49 | 98.32% | 95.50% | 9.46% | 94.04% | 84.85% | 88.92% | 89.76% | Open | DIAGNOSTIC |
| `natural-retry/long_primed_n24_profile` | 6.49 | 97.20% | 95.06% | 9.17% | 92.76% | 83.64% | 88.06% | 98.74% | Open | DIAGNOSTIC |
| `nixl-ready/nixl_readiness_n24_profile` | 6.49 | 98.31% | 94.03% | 8.65% | 92.55% | 84.35% | 89.68% | 98.74% | Open | DIAGNOSTIC |
| `observer-long64/long_primed_n64_profile` | 25.13 | 90.04% | 89.91% | 9.55% | 80.49% | 72.61% | 82.02% | 92.77% | Open | DIAGNOSTIC |
| `observer-upper48/upper_48g_n64_profile` | 19.44 | 92.27% | 93.30% | 12.99% | 86.09% | 74.60% | 80.69% | 92.27% | Open | DIAGNOSTIC |

The E→P readiness event here is a whole-request proxy acknowledgment. An E HTTP span includes host preprocessing and export. P HTTP includes queueing, embedding import and model execution. The matched E renderer span includes preprocessing and serializing the same-request processor handoff. Optimizer claims about per-image readiness require separate image-level producer events. Time left by recorded kernels does not establish that a stage can take additional model weights or prefill work without interference.

[Figure B](../mllm_epd_prefill_heavy_modality_discovery/results/region/figures/figure_b_joint_timeline.svg) · [Event CSV](../mllm_epd_prefill_heavy_modality_discovery/results/region/request_timeline.csv)

## Device resources in the same scheduled-arrival interval

These duration-weighted NVML GPM samples quantify device resources. SM active and tensor metrics are different quantities from the physical kernel union. DRAM figures are device bandwidth-utilization percentages. NVLink metrics are device totals and may contain concurrent connector traffic. Transfer time and GPU operator throughput need additional observations.

| Epoch and cohort | Device slot | SM active % | Tensor activity % | DRAM BW % | NVLink TX MiB/s | NVLink RX MiB/s | Counter ROI coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `observer-long64/long_primed_n64_profile` | encoder | 88.86 | 30.33 | 28.03 | 218.70 | 0.00 | 100.00% |
| `observer-long64/long_primed_n64_profile` | prefill | 86.19 | 73.76 | 15.38 | 1,676.74 | 214.14 | 100.00% |
| `observer-long64/long_primed_n64_profile` | decode | 5.83 | 0.38 | 6.23 | 0.00 | 1,725.98 | 100.00% |
| `native-population/full_main400_eos` | encoder | 39.93 | 13.93 | 12.68 | 132.35 | 0.00 | 100.00% |
| `native-population/full_main400_eos` | prefill | 45.43 | 39.10 | 9.38 | 1,082.59 | 132.19 | 100.00% |
| `native-population/full_main400_eos` | decode | 23.82 | 1.20 | 24.63 | 0.00 | 1,066.04 | 100.00% |
| `natural-confirmation/long_persistence128` | encoder | 86.65 | 29.90 | 27.56 | 225.43 | 0.00 | 100.00% |
| `natural-confirmation/long_persistence128` | prefill | 91.79 | 78.72 | 16.15 | 1,801.90 | 223.64 | 100.00% |
| `natural-confirmation/long_persistence128` | decode | 5.99 | 0.39 | 6.44 | 0.00 | 1,822.32 | 100.00% |
| `deflection-chunk4096/native_eos_n64` | encoder | 70.69 | 24.27 | 22.68 | 240.42 | 0.00 | 100.00% |
| `deflection-chunk4096/native_eos_n64` | prefill | 86.49 | 74.44 | 16.95 | 1,920.41 | 240.41 | 100.00% |
| `deflection-chunk4096/native_eos_n64` | decode | 43.92 | 2.27 | 45.11 | 0.00 | 2,020.85 | 100.00% |

## Original per-image transport readiness

An opted-in diagnostic polls existing CUDA IPC producer events. Each event binds the real encoder data to its original request-scoped image identifier. A successful query gives an upper bound and the most recent failed query gives a lower bound. The source bounds marker publication between save entry and return. Both the marker and the GPU data are needed for transport readiness. These intervals describe the current encoder implementation. Individual vision-kernel completion, consumer clone, and useful early LM work require separate evidence.

| Epoch and cohort | Validated requests | Native events | Matched visual rows | Matched LM span lengths | Multi-image requests | Proven positive first-image lead | Guaranteed lead p95 ms | Possible lead p95 ms | Full ready before HTTP ack median lower ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `image-ready/image_ready_n24` | 24/24 | 873 | 24 | 0 | 24 | 10 | 85.665 | 2,073.634 | 115.053 |

[Per-image readiness bounds](../mllm_epd_prefill_heavy_modality_discovery/results/region/figures/native_per_image_readiness_bounds.svg)
