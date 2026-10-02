# Physical kernel boundary evidence (diagnostic only)

All entries below are **KERNEL_CLASSIFIED**, ms/call, one separate Nsight repeat per arm, mean of48 calls on ONE cumulative-critical rank selected within that profiled run. These are NOT clean baseline decomposition fractions. Each same-rank classifier matches actual OS PID to CUPTI launch correlation. Twelve native rank/arm ranges each contain exactly48 instances of all five classes.

| Arm | selected EP rank | layout kernel | count/barrier notify | fused dispatch | cached combine notify | fused combine |
|---|---:|---:|---:|---:|---:|---:|
| mllm visual | 3 | 0.0037 | 0.6593 | 0.1073 | 0.1778 | 0.0696 |
| mllm text | 3 | 0.0038 | 0.3216 | 0.1052 | 0.2112 | 0.0718 |
| llm text | 2 | 0.0038 | 0.6249 | 0.1075 | 0.1782 | 0.0704 |

`notify_dispatch` combines count exchange/prefix computation/barriers; it is not a pure count arithmetic or wire interval. Dispatch fuses packing, copying, receive polling; combine fuses reverse copy, reduction and polling. Kernel durations may contain rendezvous waits. Host polling/allocation/launch gaps and vLLM remapping are not separately measured here. No cross-rank timestamp subtraction or sum of unrelated maxima is used.

The count/rendezvous kernel is much larger than layout arithmetic and fused payload kernels on the selected ranks in this diagnostic. This supports a count/rendezvous/tail explanation but cannot assign an exclusive percentage of the CLEAN dispatch cost. Nsight changes producer readiness; the profiled first-layer duration is not interchangeable with the clean~29.60ms waiting-rank interval. Only source-backed classification and own-lane duration are claimed.

Full per-rank/per-call records: `artifacts/nsight/kernel_classification.json`; same-rank selected summaries: `artifacts/profile_critical_rank_classification.json`.
