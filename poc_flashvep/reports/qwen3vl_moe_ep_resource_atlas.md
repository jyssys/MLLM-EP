# Qwen3-VL MoE EP resource atlas (bounded, automated)

## Executive result

**FINAL STATUS: `PARTIAL_RESOURCE_ATLAS`**.

The end-to-end vLLM driver smoke test completed on the requested TP2/DP2/EP4 configuration, and a four-rank exact-route DeepEP replay produced valid `.nsys-rep` and SQLite exports. The automated parser then discovered the Nsight schema, resolved kernel names, generated signatures, dependency labels, compatibility matrix, and figures. A full-serving Nsight kernel trace was not obtained: child-worker CUDA activity was absent from the export, while enabling NVTX capture triggered a NCCL 2.28.9 NVTX-extension segmentation fault. Consequently, this report distinguishes observed bounded replay evidence from source-inferred/full-serving claims.

## Environment and reproducibility

| item | value |
|---|---|
| GPUs | H100 80 GB, physical 1,2,3,4 only |
| visibility | `CUDA_VISIBLE_DEVICES=1,2,3,4` |
| model | Qwen/Qwen3-VL-30B-A3B-Instruct, local snapshot `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c` |
| dtype | BF16 |
| runtime | vLLM 0.20.0 V1, enforce eager |
| topology | TP2 / DP2 / EP4 / PP1 |
| MoE backend | DeepEP high-throughput, TritonExperts, linear expert placement |
| DBO / prefix cache | off / off |
| Nsight Systems | 2024.6.2.225-246235244400v0 |
| Python / PyTorch / CUDA | 3.12.13 / 2.11.0+cu129 / CUDA 12.9 |
| Triton | 3.6.0 |
| DeepEP package version | not exposed by installed module; path and replay manifest recorded |

The exact commands, model/configuration, reference commits and route-capture hash are in `reference_manifest.json` and `profiling_command.txt`.

## What was profiled

The primary capture is `resource_atlas_main.nsys-rep`/`resource_atlas_main.sqlite` generated from four ranks replaying the existing real Qwen3-VL layer-24 route capture (`799` rows, `784` visual, hidden size `2048`, `E=128`, `K=8`, EP4). The replay calls DeepEP `get_dispatch_layout`, dispatch and combine with the original BF16 hidden states/top-k IDs; it does not use synthetic routes and does not include expert GEMM. Four rank JSON files contain 3 warmups + 20 measured iterations and event timings.

Observed SQLite schema (discovered dynamically, not hard-coded) contained 224 CUDA kernels, 579 memcpy records, 765 CUDA-event records and 1,063 synchronization records. `nsys stats` generated kernel and memory summaries in `nsys_stats/`.

Per-rank median CUDA event timings (ms):

| physical GPU | layout | dispatch | combine | dispatch+combine+layout |
|---:|---:|---:|---:|---:|
| 1 | 0.0535 | 0.3030 | 0.1226 | 0.5088 |
| 2 | 0.0543 | 0.3080 | 0.1205 | 0.5106 |
| 3 | 0.0561 | 0.3105 | 0.1215 | 0.5110 |
| 4 | 0.0846 | 0.2286 | 0.1174 | 0.4596 |
| rank median | 0.0552 | 0.3055 | 0.1210 | 0.5097 |

The slowest replay rank is physical GPU 3 at about `0.511 ms` median total; GPU 4 is faster. These are event timings from the exact-route communication replay, not an end-to-end Qwen3-VL latency claim.

Kernel-name aggregation found `deep_ep::layout::get_dispatch_layout`, `deep_ep::intranode::notify_dispatch`, `deep_ep::intranode::dispatch`, `cached_notify_combine`, and `intranode::combine`. Kernel-name totals were approximately `0.215 ms` layout, `19.594 ms` dispatch-family, `2.875 ms` combine-family across captured launches; NCCL all-gather/all-reduce/barrier kernels totalled `40.114 ms` but their logical DeepEP sub-phase is unresolved. The event medians above are the primary communication numbers.

## Full-serving Nsight attempts and fallback

`run_atlas.py` successfully initialized the actual Qwen3-VL vLLM driver and returned matching greedy output tokens (`[1986, 374]`) in the successful driver-smoke attempt (`20260903_131420`, warmups 2, decode 8). However, the resulting full-serving SQLite had no CUDA kernel activity because the CUDA child-worker processes were not included by the launcher capture. An attempt with fork-before-exec tracing did not complete a usable export. An NVTX-trace attempt crashed in NCCL's `nvtxExtInitOnce_v3` while `ncclGetUniqueId`; no sudo/package mutation was performed. These attempts are preserved in sibling result directories and are summarized in `gate_summary.json`.

The local hook uses `sitecustomize.py` and read-only wrappers; no installed vLLM/DeepEP source was edited. Hook proof files show patched function boundaries, but because the final replay intentionally disables the hook and uses `trace=cuda,osrt`, no NVTX table is present. Therefore the required fine-grained full-serving phase rows are `SOURCE_INFERRED`/`UNKNOWN`, never fabricated from replay kernels.

## Resource signatures

`resource_signature.csv` and `.md` cover all requested phases. The only directly observed phase rows are bounded replay `DEEPEP_LAYOUT`, `DEEPEP_DISPATCH`, `DEEPEP_COMBINE`, an unresolved DeepEP/NCCL collective group, and auxiliary tensor kernels. Vision patch/attention/MLP/merger, LLM attention/router/expert, norm/residual, and decode sub-phases have no reliable timing in this capture and are marked `SOURCE_INFERRED` or `UNKNOWN` with low confidence. Nsight Systems alone cannot provide Tensor Core or HBM utilization here; those fields are deliberately not claimed.

The observed resource classes are dominated by communication-mixed kernels (168/224) and unresolved/elementwise auxiliary kernels (56/224). Memory-copy summary reports 539 host-to-device, 24 device-to-host, 16 device-to-device and 117 memset records; these include replay setup and are not attributed to a full-serving phase.

## Dependency and compatibility interpretation

`dependency_graph.json/.md` records hard same-request edges (vision encoder → LM prefill → dispatch → expert → combine), cross-request independence for a pending encoder versus an existing request's communication, and possible cross-image independence. `resource_compatibility_matrix.csv/.md` intentionally tags the previously measured pairs as `MEASURED_NEGATIVE`:

| pair | measured result | atlas interpretation |
|---|---:|---|
| Vision Encoder + DeepEP Dispatch | wall `-12.4%`, communication `+19.0%` | LOW / no candidate |
| Vision Encoder + DeepEP Combine | wall `-5.0%`, communication `+14.0%` | LOW / no candidate |
| Vision Encoder + Expert | wall `-8.9%` | LOW / no candidate |

These measurements are not re-labelled as theoretical HIGH opportunities. The shortlist therefore contains only conditional, low-confidence candidates (CPU-side request preparation + dispatch; a small vision merger unit + combine; decode attention + dispatch), each requiring a bounded follow-up. None is a validated positive overlap result.

## Visual streaming feasibility

**`POSSIBLE_WITH_RUNTIME_CHANGE`**. `grid_thw`/`cu_seqlens` preserve image sequence boundaries and the vision transformer can process concatenated images, but the current vLLM path returns the complete visual embedding before LM execution and exposes no per-image ready callback. Image-level streaming would require a runtime interface/scheduler change; it was not implemented.

## Related-work boundary

RESONATOR (ISCA 2026, [paper](https://doi.org/10.1109/ISCA66397.2026.00173)) and [SpaceServe](https://github.com/gofreelee/SpaceServe) study encoder/LLM resource sharing and encoder parallel plans. The local SpaceServe reference is commit `66de079af4234b27f7f82ff91d238ef1351324ba`; no public RESONATOR code was found. [Flux/COMET](https://github.com/bytedance/flux) studies fine-grained MoE communication/computation overlap; local reference commit is `19831ca2d820e3e782ed1d15d8b52d0898b78b26`. [DeepEP](https://github.com/deepseek-ai/DeepEP) remains the communication backend. This PoC only builds a measurement atlas and does not claim overlap novelty or implement any scheduler.

## Main observation and next action

The automation path is viable for CUDA/SQLite/kernel resource evidence, and it clearly identifies DeepEP communication kernels and their event-scale cost. The current blocker is not analysis but full-serving child-process/NVTX capture: without a profiler-compatible launch/capture arrangement, phase-level Qwen3-VL signatures cannot be promoted from source inference to observed evidence. The next single action is to run one bounded full-serving capture under a profiler-compatible child-process launcher (or a supported `cudaProfilerStart/Stop` range) while avoiding the NCCL NVTX extension crash, then rerun this parser. No optimization method should be designed from the replay alone.

## Artifact index

- Result directory: `poc_flashvep/deepep_revalidation/results/qwen3vl_moe_ep_resource_atlas_20260903_134900/`
- Automated parser: `poc_flashvep/resource_atlas/analyze_nsys_resource_atlas.py`
- Read-only hook: `poc_flashvep/resource_atlas/atlas_hook.py`
- Required figures: `figure_phase_timeline.png`, `figure_phase_duration_breakdown.png`, `figure_resource_class_map.png`, `figure_compatibility_heatmap.png`
- Required machine-readable outputs: `schema_inventory.json`, `resource_signature.csv`, `phase_kernel_mapping.csv`, `dependency_graph.json`, `resource_compatibility_matrix.csv`, `gate_summary.json`
