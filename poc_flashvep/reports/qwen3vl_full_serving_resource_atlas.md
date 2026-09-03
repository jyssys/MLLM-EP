# Qwen3-VL full-serving resource atlas

## Final result

**FINAL STATUS: `FULL_RESOURCE_ATLAS_COMPLETE`**

The child-process capture blocker was resolved without changing the model,
routing, expert placement, scheduler policy, or installed vLLM/DeepEP source.
A real Qwen3-VL single-image request (two warmups and eight decode tokens) was
captured from the CUDA-owning vLLM workers. The resulting
`full_serving_resource_atlas.nsys-rep` and SQLite contain 208,529 CUDA kernel
records, 25,188 memcpy records, and 49,428 NVTX records (43,244 completed
ranges). The automated, process-aware SQLite parser generated all signatures,
phase/kernel mappings, dependency and compatibility artifacts, and figures.

The Nsight launcher exits with status 143 because `--capture-range-end=stop-shutdown`
terminates the target after `cudaProfilerStop`; this is expected and the exports
are valid. The serving driver files report `ok: true` and both DP drivers
produced the same greedy tokens. The only error-like lines in `run.log` are two
non-fatal vLLM usage-telemetry JSON parsing exceptions; no CUDA, DeepEP, NCCL
model-execution, or correctness failure was observed.

## Exact environment

| item | value |
|---|---|
| model | Qwen/Qwen3-VL-30B-A3B-Instruct, local snapshot `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c` |
| vLLM | 0.20.0, V1, enforce eager |
| precision | BF16 |
| topology | TP2 / DP2 / EP4 / PP1 |
| MoE backend | DeepEP high-throughput + TritonExperts |
| placement | linear (`expert_id // 32`) |
| DBO / prefix cache | off / off |
| GPUs | `CUDA_VISIBLE_DEVICES=1,2,3,4` only (local IDs 0--3 map to physical 1--4) |
| Nsight Systems | 2024.6.2.225-246235244400v0 |
| request | one real `skimage` astronaut image, brief description, prefill + 8 decode tokens |
| warmups | 2, outside the intended capture window |

The exact command is in `profiling_command.txt`. In abbreviated form:

```text
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --backtrace=none \
  --trace-fork-before-exec=true --wait=all --stats=true --export=sqlite \
  -o full_serving_resource_atlas python poc_flashvep/resource_atlas/run_atlas.py \
  --warmups 2 --decode-tokens 8 --cuda-profiler-api
```

## Capture fix and validation

`sitecustomize.py` loads the read-only `atlas_hook.py` in each worker. After
model/NCCL/DeepEP initialization and warmups, the driver creates
`cuda_profiler_start.signal`; each CUDA-owning worker calls
`torch.cuda.cudart().cudaProfilerStart()`. A stop signal invokes
`cudaProfilerStop()` in the same worker context. This avoids the previous
NCCL `nvtxExtInitOnce_v3` crash during initialization and fixes the missing
child-worker CUDA activity. Hook proof files record `CUDA_VISIBLE_DEVICES=1,2,3,4`,
vLLM 0.20.0, and successful profiler start/stop calls. NVTX is collected only
after initialization, while `mm_processor_cache_gb=0` forces the measured image
to execute the real vision path rather than reusing warmup embeddings.

The SQLite `PROCESSES` table contains four `VLLM::Worker_DP` CUDA processes
(PIDs 1773081, 1773085, 1773089, 1773093) and two engine coordinators. A
process-aware mapper associates each CUDA kernel's `globalPid` with the nearest
NVTX worker thread namespace; this prevents the cross-worker time aliasing that
made the earlier aggregate parser over-attribute ranges. For asynchronous
DeepEP launches, stable kernel names take precedence over CPU-range containment,
because communication kernels can outlive a Python wrapper.

## Observed resource signatures

Values below are sums of observed NVTX range durations or individual kernel
durations across worker/rank instances. They are not a critical-path latency
sum and are not GPU utilization. Full machine-readable values are in
`full_resource_signature.csv`; dominant names are in
`full_phase_kernel_mapping.csv`.

| phase | aggregate NVTX wall (ms) | CUDA kernel time (ms) | kernels | class / observed evidence |
|---|---:|---:|---:|---|
| Vision patch | 2.775 | 0.038 | 4 | `nvjet_tst_*`; UNKNOWN/short compute, full-serving observed |
| Vision attention | 150.143 | 5.518 | 737 | SM90 CUTLASS FlashAttention plus `nvjet_tst_*`; COMPUTE_HEAVY |
| Vision MLP | 45.851 | 3.013 | 315 | `nvjet_tst_*`, GELU/elementwise/layer norm; COMPUTE_HEAVY |
| Vision merger | 6.135 | 0.525 | 61 | small `nvjet_tst_*`/layer norm; UNKNOWN (resource utilization not exposed) |
| LLM attention (prefill) | 4173.767 | 12.202 | 2173 | FlashAttention/`nvjet_tst_*` with mixed auxiliary activity; COMMUNICATION_MIXED |
| TP communication | diagnostic | 569.240 | 6408 | `cross_device_reduce_1stage`; FULL_SERVING_OBSERVED kernel-name aggregate |
| Router/top-k | 11621.640 | 26.602 | 6144 | exact `vllm::moe::topkGating<8,128,4,...>`; ROUTING |
| DeepEP layout | diagnostic | 23.288 | 6144 | `deep_ep::layout::get_dispatch_layout`; communication-mixed |
| DeepEP dispatch (prefill) | 2551.267 | 24.789 | 390 | `notify_dispatch` + `dispatch`; communication-mixed |
| Expert (prefill) | 3534.886 | 30.875 | 1354 | `fused_moe_kernel` and MoE alignment/activation; COMPUTE_HEAVY |
| DeepEP combine (prefill) | 599.903 | 58.429 | 586 | `cached_notify_combine` + `combine`; communication-mixed |
| Decode attention | 4019.782 | 268.963 | 60782 | decode `nvjet_tst_*`, FlashAttention and AllGather; mixed |
| Decode dispatch | 2480.701 | 1036.835 | 13842 | DeepEP `notify_dispatch`/`dispatch`; communication-mixed |
| Decode expert | 3427.065 | 232.650 | 44260 | `fused_moe_kernel` plus alignment/copy kernels; compute-heavy, HBM orientation UNKNOWN |
| Decode combine | 580.150 | 800.161 | 11901 | DeepEP cached notify/combine; communication-mixed |

The parser deliberately reports QKV and O-projection as `SOURCE_INFERRED`:
this vLLM release has no safe standalone Python boundary for those operations.
Likewise, decode sub-phases are reconstructed from generic nested markers inside
the `LLM_DECODE` wrapper. `DEEPEP_COMM_UNKNOWN_PHASE` (64.173 ms of kernels)
contains collectives whose logical sub-phase cannot be safely inferred. These
limitations are explicit rather than fabricated timings.

The Nsight kernel summary provides an independent dominant-kernel check: across
the capture, DeepEP `notify_dispatch` is 980.807 ms (6,144 launches), cached
combine is 747.644 ms (6,144), `cross_device_reduce_1stage` is 569.240 ms
(6,408), NCCL AllGather is 240.059 ms (6,176), and `fused_moe_kernel` is
157.708 ms (11,446). This confirms that dispatch/combine and TP collectives are
real full-serving communication activity, while expert execution is a real
compute kernel family.

## Answers to the critical questions

1. **Vision dominant kernels:** patch/merger use short `nvjet_tst_*` and
   elementwise/normalization kernels; vision attention is dominated by SM90
   CUTLASS FlashAttention variants and auxiliary `nvjet_tst_*` kernels; vision
   MLP is `nvjet_tst_*`/GELU/normalization.
2. **Vision attention vs MLP:** both are compute-oriented in the observed
   kernel classes; attention has substantially more FlashAttention activity,
   while MLP has smaller linear/activation kernels. TensorCore/HBM saturation
   is not measured by this trace.
3. **LLM attention/TP:** LLM attention is not compute-only: `cross_device_reduce_1stage`
   and NCCL AllGather are directly observed. TP communication is reported as a
   separate diagnostic aggregate because it is nested/asynchronous.
4. **Router:** the exact top-k kernel appears 6,144 times and totals 26.602 ms
   of kernel time; it is a small routing phase relative to the full set of
   attention/decode activity, although summed NVTX wall is not additive.
5. **DeepEP:** layout, `notify_dispatch`/`dispatch`, and cached
   `notify_combine`/`combine` kernel families are directly present in full
   serving and classified COMMUNICATION_MIXED.
6. **Prefill expert:** `fused_moe_kernel` is directly observed and classified
   COMPUTE_HEAVY; its companion alignment/activation kernels are also visible.
7. **Decode expert:** `fused_moe_kernel` remains COMPUTE_HEAVY in this trace.
   It has substantial alignment/copy activity, but Nsight Systems does not
   provide enough evidence to call it memory-bound; HBM orientation is UNKNOWN.
8. **Idle/slack:** dominant CUDA streams span roughly 4.1--4.3 s while summed
   kernel execution is only hundreds of ms, so large timeline gaps are
   observable. These are scheduling/async gaps, not proof of free SM capacity;
   `stream_activity.csv` and `timeline_summary.json` label this conservatively.
9. **Independent compatible pair:** the dependency graph marks pending
   cross-request vision work versus current-request communication as
   `CROSS_REQUEST_INDEPENDENT`, but the previously measured real pairs were
   negative and this capture does not schedule concurrent work. Candidates remain
   conditional, not validated opportunities.
10. **Prior overlap failure:** the full trace confirms that encoder and EP
    phases occupy active CUDA/communication streams concurrently. It is
    consistent with the prior measured negatives (resource interference), not
    evidence that a new overlap scheduler is safe.

## Dependency and compatibility

`full_dependency_graph.json/.md` records hard same-request ordering:

```text
VISION_ENCODER -> LLM_PREFILL -> DEEPEP_DISPATCH -> EXPERT_GEMM -> DEEPEP_COMBINE
```

It also records pending-encoder/current-communication as cross-request
independent and image-to-image encoder work as structurally possible. The full
compatibility matrix updates inferred resource labels to
`FULL_SERVING_OBSERVED`, while preserving measured negatives:

| pair | prior measured result | full-atlas interpretation |
|---|---:|---|
| Vision encoder + DeepEP dispatch | wall slowdown 12.4%, communication slowdown 19.0% | LOW / NO / MEASURED_NEGATIVE |
| Vision encoder + DeepEP combine | wall slowdown 5.0%, communication slowdown 14.0% | LOW / NO / MEASURED_NEGATIVE |
| Vision encoder + expert | wall slowdown 8.9% | LOW / NO / MEASURED_NEGATIVE |

The remaining shortlist is deliberately short and conditional:

1. CPU request preparation + DeepEP dispatch — cross-request independent and
   low direct GPU-resource risk; full-serving resource evidence, but no measured
   concurrent gain.
2. A small vision merger/projector unit + combine — potentially shorter than a
   full encoder, but memory interference and the negative full-encoder/ combine
   measurement are risks.
3. Decode attention + dispatch — cross-request independent but latency-sensitive
   and potentially HBM-contending.

No pair is labelled a HIGH opportunity. Compatibility is not inferred from
resource classes alone, and the atlas does not implement or benchmark a new
concurrent scheduler.

## Visual streaming feasibility

**`POSSIBLE_WITH_RUNTIME_CHANGE`**. `grid_thw`/`cu_seqlens` preserve image
sequence boundaries and the vision transformer receives concatenated image
sequences, but vLLM returns the complete visual embedding tensor before LM
prefill and exposes no image-level ready callback. Independent image release
would require a runtime interface/scheduling change; none was implemented.

## Related-work boundary

RESONATOR studies encoder/LLM resource sharing and encoder DP/TP choices; no
public RESONATOR repository was found (`RESONATOR_CODE_STATUS:
NOT_PUBLICLY_FOUND`). The inspected SpaceServe reference is commit
`66de079af4234b27f7f82ff91d238ef1351324ba`; Flux/COMET is commit
`19831ca2d820e3e782ed1d15d8b52d0898b78b26`; DeepEP remains the production
communication backend. Those works motivate the resource/dependency taxonomy,
but this branch only completes measurement automation and does not claim
overlap novelty.

## Artifact index

Result directory:
`poc_flashvep/deepep_revalidation/results/qwen3vl_full_serving_resource_atlas_20260903_154500/`

- `full_serving_resource_atlas.nsys-rep`
- `full_serving_resource_atlas.sqlite`
- `source_audit.md`, `reference_manifest.json`, `profiling_command.txt`, `nsys_*_help.txt`
- `full_resource_signature.csv/.md`
- `full_phase_kernel_mapping.csv`
- `full_dependency_graph.json/.md`
- `full_resource_compatibility_matrix.csv/.md`
- `full_overlap_candidate_shortlist.md`
- `visual_streaming_feasibility.md`
- `stream_activity.csv`, `timeline_summary.json`, `nvtx_phase_ranges.csv`
- `gate_summary.json`, `analysis_summary.json`, and four automated PNG figures

The reusable parser is `poc_flashvep/resource_atlas/analyze_nsys_resource_atlas.py`.
It discovers the SQLite schema at runtime, resolves StringIds, associates NVTX
ranges with child-process CUDA kernels, and never fabricates TensorCore/HBM
utilization.

**Next single action:** use this full-serving atlas to select one dependency-safe
pair for a separately controlled, single-request cross-request validation. Do
not implement a production scheduler or overlap method from the atlas alone.
