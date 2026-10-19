# Persistent execution red-team audit

Checkpoint time: 2026-10-19 10:41:23 UTC.
Worker: `/root/independent_axes`.
This report records source inspection and CPU arithmetic.
It does not report a new GPU measurement or a model-quality result.

## Outcome

The full BF16 shared core cannot remain on-chip at P2/P4/P8 under the audited representation.
This conclusion survives a deliberately generous storage bound.
It does not reject streamed persistent kernels or partial weight retention.

Streamed persistence exposes a different question: can removed dispatch and operator boundaries exceed added runtime and kernel costs?
That question remains unmeasured for Ouro.
Existing generic runtimes already implement the central execution mechanism.
The candidate verdict is `GENERIC_ONLY`, not a new parallelism claim.
The complete card is [IAX09](../candidates/IAX09_parameter_resident_persistent.md).

## 1. Distinguish three forms of persistence

| Form | What remains active | What it can save | What it does not establish |
|---|---|---|---|
| Persistent GEMM grid | A fixed worker grid for one GEMM | Tile scheduling and launch efficiency | Weight retention across the full stack |
| Full-stack persistent work | Device workers across many operators | Dispatch, boundaries, and legal overlap | All model weights remain on-chip |
| Parameter-resident recurrent core | Recurrent weights remain in on-chip storage | Selected repeated weight reads | Feasibility when the complete core exceeds storage |

These forms can coexist, but their costs differ.
Calling all three “persistent” hides the decisive storage test.

## 2. Audited local paths

The installed vLLM Ouro model repeats the same physical layers across recurrent passes.
Its unquantized linear method dispatches to `torch.nn.functional.linear` on the inspected default CUDA path.
This source inspection does not identify the final cuBLAS kernel selected at runtime.
The model defines a gate, but its inspected forward does not use that gate.
Thus, fixed vLLM and original adaptive CDB need separate baseline paths.

The CDB runner already captures and replays recurrent-stage graphs.
The graph key contains the batch shape, gate flag, and KV slot.
Static buffers retain device metadata between launches.
Its original cumulative-hazard policy calls `.item()` on each sigmoid gate value.
The policy updates survival and the exit CDF on the host.
Replacing this path with device control can save synchronization, but it creates a numerical contract.

For the inspected policy, let `p=sigmoid(gate).float()`.
It updates `cdf <- cdf + p*survival` and `survival <- survival*(1-p)`.
Host Python arithmetic and a new FP32 device implementation can differ near q=.5.
A device implementation must match the policy or classify and test the numerical change.
It must preserve minimum R2, maximum R4, output selection, and missing-depth KV behavior.
The gate cannot authorize a future pass before its inputs exist.

The local FlashInfer package reports version 0.6.14.
Its build commit is `19f1a41e6b21f0c422d775e377b6fdf9a1fc9d23`.
The source contains these generic mechanisms:

- The cuTile BF16 GEMM uses a persistent grid and SM-aware tile scheduling.
- Its inner K loop still loads A and B tiles with `ct.load`.
- TinyGEMM transfers weight tiles from global memory through its pipeline.
- The persistent-attention template joins two runners with a grid synchronization.

The 16-stage TinyGEMM pipeline uses 196,608 bytes before other shared storage.
The source computes `16*4*(16*64*2 + 8*64*2)` bytes.
This calculation is not an achieved occupancy measurement.
A new resident weight allocation must compete with existing buffers and registers.
PDL also requires a compatible launch chain.
The inspected wrapper warns that an incompatible preceding operation can cause a hang.
This audit did not enable PDL or execute these kernels.

Source paths and hashes appear in [the audit artifact](../artifacts/persistent_execution_audit.json).

## 3. Capacity and revisit accounting

Both checkpoints use d=2048, intermediate width 5632, and 16 attention/KV heads.
The physical layer counts are 24 and 48.
The dense block contains Q/K/V/O and SwiGLU gate/up/down weights.
There is no invented query/key normalization in this accounting.

`block_elements = 4*2048^2 + 3*2048*5632 = 51,380,224`

`block_BF16_bytes = 102,760,448 = 98 MiB`

The 1.4B core therefore contains 2352 MiB of dense block weights.
The 2.6B core contains 4704 MiB.
This excludes norm scales, the gate, embeddings, and the LM head.
It is a lower storage count for full-model residence.

The existing hardware probe records 132 SMs and 50 MiB L2 on physical GPUs 4 and 6.
Its raw file is `outputs/components_base14_P2_graph_v1/components.json`.
The probe timestamp is 2026-10-19T10:34:52.088103+00:00.
The parent worker ran that probe.
This worker only read its completed artifact.

For 132 SMs, all registers total 33 MiB.
All shared memory totals 29.390625 MiB under the documented 228 KiB/SM limit.
Adding the full 50 MiB L2 gives 112.390625 MiB.
Registers, shared memory, and L2 cannot act as one freely accessible pool.
Summing them still gives a useful, proposal-favoring capacity screen.
[NVIDIA Hopper Tuning Guide, sections 1.4.1–1.4.2](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html).

An even more generous full-GH100 screen uses 144 SMs and 60 MiB L2.
Its non-unified sum is 128.0625 MiB.
NVIDIA distinguishes this full implementation from the 132-SM H100 SXM product.
This is not a claim about installed hardware.
[NVIDIA Hopper architecture, architecture-unit lists](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/).

| Model | P | Dense core/rank, MiB | Dense block/rank, MiB | Intervening other layers/rank, MiB |
|---|---:|---:|---:|---:|
| 1.4B | 2 | 1176 | 49 | 1127 |
| 1.4B | 4 | 588 | 24.5 | 563.5 |
| 1.4B | 8 | 294 | 12.25 | 281.75 |
| 2.6B | 2 | 2352 | 49 | 2303 |
| 2.6B | 4 | 1176 | 24.5 | 1151.5 |
| 2.6B | 8 | 588 | 12.25 | 575.75 |

Every core/rank value exceeds both generous sums.
This test assumes ordinary uncompressed BF16 weight storage.
It does not address a new exact compression algorithm.
It also does not reject selective replication or a streamed mega-kernel.

The revisit count measures distinct other-layer weight bytes between corresponding layer visits.
It excludes KV, activations, and other weights.
It is not measured HBM traffic or an LRU miss count.
L2 policy can prioritize a selected range, but cannot guarantee all weights remain resident.
CUDA also supports cache policies on graph kernel nodes.
Thus, the graph baseline deserves the same selected-range policy.
[CUDA L2 cache control, access-policy windows and graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html).

A fixed protected subset offers a separate hypothesis.
Its maximum retained bytes must be named before timing.
It cannot receive credit for all four core reads.
The same-token activation still traverses every required layer and loop norm.
Selected tiles can help without creating a new axis of ready work.

## 4. Dynamic KV and control are not optional

Full depth-indexed KV stores separate logical state for each recurrent depth and physical layer.
For BF16, it requires `2*L*4*N*16*128*2` bytes before sharding.
This gives 0.75 GiB at N=1024 for 1.4B.
It gives 1.5 GiB at N=1024 for 2.6B.
At N=16384, these values become 12 GiB and 24 GiB.
These are capacity values, not measured attention traffic.

Persistent workers still need the required request, token, depth, layer, slot, and position metadata.
They must publish current-token KV before the corresponding attention consumer.
Adaptive execution must preserve `last_exited` state propagation, where selected by the pinned runtime.
Fewer executed loops do not imply fewer allocated KV slots.
Request completion and continuous batching need additional device control or host interaction.

The local scalar gate computation is small, but a host wait can expose latency.
Removing that wait is generic control engineering until a recurrence-specific advantage survives matched controls.
Modern CUDA graphs also support conditional device control.
The documented conditional body remains single-device.
Multi-GPU TP therefore still needs validated coordinated routing and communication.
[CUDA Graphs, conditional nodes](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

## 5. Primary-source competitors

Persistent RNNs kept recurrent weights in registers across timesteps.
Its small recurrent matrices and storage assumptions explain the relevance and the transfer limit.
This paper does not demonstrate resident storage for Ouro's complete 24/48-layer core.
[Diamos et al., ICML 2016, sections 3–4](https://proceedings.mlr.press/v48/diamos16.pdf).

MPK is the decisive generic baseline.
The paper uses SM-level tasks, device queues, semaphores, and in-kernel communication.
Shared-memory pages support cross-task prefetch rather than whole-model residency.
Its BF16 serving experiments include paged attention and continuous batching.
They are not Ouro experiments.
The paper's 64-token prompts and 1024-token outputs also differ from this campaign's context matrix.
[MPK, arXiv:2512.22219v2, sections 3–6](https://arxiv.org/html/2512.22219v2).

The inspected author revision is `ca58a02d110a278bffaed503be5e66494db2597d`, dated 2026-10-17.
Its runtime has iteration-tagged task IDs, worker queues, device KV page state, and NVSHMEM integration.
This code inspection is not an installation or a successful local MPK run.
[Pinned runtime source](https://github.com/mirage-project/mirage/blob/ca58a02d110a278bffaed503be5e66494db2597d/include/mirage/persistent_kernel/persistent_kernel.cuh).

FLUTE accelerates lookup-table quantized matrix multiplication.
It does not supply unchanged BF16 full-core residency.
Some released quantization workflows also use learned scales.
The campaign cannot use those as a new training-free BF16 mechanism.
[FLUTE paper](https://arxiv.org/abs/2407.10960v4), [pinned author repository](https://github.com/HanGuo97/flute/tree/9eb83a12d56949bbe7fe9c836ba97a67bd1e3761).

Current FlashInfer documentation lists persistent cuTile scheduling among BF16 GEMM options.
The documentation version differs from the installed package.
The local source, not the current documentation, determines installed behavior.
[FlashInfer BF16 GEMM documentation](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_bf16.html).

## 6. A distinct secondary hypothesis: repeated program descriptors

`IAX10` would compress the repeated execution program, not its weight tensors.
It would store one physical task template and instantiate request/depth/KV-slot state per iteration.
This targets descriptor traffic, instruction footprint, and capture storage.
It does not claim a new independent recurrent frontier.

The useful work remains the exact ready task graph.
Weights and historical KV remain in their original owners.
Each descriptor instance requires a depth/version tag before reuse.
Added costs include address calculation, branch control, queue state, and version tests.
The saved descriptor fraction on the critical path is unknown.

The sharp falsifier compares aliased logical unrolling with a generic loop-capable task runtime.
Use the same kernels, pointers, metadata, and legal schedule.
If both use the same compact loop representation, there is no recurrence-specific distinction.
The pinned MPK runtime already includes iteration-tagged task IDs.
CUDA conditional graphs also supply generic loops.
Thus, this hypothesis receives `GENERIC_ONLY` without an engine proposal.

Do not count IAX10 as a surviving new axis.
It is a separately identified cost target with a strong existing control.

## 7. Cheapest useful next tests

1. Measure exposed dispatch and gate waits in the existing graph baseline.
2. Keep model, q, KV policy, outputs, and required operations unchanged.
3. Compare compact loops with aliased unrolling under the same generic runtime.
4. Name any protected weight subset and compare the same cache policy on graph replay.
5. Stop before engine implementation if the optimistic savings cannot cover added persistent costs.

Apply the first test to 1.4B/2.6B and fixed/original-adaptive paths.
P2/P4 need the actual topology and a validated TP implementation.
P8 remains projected until separately authorized hardware exists.
Warm repeated layer0 timings do not measure full-stack revisit residency.

## 8. Evidence boundary and time

This segment began at 2026-10-19 10:30:01 UTC.
The recorded end checkpoint is 2026-10-19 10:41:23 UTC.
The checkpoint span is 682 seconds.
All six model/P rows passed the CPU arithmetic test.
Both report files passed the prose test with zero findings.
The segment includes primary reading, local source inspection, arithmetic, and report work.
It contains no GPU execution by this worker.
Parent GPU work overlaps this segment and must not multiply campaign elapsed time.

Two path tests failed before targeted file discovery resolved the paths.
The web tool could not open two GitHub API endpoints.
Read-only `curl` calls then retrieved the commit metadata.
This worker did not install, clone, or execute external source.

This audit claims no complete model fit, latency benefit, task-quality result, or traffic reduction.
