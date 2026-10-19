# DWDP transfer to dense recurrent Ouro

Worker: `/root/independent_axes`.
Segment began at 2026-10-19 13:26:05 UTC.
Evidence class: primary-source inspection and structural accounting.
This audit contains no GPU run, new model-quality result, or runtime change.

## Outcome

DWDP adds a valid peer-weight-prefetch control.
It does not establish a recurrence-specific ownership optimum for unchanged dense Ouro.
The closer dense prior, SiDP, already combines weight streaming with activation shipping.
No new candidate follows from this audit.
The verdict is `GENERIC_OWNERSHIP_TRANSFER_WITH_UNMEASURED_OURO_COST`.

The earlier remote-MLP result rejects its tested activation round trip.
It does not reject peer weight streaming, direct peer-load kernels, or every hybrid layout.
Those mechanisms have different costs and require different controls.

## 1. What DWDP actually implements

DWDP retains independent workers, replicates attention weights, and distributes MoE weights.
It stages remote expert weights through asynchronous peer copies and uses double buffering.
Compute waits for the local prefetch completion, not a group-wide all-gather.
The paper's target is the context side of disaggregated DeepSeek-R1 serving on GB200 NVL72.
It uses NVFP4 MoE weights and FP8 KV.
Its roofline maximum assumes enough compute to hide prefetch.
Sections 4 and Appendix A discuss source contention and compute–communication interference.
Its end-to-end study changes context provisioning and reports TTFT tradeoffs.
Those results do not predict BF16 Ouro decode latency.
[DWDP v2, sections 2–5 and Appendix A](https://arxiv.org/html/2604.01621v2).

The [arXiv record](https://arxiv.org/abs/2604.01621) lists v2 on 2026-05-12.
The inspected HTML body's date is August 24, 2026.
This audit records that date discrepancy without inferring a revision chronology from it.

The official [integration PR #12136](https://github.com/NVIDIA/TensorRT-LLM/pull/12136) merged on 2026-04-02.
This audit inspected both its merge revision and current main.
It did not treat the blog's initial support limits as immutable current-code limits.

At merge revision `e92ee4f`, `dwdp.py:334–387` exchanges IPC metadata and opens peer pointers.
Lines 531–585 copy those peer weights into local buffers with `cudaMemcpyAsync`.
Lines 157–238 allocate ping-pong buffers and order their consumption and reuse with events.
Lines 416–470 construct an NVFP4 view from local and already-prefetched buffers.
The compute kernel does not receive a peer pointer as a substitute for the remote staging buffer.
[Pinned merge implementation](https://github.com/NVIDIA/TensorRT-LLM/blob/e92ee4fe550783a5f3a51cd690d89d1fe18ea30b/tensorrt_llm/_torch/pyexecutor/dwdp.py#L531).

Current main is `321a38c`, dated 2026-10-19T12:39:25Z.
Its virtual-address design maps the local region directly and backs remote regions with local page-pool buffers.
`weight_manager.py:255–335` copies peer slices into those buffers.
Its optional batched path uses `cudaMemcpyBatchAsync`.
Lines 446–514 wait for the copy, bind the composite tensor, and protect the buffer until compute finishes.
Virtual address continuity does not mean that GEMM directly streams every remote weight from peer HBM.
[Current weight buffer](https://github.com/NVIDIA/TensorRT-LLM/blob/321a38cd0ecbc22b4931d488cf83d31244cd7db3/tensorrt_llm/_torch/modules/dwdp/weight_buffer.py#L84),
[current weight manager](https://github.com/NVIDIA/TensorRT-LLM/blob/321a38cd0ecbc22b4931d488cf83d31244cd7db3/tensorrt_llm/_torch/modules/dwdp/weight_manager.py#L255).

The inspected current setup still registers and extracts MoE expert modules.
It rejects registered layers without the expected experts module.
This source path is not an inspected dense Ouro adapter.
This audit does not assert that every current DWDP path retains the initial NVFP4-only restriction.
[Current setup, lines 357–422](https://github.com/NVIDIA/TensorRT-LLM/blob/321a38cd0ecbc22b4931d488cf83d31244cd7db3/tensorrt_llm/_torch/modules/dwdp/setup.py#L357).

## 2. A closer dense ownership prior

SiDP targets dense offline inference with a distributed FFN weight pool.
Its Weight-as-a-Service mode copies remote weights into a local cache.
Its Compute-as-a-Service mode sends activations to the owner and returns outputs.
The owner can combine incoming rows into a larger FFN batch.
Its runtime protects cache slots and staggers remote reads to reduce owner hotspots.
The paper uses a profiled batch threshold and coordinated mode changes.
Its evaluation uses much larger dense models and throughput-oriented workloads.
It does not establish low-concurrency Ouro behavior or an online per-depth mode-switch benefit.
[SiDP, sections 4.1–4.4 and 5.1](https://arxiv.org/html/2605.28095v1).

No verified SiDP repository link appeared in the inspected paper or bounded title/code query.
This is not proof that no public code exists.
The source screen relies on its explicit mechanism, not an assumed local implementation.

SiDP directly covers the choice between moving weights and moving activations.
Existing [AFD and KernelFlume controls](operator_ownership_nearest_audit.md) cover related compute/state ownership choices.
Generic TP/TSP remains the same-P sharded-compute control.
The new sources strengthen the generic comparison map.
They do not justify renaming that comparison as a new recurrent architecture.

## 3. Distinguish three data paths

| Path | Tensor movement | Required completion |
|---|---|---|
| Staged peer weights | Immutable weights to local scratch, then local GEMM reads | Copy before GEMM, compute before scratch reuse |
| Direct peer-load GEMM | Kernel loads weight operands from peer addresses | Operand availability and valid peer mappings |
| Compute-owner MLP | Input activations to owner, output activations back | Complete input, owner compute, complete return |

CUDA supports kernel access to peer memory when topology and peer-access support permit it.
That capability does not establish the performance of a dense GEMM with peer operands.
[CUDA 12.9.1, section 6.2.9.4](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#peer-to-peer-memory-access).

For a staged remote slice of M bytes, charge the source read, M link bytes, destination write, and later GEMM operand read.
Caches can change the physical HBM transactions.
These are logical operations, not measured HBM counts.
Also charge event dependencies, address metadata, allocation alignment, and buffer lifetime.
Dense TP shards do not necessarily match DWDP's contiguous expert blocks.
A dense transfer needs compatible GEMM views or charged layout/packing work.

Direct peer loads can avoid materializing the complete staged slice in destination HBM.
They still consume source memory service and interconnect bandwidth.
Kernel tiling, repeated loads, transaction size, caching, and latency determine actual traffic and stalls.
This audit neither proves that path loses nor identifies a qualifying optimized Ouro kernel.

Activation shipping charges both input and output directions.
It also charges packing, queueing, owner-side execution, and the source's return dependency.
A slow activation round trip cannot serve as a measurement of staged peer-weight GEMM.

## 4. Dense R4 repeats all required operands

Use the pinned [Ouro dimensions](../artifacts/semantics_manifest.json): D=2048, I=5632, and 24 or 48 physical layers.
There are 16 query heads and 16 KV heads.
The calculation adds no MoE sparsity or invented query/key normalization.

One BF16 MLP contains `3*D*I*2 = 69,206,016 bytes = 66 MiB`.
The Q/K/V/O matrices contain `4*D*D*2 = 32 MiB` per layer.
The unique dense cores therefore contain 2352 MiB and 4704 MiB.
Norm scales, embeddings, heads, gates, KV, and runtime scratch are additional.

Consider balanced MLP ownership over P ranks, with attention replicated.
Each worker computes its local ready batch through all four passes.
Assume that it does not retain remote MLP weights between visits.
The ideal remote payload per worker batch is `4*(1-1/P)*L*66 MiB`.
This counts one transfer per required weight per visit.
It does not multiply by batch rows because one GEMM batch can share the same weights.
It also does not predict actual cache or protocol traffic.

| Model | P | Unique MLP, MiB | Remote R4 payload/worker batch, MiB | Resident MLP bytes saved/rank before staging, MiB |
|---|---:|---:|---:|---:|
| 1.4B | 2 | 1584 | 3168 | 792 |
| 1.4B | 4 | 1584 | 4752 | 1188 |
| 1.4B | 8 | 1584 | 5544 | 1386 |
| 2.6B | 2 | 3168 | 6336 | 1584 |
| 2.6B | 4 | 3168 | 9504 | 2376 |
| 2.6B | 8 | 3168 | 11088 | 2772 |

If all P workers execute one such batch, aggregate link payload is `4*(P-1)*L*66 MiB`.
Balanced outgoing payload equals incoming payload per rank in that all-active case.
Counting both endpoint operations must not double-count the same wire payload.
Immutable weight copies need no weight-data return.
Opposite-direction copies instead serve other readers, with their own completion and resource costs.
At smaller active-worker counts, charge only actual readers and their actual owners.
Advertised aggregate bidirectional bandwidth cannot replace that traffic matrix.

A two-buffer, per-layer sharded layout adds up to `2*(1-1/P)*66 MiB` of staging per worker.
A whole-layer-owner layout can instead need two complete remote MLP buffers, or 132 MiB.
These examples exclude padding, communication metadata, activations, and graph pools.
They illustrate why gross resident-weight savings are not net free HBM.

For B ready rows, MLP input plus output payload is `2*B*2048*2 = 8192*B bytes`.
One complete 66 MiB weight transfer has 8448/B times that activation payload.
At B=8, the payload ratio is 1056.
This ratio is not a latency crossover or proof that compute-owner execution wins.

## 5. What recurrence changes, and what it does not

Weights are immutable, so future weight prefetch is legal before future hidden states exist.
That is useful potential overlap, but ordinary models also expose predictable next-layer weights.
R4 does not make the next dense computation ready before its preceding residual, norm, attention, and MLP dependencies.

Ouro stores one physical core, not four materialized copies.
Keeping every remote MLP weight after the first pass can remove later refetches.
That policy restores a complete local MLP replica and consumes the corresponding HBM.
Partial retention offers a capacity/traffic tradeoff.
An aliased unrolled executor can apply the same policy to the same repeated pointers.
No recurrence-specific ownership gain follows without a cost difference against that control.

At system C=1, a worker cannot use four recurrent passes as four independent ready requests.
At C=2/4/8, distribute the same available rows across workers without multiplying concurrency by four.
Compare local batches, complete replicas where feasible, same-P TP, and generic weight-sharing modes.
Attention, historical KV, and original gate identities remain request/depth/layer specific.

The current 1.4B/2.6B cores are much smaller than one H100's HBM capacity.
This does not guarantee that every KV and scratch configuration fits.
Any capacity-benefit claim must first show an actual admission limit that the saved weight bytes resolve.
The audit establishes no such limit or additional admitted request.
Do not transfer a large-model, offline capacity advantage without that changed assumption.

Adaptive execution may use only a subset of later passes.
Speculative weight prefetch can therefore waste bytes even though it does not speculate hidden states.
Charge those bytes and buffer occupancy.
Preserve the original gate, selected outputs, and required missing-depth KV policy.

Byte-preserving copies can retain BF16 checkpoint values.
New peer-load kernels, tensor layouts, or accumulation orders still need numerical qualification.
No source statement establishes bitwise Ouro outputs or held-out model quality.

## 6. Scope of the next discriminator

This source audit does not justify an engine.
The smallest useful follow-up first resolves these questions:

1. Does the exact workload need HBM capacity that complete local replicas cannot supply?
2. Which weights remain remote, which remain cached, and which copy or load path serves them?
3. What ready computation covers each transfer, including first/last-layer and recurrent-wrap boundaries?
4. What are the per-owner incoming/outgoing bytes under actual C=1/2/4/8?
5. Does the full path beat matched same-P TP and generic dense weight-sharing controls?

Any later experiment must retain native async scheduling, whole-R4 graphs, exact required KV, and the original numerical contract.
It must include source-side contention and local-kernel interference.
It must not infer E2E gain from advertised NVL72 bandwidth or a prefetch-only microbenchmark.
The current H100 system is not the paper's GB200 NVL72 system.
P8 remains a projection-only topology until separately authorized.

## 7. Evidence and time limits

The [structured observations](../artifacts/dwdp_dense_observations_v1.json) retain primary locators, code pins, hashes, search queries, and all six accounting rows.
The audit read official paper text and selected official code paths.
It did not install, clone, build, or execute those runtimes.
The dense SiDP paper is the one additional nearest-paper review in this segment.
The evidence and validation checkpoint was 13:33:51 UTC.
The segment lasted 466 seconds to that checkpoint.
Concurrent worker intervals do not add campaign elapsed hours.
