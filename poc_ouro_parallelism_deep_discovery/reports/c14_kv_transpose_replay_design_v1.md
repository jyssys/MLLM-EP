# C14: physical KV ownership transpose — design v1

## Decision and scope

Status: `DESIGN_ONLY_NOT_IMPLEMENTED`.
No GPU execution or transport implementation occurred.
The proposed evidence class is `PHYSICAL_KV_LAYOUT_TRANSPOSE_REPLAY`.
It is a synthetic state microbenchmark, not model E2E execution.

The specified depth-owner layout requires a physical transpose before head-sharded TP4 decode.
Exactly 75% of its unique prefix payload changes GPU owner.
This fraction does not specify link traffic or latency.
Packing, local copies, NCCL transport, and paged scatter all count.

This report examines one request, full R4, and prefix lengths 1024, 8192, and 32768.
Both Base checkpoints use 16 MHA heads, head dimension 128, and BF16 storage.
Base1.4 has 24 physical layers.
Base2.6 has 48 physical layers.
The checkpoint configurations state these dimensions.
Their source identities appear in the companion JSON.

Depth-stationary ownership differs from the physical-layer ownership in the earlier [C14 report](c14_cyclic_prefill_reopening_audit_v1.md).
That earlier report assigns 6 or 12 physical layers to each owner across all depths.
This screen instead assigns all layers of one depth to each owner.
Neither variant receives zero-cost transition credit.

## Source-backed destination contract

Installed [OuroAttention](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/ouro.py:122) divides the 16 heads across TP ranks.
TP4 therefore owns four Q heads and four KV heads per rank.
The [QKV loader](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/linear.py:1303) selects contiguous rank shards.
Rank q owns global heads `[4q,4q+4)`.

[OuroAttention](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/ouro.py:171) creates one Attention object for each recurrent pass.
Its logical cache index is `j = r*L + ell`.
The [model loop](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/ouro.py:367) retains the original recurrent normalization.
Shared physical weights do not permit shared depth KV.

The installed [FA3 backend](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/attention/backends/flash_attn.py:131) defines logical cache shape `[blocks,heads,16,256]`.
K occupies content indices 0–127.
V occupies indices 128–255.
For NHD, physical order is `[blocks,16,heads,256]`.
The [read path](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/attention/backends/flash_attn.py:877) transposes the middle axes before splitting K and V.
The [write path](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/attention/backends/flash_attn.py:1085) uses slot mapping.

The replay must use this native-compatible geometry.
It must not assume that a future native worker allocates separate contiguous tensors for all slots.
The backend also defines layer-interleaved arena orders.
Record every actual view's shape, stride, storage offset, dtype, and device before a replay.

CDB uses a different [physical representation](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_batching/cache.py:40).
It allocates separate K and V tensors with shape `[token_capacity,heads,128]`.
Thus a CDB source requires explicit K/V interleaving during packing.
The first proposed replay uses native-compatible full-head NHD sources instead.
It does not claim a completed CDB-to-native adapter.
The synthetic values need not represent checkpoint-generated activations.
This replay tests routing, copying, and layout only.

## Exact ownership and addresses

Let r, p, and q range from 0 through 3.
Let ell range from 0 through L−1.
The immutable prefix covers positions `[0,N)`.
Each state key is `(request, epoch, r, ell, position, head, K_or_V, component)`.

Before transfer, source rank p owns depth `r=p` with all 16 heads.
After transfer, destination rank q owns all four depths with heads `[4q,4q+4)`.
The transformation is a permutation of BF16 words.
It performs no arithmetic reduction or dtype conversion.

For the first replay, reserve block 0 as a poisoned null page.
Map token t to physical block `1+floor(t/16)` and offset `t mod 16`.
Use disjoint source, destination, send, and receive allocations.
Allocate L source slots and 4L destination slots per rank.

For separate contiguous NHD slots, the logical strides in BF16 elements are:

| View | Shape | Stride |
|---|---|---|
| Source slot ell | [B,16,16,256] | [65536,256,4096,1] |
| Destination slot rL+ell | [B,4,16,256] | [16384,256,1024,1] |

Here `B=N/16+1`, including the null page.
These strides define the proposed fixture, not an observed native allocation.
For an interleaved arena, the recorded actual strides replace these constants.

For head h, use `q=floor(h/4)` and `local_head=h mod 4`.
The source address uses slot ell on rank r.
The destination address uses slot rL+ell on rank q.
Both addresses retain the same token, component, and epoch.
An address equals its storage base plus twice its element offset.

The destination prefix length remains N after publication.
A later decode query at position N needs its own writable slot.
Do not publish a nonexistent current-token KV value.
This replay does not execute that query or its FA3 computation.

## Exact payload accounting

Define one ordered source-to-destination piece as C.
Define one rank's source or destination prefix payload as S.

```text
C = L*N*4 heads*128 components*2 K/V*2 bytes = 2048*L*N
S = 4*C = 8192*L*N
same-rank piece = C
remote send per rank = remote receive per rank = 3*C
cluster unique prefix = 4*S = 16*C
cluster one-way remote payload = 12*C
cluster endpoint send-plus-receive accounting = 24*C
```

The final retained aggregate KV capacity does not change.
Endpoint send-plus-receive bytes are not link bytes.
NCCL routing, protocol overhead, and intermediate hops can change physical link traffic.

The following values are structural accounting.
One MiB is 1048576 bytes.

| Base | N | S per rank, MiB | Local piece, MiB | Remote send per rank, MiB | Cluster remote payload, MiB | Cluster unique KV, MiB |
|---|---:|---:|---:|---:|---:|---:|
| 1.4B | 1024 | 192 | 48 | 144 | 576 | 768 |
| 1.4B | 8192 | 1536 | 384 | 1152 | 4608 | 6144 |
| 1.4B | 32768 | 6144 | 1536 | 4608 | 18432 | 24576 |
| 2.6B | 1024 | 384 | 96 | 288 | 1152 | 1536 |
| 2.6B | 8192 | 3072 | 768 | 2304 | 9216 | 12288 |
| 2.6B | 32768 | 12288 | 3072 | 9216 | 36864 | 49152 |

Remote receive per rank equals remote send per rank.
Base2.6 at N32768 therefore transfers 36 GiB of remote logical payload across the cluster.
No bandwidth assumption converts this value into elapsed time.

## Pack, NCCL, and scatter protocol

The proposed primary replay uses 1024-token chunks and all L local source slots per chunk.
It uses one ordered all-to-all call per chunk.
The six cells require 1, 8, or 32 calls each.
This is one fixed policy, not an autotuned transport sweep.

For each chunk, use contiguous BF16 send and receive buffers with shape `[4,L,1024,4,256]`.
The send buffer's first axis identifies destination q.
The receive buffer's first axis identifies source p.
The source head slice is generally strided across token rows.
A view alone does not perform packing.

The ordered steps are:

1. Check the immutable source epoch and the unpublished destination lease.
2. Pack each head group into its destination-major send buffer.
3. Execute NCCL all-to-all with matching counts and order on all four ranks.
4. Establish NCCL completion on the consuming service stream.
5. Scatter every receive piece into its destination depth and layer pages.
6. Record destination completion after all chunks.
7. Check all-rank success before the host marks the new ownership epoch ready.

[all_to_all_single](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/torch/distributed/distributed_c10d.py:4694) provides equal destination splits and source-ordered output.
The call reaches `group.alltoall_base`.
A grouped P2P implementation is an alternative, not a second free control.
Installed [PyNccl](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/distributed/device_communicators/pynccl.py:322) exposes send, receive, and grouping on an explicit stream.
Its methods do not pack strided views.
Reject a disabled communicator instead of treating its no-op as successful transport.

The [installed NCCL interface](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/torch/include/torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp:287) states that collectives use a separate CUDA stream.
A returned Python call does not establish GPU completion.
`Work.wait()` or `Work.block_current_stream()` orders subsequent service-stream reads.
The final event must follow that dependency and every scatter.
Only the final event wait occurs at the trial boundary.

Packing reads S and writes S per rank.
Scattering reads S and writes S per rank.
Thus these explicit kernels request 4S bytes of total reads and writes per rank.
This is not measured HBM traffic.
Communication adds its own endpoint accesses and possible internal staging.

The primary replay includes the local C piece in pack, all-to-all, and scatter.
The local piece requires no remote payload.
It does not receive zero-cost copy credit.
A future self-alias variant needs separate stride and allocation-lifetime evidence.

## Temporary memory and page capacity

The source remains allocated and immutable until transport completes.
The destination needs its full S before publication.
Each side also reserves one null page per slot.
The per-rank null-page term is `Z=131072*L` bytes for each side.

A monolithic replay needs S bytes for each send and receive buffer.
Its tensor peak is `4S+2Z`.
The primary chunked replay needs `8192*L*1024` bytes for each communication buffer.
Its tensor peak is `2S+2Z+16384*L*1024`.

| Base | N | Primary tensor peak, MiB | Monolithic tensor peak, MiB |
|---|---:|---:|---:|
| 1.4B | 1024 | 774 | 774 |
| 1.4B | 8192 | 3462 | 6150 |
| 1.4B | 32768 | 12678 | 24582 |
| 2.6B | 1024 | 1548 | 1548 |
| 2.6B | 8192 | 6924 | 12300 |
| 2.6B | 32768 | 25356 | 49164 |

These peaks exclude communicator workspace, allocator overhead, validation scratch, and metadata.
They also exclude model weights and activations.
Measure allocated and reserved memory separately if the replay proceeds.
The 80 GB device label does not establish currently free capacity.

One shared int32 block-table row needs N/4 bytes for each side.
Optional reservation for 128 later cached positions adds eight destination blocks.
That reservation adds 24 MiB per rank for Base1.4 or 48 MiB for Base2.6.
No append reservation participates in prefix transport.

A smaller layer tile G needs `16384*G*1024` scratch bytes.
It also increases collective rounds by L/G.
This is a capacity alternative, not an assumed latency improvement.
Do not silently change the primary chunk policy after results appear.

## Pointer and graph publication obligations

The replay owns a quiescent lease on all four devices and all source pages.
No serving process consumes or appends this synthetic state.
One guarded parent must own physical GPUs 4, 5, 6, and 7.
Each child must retain a singleton device mask.

Keep all source and communication buffers alive until their GPU readers complete.
Keep destination readers disabled until all writes and all-rank completion checks pass.
Preserve the source when any rank fails.
A failed transition must not publish partially populated destination metadata.

Record the following before any future graph use:

- Cache storage identity, offsets, strides, dtype, and depth-to-Attention mapping.
- Block-table identity, values, valid prefix lengths, and writable append slots.
- Producer completion events and consumer stream dependencies.
- Source lease lifetime and destination ownership epoch.

A cached graph can retain old tensor addresses.
Changing a Python cache reference does not update captured kernel arguments.
An existing decode graph cannot safely consume a newly rebound allocation without further binding evidence.
Allocate final destination storage before capture, or prove an in-place metadata update under a quiescent lease.

The installed [graph_capture context](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/distributed/parallel_state.py:1430) coordinates TP/PP capture streams.
It does not establish KV publication or pointer rebinding.
The initial screen should remain an eager transport replay.
No serving graph needs modification.
A synthetic address check does not establish a future native cached-decode binding.

## Correctness and clock checks

Use finite BF16 word patterns derived from the complete state key.
Use several patterns that encode different index digits.
One BF16 word cannot uniquely identify every coordinate.
Include both signed zeros.
Compare uint16 words, not floating-point equality.

Before timing, require these checks:

1. Check complete shape, dtype, word count, strides, and nonoverlapping leases.
2. Check every destination word against an independent coordinate oracle.
3. Check that source histories remain bitwise unchanged.
4. Check that null pages, unused pages, and guard regions retain their poison values.
5. Check round-trip reconstruction for every source depth and head.
6. Reject deliberately swapped depth, head, block, and K/V mappings.
7. Reject a skipped local piece and a signed-zero conversion.
8. Check the same ownership epoch on all four ranks.

Generate a new source epoch outside each timed replay.
Check full destination words after each trial, outside the service window.
Record validation time and scratch separately.
Do not store a hidden extra full-size GPU reference without charging its capacity.

Measure global makespan from parent release until every rank reports its final completion event.
Include IPC, packing, communication, local movement, scattering, and the publication acknowledgement.
Also report each rank's outer CUDA-event interval.
Do not compare unsynchronized event timestamps across GPUs.

Communicator initialization, allocation, source generation, and correctness checks remain separate preparation costs.
A preallocated replay is not a cold E2E transition.
Any pack/transport/scatter diagnostic intervals remain nested spans.
Do not force them into an additive 100% attribution.
No per-layer synchronization belongs inside the timed service.

Use one untimed full replay before measured trials.
Keep six model/length cells and their paired generic controls.
Use a fixed small repeat count if execution receives approval.
No output quality, natural workload, or architecture-speedup conclusion follows from this lane.

## Generic controls and interpretation

A phase-specific generic PP-to-TP baseline also requires this ownership change.
For balanced four-owner full-head sources, the remote fraction remains 75%.
Generic logical unrolling can place one full pass on each owner.
That control has the same depth-owner mapping as this replay.

A physical-layer PP control instead owns `L/4` layers at all four depths.
It still owns L full-head logical cache slots per rank.
For local slot k, its mapping is:

```text
depth-owner source: r=p, ell=k
layer-owner source: r=floor(k/(L/4)), ell=p*(L/4)+(k mod (L/4))
```

Both controls send four-head groups to the same native TP4 owners.
Match source strides, page fragmentation, chunk sizes, buffer capacity, collectives, and destination geometry.
Only the coordinate labels differ in a fully matched synthetic layout.
A label permutation alone should not create a transport advantage.

Native TP4 prefill already writes head-sharded KV.
Its no-switch control requires no ownership transpose.
Do not assign this new transfer to that native baseline.

A real depth-owner prefill normally retains a full core on each GPU.
This requires four aggregate core copies unless another weight protocol supplies them.
Native TP4 and balanced physical-layer PP can retain one aggregate core copy.
Weight reshaping, loading, residency, or retirement costs cannot disappear from a phase-specific E2E comparison.

Same-DAG expressibility does not establish equal realized cost.
Different strides, allocation lifetimes, kernels, messages, and schedules can change physical cost.
Conversely, matching all those physical choices constructs a stronger generic control than DAG expressibility alone.
The transpose itself is generic redistribution, not demonstrated recurrence-specific novelty.

The replay can measure a charged transition service on this hardware.
It can reject claims that the KV layout change is free.
It cannot establish faster prefill, native decode correctness, quality preservation, or an E2E win.
It cannot establish a distinct architecture without a separate physical mechanism.

For identical subsequent TP4 decode, a phase-specific win requires prefill savings to exceed the complete transition and other phase costs.
This audit measures no prefill saving.
A replay is decisive only if that missing transition cost determines an already supported comparison.
Otherwise, it adds a component measurement without resolving C14.

## Evidence and time record

The companion artifact is [c14_kv_transpose_contract_v1.json](../artifacts/c14_kv_transpose_contract_v1.json).
It records all six exact byte ledgers and inspected source hashes.
The earlier C14 card and reports remain unchanged.
No C07 artifact or runtime file changes.

Actual segment start: 2026-10-19 16:42:43 UTC.
Source/CPU verification checkpoint: 2026-10-19 16:52:03 UTC.
Elapsed time through that checkpoint: 560 seconds.
All six numerical ledgers and all source/config hashes passed CPU checks.
Both ownership maps passed bijection checks.
The CPU checks also covered 221184 sampled boundary scalar addresses.
All report links resolved.
The report lint found zero violations before this time-record update.
No GPU time occurred in this segment.

