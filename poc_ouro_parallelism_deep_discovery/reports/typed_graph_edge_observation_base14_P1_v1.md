# Typed graph edges: Base 1.4B P1 observation

Append-only observation, revision 1. This report does not replace earlier failure logs or the pre-probe contract.
The coordinator ran the GPU probe. This worker ran only the CPU summary and tests.

## Observed metadata

The probe uses Base Ouro-1.4B, fixed R4, P1, and physical GPU5.
The controlled request has 1,024 prompt tokens and four generated tokens.
One retained graph accounts for three observed replays.
The raw export reports stable metadata across two read-only queries.
It contains 1,154 nodes: 1,153 kernels and one memcpy.
It contains 1,153 edge records.

| Raw eight-byte record | Count | Fields |
|---|---:|---|
| `0000000000000000` | 1,057 | All fields zero |
| `0100010000000000` | 96 | `from_port=1`, `to_port=0`, `type=1`, five zero reserved bytes |

The CPU join maps all 96 non-default records from `FlashAttnFwdSm90` kernels to `FlashAttnFwdCombine` kernels.
No other non-default classification pair appears in this capture.
The CSV preserves every full kernel name and every edge-data byte.
This mapping describes captured metadata, not kernel duration.

## Nominal cuts

| Kernel-group cut | Crossing edge index | Raw bytes | Non-default? |
|---:|---:|---|---|
| 0 | None | No graph edge | Not applicable |
| 288 | 287 | `0000000000000000` | No |
| 576 | 575 | `0000000000000000` | No |
| 864 | 863 | `0000000000000000` | No |
| 1,154 | None | No graph edge | Not applicable |

These cuts separate fused kernel groups.
They do not mark exact semantic pass boundaries.
The following normalization kernel can combine work from adjacent semantic passes.
The observation does not authorize event insertion or any dependency change.

## What follows, and what does not

CUDA 12.9 defines dependency type 1 as programmatic.
Its source port 1 corresponds to the programmatic completion trigger or termination of all source-kernel blocks.
This differs from an ordinary full-kernel completion edge. [CUDA edge fields](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/structcudaGraphEdgeData.html)

The endpoint diagram still forms one chain.
That connectivity does not establish full-kernel serialization because 96 edges carry non-default semantics.
It also does not establish that those kernel pairs overlapped during any replay.
There is no measured overlap duration, kernel-group duration, or request latency in this export.
There is no speedup or new recurrence-specific architecture result.

The v3 default-only getter rejected information loss.
This export identifies the missing metadata rather than suppressing that rejection.
The default-only EventGraph remains unchanged.
The earlier v1/v2 runtime-selection failures remain separate from the v3 lossy-query failure.

## Evidence and CPU verification

Raw evidence: `outputs/native_graph_typed_edges_base14_P1_v1/rank0_replayed_graph0_edges.json` and its sibling DOT.
The summarizer checks the DOT hash, snapshot fingerprint, endpoint multiplicities, edge indices, and raw/named field agreement.
Its derived files are:

- `analysis/typed_graph_edge_summary_base14_P1_v1/summary.json`
- `analysis/typed_graph_edge_summary_base14_P1_v1/edges_with_kernels.csv`

Four new CPU tests pass. The combined four-module regression passes all 33 tests.
Negative cases reject a changed DOT, changed raw bytes, and inconsistent named fields.
No frozen raw output changes.

The bounded source-lead check and CPU summary share one recorded worker continuation.
Continuation start: 2026-10-19 13:54:36 UTC.
The first CPU-summary checkpoint is 2026-10-19 13:57:34 UTC.
Evidence cutoff: 2026-10-19 14:02:05 UTC.
The recorded continuation is 449 seconds through that cutoff.
This worker interval is not additional campaign elapsed time.
