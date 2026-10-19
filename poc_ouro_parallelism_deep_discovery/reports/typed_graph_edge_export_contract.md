# Read-only typed graph edges

Revision 1. Implementation and CPU validation only. This worker ran no live typed-edge export.
The coordinator started the guarded probe before handoff. Its result is outside this report.

## Separate failures

The runtime-selection report records the earlier two-library failure and its bounded correction.
That correction does not explain or remove CUDA error 402 in the native v3 attempt.
The v3 traceback reaches the edge-data fetch with `edgeData=NULL`.
The default-only query correctly refuses a semantically lossy result.
No valid event interval follows from that failure.

The existing event observer remains strict and unchanged in this task.
All native event v1/v2/v3 logs remain intact.
This implementation adds no event node and changes no dependency.
It does not disable programmatic dependent launch, or PDL.

## API and ABI

CUDA 12.9's `cudaGraphEdgeData` contains eight unsigned bytes.
Offsets 0, 1, and 2 hold `from_port`, `to_port`, and `type`.
The five reserved bytes begin at offset 3.
A zero record denotes ordinary full serialization with memory visibility.
Nonzero ports and dependency types carry different semantics. [Official edge structure](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/structcudaGraphEdgeData.html)

`cudaGraphGetEdges_v2` supports an all-NULL count-only query.
Its data query needs a metadata buffer to retain non-default annotations.
A NULL metadata buffer can produce `cudaErrorLossyQuery`.
Graph access also requires external synchronization against concurrent graph changes. [Official getter](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/group__CUDART__GRAPH.html)

The new runtime wrapper binds only identification and getter functions.
It reuses the coordinator's Torch-linked runtime selector and `RTLD_NOLOAD` policy.
It checks CUDA 12.9 and the host structure layout.
No graph-launch, event, graph-update, or dependency-mutation API belongs to its binding table.

The export retains all eight bytes as a byte list and a hexadecimal string.
It also records every named field, including each reserved byte.
Unknown field values remain unknown. Nonzero reserved bytes remain visible.
The exporter never converts an annotated edge into an ordinary fence.

## Worker and artifacts

`TypedEdgeGraphDebugProbe` subclasses the existing `GraphDebugProbe`.
It reuses that worker's retained graph objects and observed replay counts.
It does not select a graph by node count alone.
After the tiny request completes, it exports the same graph identities that actually replayed.

Each DOT receives a sibling `*_edges.json` file.
The file contains every top-level node, edge endpoint, and edge-data byte.
It preserves API enumeration indices and duplicate endpoint records.
Two getter snapshots must produce the same canonical metadata fingerprint.
Enumeration order alone does not change that fingerprint.
Child graphs do not receive a recursive inspection in this bounded implementation.

The driver adds one opt-in flag:

```text
scripts/native_graph_dag.py --size 1.4B --tp 1 --edge-metadata
```

Use the existing GPU guard and a new output directory.
Without this flag, the existing worker and RPC path remain unchanged.

## Nominal cuts and interpretation

The exporter compares DOT handles and endpoint multiplicities with the live API snapshot.
It applies cut analysis only when the repeated kernel pattern passes the existing parser checks.
Each cut lists crossing edge indices and any non-default edge indices.
Graph endpoints receive a null edge classification, not an invented incoming or outgoing edge.
Unsupported DOT structure does not discard the exported typed records.

The old P1 DOT supplies nominal positions `[0, 288, 576, 864, 1154]`.
Which interior positions cross non-default edges remains `UNKNOWN` until the guarded typed export runs.
The v3 rejection alone does not locate those edges.
The old DOT's endpoint labels do not preserve all eight metadata bytes.

A non-default edge can invalidate an ordinary full-completion interpretation of the endpoint chain.
Its existence does not measure overlap duration, hidden latency, or removable time.
These remain fused kernel-group cuts, not exact semantic pass boundaries.
No event insertion or forced full fence receives authorization from this export.

## CPU checks and limits

Nine new CPU tests pass. The combined typed-edge, event, and DOT suites pass all 29 tests.
The synthetic strict-query test rejects error 402. Its typed-buffer counterpart retains all metadata.
All 2,048 byte-position/value cases round-trip without loss.
Tests also cover unknown values, reserved bytes, read-only bindings, fingerprints, and cut classification.
Three Python modules pass syntax compilation.

A CPU C fixture checks the installed CUDA 12.9 `driver_types.h` layout and enum values.
The first compile attempt failed because the runtime wheel omits `crt/host_defines.h`.
The passing compile uses the existing Triton CRT support include after the CUDA 12.9 include directory.
That support comes from the installed CUDA 12.8 compiler headers.
The tested edge declaration still comes from the explicit CUDA 12.9 header.
The CPU fixture links no CUDA runtime and makes no CUDA call.

Actual segment start: 2026-10-19 13:46:34 UTC.
Validated evidence cutoff: 2026-10-19 13:54:36 UTC. Recorded interval: 482 seconds.
Only final metadata publication follows this cutoff.
This worker interval is not additional campaign elapsed time.
