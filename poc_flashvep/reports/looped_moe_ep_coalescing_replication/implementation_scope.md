# Implementation scope

## Native execution bridge

`poc_flashvep/looped_moe_ep/runtime.py` contains the TRUE EP bridge.
`PhysicalEP.__init__` retains exact owner-local expert parameters.
`PhysicalEP._run` performs count exchange, dispatch, native GEMMs, combine, and source restoration.
`patch_class` replaces the routed expert expression in memory.
It does not edit the external native source files.

The IFM path keeps its original expert forward function.
The Foil path uses its original three grouped GEMMs and activation.
The source rows preserve original top-2 multiplicity and scores.
Unit checks include duplicate expert IDs and token restoration.
GPU fidelity checks use the actual checkpoints, not fixtures.

The official native adapter comes from `poc_flashvep/looped_moe_headroom/adapters.py`.
The study reuses that adapter without changes.
It also reuses the existing verified checkpoint pins and fixed public sample manifest.

## Measurement and controls

| File | Responsibility |
| --- | --- |
| `worker.py` | Official loading, native references, sharding, fidelity, and completion checks |
| `measure.py` | Clean complete-forward matrix, hardware-capped generation, and separate heavy traces |
| `ready_probe.py` | Independent native request frontiers and all-rank ready snapshots |
| `operator_controls.py` | Actual expert-shape scaling, independent overlap, and exact replica allocation |
| `steady.py` | Uninstrumented repeated generation and first-shape timing diagnosis |
| `launch.py` | One process per physical GPU, bounded attempts, logs, and owned-burner restoration |
| `campaign.py` | Dependency-ordered EP4/EP8 coverage and completed-run resume |
| `oracles.py` | Bounded causal policy replay and memory-constrained placement estimates |
| `analyze.py`, `diagnostics.py`, `characterize.py` | Evidence extraction and qualified statistics |
| `finalize.py` | Final tables, figures, and reports from completed campaigns |
| `archive.py` | Lossless evidence archives and verified extraction |

## Intentionally absent components

No expert computation is skipped.
No prior expert output substitutes for a new computation.
No future hidden state or authoritative route is fabricated.
No candidate replica-aware dispatcher or coalescing optimizer enters the native model path.
The ready observer is characterization infrastructure only.
The operator overlap control is not an asynchronous serving implementation.

There is no new TP, PP, loop pipeline, learned policy, or quantization path.
The current native model paths use full-prefix recomputation.
They do not provide cached decode, a paged-KV scheduler, or a client-facing server.
Hardware-generation latency excludes a network protocol and text response serialization.
No claim compares this research executor with optimized production DeepEP serving.
