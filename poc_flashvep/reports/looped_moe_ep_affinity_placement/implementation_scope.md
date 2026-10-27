# Implementation scope

## Reused components

The study reuses the previous TRUE EP expert-sharding and NCCL transport bridge.
It retains the official xLLM and Foil model implementations.
Native attention, router expressions, selected-score normalization, residuals, and recurrent execution remain unchanged.
The exact native checkpoints remain on disk without modification.

## New study code

| Area | Modules | Scope |
|---|---|---|
| Native evaluation | `adapter.py`, `quality.py`, `prepare_data.py`, `nll.py` | Official scorers, FA3 compatibility, and pure shifted-token loss |
| TRUE EP measurement | `worker.py`, `measure.py`, `kernel_profile.py` | Actual EP4/EP8 timing and separate observers |
| Route discovery | `collect_routes.py`, `affinity.py`, `prediction_controls.py`, `surrogate_controls.py` | Request-held-out correlation and null controls |
| Fixed placement | `placement.py`, `oracle_summary.py` | Same-memory trace models, not a new executed placement |
| Generic controls | `ep.py`, `generic_controls.py`, `aggregate_generic.py` | Exact current-router source and metadata ablations |
| E2E percentages | `attribution_markers.py`, `attribution_partition.py`, `e2e_attribution.py` | Exclusive complete-rollout event spans |
| Evidence and release | `reporting.py`, `evidence_bundle.py`, `finalize.py`, `publish.py` | Evidence tables, completeness gates, and scoped Git release |

## No new loop-aware engine

No proposed loop-aware placement passes the 1.15× actionable oracle gate in advance.
The study does not implement expert skipping, output reuse, recurrent speculation, or loop partition pipelines.
It does not change precision or train a model, gate, or predictor.
The only transport ablations are generic controls under replicated dense-state ownership.

## Added generic metadata control

Every rank already holds current router outputs in this bridge.
The count-collective control derives the full count matrix from those existing outputs.
Its correctness observer still executes a real All-Gather for comparison.
It checks every recurrent invocation along a complete generated trajectory before clean timing.
The clean trajectory must reproduce those output tokens.
The timed path removes that redundant All-Gather.
Dispatch, owner-local expert GEMMs, combine, and restoration remain real distributed operations.
The source-owner combination uses no future router information.
It is not free token migration for a KV- or attention-sharded serving engine.

## Validation levels

CPU tests establish interval arithmetic, formula identity, and Git-scope safety only.
They do not establish GPU timing, true expert dispatch, or model accuracy.
Actual model tests have separate saved evidence and completion markers.
Publication gates reject missing mandatory GPU results.
Unimplemented wall-time oracles remain explicitly unavailable.
