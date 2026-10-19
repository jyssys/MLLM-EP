# Inter-pass capacity screen: independent review

Decision: accept the conditional bound and its stated limitations.
No counterexample appears inside the precise bypass-cache model.
No formula correction is necessary.
This review does not reject all distributed ownership optima.

## Mathematical check

Let `R=min(C,W)`.
At each pass start, the cache contains at most R required bytes.
Every other distinct required byte must enter the cache hierarchy during that pass.
The model counts that entry even if a prefetch performs it.
Thus each warm pass needs at least `W-R` counted bytes.
A cold first pass needs W bytes.

Both lower bounds are attainable in this model.
Permanently retain any fixed R-byte subset and bypass every other access.
This construction gives `D*W-(D-1)*R` cold bytes and `D*(W-R)` warm bytes.
The reduction fractions use the stipulated no-cache denominator `D*W`.
They do not use a measured native-run denominator.

All five supplied CPU tests pass.
The exact search covers 168 cold/warm toy cases, including the warm-state choice.
Six additional selected edge cases also match the formulas.
An independent calculation reproduces all 24 artifact records.

A simple boundary check confirms why the two initial states must remain separate.
For `W=1,C=1,D=2`, the cold minimum is one fetch and the warm minimum is zero.
Applying the cold minimum to every cached decode invocation would fail.
The reviewed report explicitly avoids that extension.

## Dimensions and hardware provenance

I read the two pinned configs and safetensors headers without loading model tensors.
Every relevant projection has BF16 storage and the expected shape.
Each layer has four `2048×2048` attention matrices.
Its MLP has two `5632×2048` matrices and one `2048×5632` matrix.
This gives 102,760,448 bytes, or 98 MiB, per layer.

The 1.4B pin contains 168 such tensors across 24 layers, totaling 2,466,250,752 bytes.
The 2.6B pin contains 336 across 48 layers, totaling 4,932,501,504 bytes.
The config hashes match the screen artifact.
The matrix dimensions divide evenly for TP1/2/4/8.
This does not establish a native TP8 implementation or measurement.

The L2 evidence traces directly to [components.json](../outputs/components_base14_P2_graph_v1/components.json), recorded at 2026-10-19 10:34:52.088103 UTC.
It records 52,428,800 bytes for physical GPUs4 and6.
The [probe source](../scripts/component_probe.py) obtains `L2_cache_size` from device properties and records UUIDs.
The property reports hardware capacity, not a measured retention rate.
The screen explicitly assumes the same 50 MiB capacity on other ranks.
It does not claim direct capacity observations for GPUs5/7 or a 50 MiB persisting set-aside.

## Scope check

The report does not extend its result to actual HBM bytes, copies-arm timing, or E2E latency.
Those exclusions are necessary.
Native kernels can have tiled rereads, multiple cache levels, concurrent traffic, and different replacement policies.
Actual copies runs can also retain weights across decode invocations.
Neither logical reduction fraction identifies the incremental gain over those runs.

A byte-count fraction does not determine exposed stall time or request latency.
The screen also excludes register/shared-memory residency, compression, reordering, and extra ready rows.
It cannot reject mechanisms that change those assumptions or another ownership cost.
Its selective-residency and distributed-optimum qualifications remain appropriate.

Inputs: [report](interpass_cache_capacity_screen.md), [checker](../scripts/interpass_cache_screen.py), and [artifact](../artifacts/interpass_cache_capacity_screen_v1.json).
No source search, GPU call, raw-output edit, or input-file edit occurs in this review.
Worker start: 2026-10-19 14:15:53 UTC.
Arithmetic and header-check cutoff: 2026-10-19 14:18:04 UTC.
Final verification cutoff: 2026-10-19 14:19:04 UTC.
The worker segment spans 191 seconds through that cutoff.
This worker interval is not additional campaign elapsed time.
