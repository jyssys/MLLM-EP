# Implementation scope

The primary engine remains the pinned RLT implementation.
This study does not introduce an attention kernel, model engine, TP path, or PP runtime.
The original 24-layer weights, RoPE, normalization, BF16 precision, and KV layout remain unchanged.

## Runtime changes

The preserved grouped-prefill patch adds a configurable prefill loop count.
The new candidate permits G3.
It also defers incompatible recurrent spans from the same batch.
This prevents a single execution descriptor from mixing G3 with its G1 remainder.
The change preserves the native scheduler, allocator, executor, attention, and sampler.
The complete candidate diff and hash are in `qualified_runtime_candidate.json`.

No live Adaptive C-only, G-only, or Joint policy exists yet.
The legal-action module and bounded next-token schedule are offline definition tools.
They do not change serving behavior.

## Measurement and analysis

The new package is `poc_flashvep/ouro_adaptive_chunk_loop/`.
It supplies independent worker dispatch, ownership-safe burner control, source locks, and result checks.
Each movable profile block runs all compared policies within one physical GPU context.
The original factorial plan remains unchanged.
`quality_and_surface_dispatch_v1.json` records the supplemental worker-affinity rule.

The cost observer records host submission, CUDA events, current request phases, and complete-token labels.
Labels can use later completion events during offline fitting.
Online feature extraction excludes those future labels.
Schema 2 also records current host load and pending submission-event count.
Earlier records retain missing-feature flags.
No earlier raw row is rewritten.

## Diagnostic-only changes

Common-prefix replay can force already frozen output prefixes.
This mode is never used for serving or benchmark quality.
The MLP geometry probe uses a fixed captured input and the original weights.
Its FP32 reference is diagnostic only.
It does not replace BF16 inference.

## Quality and performance separation

The quality pilot uses full natural answers with the preserved Base-model prompt recipe.
Its frontend decoding overhead is not a clean serving-latency baseline.
The cost grid uses bounded natural outputs with CUDA-event observation.
Its timings are diagnostic, not headline SLO capacity.
Qualified online serving remains conditional on later gates.
