# Staged submission cost: prospective diagnostic

## Observed signal

Huginn medium-load Native PD has a median P device enclosure of `590.493155 ms`.
Loop31 has a median enclosure of `1268.803140 ms` in the matched N128 trace.
The latter moves only one of 32 recurrent passes to D.
This difference needs a runtime control before any claim about the split axis.
Device enclosures include submission gaps and other interleaved work.
They are not exclusive kernel times.

## Ranked hypotheses

1. Staged scheduling creates one host submission per recurrent pass.
   A native multi-loop submission should reduce that cost without changing active recurrent work.
2. Snapshot construction and transport add a separate cost.
   Staged Native PD versus full Snapshot PD should isolate that effect.
3. Batch composition changes the per-request enclosure.
   Isolated calibrated service and packed row observations should separate that effect from request queueing.

These are testable hypotheses, not established causes.
The predeclared medium-load transport controls now receive earlier execution priority.
Their configurations and arrival traces do not change.

## Optional native submission candidate

The candidate uses the native multi-loop execution interface.
It applies to complete legal prompt regions only.
It stops at the request's export cut.
It requires equal starting frontiers across all selected remaining depths.
Partial chunks retain the existing one-loop wavefront path.

Prelude remains a separate submission.
This preserves the staged hidden-state storage and final-depth Coda-KV path.
The candidate does not use Atomic Prelude to skip the required full prompt-state buffer.
It does not combine Prefill and Decode rows.
It does not change weights, precision, Attention, KV layout, or recurrent depth.

The explicit configuration flag is `contiguous_prompt_loop_burst=True`.
Existing configurations omit this flag and retain their original execution.
New candidate runs use the `m_` namespace.
An N8 exact-state pilot runs for both models before any performance expansion.
Separate N32 qualification remains mandatory.

This candidate is a generic scheduling optimization for both split axes.
It is not Loop-specific novelty.
Any later comparison must apply the same candidate and tuning policy to both axes and their transport controls.
Results from different execution epochs must not become a same-runtime speedup claim.
