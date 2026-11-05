# Attribution contract: runtime, recurrent rows, and logical layers

## Keep the numerical correction in every arm

The qualified Down projection executes fixed 256-row tiles.
Huginn also uses this correction for its adapter.
The original source is `poc_flashvep/looped_pd_b_numerical_diagnosis/patches.py`.
This campaign does not change that correction.
A full logical prompt does not imply one large Down GEMM.
Other projections do not necessarily use the same tiling.

Record actual packed query geometry and recurrent graph calls.
Calculate the fixed-tile work separately from semantic recurrent rows.
These calculations do not replace GPU kernel timing or FLOP counters.
Internal cuBLAS kernel decomposition remains unknown without a valid trace.

## Matched-row examples: structural accounting only

For one Huginn request, Loop31 moves 2048 recurrent rows.
Token1984 moves 64 × 32 = 2048 recurrent rows.
At a single-request batch, Loop31 requires eight 256-row Down tiles per recurrent layer.
Token1984 requires 32 such tiles per recurrent layer.
This is a fourfold difference in fixed-tile rows for that component.
It is not a fourfold difference in total FLOPs or latency.
Actual multi-request batching changes this ratio and must use recorded query geometries.

For Ouro, Loop3 moves 4096 recurrent rows.
Token3072 moves 1024 × 4 = 4096 recurrent rows.
Both require sixteen fixed Down tiles per recurrent layer at a single-request batch.
They still differ in Attention context, graph replay count, and hidden-state payload.

Token suffix queries also read their completed causal prefixes.
Loop continuation queries span the entire prompt at fewer depths.
Equal recurrent rows therefore do not imply equal Attention work.
Token removes more of that suffix Attention work from P as well.
Do not account for D alone and call the comparison equal total work.

## Huginn boundary work

Loop31 moves the last recurrent pass and required prompt Coda-KV work to D.
The 3.125% figure counts recurrent rows only.
It excludes Prelude, Coda, injection, padding, and scheduling.
P service demand can therefore change by more than 3.125% without skipping work.
Near saturation, a small service-demand change can also cause a larger queue-tail change.
These are mechanisms to test, not measured explanations of the old 10% result.

The new comparison must measure P/D service, queueing, transfer, graph behavior, and admission together.
The event-driven model must predict held-out static runs before it supports a queue-amplification claim.
Serial calibration alone does not establish online queue relief.

## Logical-layer control

Ouro Loop3 is an unrolled recurrent-layer boundary after 72 logical core layers.
Huginn Loop31 is a boundary after two Prelude layers and 124 logical core layers.
Aliased recurrent weights do not require physical duplication in an unrolled representation.
The same boundary can retain the same KV, hidden state, injection state, ownership, and graph calls.
That case is EQUIVALENT_MECHANISM.

A Token-versus-Loop improvement can support a useful partition choice.
It does not establish exclusive Loop-specific novelty against an equivalent logical-layer implementation.
A gain caused by the fixed-row numerical patch must remain a property of this qualified runtime.
It must not become an unconditional claim about Looped Transformer architecture.
