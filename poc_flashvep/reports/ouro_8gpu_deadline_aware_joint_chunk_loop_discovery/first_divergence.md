# First-divergence investigation

Status: one historical case attributed by a same-input GPU operator replay.
Other cases and benchmark quality remain unqualified.

The failing command is:

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_adaptive_chunk_loop.legacy_feedback --assert-exact
```

The command reads the original GPU outputs.
It fails with `BATCH_TOKEN_PARITY_FAILURE_REPRODUCED_FROM_FROZEN_GPU_TRACE`.
It is a trace check, not a new free-running GPU reproduction.
The audit covers 96 paired records and finds 86 exact sequences.
It includes twelve cohorts, not the prior report's ten-cohort 72/80 table.
Eight distinct request IDs differ.
The earliest listed mismatch is `bg_11_3`, output index 8.
All compared depth traces agree.

The new diagnostic plan freezes Atomic, Grouped, and Split comparisons for this request.
It captures recurrent inputs/outputs, selected operator outputs, and K/V rows.
Graph ON/OFF and multi-stream/single-stream form four diagnostic lanes.
Graph replay does not execute Python operator hooks.
Therefore operator-level traces belong to the eager control lane.
Cloning and CPU export can change scheduling timing.
These runs cannot supply clean serving latency.

## Required interpretation

A BF16 logit tie does not identify the first cause.
Different hidden states can originate before the final LM head.
Batch membership, historical KV, graph inputs, and buffer publication must remain visible.
A strict-synchronization diagnostic is not a production fix.
No unconditional synchronization may silently enter the measured baseline.

## New GPU localization

The `bg_11_3` case again diverges at output index 8.
The captured difference starts before its first output token.
An initial last-row probe first detected R3, layer 1 attention.
That was not the first causal operation.
The observer initially lost earlier chunk rows when it reused the same trace key.
The fixed diagnostic preserves each chunk with absolute position keys.
This change affects observation only, not model execution.

Across all 288 prompt positions, the first captured difference is R3, layer 0 MLP.
The zero-based depth index is 2.
Its input normalization, Q/K/V projections, attention result, and MLP input match exactly.
The MLP output differs at 47 positions.
The first affected position is 61.
The maximum absolute difference across those positions is 0.000244140625.
Relative L2 is 0.000040208939026342705.
Graph ON and OFF produce the same first differing operation.
The earlier stream controls also retain the same captured divergence.

Atomic executes position 61 in a 1,024-row batch.
Grouped executes it in a 48-row batch.
The same captured BF16 MLP input and the same pretrained weights reproduce both results exactly.
The 1,024-row replay matches Atomic.
The 48-row replay matches Grouped.
Their position-61 output difference has maximum absolute error 0.0000152587890625.
Its relative L2 is 0.00001785154927347321.
The replay contains no attention, KV access, staged-state transfer, or scheduler.

This supports `EXPECTED_BF16_BATCH_ORDER_SENSITIVITY` for this specific first divergence.
It does not prove the cause of every historical mismatch.
It does not prove benchmark accuracy equivalence.
No FP32 operation replaces BF16 serving arithmetic.
The FP32 calculation is a diagnostic reference only.

Evidence: `attention_boundary_p1_attention_boundary_graph0_v1_atomic_vs_grouped.json` and the matching graph-ON file.
The operator replay is `raw/p1_MLP_same_input_geometry_v1/mlp_geometry_replay.json`.
The local tensor files remain under the corresponding raw run directories.
Tensor-index files preserve per-tensor hashes.
