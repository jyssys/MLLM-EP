# First-divergence investigation

Status: historical failure reproduced. New GPU diagnosis is pending Phase 0 controls.

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

The current root-cause category is `UNKNOWN`.
No evidence yet excludes a lifecycle or stream bug.
No new quality-preserving claim is permitted.
