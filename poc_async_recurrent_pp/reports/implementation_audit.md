# Emulator validation history

The first diagnostic version retained effective stage inputs for the next prediction.
This bookkeeping would repeatedly recycle stale input predictions instead of retaining newly computed preceding-stage outputs.
It would discard useful completed work after the first approximate round.
The specification's `x[t,r,j]` denotes a computed stage output. The corrected emulator follows that definition.

The audit detected this defect before candidate task-quality evaluation.
The first diagnostic attempt remains in `outputs/diagnostics_v1/` with its source snapshot and logs.
Its approximation metrics are invalid for the intended PARPP candidate. They must not support a quality verdict.
The exact task-sanity job remained valid because it used no approximation hook.
Only the CPU controller stopped during this correction. The active exact task job continued.

The corrected version retains each stage's computed output after its six physical layers complete.
At the next round boundary, the predictor reads these previous-round outputs.
It still freezes all current-round inputs before any current-round stage executes.
A regression test distinguishes newly computed outputs from recycled predictions.
The second diagnostic version repeats reference parity, cache audits, and intervention measurements.

This correction changes emulator bookkeeping. It does not change model weights, precision, attention, routing, or acceptance semantics.
No prediction coefficient uses evaluation labels or future evaluation activations.
