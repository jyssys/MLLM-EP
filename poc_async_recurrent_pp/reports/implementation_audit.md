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

## Batched generation driver

The first custom generation driver compacted the batch after EOS.
Long reference validation found output differences in three of four smoke requests.
The first request ended after 364 tokens. Other requests first diverged at token indices 377, 415, and 433.
The compacting driver therefore does not supply primary quality evidence.
Its pilot results and failed parity artifacts remain available without a quality claim.

A smaller EOS-boundary probe compared the same cached forward with batch sizes four and three.
All 96 selected key/value tensors matched before the forward.
Inputs, masks, and position IDs also matched for the remaining rows.
That first forward produced bitwise-equal logits and observed first-layer components.
This probe did not isolate the later divergence's internal cause.
The analysis does not claim that immediate cache corruption or one specific kernel caused the failure.

The primary evaluator now calls the official `model.generate()` for every arm.
Only the stage-boundary hooks change candidate hidden inputs.
The exact wrapper uses the same native fixed-batch generation and EOS handling as the official reference.
An observer never changes a stopping decision. It records completed outputs and returns an all-false stop condition.
The new long parity gate must pass before any new quality wave starts.

Native fixed-batch generation can update inactive padding KV after a request ends.
These entries are not logical generated positions. They do not alter any active request or enter another request's history.
The report distinguishes this official padding behavior from erroneous live-token writes.
No generation token appears after a request's first EOS.

The corrected quality runs preserve the original questions, predictor coefficients, cap rule, and greedy settings.
They use new output directories. Earlier files remain unchanged.
No accuracy comparison from the rejected driver contributes to the final quality gate.

## Decimal parser regression

A hand-written regression case exposed a decimal parsing defect in the inherited explicit-answer fallback.
The old integer-task pattern cut `$1,234.50$` at its decimal point.
The regression test failed before the parser correction.
The corrected fallback preserves the complete final line or a complete delimited expression.
It still gives balanced boxed answers priority.

This correction uses a generic hand-written formatting case, not a benchmark answer.
The final analysis rescans every valid native generation with the same parser version.
It retains each generation's original encoded grade and records any grade change separately.
No prompt, predictor, model setting, sample selection, or output token changes.
