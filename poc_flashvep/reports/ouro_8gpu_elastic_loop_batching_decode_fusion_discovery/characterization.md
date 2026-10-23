# What grouped decode changes

## Submission grouping is not model-work reduction

D1 returns to scheduling after each shared-core pass.
D2 returns after two legal passes. D4 can retain a batch across all four passes.
Each member still needs the same four depth-indexed KV writes and the same shared-core computation.
Native CUDA Graph replay remains per recurrent pass.
Fewer host submissions therefore do not imply fewer model layers or fewer row-level recurrent passes.

Grouping can reduce descriptor construction and host scheduling boundaries.
The native async path can already prepare and submit work while the GPU executes earlier work.
Only an exposed critical-path saving can improve completed-token latency.
Host duration and CUDA service duration cannot simply be added.

## Ordinary cohort versus aligned decode frontier

The ordinary hardware cohort begins with native split prefill.
Early requests can decode while later requests still prefill.
The last-first-output clock removes first-token time from the rate denominator.
It does not make every request occupy the same recurrent depth afterward.
The resulting phase fragmentation can persist into the measured decode interval.

At P2048/N16, the ordinary cohort shows a reproducible grouped-decode throughput advantage.
The aligned-frontier control changes that interpretation.
It holds already-prefilled members until every member completes first-output readback and host commit.
Frontend delivery can follow that commit within the same engine step.
Every member then begins decode from the same recurrence boundary.
D1 becomes faster than D2/D4 in that controlled state.

This result separates a batch-state effect from a pure loop-fusion benefit.
It does not remove the ordinary-cohort measurement.
It also does not prove that a frontier barrier improves complete-serving TTFT.
The barrier deliberately delays early requests before the decode window.
It is a component control, not a proposed serving policy.

## Why KEEP can help one row but hurt another

With newly eligible rows, KEEP preserves the old cohort's smaller matrix shape.
That can shorten the old cohort's output latency while delaying the new cohort.
YIELD can combine rows into fewer larger recurrent executions.
That can improve all-row completion while increasing the old cohort's token latency.
This is a scheduling trade-off, not automatically a loop-specific fusion benefit.

The no-reform D1 ablation preserves the same old-row priority without a grouped submission.
It separates membership preservation from host-call amortization.
The fully warmed component intervention also excludes first-time graph capture.
Only the actual v2 results support its timing comparison.

## Research boundary

Useful ELB needs more than a fixed-group throughput gain from one fragmented cohort.
It needs recurring states where keeping the batch adds value beyond equally informed fixed and event controls.
It also needs opposing states where reform helps, with no fairness or work mismatch.
Low-request completed-token latency remains the primary target.
High-N packing effects remain secondary evidence.
No task-quality, capacity, or novel-method conclusion follows from batch-state accounting alone.
