# Adaptive ELB decision

Implementation status: NOT_RUN.
The P2 promotion gate does not establish a useful incremental region beyond optimized native controls.
This study therefore does not implement a larger selector or an adaptive Mixed scheduler.

## Considered causal mechanism

The simplest candidate would keep a pure decode batch when no useful alternate work is currently ready.
It would yield when ready rows can improve packing or when the common legal group falls to one.
Only current queues, event readiness, depth, and measured historical costs could enter that decision.
Future arrivals and future actual GPU duration would remain unavailable.

The legal group would still clip to every member's remaining R4 depth.
Mixed rows would retain native G1 semantics.
A row reaching R4 would finish CODA before it executes the next token.
The same complete-token fairness rule would apply to every comparator.

These are design constraints, not a deployed or timed policy.
No decision-change fraction, adaptive speedup, or quality result exists for this unimplemented selector.

## Why implementation stops

The measured low-N async/graph path loses throughput when submissions group.
The synchronous benefit is smaller than the benefit of existing native asynchronous execution.
The ordinary N16 Refill benefit disappears under aligned-frontier and native Mixed controls.
Native Mixed already supplies the relevant prompt/decode co-packing.
Changing the selector without a remaining physical advantage would not establish an incremental method.

## Conditions to reopen

1. Find a recurring legal state where grouped execution beats optimized D1 or native Mixed under the same resources.
2. Confirm the completed-token improvement across independent starts and source families.
3. Find an opposing state where reform helps under the same fairness rule.
4. Show a causal selector that chooses different executed plans from tuned fixed D and generic event-only D1.
5. Preserve numerical, KV, lifecycle, natural-stop, and full-answer correctness.
6. Measure real natural-serving and SLO capacity before a method claim.

A combined multi-loop CUDA Graph is a separate untested backend control.
It is not an implemented ELB policy and does not require resuming the paused Loop PP project.
