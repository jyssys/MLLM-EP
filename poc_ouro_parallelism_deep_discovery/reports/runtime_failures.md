# Runtime failures and bounded diagnosis

All failed attempts remain in their versioned output directories.
No failed timing contributes to an architecture result.

## Capture wrapper

The first 1.4B smoke attempted to assign a field of CDB's immutable exit policy.
It failed with `FrozenInstanceError` before generation.
The repair creates a replacement policy with the same original gate and updates both engine and runner references.
The second attempt passes the tiny fixed/adaptive checks.
This repair changes wrapper configuration, not checkpoint weights or CDB source.

## Operator launcher

The first P2 operator attempt contained an unmatched closing parenthesis.
Python rejected it before the GPU test.
The repaired script passes compilation and the real P2 operator test.
Future probes must pass compilation before resource reservation.

## Native TP graph registration

The first Base1.4B R4 TP2 baseline fails after graph capture.
The exact error is `custom_all_reduce.cuh:455 invalid argument`.
The run produces no valid baseline timing.
The resource guard restores the verified owned burners afterward.

The ranked hypotheses are:

1. Expandable allocator incompatibility. The scoped guard injects `expandable_segments:True`, unlike the older native baseline guard. If causal, disabling it should repair graph registration.
2. Noncontiguous physical device mapping. If causal, the result should change between physical 4/6 and 4/5 with other settings fixed.
3. Interpreter or library-path mismatch. If causal, the original native interpreter should pass when the isolated launcher fails under matched allocation and mapping.

`scripts/custom_ar_repro.py` constructs the actual custom communicator and registers a graph-allocated input.
It asserts the expected reduced tensor, not merely a successful process exit.
It avoids model loading and preserves all model-level failure logs.
Each diagnosis changes one setting at a time.
No driver reset, global package upgrade, or foreign-process termination is permitted.

The reduced test reproduces the failure with expandable allocation on physical GPUs 4 and 6.
The same test passes after only the allocator setting changes to `expandable_segments:False`.
It checks the reduced tensor after CUDA Graph replay.
The output equals the expected sum.

The native baseline launcher now disables expandable allocation before Torch imports.
This setting applies only to the campaign's native baseline process.
It does not change the shared CDB runtime or another environment.
The second full model test passes graph registration with the scoped allocator setting.
It then fails during sampler warm-up because the isolated process cannot find the existing Ninja executable.
The next repair adds the native environment's binary directory to that process's PATH.
It installs no package and changes no shared environment.
The third attempt then executes its first request but rejects serialization of absent optional metrics.
The wrapper now permits absent metrics and explicitly enables the runtime's standard request statistics.
The fourth model attempt completes all frozen hardware shapes and parity capture.
It preserves custom All-Reduce, CUDA Graphs, BF16, and the pinned checkpoint.
The earlier failed attempts remain excluded from timing analysis.
Custom All-Reduce and CUDA Graphs remain enabled.
No weaker baseline replaces the failed optimized baseline.
