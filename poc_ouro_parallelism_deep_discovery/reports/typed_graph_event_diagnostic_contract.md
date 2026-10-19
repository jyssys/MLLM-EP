# Typed-edge graph event diagnostic

This is an observer experiment, not an inference method.
Its scope is one authorized P1 GPU and a four-output controlled request.
It uses the original native model operators and Torch replay callback.
It does not establish natural quality or candidate speedup.

## Changed observation boundary

The earlier event diagnostic refused the native graph's nondefault edges.
The read-only typed export then found96 programmatic dependencies.
All96 connect an attention core to its split merge.
The three nominal interior kernel-group cuts cross default edges instead.

The new diagnostic retains every original edge and all eight metadata bytes.
It adds one event at each of five declared cuts.
It refuses a cut that crosses a nondefault edge.
It also refuses unknown edge semantics and nonzero reserved fields.
It checks the original typed fingerprint after restoration.
It does not convert a programmatic dependency to a full fence.

The [CUDA12.9.1 guide](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#cuda-graphs) distinguishes default full dependencies from programmatic dependencies.
The installed Torch documentation requires another instantiation after a raw graph changes.
The diagnostic calls the original graph's `instantiate` method after insertion and restoration.

## Measurement and checks

The runner retains three clean requests before and after the observation.
It also retains matched numerical captures before, during, and after insertion.
Each capture includes full logits, final hidden states, and sampled written KV.
The numerical comparison remains pending until the GPU experiment and independent comparison finish.

External CUDA events enclose complete replay calls in the separate observer lane.
Internal events retain only the last observed replay.
The first instrumented replay includes mutation and reinstantiation.
It cannot estimate normal event overhead.
The later replay envelopes can supply a bounded same-process observer comparison.

The cuts delimit native kernel groups, not exact semantic recurrent passes.
A fused normalization crosses each nominal pass boundary.
Preserved dependency metadata does not measure kernel overlap duration.
These event intervals must not become an additive request breakdown.
They must not replace a critical-path timeline.
Multiplication of the last replay by the request's replay count is forbidden.

The initial CPU suite passed32 tests across the existing and new graph modules.
These tests execute no CUDA.
They check metadata preservation, prohibited cuts, node identity, and the existing graph restoration protocol.
GPU insertion, numerical comparison, and observer validity remain `NOT_RUN` at this contract checkpoint.
