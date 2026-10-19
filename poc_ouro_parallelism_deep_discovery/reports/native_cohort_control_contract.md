# Native fixed-depth cohort control

Status: protocol frozen before result inspection.
This control measures an existing generic baseline.
It does not implement a new architecture.
It does not repeat the separate adaptive CDB factorial experiment.

## Purpose

Candidate ready-work batching must beat competent native batching at the same GPU count.
The existing B1 teacher hook cannot safely force several asynchronously admitted requests with one sampler-call counter.
This control therefore uses native per-request `allowed_token_ids`.
Each request permits exactly one frozen non-EOS token.
The output cap is 128 tokens with EOS disabled.
This creates matched hardware work, not task accuracy.

## Frozen scope

The [active protocol](../artifacts/native_cohort_protocol_v2.json) fixes both Base checkpoints and fixed R4.
It fixes C=1/2/4/8 and context lengths 1024/4096.
Each TP1/2/4 job retains BF16, native asynchronous scheduling, and CUDA Graphs.
Chunked prefill, prefix caching, and expert parallelism remain disabled.
Each rank allocates 52 GiB for KV.
All compiler-cache roots remain job-local.

The preserved v1 protocol proposed 32 GiB.
The pre-execution capacity check needs 49.5 GiB for Base2.6B TP1 at C8, context 4096, and 128 output tokens.
No v1 job ran.
The active v2 protocol therefore uses 52 GiB uniformly.
This prevents the known KV-capacity deficit from defining the intended low-concurrency compute comparison.
Actual admission and allocated memory still require observation.

C means the submitted closed-cohort size.
It does not assert simultaneous scheduler admission or a constant GPU batch size.
The native offline frontend submits requests separately.
A separate observation lane records actual scheduled membership and sampler tensor sizes.
That lane contributes no headline timing.

## Metrics and limitations

Clean wall time covers one complete native `generate` call.
It includes frontend submission and completion of every request in the cohort.
Output TPS is total controlled output tokens divided by that wall time.
This is finite-cohort throughput, not steady-state serving throughput.
Per-request native metrics remain available separately.

Each cell uses three short warm-up runs and five complete repetitions.
The runner randomizes configuration order within each repetition.
It retains both repeated calibration prompt shapes.
They do not become eight independent prompt populations at C8.
Do not use these results as a broad request-level confidence claim.

The new constant-token output differs from the earlier changing-teacher workload.
Do not interpret their latency difference as a system speedup.
Natural quality, candidate performance, and adaptive behavior remain `NOT_RUN` in this control.
