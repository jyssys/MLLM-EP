# Failures and limits

## Initial calibration engine lifecycle

The v1 harness closed an engine but retained the caller's Python reference.
Its 54 GiB KV pool remained live when the next engine allocated another pool.
The quiet-device job and first four parallel calibration jobs failed with CUDA OutOfMemoryError.
The model, attention kernels, and grouped scheduler did not change.
The failed job identities, partial measurements, and tracebacks remain in their original directories.

The corrected harness deletes the caller reference, runs garbage collection, and releases unused allocator blocks before creating the next engine.
The dispatcher now returns a nonzero status if a job fails.
That status stops a shell stage instead of silently continuing a failed calibration.
Corrected v2 job IDs preserve the initial failures rather than overwriting them.
No failed lane qualifies an SLO threshold or serving capacity claim.

## Historical limits remain

The prior 112/120 batched sequence match is not universal parity.
The prior all-capped 128-token lane is not complete-answer quality evidence.
The old Atomic-first and cold-JIT effects remain historical limitations.
This study must use counterbalanced order and separate complete-EOS results.

## No-chunk pilot cleanup

The first no-chunk pilot completed its eight natural responses.
Its temporary one-prompt selector retained a bound scheduler reference after engine close.
The next staged-G4 engine then failed to allocate its KV pool.
The partial no-chunk measurement and failure remain preserved under `traffic_pilot_d_v1`.
The correction restores the native selector and releases both temporary references before engine destruction.
This changes cleanup between cases, not timed model operations.

## Queue transition after Poisson

All sixteen registered Poisson job blocks completed.
The final Mixed C128 cell at rate 1 req/s retained two request timeouts.
Those requests stay in the offered denominator.
Both requests reached the declared 120-second drain limit before completion.
Their queue delays exceeded 100 seconds.
The raw timeout records and incomplete work remain available.
The analysis must not describe this lane as error-free.

The stage wrapper then returned a shell syntax error before starting the burst queue.
The current wrapper passed `bash -n` when execution resumed.
The wrapper file changed during the earlier long-running invocation.
Shell input rereading is a possible cause, not a verified model or scheduler failure.
The resumed controller starts the untouched burst and profiling manifests.
Completed Poisson cells do not repeat.
The resource exit handler restored the recognized burners during the interruption.
