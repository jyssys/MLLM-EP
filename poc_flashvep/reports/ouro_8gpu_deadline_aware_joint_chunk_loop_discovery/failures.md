# Failures and limitations

## Frozen batched-parity failure

The trace assertion reproduces saved output mismatches.
It does not establish a cause.
The original outputs and historical verdicts remain unchanged.

## Source-inventory probe

The first inventory command assumed `/sys/fs/cgroup/cpu.max` existed.
This host does not expose that path.
The probe failed before it wrote a source lock.
The corrected probe records available v1/v2 quota files.
An absent quota file does not prove unlimited CPU capacity.
The GPU calibration continued independently.

## Calibration coverage

The registered contention plan has two independent starts per device and parallelism level.
Each start has three warmed cohorts at two contexts.
Only the first context has an explicit group-start barrier.
The second context can have partial start skew across workers.
Report actual measurement intervals before interpreting simultaneous-worker effects.
This limitation must not silently become a claim of perfectly aligned exposure.

## Diagnostic observer limits

Tensor clones can change scheduling timing.
Graph replay bypasses Python operator hooks.
Heavy diagnostics do not supply clean serving latency.
Isolated numerical agreement cannot certify full-answer quality.

## Not yet executed

Live Adaptive C-only, G-only, and Joint are NOT_RUN.
Qualified serving capacity and adaptive benchmark accuracy are NOT_RUN.
The 114-case fixed-policy natural quality pilot is complete.
No new TP/PP implementation is authorized in this campaign.

## G3 boundary incompatibility

The validator-only G3 source combined G3 rows with G1 remainder rows.
The actual executor rejected that descriptor.
The candidate guard now defers incompatible rows.
The red/green regression and complete diff remain preserved.

## Controlled-probe observation key

An early observer retained only the final prompt chunk under a repeated key.
That capture could not establish the first affected prompt position.
New diagnostic IDs record absolute chunk boundaries and all 288 positions.
The early capture remains unchanged.

## Python invocation errors

One MLP registration used the base Python without Torch and failed before any GPU lease.
The corrected command used the pinned environment.
One local unittest command lacked the RLT PYTHONPATH and failed during import.
The corrected path-qualified suite passed.
Neither failure is reported as a model execution result.

## Heavy-profile harness lifetime

The first GPU0/GPU1 observer controls retained a bound runner method after profiling.
That reference retained the prior engine's 54-GiB KV pool.
The next engine allocation failed with CUDA OOM.
This was a diagnostic harness lifetime error, not evidence that one native engine exceeds H100 capacity.
The corrected harness releases profiler and bound-method references before creating the next engine.
New v2 attempt IDs completed all six policies on both affected devices.
Failed v1 logs and partial results remain intact and are excluded from matched summary tables.

## Predictor and headroom limits

The selected global margin fails prefill-specific held-out coverage.
The many decode quanta hide this failure in the aggregate.
No threshold or held-out margin was relaxed.
Full-token stock-policy predictions include later interruptions.
They cannot directly support a different action's deadline forecast.
Phase 4 therefore remains a partial offline characterization.
No live adaptive or sustainable-capacity gain is fabricated.

## Observer source coverage

Raw observations retain observer source hashes and schema versions.
Observer fields evolved during early fixed-policy collection.
The final source and initial committed source are available.
Not every intermediate observer source body has a separately committed snapshot.
This limits exact reconstruction of some early observer variants.
Within each process-start block, all six policy cells use the same loaded observer.
No old raw row is rewritten to conceal that limit.

## Instrumentation parity

Only 832/864 request sequences match in clean/event controls.
Lengths match for 855/864 requests.
The latency difference can therefore include changed natural work and packing geometry.
The report does not label that difference as pure observer overhead.
One BF16 geometry reproduction does not explain every control mismatch.
