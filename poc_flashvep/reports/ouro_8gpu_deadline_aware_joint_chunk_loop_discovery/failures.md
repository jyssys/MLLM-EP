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
Qualified serving capacity and new benchmark accuracy are NOT_RUN.
No new TP/PP implementation is authorized in this campaign.
