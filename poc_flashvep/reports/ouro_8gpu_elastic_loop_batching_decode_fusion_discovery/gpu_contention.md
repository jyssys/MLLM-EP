# Devices, ownership, and parallel interference

Eight independent single-GPU workers use physical GPUs0–7.
Each process sees only its assigned device as logical CUDA0.
Every worker checks the actual CUDA UUID against the frozen physical-device map.
No distributed process group forms.
This campaign does not measure an eight-GPU model-parallel forward.

The resource manager checks UID, session ownership, approved burner logs, and active CUDA descendants.
It signals only verified owned burner sessions.
It refuses an unrelated active job instead of terminating that job.
The initial device snapshot contains only the known owned burners.
Resource transitions remain in `resource_ledger.jsonl`.

## Calibration

The new calibration uses32 process starts and192 measured cohorts.
Each GPU runs the same N4/P256 and N8/P2048 cells at1/2/4/8 worker counts.
Every count/device cell has one independent process start and three within-process repeats.
A warmed barrier aligns the first measured cohort group.
The second context retains the same worker group but does not receive a second global barrier.
That limits the precision of its simultaneous-load interpretation.

Relative to the same GPU's single-worker reference, mean throughput drift ranges from-0.6278% to+0.3457%.
These small observed drifts do not prove universal interference neutrality.
They do not cover every8K or N16 memory-pressure condition.
No token-level bootstrap substitutes for another calibration start.

## Operating lanes

The main screen crosses D1/D2/D4 on every GPU.
Policy order rotates across physical devices.
The queue assigns the next registered job to each available worker.
Completed immutable hashes prevent duplicate execution on resume.
Failed attempts remain unchanged and require reviewed new IDs.

The decisive quiet lane reserves all eight authorized GPUs and runs one measurement worker at a time.
It uses three independent starts on each of GPUs0 and4.
Unused devices omit burners only during declared quiet or interference-isolation controls.
This avoids burner-related node-power and host-load changes during confirmation.
Ordinary unused authorized devices retain owned burners.

Each worker uses16 assigned CPU cores, one OMP thread, one MKL thread, and separate per-GPU kernel-cache paths.
The analysis process uses cores128–143.
Each process retains its own CUDA context, model load, warm-up, log, source snapshot, and run ID.
Existing warm per-GPU compiler caches remain enabled.
The experiment does not claim cold-JIT serving latency.

`GPU_policy_rotation.json` records actual device and policy assignment.
`contention_drift.csv` preserves every calibrated ratio.
`evidence_validation.json` checks recorded cross-GPU timing-window overlap in quiet runs.
That overlap check does not claim zero operating-system or startup background activity.

The final handoff restores owned burners on all eight devices.
