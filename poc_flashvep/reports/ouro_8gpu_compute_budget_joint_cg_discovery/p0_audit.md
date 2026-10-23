# Phase 0 evidence and scope

The authoritative specification is `ouro_8gpu_compute_budget_joint_cg_optimization_poc_spec.md`.
The complete specification has 466 lines.
The campaign uses a separate branch, package, report directory, and external runtime worktree.
It preserves the previous deadline study and its BLOCKED verdict.
That verdict concerned online deadline control, not the Joint action space.

## Historical evidence

The previous campaign collected 1,728 cohorts and 530,826 quantum records.
Its global quantum model hid poor prefill coverage behind many easy decode samples.
Atomic prefill coverage was 79.141%.
Staged prefill coverage was 90.901%.
The old next-token target also included future stock-policy interruptions.
The new controller uses neither old model as its control signal.

Historical fixed-policy quality and numerical findings remain historical evidence.
They do not establish correctness or accuracy for the new adaptive policy.
The previous aggregate timing collection also included queue waits before its measurement boundary.
New service events start after native request-dependency waits.

## Preservation and permissions

The original staged binary diff has SHA256 `7018be31bd8a517574711055a72afbd6ae90737e4a585578c98ec255dbf7e1d9`.
Repeated checks found the same hash during this campaign.
The new branch does not reset, clean, or stash unrelated changes.
The user authorizes normal commits and the final branch push.
The stopped native-vLLM Loop PP project remains outside this study.

All eight physical devices are independent single-GPU workers.
Each worker checks its actual GPU UUID against the frozen device map.
Each worker exposes exactly one CUDA device and does not initialize distributed model execution.
Resource control identifies owned burner sessions before sending signals.
It does not signal unrelated GPU processes.

## Measurement registration

The contention queue has 64 device/start jobs.
It covers one, two, four, and eight simultaneous workers on every physical GPU.
Each device/concurrency condition has two independent model-process starts.
Every start measures three warmed cohorts at each of two legacy workload shapes.
These inputs provide continuity, not fresh blind evidence.

Worker g uses CPU IDs `16*g` through `16*g+15`.
This disjoint affinity is the same for solo and parallel conditions.
The first workload shape has a warmed group-start barrier.
The second workload shape has no separate barrier.
The analysis retains that limitation.

The 3% drift threshold triggers a review.
It does not prove neutral interference below that threshold.
Diagnostic concurrent collection can continue after review.
Clean serving requires a separately reserved low-concurrency lane when drift remains nonneutral.
Otherwise unused GPUs remain reserved and quiet during isolation controls.
Owned burners run during other unallocated periods.

The fresh input manifest has 1,099 prompt variants.
Every source family belongs to only one registered partition.
No family overlaps the previous study's manifest.
TRAIN, VALIDATION, CALIBRATION, NEW_HELDOUT, and FINAL_ONLINE remain distinct.
The final online partition is not used to tune cost margins or policy choices.
