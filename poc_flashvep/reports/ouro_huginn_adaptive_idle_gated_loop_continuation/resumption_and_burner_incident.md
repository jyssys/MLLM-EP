# Resumption and burner incident

## Observed state

The resumption check started at 2026-11-06 18:58:13 KST.
No GPU experiment controller or serving worker remained active.
The quantum-calibration controller completed its configured list before the check.
The watchdog and deadline guard remained active. Neither process selects new research experiments.
Therefore, their presence did not establish continued research progress.
The fixed campaign deadline remains 21:05:11 KST. This resumption does not reset it.

Seven approved burners remained active on GPUs 1 through 7.
GPU 0 had no GPU process and reported zero allocated memory.
The shared burner log recorded `EADDRINUSE` on TCP port `43095` at 15:46:24 KST.
That error stopped one engine during the eight-GPU restoration attempt.
The exact reason for the port collision remains unconfirmed.

## Monitoring defect

The idle restoration branch required an empty process list across all eight GPUs.
Seven working burners prevented this branch from retrying the missing burner.
The monitor now checks the ownership of every active GPU process.
It restores only missing burners when all active GPU processes belong to approved burners.
An unrelated GPU process prevents automatic restoration.
The exclusive quiet-lane lease prevents restoration during an experiment.

Three CPU tests passed for partial restoration, full occupancy, and unrelated-process protection.
The tests do not establish successful GPU engine initialization.
Closeout must check actual GPU processes separately from worker-start acknowledgments.

## Research resumption

The forced mixed-cut qualification controller restarted in its existing namespace.
Completed immutable receipts remain unchanged.
The unfinished N32 tests continue before Native load and A/A tail controls.
All approved burners stop before isolated GPU measurements.
No idle interval counts as GPU experiment time or evidence of adaptive serving.
