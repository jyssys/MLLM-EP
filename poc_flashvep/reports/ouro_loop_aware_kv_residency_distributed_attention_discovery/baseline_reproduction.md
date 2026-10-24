# Baseline reproduction and interference

The native model ran in all 22 feasible B/L cells of the registered discovery grid.
Each cell has two source-disjoint process starts and two technical repetitions.
`baseline_summary.csv` reports those results.
The table does not treat technical repetitions as independent requests.

The calibration compares one, two, four, and eight independent workers.
It uses P8192/B4 and common warm barriers.
Width order reverses in the second independent start.
The largest GPU0 ratio to its isolated reference is 1.00223×.
That observed deviation is below the registered 3% screen.
Other GPUs describe device spread, not a paired isolation experiment for every device.
This test does not certify zero rolling-startup interference.

The light memory observer compares the same final-source P8192/B4 workload.
Observer OFF mean ITL is 31.2365 ms.
Observer ON mean ITL is 31.3147 ms.
The observed difference is 0.250%.
Each arm has one process and three technical repetitions.
This is an overhead diagnostic, not a precise population confidence bound.

Primary layout pairs also use a Quiet Lane with no other managed GPU workload.
The 64K follow-up tests a variable parallel E2E result instead of selecting its fastest trial.
The original parallel measurements remain unchanged.
