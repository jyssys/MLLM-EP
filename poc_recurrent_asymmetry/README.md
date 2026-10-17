# Recurrent Memory–Compute Asymmetry Discovery

Verdict: **GENERIC_ONLY**.
No new recurrent parallelism engine follows from this study.

Read the [final report](reports/final_report.md) first.
The [reproduction guide](reports/reproduce.md) lists scripts, pinned environments, and raw evidence.
The [timing contract](reports/timing_contract.md) separates clean wall time, component intervals, and modeled schedules.

The exact Ouro-1.4B R4 reference passes its regression checks.
Native TP controls and smaller-chunk execution remain numerically `REFERENCE_PARTIAL`.
The study does not claim a measured candidate E2E speedup.

The [measured timeline](figures/actual_trace_timeline.png) uses one actual GPU clock.
Its heavy observer fails the clean-latency gate.
The [modeled schedule](metrics/presentation_v1/figures/modeled_schedule.png) is a separate simulation.
Neither figure establishes a new four-GPU runtime speedup.
