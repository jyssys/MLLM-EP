# Timing and attribution contract

Only physical GPUs4–7 are in scope.
Logical CUDA ranks map to the selected physical mask in order.
TP1 uses GPU4. TP2 uses GPUs4–5. TP4 uses GPUs4–7.

Idle assigned GPUs use verified project burners.
Each experiment guard stops only those burners and restores them after every exit.
Foreign processes remain untouched.

Clean timing excludes model loading and graph capture.
It measures real natural generation to EOS or the frozen cap.
A capped request remains in all throughput and censoring ledgers.
Forced output is permitted only in a separately labeled work-shape replay.
It cannot provide benchmark accuracy.

The native runtime is dense TP, not MoE EP.
Its reductions and logits gathering are real collectives.
There is no expert Dispatch/Combine All-to-All in this model.
FlashAttention kernel names containing Combine are not EP Combine.

Three timing quantities remain separate:

1. Clean request wall time.
2. Per-GPU kernel service time.
3. Non-overlapped critical-path time.

Kernel service sums cannot define an additive E2E breakdown.
Per-rank CUDA clocks are not globally aligned without calibration.
Missing attribution remains PARTIAL or UNRESOLVED.
Selected-state fidelity hooks deliberately add synchronization and CPU copies.
Their wall times are not headline serving latency.

Physical TP×DP throughput requires concurrent engines and a measured cohort wall.
The campaign cannot sum isolated TPS values to invent a four-GPU result.
Offered concurrency and actual residency remain separate fields.
Request bootstrap intervals cannot measure between-session throughput uncertainty.

No new framework can start before the quality gate passes.
If that gate fails, only a small labeled exploratory hardware audit is permitted.
Untested configurations remain NOT_RUN.
