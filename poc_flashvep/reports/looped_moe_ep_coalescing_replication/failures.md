# Failures and untested conditions

No completed GPU cell failed native-versus-EP fidelity.
Failure logs remain in the archives if an attempt produced one.
One CPU watcher initially imported Torch from the analysis-only Python environment.
The watcher failed before GPU launch.
The fix delayed the GPU-only import until the worker phase.
The original watcher log remains preserved.

Foil's first-occurrence generation timing failed the observer-parity check.
The timing report preserves that failed check and the warmed follow-up.
It does not reinterpret the faster heavy trace as an optimization.

The generation replay is not independently validated asynchronous serving.
The ready observer adds control broadcasts and readiness votes.
Its sampled queue eligibility is scheduler-dependent.
The replica model assumes row-proportional expert span savings.
It does not model a new replica-aware collective implementation.
The same-window coalescing comparison can overstate incremental gain.
The best-window generic comparison removes that apparent gain here.

DeepEP production comparison: NOT_RUN, unavailable in the validated Torch 2.11 environment.
TP and hybrid baselines: NOT_RUN, optional and outside the thin EP bridge.
Optimized A/B dispatcher: NOT_RUN, headroom gate not met.
Candidate natural task accuracy: NOT_RUN, no candidate implementation.
NCCL wire traffic and hardware link counters: NOT_MEASURED.
Globally synchronized rank waiting and complete clean critical-path attribution: PARTIAL.
No approximation, smaller proxy checkpoint, or architecture substitution fills these gaps.
