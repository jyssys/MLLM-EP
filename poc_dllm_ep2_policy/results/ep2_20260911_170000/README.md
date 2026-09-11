# Result root

This directory contains raw and derived artifacts for the EP2 mechanism screen.

- `runtime_audit*`: per-rank ownership, manager, profiler, and output evidence.
- `scaling`: 3 independent runs/backend, 30 timed repetitions/shape.
- `request_runs_warm5`: clean randomized-order request controls used for the
  static latency result.
- `trajectories`: observer-heavy structural traces, including the fixed-length
  gen=256 paired backend trajectory and batch=4 control.
- `analysis`: reproducible CSV summaries, oracle JSON, and required plots.

Primary decision numbers are in `analysis/summary.json`. Raw profiler traces
are retained locally but excluded from Git because of their size.
