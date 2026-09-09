# GPU work paused by user — 2026-09-08

Historical checkpoint: superseded by explicit user reauthorization on 2026-09-09.
GPU work may resume on physical 4,5,6,7 only. Idle/final burn is authorized on
that same subset and must not overlap research measurements.

The user must return physical GPUs 4–7. Finish only the already-running clean
Qwen3-VL comparison if it completes promptly, then release all owned GPU workers.
Do not start another GPU experiment, correctness replay, model import that may
initialize CUDA, or burn. Resume GPU work only after new explicit user authority.

CPU trace analysis, source/literature auditing, reporting and safe CPU compilation
may continue. `run_cpu_only.sh` hides all GPUs. The old `run_scoped.sh` now refuses
execution while this marker exists, preventing accidental deferred GPU launches.
This is a user-directed pause, not an environment failure or research NO_GO.
