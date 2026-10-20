# Completion status

Verdict: BLOCKED at C2 full-model numerical qualification.
This is the specified stop condition, not a completed three-way serving campaign.

## Implemented and physically tested

- Four-process NCCL TP4 with actual weight/head shards and full depth-indexed KV.
- Operator and short-model numerical checks, including CUDA Graph replay.
- Longer aligned numerical diagnosis and an unsharded bitwise cache control.
- Common four-process Layer PP and Loop PP with persistent CUDA IPC transport.
- Bitwise bounded PP correctness and physical different-request kernel overlap.
- Twenty-four qualified PP-only cohort measurements across two shapes and two runtime paths.
- Separate native vLLM TP4 numerical and capped natural-generation probes.
- Nsight captures, CUDA API counts, communication accounting, and allocator memory records.
- Fifteen CPU evidence tests. See `tests.json` for the actual test run.

## Failed attempts retained

- BF16 row reduction: short hidden-state tolerance failure.
- FP32 row/column controls: longer hidden/logit tolerance failure.
- Attention head-geometry diagnostic: reduced error but still failed.
- Native vLLM first launch: child PATH could not find existing ninja.
- Native TP4 longer internal values: frozen gate failure despite token agreement.
- Orphan worker incident: later control runs excluded and repeated after cleanup.
- First clean TP Nsight attempt: profiler shutdown interrupted worker teardown. The retry completed.

## NOT_RUN

The qualified 21-row TP4-versus-PP serving matrix and optimized graph-mode vLLM serving did not run.
DP4 context, dynamic-depth runtime, new adaptive scheduler, capacity sweep, and task accuracy benchmarks did not run.
No R2/R3 sweep ran.

The source base commit is in `preflight.json`.
The result commit is the commit that adds this status file and the scoped new directories.
Use `git log -1 --format=%H -- poc_flashvep/reports/ouro_tp4_parallelism_discovery/status.md` to obtain it.
The final handoff identifies the exact result commit.
Only this PoC's files enter the new commit.
The user's final instruction authorizes push. `preflight.json` records that interpretation.
