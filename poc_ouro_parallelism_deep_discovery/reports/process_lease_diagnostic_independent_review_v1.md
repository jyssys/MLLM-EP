# TP2×DP2 process-lease diagnostic review

## Outcome

The targeted preflight has no source blocker for one guarded diagnostic startup.
The v1 root cause remains unresolved.
This review does not permit a weaker lease check.
No GPU or model ran during this review.

The failed startup did not retain the rejected PID.
Its worker saved ownership only after the strict lease.
Both failure files therefore lack the information needed to identify the rejected process.
The original failures remain unchanged at `outputs/native_tp2_dp2_cohort_base14_v1/`.

## CPU observations

The first nested CPU check passed for frontend, EngineCore, and rank roles.
Their PIDs were 3387908, 3388075, and 3388178.
All three passed logical guard equality, initial `/proc` guard equality, and the existing strict self lease.
All three reported `torch.cuda.is_initialized() == False`.
The check used installed `get_mp_context`, `set_process_title`, and native Python.
It did not construct `WorkerProc` or load a model.

The timestamped recheck also passed for all three roles.
Its command, output, UTC interval, and pre-run hashes are in [the retained artifact](../artifacts/process_lease_spawn_chain_cpu_v1.json).
The command is:

```bash
/home/esjung/.venvs/cp-composition-vllm026/bin/python -m poc_ouro_parallelism_deep_discovery.scripts.process_lease_spawn_chain_cpu_v1 --output /home/esjung/MLLM-EP-ouro-deep-discovery/poc_ouro_parallelism_deep_discovery/artifacts/process_lease_spawn_chain_cpu_v1.json
```

The artifact path must be new for a rerun.
The working directory is `/home/esjung/MLLM-EP-ouro-deep-discovery`.
The [retained script](../scripts/process_lease_spawn_chain_cpu_v1.py) contains the subprocess code.
Its child environment hides all CUDA devices.

A separate CPU child deliberately inserted the guard through `os.environ` after exec.
The new diagnostic reported logical equality and initial `/proc` absence.
The unchanged strict lease rejected PID 3393415 with the expected guard-identity error.
The diagnostic distinguishes these observations in this injected pattern.
It does not reproduce the native failure or establish its cause.

The two diagnostic tests and 20 TP2×DP2 tests passed in native Python.
The combined run took 0.524 seconds.
These CPU tests do not establish native engine startup correctness.
The [review artifact](../artifacts/process_lease_diagnostic_review_v1.json) retains the observations and hashes.

## Installed-source feasibility

The installed `LLM.collective_rpc` accepts a callable with an additional worker argument.
See `vllm/entrypoints/llm.py:560–589`.
`MultiprocExecutor.collective_rpc` serializes the callable with cloudpickle.
See `vllm/v1/executor/multiproc_executor.py:343–377`.
Its worker loop invokes the decoded callable with the worker wrapper first.
See the same file at lines 986–1000.
Thus, `native_worker_snapshot(worker, expected)` matches the inspected dispatch.
The extra argument does not require a shared worker patch.

The inspected EngineCore and TP worker constructors use `get_mp_context().Process`.
See `vllm/v1/engine/utils.py:131–215` and `vllm/v1/executor/multiproc_executor.py:662–720`.
The inspected local path does not apply the Ray environment allowlist.
This limited finding does not exclude changes elsewhere during native startup.

## Privacy and authority

[The diagnostic helper](../process_lease_diagnostics.py) permits exactly three public guard keys.
It exports key names and presence/equality booleans.
It exports no environment values.
It checks UID before reading another declared process's environment.
It also records PID, PPID, UID, PGID, SID, and process start time.
The start-time parser selects field 22 from `/proc/PID/stat` correctly.

[The worker](../scripts/native_tp2_dp2_cohort_worker.py) records frontend observations before and after LLM construction.
It records rank RPC self observations and external observations of the declared PIDs.
It saves ownership before the strict check.
The UID, PGID, SID, initial guard, and ancestry checks remain required before `LOADED`.
No cohort timing starts if that check fails.
The diagnostic records authorize no cleanup, signaling, or process ownership.
A snapshot is not an atomic lease or a PID-reuse proof.

## Remaining test

One guarded native startup must identify the rejected PID and its role.
Logical equality with initial `/proc` failure would identify a snapshot discrepancy.
Logical and initial failure in a rank would instead identify missing or changed propagation.
Neither observation alone identifies the responsible source operation.
PID disagreement between rank RPC and ownership requires separate investigation.
The strict failure must remain fatal.

The observation interval began at 2026-10-19 16:48:03 UTC.
The retained review observations ended at 16:53:24 UTC, a 321-second interval.
Report preparation continued after that cutoff.
This interval overlaps coordinator work and is not additional campaign elapsed time.
