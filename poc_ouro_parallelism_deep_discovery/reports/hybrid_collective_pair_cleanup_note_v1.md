# Collective-pair P2: incomplete cleanup diagnostic

## Current result

The P2 attempt remains NOT_ADMITTED.
A read-only check found six PAIR lines and rank-0 success with six records.
It found no outer-guard status and no rank-local cleanup status.

The observation falls within 2026-10-19 18:27:43–18:28:37 UTC.
Main reports rank-0 success at 18:26:27 UTC.
The workers' continued lifetime comes from Main's observation.
This review does not inspect or signal either process.

The earlier source/CPU review did not establish successful GPU teardown.
The current completion check supplies a red signal.
It does not identify the blocked cleanup call.

## What the source establishes

The [probe](../scripts/hybrid_collective_pair_probe.py#L191) calls CustomAllreduce.close before process-group destruction.
It writes local status only after both calls.
Thus, absent local status cannot distinguish their internal locations.

Installed CustomAllreduce.close has three native steps.

1. Dispose the custom-AR object.
2. Free this rank's metadata buffer.
3. Free this rank's staging buffer.

The Python method has no collective or timeout.
The wrappers call opaque `torch.ops._C_custom_ar` functions.
Their installed Python source does not establish which native step can block here.

Default process-group destruction shuts down all registered groups in reverse group-name order.
The 120-second group timeout does not establish a timeout for every native destructor.

Installed vLLM GroupCoordinator.destroy uses a different order.
It destroys device and CPU groups before the device communicator.
CudaCommunicator.destroy then releases its CustomAllreduce reference.
The custom object's destructor invokes close.

This difference supports an isolated ordering test.
It does not prove that the probe's order caused this failure.

## Smallest defensible retry diagnostic

Do not instrument the active workers.
Use a fresh output directory and a separately frozen retry source.

Keep the measured pair bodies, inputs, graph captures, trials, and correctness gates unchanged.
Add CPU-only BEFORE/AFTER markers around each cleanup boundary.
Each marker must retain rank, boundary name, and a monotonic host timestamp.

Main selected one teardown-only retry with a different cleanup order.

- NCCL subgroup destruction.
- Gloo WORLD destruction.
- Custom-AR close.
- Final status write.

A 30-second one-shot `faulthandler.dump_traceback_later` can report the blocked Python call site.
Give each rank its own open trace file.
Cancel the watchdog after cleanup.
This diagnostic needs no CUDA event query, device synchronization, signal, or active-process attachment.

The markers can distinguish NCCL destruction, WORLD destruction, and custom-AR close.
If close blocks, its Python frame can distinguish dispose from the two buffer-free wrappers.
Native C++ cause remains unresolved without further evidence.

The selected order more closely follows installed vLLM teardown.
Custom-AR close passes rank explicitly to both buffer-free calls.
It performs no process-group query.
Thus, the Python source does not require a live Gloo group for close.
The retry remains a hypothesis test, not a claimed fix.
It permits no unrelated workload change or further GPU retry.

## Admission and preservation

Rank-0 success is insufficient.
Require outer-guard success and every rank's error-free local cleanup record.
Retain the first attempt even if a later retry succeeds.
No timing result from this incomplete attempt enters an admitted table.

This task performs no GPU call, process mutation, active source edit, or latency analysis.
The [structured note](../artifacts/hybrid_collective_pair_cleanup_note_v1.json) retains source and observation hashes.

## Actual work

The actual interval is 2026-10-19 18:27:43 UTC through 2026-10-19 18:31:33 UTC.
The active interval through this checkpoint is 230 seconds.
No prior pause interval enters this work.

