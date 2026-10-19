# Owned GPU handoff failure

## Result and scope

The first native long-context job failed before model initialization.
An owned burner entered GPU4 after the new guard checked that device.
The queue used an intermediate status file as its release condition.
That file did not mark completion of the outer cleanup routine.

This failure is not evidence of a model context limit.
It is not evidence of a foreign user's interference.
The queue now requires both guard records before it starts a dependent job.
No legacy guard file changed.

## Evidence sequence

The raw records remain unchanged.

| Observation | Raw record |
| --- | --- |
| The preceding quality job restored GPU4 worker3045208. | `outputs/quality_native_async_base26_P4_v1/guard_status.json` |
| At14:56:19.535724UTC, the new guard found that worker but no GPU4 context. | `outputs/native_longctx_base14_P1_v1/gpu_policy.json` |
| The preceding outer cleanup later restored GPU4 again. | `outputs/quality_native_async_base26_P4_v1/guard_cleanup_validation.json` |
| The new model saw only20.98GiB free, below its72.85GiB initialization requirement. | `outputs/native_longctx_base14_P1_v1/console.log` |

The preceding cleanup record names launcher3045402 and `scoped_idle_burner_restore:[4]`.
Its original pending restore list contains GPUs5,6,7.
The new guard already stopped the first GPU4 worker.
The preceding cleanup then treated GPU4 as idle and started another owned burner.
The intermediate status file permitted this overlap between two guard lifetimes.

## Dependency and correction

The reused shared guard writes `guard_status.json` before its caller executes the final cleanup.
The caller then writes `guard_cleanup_validation.json`.
The campaign queue previously checked only the first file.

The [release helper](../resource_release.py) now requires these conditions:

1. Both files exist.
2. The shared guard reports success.
3. The final cleanup reports no remaining owned GPU contexts.

This is an artifact-based prerequisite for this queue.
It is not a replacement for the next guard's live ownership check.
It does not authorize termination of a foreign process.

The [CPU regression](../tests/test_resource_release_cpu.py) covers an intermediate-only status, residual owned contexts, and a failed prerequisite.
The initial status-only helper failed two of three tests.
The corrected helper passes all three tests.
The two frozen-protocol tests also pass.

## Physical retry and separate failure

The second run passed initialization and completed all24 scheduled clean trials.
It also exported actual allocated and reserved memory.
The final diagnostic then raised a separate `teacher_index` attribute error.
That exception belongs to the reused capture helper, not the clean generation calls.

The second guard still reports failure.
The complete job must not receive a success label.
Its timing records remain partial evidence until independent checks establish their scope.
Its post-timing finite-value validation remains unresolved.

The diagnostic adapter needs a separate bounded correction.
That correction must not install a teacher sampler or change the native generation path.
All failure logs remain available for reproduction.

## Interpretation

The observed retry supports the handoff correction in this workflow.
It does not prove freedom from every resource race.
No model speedup, architecture result, or context-capacity conclusion follows from this debugging result.
