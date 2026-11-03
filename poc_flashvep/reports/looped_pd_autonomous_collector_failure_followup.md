# Activity collection: retained failures and next controls

## Evidence classes

The following results describe instrumentation failures.
They do not establish model inaccuracy or Prefill compute saturation.
The cause of a CUDA launch failure remains unconfirmed.

## Completed bounded capture

Run: `native_joint_huginn_start1_bounded128_supervised_retry`.

The native CUDA collector retained 82,952 kernel events over 6.419574 seconds.
The measured kernel union was 84.170627%.
The 128-request wall time was 109.055238 seconds.
The matched clean run took 100.05724 seconds.
The wall-time difference was +8.992848%.
This exceeds the existing 5% perturbation limit.

All background outputs matched the clean run.
All eight additional Prefill probes matched their native reference.
Expected and observed recurrent work both equaled 10,551,296 rows.

Within the same capture interval, Decode ITL p99 was 271.056213 ms.
The frozen limit was 198.378139 ms.
The corrected P-ready queue had negative half-window growth.
Therefore, this run did not pass the joint acceptance gate.

## Failed extended capture

Run: `huginn_native_tail_capture_probe16_128`.

This control delayed profiler finalization until worker close.
The collector started after warm-up and after measured arrivals began.
The run failed during collection, before normal finalization.
The log contains completed requests through sample 38.
The first reported error was `CUBLAS_STATUS_EXECUTION_FAILED` in Huginn Coda MLP execution.
Later synchronization calls reported an unspecified CUDA launch failure.

CUDA errors can surface after the operation that caused them.
The reported Coda operation is not proven to be the first faulty operation.
This failure does not support a profiler-finalization-only explanation.

Run: `native_joint_ouro_start1_bounded128`.

The run also failed after collection started.
Its final reported failure occurred in `ready_event.query()` during transfer release.
The collector did not record a normal collection-stop event.
This identifies a failure boundary, not a proven root cause.

## Distinct source-version guard failure

Run: `bulk_joint_ouro_start1_probe16_128_exact_probe_reference`.

This reference job failed before model execution.
The source guard detected a concurrent edit to `prefill_graph.py` during worker startup.
The guard rejected the run as designed.
This is an experiment-control error, not measured output divergence.
The failed namespace and its logs remain intact.
A new reference namespace is required before the probe can pass correctness.
Subsequent source edits require the same exclusive lane lock as GPU launches.

## Ranked hypotheses for the CUDA collection failures

1. Collection interacts with native GPU graph replay or its synchronization.
   A model-path test without NIXL can distinguish this from a transport-only failure.
2. Collection interacts with NIXL, CUDA events, or source-buffer lifetime.
   A matched P-only collection must remain stable before this explanation gains support.
3. Collector allocation or host processing changes memory or submission pressure.
   Memory records and collector-duration controls can test this explanation.
4. A pre-existing execution race appears under changed timing.
   Repeated clean controls and synchronized diagnostic controls can test this explanation.

No hypothesis is confirmed by the current failure locations.
Do not expand the failing collection path without a bounded diagnostic control.

## Evidence locations

All run directories are under:

`looped_pd_autonomous_prefill_compute_decode_headroom/raw/`

The matching logs are under:

`looped_pd_autonomous_prefill_compute_decode_headroom/logs/`

This report preserves failed runs as failures.
No empty or failed trace is converted into measured idle time.
