# Preserved failures and limitations

## Unsupported first constructor

`p1_smoke_v1_gpu0` failed before generation.
Native RLT rejects async scheduling with exit mode `ouro`.
The reviewed retry selects native `ouro_delayed` with threshold1 and fixed maximum4.
It cannot perform model early exit.
The original failure remains unchanged.

## Batch-wide numerical alignment

The first numerical harness aligned complete CODA batches across policies.
Different native batch membership made that alignment invalid at N8.
The revised harness aligns request ID, output index, and token position.
The old failed attempts remain preserved.
No numerical threshold changed.

## Mutable telemetry hash

Early workers included dispatcher-owned telemetry in completed artifact hashes.
The dispatcher could append a final sample after the worker wrote its marker.
This produced a hash error without changing numerical output files.
The validator now excludes that mutable telemetry channel from worker completion checks.
The final evidence manifest hashes telemetry after process exit.
Immutable result and source hashes remain mandatory.

## Graph-reference leak in timing harness

All eight `p2_screen_v1` attempts failed while constructing their second engine.
A local graph reference retained the previous cache after engine teardown.
The next engine could not allocate another 54-GiB KV pool.
The correction deletes that local reference before collection.
The reviewed `p2_screen_v2` queue completed all eight jobs and 672 measurement cohorts.
The failed attempts do not supply headline performance.
This is a harness lifecycle correction, not a faster scheduling method.

## Evidence limits

The first two natural-stop jobs supplied a Transformers BatchEncoding container where RLT expects integer token IDs.
Both failed at validation before generation.
The reviewed v2 extracts the unchanged `input_ids` list and completes all twelve probes.
The correction does not change the prompt template or model settings.

The first eight paired interventions did not warm every possible row width.
Some newly formed batches triggered CUDA Graph capture inside the measured interval.
Their large timing ratios are confounded and do not support a scheduling speedup.
The reviewed v2 warms every width from1 through the combined row count.
It also requires zero new graph captures during each measured intervention.
The original v1 outputs and timings remain preserved and excluded from the primary intervention table.

The structural screen uses capped free-running hardware probes.
It is not GSM8K/MATH answer-quality evidence.
Numerical value checks sample KV positions rather than every cache element.
Repeated cohorts within one process are not independent process starts.
The same input family repeats across physical devices.
High-concurrency packing effects must not become a low-concurrency latency claim.
Natural open-loop capacity and full-answer accuracy require separate conditional gates.

The original Nsight marker describes first-output delivery too broadly.
It actually starts after native readback/host commit, before frontend delivery in that engine step.
The raw marker remains frozen. Reports state the precise clock interpretation.
