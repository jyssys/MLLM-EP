# Initial D-local Prefill probe latency

## Measured observations

The first additional D-local Prefill request took much longer than later requests.
These durations include scheduler waiting and all host-visible execution costs.
They are not isolated GPU compute times.

| Run | Completed probes | First probe, ms | Later probe range, ms |
|---|---:|---:|---:|
| `graph_joint_ouro_start1_probe16_128` | 4 | 4179.065 | 121.138–131.015 |
| `long_deep_ouro_start1_probe16_128` | 7 | 2808.412 | 94.669–97.705 |
| `long_deep_huginn_start1_probe16_128` | 3 | 2469.369 | 131.095–139.938 |

Each probe uses 16 input tokens and produces one output token at full fixed depth.
The raw durations come from `finish - time` in each run's `D_probe_requests.jsonl`.

## Competing explanations

1. A previously unused attention shape requires kernel preparation on its first execution.
2. A previously unused allocation shape requires memory or workspace preparation.
3. The first probe encounters more scheduler waiting than later probes.

No native graph-capture event overlapped the first probe in `graph_joint_ouro_start1_probe16_128`.
This observation excludes that recorded capture mechanism, not all forms of initial kernel preparation.

## Controlled test

The new control executes one real D-local Prefill request before measured arrivals.
It uses the same probe geometry in both the no-probe baseline and the probe candidate.
The native scheduler, KV allocator, model operations, and fixed depth remain unchanged.

The warm-up request must complete before the measured work-counter snapshot.
Its output and depth are recorded separately.
It does not contribute to throughput, SLO attainment, or measured probe counts.

An eight-request gate checks output parity, KV transfer, and recurrent work accounting before a larger run.
The test uses a new namespace and preserves the original cold-probe evidence.

## Interpretation limits

A reduction in the first-probe delay would identify a warm-up-sensitive cost.
It would not identify compilation specifically without further evidence.
It would not prove P compute saturation or sustainable serving capacity.

The same-run P saturation, D memory, D SLO, and actual probe gates remain mandatory.
The final report must show cold and warmed conditions separately.
