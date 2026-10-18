# Timing and execution contract

## Evidence categories

| Category | Evidence | Allowed interpretation |
|---|---|---|
| Physical execution diagnostic | Actual decoder-layer calls, online gate events, completed KV copies | Actual loop omission and defined KV propagation |
| Numerical reference | Teacher-aligned logits and full KV arrays | Comparison with registered numerical limits |
| Quality generation | Natural greedy output, answer extraction, completed request wall time | Accuracy and observed execution work for this configuration |
| Live DP4 serving | NOT_RUN | No serving speedup or imbalance claim |
| Cost-aware simulation | NOT_RUN | No realistic oracle headroom claim |
| Zero-cost oracle | NOT_RUN | No structural scheduling gain claim |

## Request timing

The quality worker loads the model before all scored requests.
It runs a separate unscored warmup.
It synchronizes the GPU before and after each complete quality request.
`wall_ms` measures that host interval around `generate_batch`.
It excludes model loading and package installation.

`TTFT_ms` uses the engine's host request creation and first-token timestamps.
`decode_span_ms` uses its host first-token and completion timestamps.
Neither field is an isolated CUDA-event service interval.
The trace observer adds Python work to every recorded decode exit.
These quality-run times do not establish clean performance or instrument overhead.
The raw `peak_memory` field is the worker's cumulative allocated-memory peak.
The runner does not reset that counter before each scored request.
It is not an independent per-request peak measurement.

The study does not compare observer wall times as a new scheduling speedup.
It does not multiply one graph replay or one loop measurement by a logical work count.
It does not measure a live queueing service or a rank-idle witness.

## Execution units

The first generated token comes from full-depth prefill.
Each subsequent token has one observed online exit event.
The event's one-based depth records the actual number of recurrent core applications.
Each core application executes all 24 physical decoder layers.

The quality runner checks two identities for every request:

1. `prefill_recurrent_steps = 4 * prompt_tokens`.
2. `recurrent_steps = sum(actual_decode_depths)`.

It also requires one exit event per generated token after the first.
Every fixed-R4 event must record depth four.
Generated lengths may differ between fixed and adaptive policies.
No worker forces tokens after EOS to equalize those lengths.

## Work reduction denominator

The reported adaptive decode work reduction uses its own naturally generated positions:

```text
1 - sum(adaptive executed decode loops) / (4 * adaptive decode token count)
```

This denominator asks how much work four loops would require for those same positions.
It is not the total work of the separately generated fixed-R4 answer.
It is not an E2E speedup or a cross-GPU placement benefit.

## Observer validity

The bounded KV diagnostic synchronizes before it compares copied rows.
Its many completion waits intentionally contaminate wall latency.
The study uses that run only for execution and cache checks.

No hook-bypass timing comparison qualifies a clean latency result in this study.
No per-loop CUDA timing surface qualifies a CDB simulator.
These missing measurements remain `NOT_RUN`, not zero-cost assumptions.

## Statistical unit

Quality comparisons pair the two policies by question ID.
The bootstrap resamples those question IDs 20,000 times with seed 20261018.
It does not resample individual tokens, recurrent calls, or layers as independent serving requests.
The reported pilot intervals are nominal percentile intervals.
They are not anytime-valid intervals across repeated looks.
