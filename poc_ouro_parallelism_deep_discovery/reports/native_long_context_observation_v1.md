# Native context scaling: completed controls

## Result and boundary

All six attempt 3 jobs completed on authorized H100 GPUs.
The matrix contains 144 clean trials across 24 cells.
Each cell contains two frozen prompt patterns with three repeats each.
Each request produces 128 controlled constant-token outputs.
This is native fixed-R4 hardware evidence, not natural reasoning quality or candidate speedup.

Both Base models complete 32768-token prompts at TP1, TP2, and TP4.
The previous local 10240-token setting therefore does not establish a model context limit.
This result does not extend the checkpoint's supported context length.
No RoPE setting changes.

## Live API wall time

Values are cell medians in milliseconds.
The interval encloses one native generate API call.
It includes prompt processing, 128 controlled outputs, sampling, and completion.

| Base model | TP | N1024 | N8192 | N16384 | N32768 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.4B | 1 | 1045.545 | 1457.683 | 2011.687 | 3373.688 |
| 1.4B | 2 | 927.228 | 1158.597 | 1474.309 | 2212.332 |
| 1.4B | 4 | 847.197 | 951.656 | 1118.899 | 1541.552 |
| 2.6B | 1 | 2076.056 | 2908.533 | 4026.822 | 6739.666 |
| 2.6B | 2 | 1820.040 | 2258.939 | 2884.913 | 4391.797 |
| 2.6B | 4 | 1578.830 | 1846.560 | 2198.798 | 3050.169 |

The mean per-pattern TP1/TP4 wall speedups at N32768 are approximately 2.1883× and 2.2098×.
The corresponding request-bootstrap 95% intervals are [2.1863, 2.1904] and [2.2087, 2.2110].
These intervals resample only two frozen patterns after within-pattern medians.
They do not establish uncertainty across a broad workload population.
No layer or token becomes an independent serving sample.

## Cached decode timestamp metric

The following TPOT values use native first-token and last-token timestamps.
The denominator is 127 inter-token intervals, not 128 output tokens.
This metric is not isolated GPU attention or collective service time.

| Base model | TP | N1024 TPOT ms | N32768 TPOT ms |
| --- | ---: | ---: | ---: |
| 1.4B | 1 | 7.929 | 16.190 |
| 1.4B | 2 | 6.935 | 11.358 |
| 1.4B | 4 | 5.934 | 8.317 |
| 2.6B | 1 | 15.772 | 32.272 |
| 2.6B | 2 | 13.532 | 22.274 |
| 2.6B | 4 | 11.770 | 16.431 |

Context growth increases measured native latency.
Native TP also improves the observed scaling for this workload.
Neither observation identifies a recurrence-specific removable cost.
The experiment does not isolate attention from other exposed work.

## Configuration and memory

TP1 uses GPU 4.
TP2 uses GPUs 4,6.
TP4 uses GPUs 4,5,6,7.
All participating devices have the validated NVLink topology.

Every arm retains the pinned Base checkpoint, BF16, fixed R4, and native asynchronous scheduling.
Chunked prefill, prefix caching, and expert parallelism remain disabled in this frozen control.
The native maximum length and batched-token cap are 32896.
This includes 32768 prompt tokens and 128 controlled output tokens.

Each TP group receives 52 GiB of aggregate KV reservation.
Per-rank reservations are 52, 26, and 13 GiB at TP1, TP2, and TP4.
The model retains 96 or 192 depth-indexed cache slots.
The experiment never aliases distinct live depth states.

| Base model | TP1 job peak GiB | TP2 maximum-rank job peak GiB | TP4 maximum-rank job peak GiB |
| --- | ---: | ---: | ---: |
| 1.4B | 56.196 | 28.487 | 14.725 |
| 2.6B | 58.619 | 29.590 | 15.453 |

These values are PyTorch peak allocated memory across each complete job.
They span all contexts and diagnostic capture.
They are not independent per-context peaks, HBM traffic, or total device-resident memory.
Reserved-memory records remain in the raw output.

## Validation and limitations

All six guards and runner statuses report success.
All 14 rank-level post-timing captures contain finite sampled values.
Those captures include four logits/hidden calls and selected written KV features.
They do not validate every cache element or natural outputs.
Forced token identity alone does not establish numerical or task correctness.

Attempt 1 failed during the owned resource handoff.
Attempt 2 completed clean trials but failed during diagnostic export.
Neither attempt enters the attempt 3 comparison table.
Both failures remain available.

The [validated summary](../analysis/long_context_complete_v2/summary.json) records every input hash and denominator.
Its CSV files retain all 144 trials and per-pattern comparisons.
Its editable SVGs retain context scaling for API wall, first-token latency, and cached decode TPOT.
The earlier complete_v1 memory field has an explicit scope correction.
No numerical value changed in that correction.

## Consequence for discovery

The completed controls remove a supposed capacity gap based only on the old local length cap.
They do not reject every ownership transformation or capacity regime.
They strengthen the conventional baseline that a proposed architecture must beat.
No qualifying architecture follows from this control alone.

## Reproduction

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_ouro_parallelism_deep_discovery.scripts.analyze_native_long_context --attempt 3 --output analysis/long_context_complete_v2
```

Use a new analysis output path for a repeated analysis.
Guarded GPU commands reside in `artifacts/native_long_context_queue_v3.json`.
The [pre-run contract](native_long_context_control_contract.md) preserves the frozen workload.
