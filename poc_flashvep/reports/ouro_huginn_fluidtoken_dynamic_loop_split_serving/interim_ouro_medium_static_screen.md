# Interim Ouro medium-load static screen

## Scope

These are three completed N128 executions from one independent start.
All three use the same 4096/32 controlled workload and arrival seed `2026110402`.
Nominal QPS is `3.4569045`.
Realized arrival QPS is `3.0123088947538035`.
This field describes the finite scheduled trace, not actual coordinator send starts.
The final inventory reports the distinct submission rate and submission lateness.
Each run completed all 128 requests.
The split runs passed exact output parity, full recurrent work, and the native ACK/release audit.

The event-marker overhead control remains pending.
These results are provisional performance evidence, not a final speedup claim.

## Measured latency

All latency values in this table use milliseconds.
Quantiles come from individual request durations.
Queue p95 and Service TTFT p95 do not sum to TTFT p95.

| Method | TTFT p95 | Queue p95 | Service TTFT p95 | TPOT p95 | ITL p99 | Maximum ITL | Joint goodput, requests/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native PD | 642.873149 | 432.664993 | 258.220534 | 36.973340 | 40.021336 | 46.139101 | 2.955757 |
| Loop3 | 624.466204 | 210.533844 | 502.508031 | 47.684946 | 122.431217 | 188.993292 | 2.946007 |
| Token3072 | 676.328326 | 210.167821 | 531.567258 | 49.136073 | 104.477151 | 126.173019 | 2.953331 |

The frozen ITL p99 limit is `65.08287642151119 ms`.
Both static splits fail this limit.
Loop3 passes the stricter TPOT margin of `48.3363260853697 ms`.
Token3072 exceeds that margin, although it remains below the full TPOT threshold.
All three attain the request-level TTFT-plus-TPOT joint SLO for all 128 requests.
Joint goodput does not include the separate token-tail gate.
Thus a passing joint-goodput result does not certify Decode tail protection.

## Work and transport

Both splits move `4096` recurrent token rows per completed request.
Each split completes 128 actual offloaded serving requests.
Each split therefore assigns `524288` recurrent token rows to D.
These are not D-local synthetic probes.

Both runs record `309237645312` snapshot KV bytes across measured requests.
Loop3 adds `2147483648` hidden-state bytes.
Token3072 adds `524288` hidden-state bytes.
These are measured transport payload counters, not peak resident HBM.
Native streaming lacks these snapshot records. Its transferred payload is not zero.

Equal recurrent rows do not imply equal causal Attention work.
The structural D pair counts are `1074003968` for Loop3 and `1879310336` for Token3072.
These counts cover one physical Attention layer across all measured requests.
They are not measured FLOPs or a physical-compute saturation result.

## Warm-up limitation

Each run records one graph capture after the measurement marker.
The capture host spans are `0.302407`, `0.347388`, and `0.230976` seconds, respectively.
The order follows the latency table.
These spans include synchronization and capture preparation. They are not pure GPU kernel time.
The common warm-up policy therefore does not certify fully prewarmed execution.
No component percentile or overlapping capture span is subtracted from the reported TTFT.

## Source evidence

- `raw/f_ouro_medium_native_start1_128/serving_metrics.json`
- `raw/f_ouro_medium_loop3_start1_128/serving_metrics.json`
- `raw/f_ouro_medium_token3072_start1_128/serving_metrics.json`
- The corresponding `service_latency_requests.json` and `split_work_and_payload.json` files retain per-request accounting.
- The corresponding `graph_captures_*.jsonl` files retain native capture observations.
- `state/primary_screen.json` retains the correctness and lifetime verdicts.

No confidence interval exists for this single-start screen.
Neither split demonstrates SLO-safe superiority in this condition.
The remaining cuts, loads, transport controls, and online decisions still require measurement.
