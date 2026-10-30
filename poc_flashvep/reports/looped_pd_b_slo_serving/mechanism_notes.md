# Mechanism observations and limits

These observations use completed runs only.
They do not establish accepted serving capacity or original-model losslessness.
The final rate sweep and serving output gates remain authoritative.

## Post-prefill handoff is not the whole transfer cost

The first Huginn W1 PD_G1 point uses200 requests, nominal0.2requests/s, and physical GPUs0,1.
The observed mean uncovered handoff is11.914ms. Its request P95 is26.280ms.
The sum of these handoff intervals is0.08999% of summed request E2E time.
The denominator is overlapping request wall time, not cohort makespan.

This small share limits only an oracle that removes this specific post-prefill interval and changes nothing else.
It does not bound every possible pipeline effect.
Transfer backpressure can also affect the earlier prefill phase, batch composition, and subsequent admission.
The final timeline must check these effects separately.

## Full-depth execution accounting

The structural expectation is `R × sum(prompt_tokens + generated_tokens − 1)` recurrent-row executions.
The counter measures submitted recurrent rows. It does not measure FLOPs or GPU latency.

| Completed run | Observed recurrent rows | Structural full-depth expectation | Evidence |
|---|---:|---:|---|
| Ouro W1 PD_G1, rate1 | 765740 | 765740 | GPU-runtime submission counters and token trace |
| Ouro W1 B, rate1 | 765596 | 765596 | GPU-runtime submission counters and token trace |
| Huginn W1 PD_G1, rate0.2 | 6034592 | 6034592 | GPU-runtime submission counters and token trace |

Ouro B generates24516 tokens. Its matched PD_G1 control generates24552 tokens.
The difference is36 generated tokens and144 recurrent rows.
It is not an early-exit optimization. Every recorded Ouro output depth remains4.
The unequal outputs prevent equivalent-work lossless performance acceptance.

## Fragmentation does not reduce KV payload

| Ouro W1, rate1, 200 requests | PD_G1 | B | Evidence |
|---|---:|---:|---|
| Transport bytes sent | 131400237056 | 131400237056 | Cohort counter delta |
| Completed transfer fragments | 2058 | 3242 | Cohort counter delta |
| Cumulative host-observed transfer duration, s | 49.138 | 62.796 | Overlapping observation intervals |

B sends the same payload in more fragments in this point.
The cumulative duration is not DMA time, cohort makespan, or an additive E2E component.
It cannot establish that communication causes the entire performance difference.
The source also waits for queued descriptors and producer events before another legal scheduling step.
That can introduce polling and descriptor overhead. Separate diagnostic traces must measure it.

Sources are the named raw runs' `before_measured.json`, `after_measured.json`, `requests.jsonl`, and `closed_stats.json`.
The final tables retain all rates and both models. These selected mechanism observations do not replace them.
