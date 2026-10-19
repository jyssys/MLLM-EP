# Depth-pinned KV owners: a capacity screen

Status: structural accounting from actual original-gate calibration traces.
This is not a GPU timing experiment or a new scheduler.
It supplements the C01/C14 ownership screen.
It does not reject every depth-aware placement.

## Question

Suppose each GPU retains the complete shared core.
Each logical depth has one fixed compute owner and KV owner.
A token visits those owners in recurrent order.
Can this ownership choice create an additional execution resource merely because four depths exist?

The comparison pool has the same number of full-core workers.
It permits ideal work-conserving use of those workers.
It is not an implemented CDB runtime.
Neither comparison receives measured throughput from this calculation.

## Declared model

Every core task costs one normalized time unit.
The model ignores batching, communication, gates, output sampling, and state-publication costs.
These exclusions favor a simple depth-pinned pipeline.
They do not supply a physical latency bound.

Let `v[r] = Pr(executed depth >= r)`.
Let `o[r]` be the sole owner of depth r.
The normalized work at rank g is `w[g] = sum(v[r] for o[r] == g)`.

```text
total work/token = sum(v)
ideal pool capacity = P / sum(v)
pinned-owner capacity <= 1 / max(w)
finite-population capacity <= C / sum(v)
```

The capacity unit is tokens per normalized core-task time.
It is not tokens/s.
The script searches every distinct partition of four depths among the permitted workers.
Equivalent GPU-label permutations share one partition.
This search optimizes only the declared static service-capacity model.
It is not an optimum for physical distributed inference.

The [script](../scripts/depth_affinity_capacity_screen.py) checks the input work-count identity.
Five CPU tests pass.
The [machine output](../analysis/depth_affinity_screen_v1/screen.json) retains all assumptions and finite-C results.

## Fixed R4 control

Fixed R4 has `v = [1,1,1,1]`.
At P2 and P4, a balanced depth-owner assignment matches the ideal pool's service-capacity ceiling.
It does not exceed that ceiling.
At C1, the four dependent tasks remain serial under this layout.
Sending each task to a different GPU does not make its input available earlier.

P8 is only an analytic scenario.
Four unsplit logical depth tasks can use at most four distinct depth owners in this particular layout.
Splitting a core task introduces operator parallelism and requires the corresponding TP comparison.
No P8 execution occurred.

## Original adaptive traces

The source cohort contains eight calibration requests per domain and model.
The original gate uses q=0.5, minimum R2, maximum R4, and `last_exited`.
Prefill remains full R4.
These are natural free-running traces, not a new candidate's output.

The table reports saturated pinned capacity divided by the ideal pooled capacity.
Intervals use 4000 request-level bootstrap samples.
The script recomputes the best static partition within each sample.
Tokens are not independent statistical samples.

| Model | Domain | P | Structural capacity ratio | Request-bootstrap 95% CI |
|---|---|---:|---:|---|
| Base1.4B | GSM8K | 2 | 0.78622 | [0.76896, 0.80148] |
| Base1.4B | GSM8K | 4 | 0.77307 | [0.75604, 0.78851] |
| Base1.4B | MATH-500 | 2 | 0.79123 | [0.78595, 0.79578] |
| Base1.4B | MATH-500 | 4 | 0.73250 | [0.71260, 0.76082] |
| Base2.6B | GSM8K | 2 | 0.75929 | [0.75378, 0.76605] |
| Base2.6B | GSM8K | 4 | 0.75528 | [0.75000, 0.76247] |
| Base2.6B | MATH-500 | 2 | 0.79067 | [0.77576, 0.80447] |
| Base2.6B | MATH-500 | 4 | 0.70606 | [0.68952, 0.72708] |

These ratios are not measured slowdowns or throughput speedups.
They characterize the stipulated unit-cost service model.
The current traces concentrate most exits around R3.
Earlier depths therefore receive more visits than later depths.
Static depth pinning can create imbalance even when the fixed-R4 assignment is balanced.

At low C, the finite-population constraint can dominate both ceilings.
Equal ceilings at C1 do not establish equal measured latency.
They also do not establish positive pipeline headroom.

## What the screen does not prove

Real batch sizes can change core costs.
Attention cost can vary with context and active KV state.
Communication, copy contention, and output boundaries can add exposed time.
An actual pooled runtime also needs legal state access and ready tasks.
This screen does not grant free migration to that runtime.

A layout can split a depth, replicate selected state, change exact ownership, or use a different task granularity.
Those variants need their own DAG, capacity, and physical-cost tests.
The result does not exclude them.
It rejects an argument that four recurrent depth labels alone create four independent useful workers.

## Next useful test

A candidate must name a benefit that the same-resource alias-aware pool cannot already obtain.
It must preserve the current token's causal chain and the selected depth-specific KV.
No engine implementation follows from this structural screen.
