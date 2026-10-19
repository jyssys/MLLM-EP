# R1 and R4 hardware counterfactual

## Scope

This experiment changes recurrent execution depth, not model training or weights.
R1 is not a quality-equivalent replacement for R4.
It is a hardware ablation for native TP scaling.
Both arms use asynchronous native scheduling, BF16, and the original shared physical core.

The workload contains two frozen prompt shapes at each context.
Each condition uses five repeats per shape and 128 forced output tokens.
The output policy removes generated-length variation from this hardware comparison.
It does not measure benchmark accuracy.

## Cached-decode scaling

The table reports TP1 TPOT divided by TP4 TPOT within the same model, depth, and context.
It does not compare quality across depths.

| Model | Context tokens | R1 TP4 speedup versus R1 TP1 | R4 TP4 speedup versus R4 TP1 |
|---|---:|---:|---:|
| 1.4B | 1024 | 1.1653× | 1.3478× |
| 1.4B | 4096 | 1.2252× | 1.4322× |
| 1.4B | 8192 | 1.4570× | 1.5089× |
| 2.6B | 1024 | 1.4288× | 1.3475× |
| 2.6B | 4096 | 1.5189× | 1.3684× |
| 2.6B | 8192 | 1.6002× | 1.4567× |

The direction differs between the two model sizes in these tests.
The 1.4B R4 arm scales better than its R1 counterfactual.
The 2.6B R4 arm scales worse than its R1 counterfactual.
These observations do not establish a common recurrence-specific TP penalty.

R4 also changes executed depth, graph structure, and depth-indexed KV state.
The comparison does not isolate those effects from recurrent execution.
Layer count, TP primitive counts, and memory capacity do not by themselves identify a removable latency cost.

## Uncertainty and evidence

Configuration order is sequential.
Only two independent request shapes enter each condition.
The experiment does not establish broad request-level confidence.
No adaptive or candidate-architecture result appears in this table.

Raw TP1/2/4 measurements and TTFT/E2E ratios are in `analysis/recurrence_comparison_02/recurrence_controls.csv`.
The editable scaling plots and JSON summary are in the same directory.
The inputs are `analysis/async_checkpoint_01` and `analysis/async_R1_checkpoint_01`.
The first analyzer failed because CSV column names depended on TP degree.
Its partial output remains in `analysis/recurrence_comparison_01` with an explicit failure note.

The next discriminating control is full-core alias versus physical-copy execution with isolated compilation caches.
That control also needs numerical and lowered-kernel checks.
It cannot establish HBM traffic without valid hardware counters.
