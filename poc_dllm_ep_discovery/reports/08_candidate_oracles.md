# Oracle tournament

## Conservative results

| oracle | GSM8K clean E2E | HumanEval clean E2E | evidence |
|---|---:|---:|---|
| perfect current fragmentation removal, raw | 3.326% | 2.858% | measured expert time minus local count-matched p10 envelope |
| perfect current fragmentation removal, p99 robust | **3.311%** | **2.852%** | group-p99 winsorized timing and trimmed envelope |
| hypothetical live-row compaction, expert-only | 7.108% | 6.481% | model sensitivity; not a new method and not measured Epoch |
| post-compaction perfect fragmentation residual | **2.119%** | **1.134%** | robust optimistic residual |
| feasible tiny-shape path at 50% residual capture | **1.060%** | **0.567%** | assumed, not implemented |

## Oracle construction

For current fragmentation, every `(dataset, layer, rank)` observation is compared with the 10th-percentile expert time among nearby pair-count observations. Required expert assignments, FLOPs and communication remain. This is deliberately optimistic: it removes all positive deviation from a local lower envelope, including noise not proven to be shape-removable.

For compaction, exact captured live-row routes are retained and shape-model timing is predicted for each rank; critical time is the maximum owner rank. The residual is predicted compact time minus a compact pair-count-matched lower envelope. P99 trimming removes the largest group-local timing observations before fitting/enveloping.

## Gate

Every novel candidate remains below 5%, so all are KILL under the user-defined tournament. No live prototype, TP4 transfer or online scheduler is authorized by the evidence.

