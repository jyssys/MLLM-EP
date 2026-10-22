# Dynamic pilot: measured placement differences

## Scope

This pilot contains 64 GSM8K and 50 MATH questions.
All three completed custom runs use the same canonical input IDs and live gate.
They use q=0.5, R2-R4 decode, R4 prefill, and last_exited KV.
They use the same no-refill token waves and one-request stage calls.
Each request generates freely with a 4096-token cap.
The custom TP4 pilot was still running when this file was written.
The larger 400-question target remains pending.

SD, LD, and OD generated identical output token IDs on all 114 questions.
They also recorded identical full exit-depth sequences on all 114 questions.
Their observed scores are therefore identical: 44/64 GSM8K and 31/50 MATH.
This observed agreement does not prove population-level accuracy equivalence.
Full task-wide KV snapshots were not collected.
Separate four-request cache probes matched all 96 logical KV slots bitwise.

## Task-observer service times

| Arm | GPUs | Sum of 29 prepared-service cohorts, s | Sum including resident-model cohort preparation, s |
|---|---|---|---|
| SD | GPU4 | 1869.092959 | 1946.021585 |
| LD | GPU4-7 | 1519.400906 | 1594.667514 |
| OD | GPU4-7 | 1682.233975 | 1717.982080 |

These are actual task-observer clocks, not clean hardware timing.
The cohort sum is not the entire experiment command wall time.
The service clock excludes request-bound graph preparation.
The preparation-inclusive clock excludes model loading.
No repeated clean-run confidence interval is available for this pilot table.
The separate identical-work replay will assess repeat variability.

OD took longer than LD on this completed pilot.
That result does not support an unconditional Loop-PP advantage under Dynamic Depth.
It also does not identify a unique cause of the elapsed-time difference.

## Observed work and logical copies

| Metric | LD | OD |
|---|---|---|
| Physical layer calls, logical rank 0 | 1,268,550 | 1,745,664 |
| Physical layer calls, logical rank 1 | 1,268,550 | 1,745,664 |
| Physical layer calls, logical rank 2 | 1,268,550 | 1,462,080 |
| Physical layer calls, logical rank 3 | 1,268,550 | 120,792 |
| Across-rank layer-call CV | 0.000000 | 0.530287 |
| Across-rank layer-call max/mean | 1.000000 | 1.376110 |
| Across-rank token-layer-work CV | 0.000000 | 0.358316 |
| Across-rank token-layer-work max/mean | 1.000000 | 1.254138 |
| Remote hidden-state tensor bytes | 4,724,670,464 | 879,792,128 |
| Remote exit-KV tensor bytes | 0 | 15,634,071,552 |
| Remote exit-KV copy calls | 0 | 79,519 |
| All explicit remote tensor bytes, including token return | 4,725,251,440 | 16,514,444,656 |

Logical ranks 0,1,2,3 map to physical GPUs4,5,6,7.
The stage-body counters exclude setup and graph construction.
Layer-call counts do not include every gate, coda, copy, or controller operation.
Token-layer work additionally weights each call by its input token count.
Neither metric equals measured GPU compute time.

The byte counters record actual copy calls and tensor shapes.
They do not measure transport wire bytes or effective bandwidth.
LD repairs deeper depth slots through local copies.
Zero remote exit-KV bytes therefore does not mean zero cache-repair cost.
OD must publish current exit-loop rows to later loop owners.
One such full-layer K/V row occupies 196,608 bytes.
The observed 79,519 remote copy calls account for the recorded 15,634,071,552 exit-KV bytes.

OD reduced hidden-state transfers but introduced larger remote exit-KV traffic.
Its fixed loop placement also produced unequal executed work.
These observations support two physical mechanisms that require timeline review.
They do not prove that payload bandwidth alone caused the elapsed-time difference.
Transfer latency, acknowledgements, queue waits, and compute overlap can also matter.

## Evidence

- `raw/quality_SD_pilot_v1/`
- `raw/quality_LD_pilot_v1/`
- `raw/quality_OD_pilot_v1/`
- `placement_work_by_rank.csv`
- `placement_accounting_summary.csv`
- `executed_work_audit.csv`
- `behavioral_pair_summary.csv`

The raw runs are `LIVE_MEASURED`.
Their counters provide structural work and logical-payload accounting from actual execution.
They do not provide an offline speedup estimate.
The final report must review controlled replay, the 400-question extension, and separate heavy traces.
