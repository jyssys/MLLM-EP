# Physical memory and communication

## Ownership

Logical ranks 0,1,2,3 map to physical GPUs4,5,6,7.
Single uses physical GPU4 only.
Runtime UUID assertions validate this mapping.
The model is dense. There is no MoE Dispatch/Combine All-to-All in this study.

Layer PP owns six physical layers per GPU and all four depth slots for those layers.
Loop PP owns all 24 physical weights per GPU and one loop's KV slots.
Custom TP owns four attention heads and one quarter of the FFN channels per GPU.
Its embedding and LM head remain replicated.
Native TP uses vocabulary-parallel embedding and head operators.

## Measured allocation

Each value is a maximum over the selected completed cohorts or native setup records.
One GiB is 2^30 bytes.
The CSV retains every rank and exact byte count.

| Arm family | Weight GiB/rank | Reserved KV GiB/rank | Service peak allocated GiB/rank | Preparation peak GiB/rank |
|---|---|---|---|---|
| S4 / SD | 2.6722 | 13.0276 | 16.2633 | 16.2619 |
| Custom T4 / TD | 0.9496 | 3.2569 | 4.3031 | 7.5386 |
| L4 / LD | 0.9493 | 3.2569 | 4.4142 | 4.4129 |
| O4 / OD | 2.6722 | 3.2569 | 6.1377 | 6.1364 |

The full-period custom peak must include the larger preparation peak when applicable.
The TP preparation peak is therefore not 4.3031 GiB.
These are PyTorch allocation peaks, not continuous NVML HBM peaks.
Graph-only workspace was not independently isolated.
No safe OOM sweep or maximum-concurrency capacity claim is made.

| Native control | Weight GiB/rank | Reserved KV GiB/rank | Setup/graph peak allocated GiB/rank |
|---|---|---|---|
| TP1 | 2.6722 | 15.9961 | 18.8239 |
| TP4 | 0.6683 | 3.9990 | 4.8219 |

Native rows measure setup/graph peaks, not full-task HBM peaks.
Their pool budgets differ from the custom exact-capacity allocation.
Reserved pools, valid logical KV, and device-level HBM are different quantities.
Do not infer a memory-capacity advantage by comparing those unlike quantities.

Sources: [reviewed_memory.csv](reviewed_memory.csv), [memory_by_rank.csv](memory_by_rank.csv), and raw setup/ownership records.

## Observed explicit PP payloads

These counters describe the complete 400-question canonical views.
They count tensor payload at actual copy calls.
They do not measure link wire traffic, protocol overhead, or effective bandwidth.

| Arm | Remote hidden bytes | Remote exit-KV bytes | Remote exit-KV receive calls | All explicit remote tensor bytes |
|---|---|---|---|---|
| L4 | 17,654,231,040 | 0 | 0 | 17,655,661,808 |
| O4 | 3,530,846,208 | 0 | 0 | 3,532,276,976 |
| LD | 14,013,427,712 | 0 | 0 | 14,014,785,968 |
| OD | 2,648,489,984 | 37,006,934,016 | 188,227 | 39,656,782,256 |

The total includes the smaller explicit token payload.
Static O4 sends one fifth of L4's hidden payload.
Dynamic OD reduces hidden payload but adds much larger remote KV publication.
Zero LD remote exit-KV bytes does not mean zero local repair work.

One OD exit-row bank contains 196,608 BF16 bytes across all 24 physical layers.
An R2 exit sends that bank to two deeper loop owners.
An R3 exit sends it to one deeper owner.
The next token waits for acknowledgement from the required owners.
This preserves the selected last_exited policy.

The export and import routines use separate per-layer K/V copy operations.
Their packing, completion waits, and control messages can matter independently of payload size.
The trace does not isolate pure wire movement from those costs.

See [stage.py](../../ouro_8arm/stage.py#L52) and [request_runtime.py](../../ouro_8arm/request_runtime.py#L111).
The complete accounting is [placement_accounting_summary.csv](placement_accounting_summary.csv).

## Dynamic work placement

| Physical GPU | OD layer calls | OD token-layer work |
|---|---|---|
| 4 | 4,084,368 | 6,678,648 |
| 5 | 4,084,368 | 6,678,648 |
| 6 | 3,366,504 | 5,960,784 |
| 7 | 284,784 | 2,879,064 |

OD layer-call CV is 0.5311 and max/mean is 1.3822.
Its token-layer-work CV is 0.2828 and max/mean is 1.2035.
LD distributes these counted layer contributions evenly.
Neither count is a FLOP measure or a GPU-time percentage.
Prefill sequence length and attention cost remain different from one-token decode work.

Rank host spans include queue waiting and shutdown messages.
Their similar end times do not establish equal GPU compute time.
Use the measured activity timelines for actual GPU participation.

## TP communication

Custom TP uses real FP32 row-partial NCCL All-Reduce followed by BF16 conversion.
The representative trace records 24,704 NCCL kernels on each of four ranks.
Native TP4 records 6,369 custom reduction kernels and 33 NCCL kernels per rank.
Native custom reduction kernels are not labeled as NCCL.

Kernel counts are not logical collective-call counts.
The executed-loop-derived row-collective accounting in comm_profile.csv is a separate quantity.
Its tensor bytes are not network wire bytes.
NCCL kernel duration can include readiness and synchronization waits.
Do not divide payload by full-stage or waiting-inclusive kernel duration and call it true bandwidth.

No sum of different ranks' component maxima is used as an E2E critical path.
Sources: [comm_profile.csv](comm_profile.csv), [gpu_activity.csv](gpu_activity.csv), and [profiling_analysis.md](profiling_analysis.md).
