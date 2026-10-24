# Benchmark summary

The table contains LIVE_MEASURED clean bounded natural-generation cohorts.
P2048 and token chunk C512 remain fixed.
N counts background decode requests. One additional long prompt raises total concurrency to N+1.
Each cell uses three independent process/GPU blocks and two repetitions per block.
TTFT is the mean target-request TTFT.
ITL P99 is the mean cohort P99 of background gaps, not a pooled population quantile.
Wall time is the drained cohort makespan, not per-request E2E.

| Model | Background N | Arm | G | Target TTFT ms | Background ITL P99 ms | Cohort wall s |
| --- | --- | --- | --- | --- | --- | --- |
| huginn | 1 | group | 2 | 529.602 | 231.028 | 3.772 |
| huginn | 1 | group | 4 | 472.686 | 227.164 | 3.713 |
| huginn | 1 | hybrid | 2 | 557.480 | 144.670 | 3.573 |
| huginn | 1 | hybrid | 4 | 549.562 | 141.780 | 3.563 |
| huginn | 1 | mixed | 1 | 547.723 | 141.271 | 3.555 |
| huginn | 4 | group | 2 | 545.620 | 252.739 | 4.071 |
| huginn | 4 | group | 4 | 478.086 | 373.641 | 3.977 |
| huginn | 4 | hybrid | 2 | 565.291 | 148.613 | 3.830 |
| huginn | 4 | hybrid | 4 | 562.083 | 148.552 | 3.822 |
| huginn | 4 | mixed | 1 | 565.190 | 148.860 | 3.817 |
| ouro | 1 | group | 2 | 413.145 | 153.437 | 1.892 |
| ouro | 1 | group | 4 | 407.280 | 202.308 | 1.885 |
| ouro | 1 | hybrid | 2 | 407.588 | 100.388 | 1.884 |
| ouro | 1 | hybrid | 4 | 419.140 | 103.401 | 1.895 |
| ouro | 1 | mixed | 1 | 405.249 | 100.130 | 1.884 |
| ouro | 4 | group | 2 | 411.843 | 153.328 | 1.936 |
| ouro | 4 | group | 4 | 410.440 | 207.619 | 1.937 |
| ouro | 4 | hybrid | 2 | 414.727 | 108.545 | 1.938 |
| ouro | 4 | hybrid | 4 | 408.125 | 102.127 | 1.929 |
| ouro | 4 | mixed | 1 | 413.335 | 104.166 | 1.937 |

Natural generation permits EOS, but the fixed screen applies an output cap.
All 2,025 fixed-matrix request instances reached that cap.
None of those mechanism requests demonstrates complete natural-response attainment.
Capped output is not successful complete-response attainment.
The raw request file retains token IDs, stop reasons, and all delivery timestamps.
No finite cohort becomes a sustainable serving-capacity claim.
The exploratory screen, clean confirmation, and heavy profiles remain separate evidence lanes.
