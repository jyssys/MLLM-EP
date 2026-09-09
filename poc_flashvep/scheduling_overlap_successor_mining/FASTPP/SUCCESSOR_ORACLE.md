# Successor oracle ledger

[ORACLES.md](ORACLES.md) distinguishes the calculated finite-policy lower
envelope, measured stage-cost diagnostics, and missing true request oracles.

Completed broad native Qwen3 PP4 screen: **0.587% request-mix / 2.351%
equal-workload** additional finite-policy E2E. Attained fixed-arrival SLO
goodput envelope peaks at 12.31% on nine thresholds, below the strong 15% gate;
this is not a maximum sustainable request-rate capacity result.

Exact best-chunk, arbitrary partition, joint chunk/partition and native PP×EP
request oracles: **NOT ESTABLISHED**. ALP's own predicted candidate enumeration
is not independent measured ground truth. A real static cost-partition control
regresses rather than realizing its positive stage proxy.

The five-existing-knob, fully warmed control is complete: 15 independent engines,
1,440 measured requests, common KV capacity, three randomized restart blocks.
Chunk128/512/2048, greedy and ALP were tested. PP-only/chunk2048 wins both
workloads by median mean-request E2E; additional finite selection envelope is
**0%**, including all three leave-one-restart-out choices. This is not a bound
on untested schedules. The nine fixed-arrival SLO thresholds remain a separate
metric; they are not an arrival-rate capacity sweep.

ALP versus PP-only median paired E2E reductions are **-21.54% steady / -19.24%
bursty**, all three pairs worse. Existing greedy recovers 94.97%, 18.12%, 82.77%
of ALP's bursty excess in the three blocks; choosing existing PP-only eliminates
that measured excess outright. A knob-remediable loss is not successor novelty.
Short answer checks pass in all 15 engines. Free-continuation equality is not
certified, so these are descriptive request controls, not a quality-matched
successor speedup claim.
