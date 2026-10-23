# Completed-token latency and throughput

All entries below are LIVE_MEASURED hardware probes.
They are not modeled speedups or full-answer serving-capacity results.
Every forwarded token executes Static R4.
Generation is greedy and free-running, but EOS is ignored to hold the diagnostic output count equal.

## Quiet low-request confirmation

Each point uses six process/GPU blocks: three independent starts on each of two GPUs.
Each block contains three repeated128-token cohorts.
Initial prefix length is P. Context grows during decode.
ITL entries are means of process-level mean adjacent output gaps.
Speedup uses the matched completed-token rate, with10,000 paired process-block bootstrap draws.

| P / N | D | Mean ITL ms | Output tokens/s | Rate speedup vs D1 |95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
|256 /1 |1 |17.601 |56.816 |1.0000× |[1.0000,1.0000] |
|256 /1 |2 |17.790 |56.211 |0.9894× |[0.9888,0.9899] |
|256 /1 |4 |17.848 |56.030 |0.9862× |[0.9849,0.9872] |
|2048 /4 |1 |22.134 |180.717 |1.0000× |[1.0000,1.0000] |
|2048 /4 |2 |22.388 |178.668 |0.9887× |[0.9881,0.9892] |
|2048 /4 |4 |22.474 |177.985 |0.9849× |[0.9844,0.9853] |
|8192 /1 |1 |28.441 |35.162 |1.0000× |[1.0000,1.0000] |
|8192 /1 |2 |28.668 |34.882 |0.9921× |[0.9916,0.9924] |
|8192 /1 |4 |28.762 |34.769 |0.9888× |[0.9885,0.9892] |

The reference interval is identically1 because each block divides by itself.
It is not an uncertainty interval for absolute D1 throughput.
The repeated source family limits population generalization.
The low-N results do not pass the prospective3–5% physical grouping screen.

## High-N ordinary cohort: preserve the positive observation

At P2048/N16, the ordinary quiet cohort retains native prefill-induced phase fragmentation.
Its six-block results are:

| D | Mean ITL ms | Output tokens/s | Speedup vs D1 |95% interval |
| ---: | ---: | ---: | ---: | --- |
|1 |35.332 |458.202 |1.0000× |[1.0000,1.0000] |
|2 |30.792 |525.426 |1.1467× |[1.1456,1.1477] |
|4 |30.986 |518.229 |1.1310× |[1.1303,1.1317] |

This is a real cohort measurement.
It is not evidence that host-loop fusion alone improves latency by14.67%.
Different recurrent batch composition remains part of that intervention.

## Same-frontier decode control

This lane uses eight process/GPU blocks and three64-token repeats.
Every member waits until the entire cohort completes its first output readback and host commit.
That internal commit can precede frontend delivery within the same engine step.
The resulting P2048/N16 decode frontier is aligned across policies.

| D | Mean ITL ms | Output tokens/s | Speedup vs matched D1 |95% interval |
| ---: | ---: | ---: | ---: | --- |
|1 |30.363 |535.304 |1.0000× |[1.0000,1.0000] |
|2 |30.726 |528.689 |0.9876× |[0.9867,0.9883] |
|4 |31.033 |523.697 |0.9783× |[0.9778,0.9788] |

The ordinary and aligned lanes have different initial scheduling histories.
The quiet ordinary table also has a different output cap.
Do not treat a cross-lane ratio as a paired complete-serving speedup.
The within-lane comparison isolates the missing pure grouping advantage.
The barrier is not a deployable TTFT improvement claim.

## Native and fairness controls

`baseline_results.csv` and `benchmark_summary.csv` retain actual native Mixed and completion-fair controls.
Native Mixed keeps existing one-loop prompt/decode co-packing.
It is not an ELB implementation.
Matched Mixed D1/D2/D4 confirmation and execution-mode ablations remain separate rows in those tables.
No gain against a weaker fairness control alone qualifies a new method.

At P2048/N16, matched native Mixed D1/D2/D4 rates are534.611/528.282/524.124 tokens/s.
D2/D1 is0.9882× [0.9876,0.9886]. D4/D1 is0.9804× [0.9799,0.9809].
These use three same-GPU process blocks, rotated policy order, and the same64-token cap.
The earlier positive ordinary Refill signal does not survive this strong native control.

## Unevaluated headline metrics

Natural full-answer TTFT/E2E, request-level SLO goodput, and sustainable request rate remain NOT_RUN.
The diagnostic decode rate excludes initial prefill and first-output time.
It must not be relabeled as complete-serving capacity.
GSM8K/MATH-500 accuracy remains NOT_RUN.
See `quality.md` and `timing_contract.md` for the exact scope.

Raw tables include the full14-shape screen and every individual cohort.
P8192/N16 is omitted because it exceeds the common KV capacity.
No cache capacity change or fabricated result fills that cell.
