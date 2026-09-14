# RefineEP microbenchmark

Status: `NOT_RUN_GATE_BELOW_12_PERCENT`.

`REFINEEP_MICROBENCH.csv` intentionally contains a single status row rather
than fabricated kernel measurements. The existing-path replay is complete,
but the credible incremental request oracle is 2.29% on GSM8K and 2.10% on
HumanEval. The spec requires 12% before CUDA work.

Consequently there is no claim that RefineEP beats normal DeepEP, low-latency
DeepEP, or any fused kernel. The measured operator winner is existing
low-latency DeepEP for all 25 real and all 15 controlled shapes.
