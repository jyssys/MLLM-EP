# Exact KV memory accounting

The model has 1,434,652,673 parameters.
Unique weight storage is 2,869,305,346 bytes.
BF16 K and V require `2 × 2 × 24 × 4 × 16 × 128` bytes per context token.
The result is 786,432 bytes, or 0.75 MiB per token per sequence.
One physical pool block stores 16 positions for all 24 physical layers.
It occupies 3 MiB across K and V.
The measured 18,432-block pool occupies 54 GiB total.
It already includes all four depths.

| Prompt tokens | Independent processes | Mean TTFT, ms | Mean delivered ITL, ms | Logical KV, GiB | Peak allocated, GiB |
| --- | --- | --- | --- | --- | --- |
| 1024 | 2 | 201.82 | 18.759 | 0.750 | 57.113 |
| 2048 | 2 | 380.09 | 20.359 | 1.500 | 57.113 |
| 4096 | 2 | 754.13 | 22.961 | 3.000 | 57.113 |
| 8192 | 2 | 1521.89 | 28.802 | 6.000 | 57.113 |
| 16384 | 2 | 3023.05 | 39.989 | 12.000 | 57.113 |
| 32768 | 2 | 6152.68 | 62.748 | 24.000 | 57.113 |
| 65408 | 2 | 14298.74 | 108.342 | 47.906 | 57.113 |


B1/16K has 12 GiB of logical KV.
B1/32K has 24 GiB.
B1/65,408 has 47.906 GiB.
B2 at that last length needs 95.812 GiB before weights and workspace.
That payload cannot fit on one H100 in this experiment.
B4/32K and B8/16K each need 96 GiB before other allocations.
These are payload constraints, not allocator-fragmentation claims.

Small B/L states leave much of the common pool empty.
Reducing that reservation does not compress valid KV.
It also reduces the number of requests that the pool can admit.
The size-aware pool control charges this capacity trade-off.

`memory_events.csv.gz` distinguishes actual allocator bytes from issued written-prefix metadata.
Async metadata can precede producer completion.
It is not a hardware traffic measurement.
The fixture capture reads initialized pages in the producing CUDA stream.
`capacity_admission.json` records actual allocator acceptance and first rejection.
Those empty reservations are not completed concurrent inference.
The successful grid records the largest completed batch within the tested sweep.
