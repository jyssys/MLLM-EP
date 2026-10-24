# Distributed attention correctness

The tested groups contain two, four, or eight physical GPUs.
Each rank has its own CUDA context and NCCL process group.
Rank identity files record physical GPU IDs and UUIDs.
The rank-local valid K/V hashes match original model-derived BF16 values.
Each valid context position belongs to exactly one rank.
The final partial page remains length-masked.

The operator uses actual query broadcast and output/LSE all-gather.
Each rank executes optimized FA4 only over its local KV.
The stable merge combines normalized local outputs through logsumexp weights.
The query has one causal position.
Every retained prefix key is visible to that position.
This validates decode attention, not multi-query prefill masks.

Completed paired operator rows: 4464.
Maximum recorded relative L2: 0.0034308868926018476.
Maximum recorded absolute error: 0.0078125.
The registered limits are 0.03 and 0.125, respectively.
The limits did not change after measurement.
Both eager and captured distributed outputs enter the error maximum.

The main matrix samples four depths and physical layers 0, 12, and 23.
It does not run all 24 layers as a distributed model.
Batch B uses B copies of one real captured query/KV state.
It measures operator geometry, not independent natural serving requests.
The fixtures come from original pretrained forward execution, not random weights.

The edge campaign tests zero-length local shards and partial pages.
Its cropped prefixes are primitive correctness tests.
They are not model quality evaluations.
All-empty global attention is outside the model's valid causal state.
No attention or KV approximation is used.
