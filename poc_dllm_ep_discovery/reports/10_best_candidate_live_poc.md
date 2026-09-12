# Best-candidate live PoC disposition

No live method prototype was implemented.

The strongest novel perfect oracle is current expert-fragmentation removal at 3.311% GSM8K and 2.852% HumanEval. The more defensible post-Epoch residual is 2.119% and 1.134%; a 50%-capture mechanism would yield only 1.060% and 0.567% before its own overhead.

The working contract prohibits complex implementation below 8% credible E2E headroom. A live coalescer would introduce queueing and metadata costs, while a specialized kernel would require significant engineering and correctness validation. Both can only recover a fraction of an already sub-5% perfect bound.

Accordingly:

- no custom kernel;
- no expert queue/coalescer;
- no online serving scheduler;
- no TP4 control, because no strong EP4 fragmentation candidate exists;
- no quality-changing experiment.

This is a gate-driven stop, not an environment block.

