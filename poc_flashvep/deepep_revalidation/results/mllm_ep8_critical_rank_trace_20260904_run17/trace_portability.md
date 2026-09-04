# EP8 trace portability

The route capture is self-contained for offline analysis: each measured wave/layer/TP shard stores token positions, token IDs, Vision/Text label, top-k logical expert IDs, router weights, destination EP rank, and sampled FP16 hidden vectors (layers 16/24/40). Per-rank DeepEP/Triton timing, model config, placement map, workload manifest, and environment are included.

The mapping is EP8-specific. Reuse on four GPUs requires an explicit remap and must not be presented as an EP8 rerun. All coalescing values are trace-driven oracle/count-cost estimates; no route or model execution was changed and no EP8 expert-output equivalence was captured.
