# SERE adversarial screen

[SERE](https://arxiv.org/abs/2602.07616) already explicitly exploits batch-level
expert redundancy. A B1 failure alone cannot overturn its batch-decoding claim.
[XShare](https://arxiv.org/abs/2602.07265) explicitly considers batch-aware expert
selection, heterogeneous requests and distributed budgets. Merely adapting the
primary union or applying a per-batch budget is not an established novelty gap.

Still unresolved: whether generated-output changes and evolving natural request
completion cause a material quality/speed limitation that survives calibration,
threshold and batch controls. This is a question, not a claimed first observation.
No successor implementation or MLLM-specific assertion is justified yet.
