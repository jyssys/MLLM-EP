# Candidate-specific prior-art attack

[Layered Prefill](https://arxiv.org/abs/2510.08055) already addresses repeated
weight traffic, layer-axis scheduling and TTFT/TBT tradeoffs; adaptive group
selection is discussed. “Tune groups for workload” alone is not a clean gap.
[Bullet](https://xianweiz.github.io/doc/papers/26asplos_bullet.pdf) addresses
layer-level prefill/decode SLO scheduling and resource contention, and
[TriInfer](https://mlsys.org/media/mlsys-2026/Slides/3756.pdf) addresses MLLM
encode/prefill/decode scheduling. Merely adding image-encoder cost is insufficient.

A potential successor would need a directly measured residual from an execution
dependency these methods retain, beyond the best existing groups and chunks.
No such material residual is established here. The common PRIOR_ART_MATRIX.md
records source scope and publication-status qualifications; no first-observation
claim is made.

The repeated full-warmup comparison is particularly consistent with the base
paper's §5.9: relaxed TBT permits chunk2048 and cap4 to approach one another.
Thus our much smaller gap against chunk2048 than against chunk512 is not a
newly discovered limitation or evidence that an MLLM assumption failed.
