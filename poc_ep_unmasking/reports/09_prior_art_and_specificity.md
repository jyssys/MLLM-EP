# 09 — Prior art and specificity

- [Learning Unmasking Policies for Diffusion Language Models](https://arxiv.org/abs/2512.09106)
  trains a confidence-driven unmask selector with RL. A training-free,
  constrained physical-cost tie break would be algorithmically different,
  but "change unmask order" alone is not novel.
- [Epoch](https://arxiv.org/abs/2609.09748) compacts fresh/live expert-parallel
  work while keeping dense logical sequence state. It is the necessary
  physical substrate for crediting any finalized-row saving. We did not run
  measured Epoch.
- [REFLEX](https://arxiv.org/abs/2608.01784) keeps router ranking but changes
  token-level expert computation budgets with refinement state. This proposed
  unmask approach would not change router/top-k, but the current data do not
  show an independent physical Pareto point.
- TEAM changes diffusion work creation/acceptance and expert selection;
  that is a different semantic mechanism. SERE is only an analogy for
  quality-constrained system-cost allocation.

No exact published collision with the narrow idea "confidence-admissible
unmask choice resolved by predicted physical EP marginal cost" was found in
this targeted audit. Novelty cannot be promoted without measured additional
MoE+EP value over confidence-only, shuffled EP cost, and generic NFE effects.
Those controls were correctly not run after the early economic gate failed.

The current structural finding is unfavorable: same transfer count and
top-k preserve total live expert rows after idealized compaction, leaving
only rank/expert layout changes. The previous best-static EP4 matched batching
result showed those changes have weak wall-time impact. Thus an attractive
objective is not yet a paper-level systems method.
