# Candidate-specific adversarial prior-art check

[NanoFlow](https://www.usenix.org/conference/osdi25/presentation/zhu-kan)
already searches operation overlap and resource allocation and documents
load-dependent limitations. “More splitting” or “overlap communication” is not
a successor problem by itself.

[DynaFlow](https://arxiv.org/abs/2605.21603) directly studies context-sensitive
intra-device execution strategies. [TokenWeave](https://arxiv.org/abs/2505.11329)
addresses wave-aware splitting and overlap, including small-token MoE overhead.
[Bullet](https://xianweiz.github.io/doc/papers/26asplos_bullet.pdf) uses
contention-aware layer/resource scheduling, while
[FinDEP](https://arxiv.org/abs/2512.21487) searches attention/shared-expert
order and two levels of task partitioning in a disaggregated deployment.

These are close collisions for a generic dynamic NanoFlow-plan successor.
A material residual after best appropriate static plans, with a distinct
dependency/resource variable and directly measured request benefit, is needed.
Different models/hardware alone do not supply novelty. No such orthogonal
causal limitation is established in the present bounded native diagnostics.
