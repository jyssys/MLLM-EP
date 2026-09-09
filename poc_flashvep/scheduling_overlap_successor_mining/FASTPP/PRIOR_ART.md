# Candidate-specific prior-art attack

[FastPP](https://www.usenix.org/conference/osdi26/presentation/hwang) already
models attention context, online prediction and rebalancing. Its actual
predictor must not be replaced by a token-only straw baseline.
[VPP](https://arxiv.org/html/2608.26523v1) already studies the chunk/layout axis
using folded virtual stages and cross-request packing. It is an August 2026
preprint with a different Ascend deployment and one-token-output evaluation;
those differences are limitations, not automatic novelty for our setup.

[DynaFlow](https://arxiv.org/abs/2605.21603) and
[Bullet](https://xianweiz.github.io/doc/papers/26asplos_bullet.pdf) also cover
context-sensitive execution and prefill/decode contention. A successor needs a
new, material variable/dependency beyond simple predictor retuning and existing
partition/chunk knobs. Current evidence does not establish it. See the common
PRIOR_ART_MATRIX.md for the technical comparison and explicit scope.
