# Prior work and novelty boundary

[TEAM](https://proceedings.mlr.press/v306/wei26r.html) motivates temporal and spatial routing measurements.
Its target is MoE diffusion decoding.
Its accepted-token and masked-token opportunities do not establish recurrent-pass redundancy.
This study tests actual expert inputs, outputs, and changed-model trajectories instead.

[xLLM-Loop](https://github.com/ifm-ai/xllm-loop/tree/df0ed1ea415b55f0a944a291f90da9d47c98b5d5) supplies the IFM architectures and native loader.
The loader explicitly uses full-prefix generation without a KV cache.
This study does not claim that generic prefix caching is new.
It does not compare this runtime with an unmeasured optimized cached-decode engine.

[Foil](https://github.com/SR-A-W/how-to-loop-moe/tree/0f9cc7351658c2437f75379e492f3920f0a0ec19) already provides routing probes and masking tools.
Foil-1 shares its recurrent experts and uses pass-specific attention.
Its architecture is not an IFM architectural twin.
This study does not claim routing inspection or expert masking as a new method.

Contribution-based expert omission is a generic sparse-model idea.
C1 does not use recurrence information.
O1 requires the baseline expert output before it can rank contributions.
Neither mechanism establishes a new loop-specific execution architecture.

C2 is the recurrence-specific test in this study.
It stores raw expert outputs at a physical site and token position.
It uses the current route weight after a same-expert match.
Its evidence must include changed-model quality and timing costs.
High routing overlap alone is insufficient.

No new scheduling, EP, TP, PP, quantization, or training method is claimed.
The final verdict applies to the measured native B1 conditions and tested candidate grid.
It does not prove that every future expert policy lacks headroom.
