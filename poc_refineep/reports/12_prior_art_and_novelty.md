# Prior art and novelty audit

## Direct boundaries

DeepEP already exposes throughput-oriented normal dispatch/combine and a
latency-oriented inference path. The current project benchmarked the validated
legacy BF16 intranode contract; the current upstream has since consolidated
newer functionality in V2, so absence of a local V2 measurement is not a
novelty gap ([DeepEP repository](https://github.com/deepseek-ai/DeepEP),
[legacy interface documentation](https://github.com/deepseek-ai/DeepEP/blob/main/docs/legacy.md)).

Epoch/FreshLane answers which live/new/refresh-required rows should execute.
This PoC uses that concept only to derive a compacted sensitivity worklist and
does not claim measured Epoch performance ([Epoch paper](https://arxiv.org/abs/2609.09748)).

The proposed mechanisms also face strong generic kernel collisions:

- StreamEP already describes streaming-tile dispatch/combine, GPU-side
  metadata, BF16 H100/NVLink specialization, and a single-launch style path
  ([official repository](https://github.com/evolutionaryscale/StreamEP)).
- FlashMoE fuses dispatch, expert computation, and combine in a GPU-resident
  persistent kernel and explicitly targets launch/synchronization overhead
  ([project page](https://flash-moe.github.io/),
  [NeurIPS 2025 paper](https://papers.nips.cc/paper_files/paper/2025/file/918d938bd209e5b56072777366f8a211-Paper-Conference.pdf)).
- mKernel similarly provides a persistent, GPU-driven fused MoE path rather
  than leaving fixed buffers or persistent execution unexplored
  ([official repository](https://github.com/uccl-project/mKernel)).
- UniEP pursues a unified expert-parallel megakernel with deterministic token
  ordering, albeit in a training-focused setting
  ([paper](https://arxiv.org/abs/2604.19241)).

## Novelty conclusion

The empirical characterization—block-diffusion liveness compaction produces a
large-to-small continuum inside one request—is useful and potentially distinct.
The proposed systems mechanism is not clean enough: existing DeepEP LL covers
the entire observed continuum, while fixed buffers, layout fusion, persistent
execution, and streaming dispatch/combine are all established mechanism
families.

Therefore a paper cannot be supported by “dLLM-specific small/medium kernel”
alone. A future direction would need either a new exact data contract that
changes the physical lower bound or a regime where the best existing path
leaves >=12% request headroom.

## Sources

- [DeepEP official repository](https://github.com/deepseek-ai/DeepEP)
- [DeepEP legacy documentation](https://github.com/deepseek-ai/DeepEP/blob/main/docs/legacy.md)
- [Epoch](https://arxiv.org/abs/2609.09748)
- [StreamEP official repository](https://github.com/evolutionaryscale/StreamEP)
- [FlashMoE project](https://flash-moe.github.io/)
- [FlashMoE paper](https://papers.nips.cc/paper_files/paper/2025/file/918d938bd209e5b56072777366f8a211-Paper-Conference.pdf)
- [mKernel official repository](https://github.com/uccl-project/mKernel)
- [UniEP](https://arxiv.org/abs/2604.19241)
