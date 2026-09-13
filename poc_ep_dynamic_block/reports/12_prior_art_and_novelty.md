# Prior-art and novelty audit

Audit date: 2026-09-14. This audit separates semantic block scheduling from the
hypothesized distributed sparse-MoE control signal. It does not treat changing
`block_length` by itself as novel.

## Direct collisions

| Work | What it already does | Consequence for this PoC |
|---|---|---|
| [AdaBlock-dLLM (ICLR 2026)](https://openreview.net/pdf?id=0Cv9PwL7cI) and [official code](https://github.com/lgxi24/AdaBlock-dLLM) | Training-free adaptive block sizing from delimiter-token confidence / semantic volatility. The audited repository is commit `491b20580ad90f6019f29e045f484b538581101a`; `compute_block_length` chooses a delimiter-defined next block under a confidence threshold. | A confidence-triggered variable-B controller is a direct collision. A surviving contribution must show that measured EP shape has predictive value beyond semantic confidence and must beat a semantic-only controller. |
| [DSB: Dynamic Sliding Block Scheduling](https://arxiv.org/abs/2602.05992) | Training-free dynamic sliding block size based on semantic difficulty, with a matching cache design. | Dynamic size, sliding boundaries, and quality/efficiency gains are already occupied. The present work cannot claim novelty from those mechanics. |
| [Adaptive Block Diffusion](https://arxiv.org/abs/2606.29275) | Trains over a distribution of prefix-window configurations so inference can operate off the fixed training grid without quality collapse. | It attacks the training/inference mismatch rather than EP runtime choice, but removes any claim that arbitrary B is universally training-free and safe. Off-grid quality behavior must be measured, not assumed. |
| [Block Diffusion (ICLR 2025)](https://openreview.net/pdf?id=tyEyYT267x) | Establishes block diffusion as the interpolation between autoregressive and diffusion generation, with flexible-length generation and KV caching. | “B controls AR-ness” is background, not a contribution. |
| [LLaDA2.0](https://arxiv.org/abs/2512.15745) and [official repository](https://github.com/inclusionAI/LLaDA2.X) | Scales block diffusion to the 100B sparse-MoE LLaDA2.0-Flash substrate; the model was trained through changing block regimes. | This is the primary model fact, not evidence that every inference-time B is quality equivalent. Headline evidence must remain on the downloaded 100B checkpoint. |

## Systems boundary

| Work/system | Relevant mechanism | Boundary |
|---|---|---|
| [dInfer](https://github.com/inclusionAI/dInfer) | Batched block diffusion, KV-cache reuse, SGLang-backed LLaDA2 execution. The official example exposes a global `block_length`; the stock benchmark path audited here contained a config-42 assignment that forced B=32. | This PoC adds a bounded variable-B correctness and schedule diagnostic, not a production scheduler. |
| [Epoch](https://arxiv.org/abs/2609.09748) | Compiles a diffusion block, removes dead-position routed work, and transports a FreshLane through EP. | Epoch changes which rows execute. This PoC changes the next diffusion block boundary / physical wave shape. Any residual claim must survive an Epoch-like live-row compaction sensitivity analysis and must not relabel dead-work removal. |
| [vm-project dLLM plugin](https://github.com/vllm-project/dllm-plugin) | Integrates block-diffusion scheduling into a serving runtime; block size is currently configured globally. | Continuous batching and runtime integration are adjacent. A queueing-free offline BCT result cannot be called online-serving speedup. |

## Novelty gate used here

A paper-level result requires all of the following measured links:

1. B changes physical routed-EP geometry after controlling for endpoint,
   decoder effort, and `B × mini_batch_size`.
2. Different requests or completed-block states have different quality-safe
   system-optimal B values.
3. An EP-only observable (remote bytes, fanout, rows/expert, tiny-expert share,
   rank-load CV, or component ratio) predicts the safe next B beyond semantic
   confidence/acceptance features.
4. An actual-trajectory dynamic schedule beats every relevant fixed-B
   quality-latency Pareto point. Independent per-block latency summation is not
   accepted as an oracle.
5. The advantage remains material after schedule fragmentation, cache/layout
   changes, controller overhead, and an Epoch-like compaction sensitivity.

Failure of any of links 2–4 reduces the result to characterization or NO-DYNAMIC-NEED,
even if block size strongly changes latency.
