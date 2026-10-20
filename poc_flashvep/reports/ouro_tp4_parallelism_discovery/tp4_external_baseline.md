# Native vLLM TP4 attempt

The installed isolated environment contains vLLM 0.26.0 with native Ouro support.
The [official Ouro project](https://ouro-llm.github.io/) identifies this release as the last native Ouro release.
The environment required no package replacement.
The first attempt failed because the child PATH did not include the installed `ninja` executable.
The retry used the interpreter's existing bin directory. It completed the numerical captures.

The native model has 24 physical layers and 96 separate attention/KV slots.
All four workers own four local attention/KV heads and execute target-model compute.
`external_ownership.json` records UUIDs, weights, cache shapes, and storage.
The native model uses actual TP collectives. It is not a replicated-model substitute.

## Execution contract

- BF16, fixed R4, TP4, EP disabled, PP1, DP1.
- Maximum model length 4096 and KV pool 4 GiB per rank.
- Whole-prompt prefill, no chunked prefill or prefix caching.
- Eager execution for numerical observation, not an optimized CUDA Graph serving benchmark.
- FlashAttention 3, native paged KV and fused operations.
- Native vocabulary sharding, unlike the custom replicated output head.
- Separate PyTorch, Transformers, NCCL, scheduler, and kernel versions.

The engine initialization took 29.414 s.
That value is setup time, not request latency.

| P/O | Max logit absolute error | Max logit mean error | Max hidden relative L2 | Sampled KV relative L2 | C2 |
| --- | --- | --- | --- | --- | --- |
| 128/8 | 0.250000 | 0.045335 | 0.027809 | 0.009527 | True |
| 512/32 | 10.562500 | 0.764408 | 0.498974 | 0.009012 | False |
| 2048/32 | 1.937500 | 0.265983 | 0.176108 | 0.008571 | False |

The sampled KV comparison covers all 96 slots and three positions: 0, P−1, and P+O−2.
It does not compare every stored external KV element.
Free-running greedy IDs match 8/8, 32/32, and 32/32 tokens on the three shape probes.
All three outputs stop at their length cap.
These are bounded numerical probes, not reasoning accuracy or serving performance.

The external serving matrix is NOT_RUN because the longer numerical comparisons fail the frozen gate.
Consequently, there is no production-oriented TP4-versus-PP performance result.
The eager observer result must not be called an optimized TP latency baseline.
Native vLLM's internal differences also preclude an engine-matched claim.
