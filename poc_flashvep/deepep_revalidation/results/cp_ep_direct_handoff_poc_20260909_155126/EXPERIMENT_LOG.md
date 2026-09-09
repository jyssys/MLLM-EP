# Experiment Log

## 2026-09-09

1. Verified physical GPU mapping/topology and exclusive use of GPUs 4–7 for all research runs.
2. Audited local vLLM PCP/EP source. Native Qwen PCP is unavailable because no installed attention implementation advertises PCP support; the MoE side nevertheless contains an explicit PCP gather/reduce-scatter bridge.
3. Audited current SGLang, Megatron-LM, and TensorRT-LLM source commits and primary CP literature.
4. Proved the Ulysses and ring/pass-KV tensor-layout contracts and derived the exact direct-transport route-dependency lower bound.
5. Implemented actual-Qwen BF16 CP attention replay and confirmed exact query-block correctness.
6. Ran randomized 30-repetition CP1/2/4 crossover sweeps at 4K–64K.
7. Implemented actual-Qwen CP4→DeepEP-EP4 boundary replay with real attention/router/expert weights and captured Qwen3-VL hidden distributions.
8. Measured early/middle/late layers and 4K–64K boundary breakdowns.
9. Implemented exact one-block-lookahead streaming with 64/128/256/512/1024 token blocks, same-chunk/no-overlap control, global/per-shape warmup, and randomized ordering.
10. Reproduced the negative result across layers 4, 24, 47 and independent restarts.
11. Detected possible per-block CUDA-event observer tax; added a clean-timing path and reran 8K/32K/64K at 256/1024 blocks for 30 repetitions. Removing detailed events did not reveal a gain.
12. Replayed token-count-matched Vision/Text/Mixed activation distributions. No reliable modality-specific effect remained at 32K.
13. Generated rank-critical CSV summaries and applied the performance, correctness, analytical, and request-integration gates.

Burn workloads were not run during any research measurement and are excluded from GPU-time accounting.
