# MLLM MoE EP collective-granularity gap PoC

This bounded project measures whether a large TRUE-EP4 prefill collective creates a transient decode stall, whether ordinary token chunking trades that stall for repeated expert work, and whether the public Layered-Prefill semantics leave a full-prompt per-layer EP collective gap.

It uses only physical GPUs `2,3,6,7`, Qwen3-VL-30B-A3B-Instruct BF16, TP2-DP2-EP4, linear expert ownership, DeepEP high-throughput dispatch/combine, top-k 8, no DBO, no EPLB, and no prefix cache. It does not implement the proposed communication-only chunking method.

See `reports/reproduction.md` for the measured commands and `reports/final_summary.md` for the evidence labels and verdict.
