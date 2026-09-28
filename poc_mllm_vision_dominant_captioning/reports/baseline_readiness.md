
# Baseline readiness

Status: **READY_TRUE_EP2_EP4_RECONFIRMED**.

- Qwen3-VL-30B-A3B-Instruct revision `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`, BF16, d=2048, m=768, 48 sparse-MoE layers, 128 experts, K8.
- EP4: TP2-DP2-EP4-PP1, DeepEP high-throughput, linear 32 experts/rank; observed EP worlds `[4]`, invalid IDs 0.
- EP2: TP1-DP2-EP2, DeepEP high-throughput, linear 64 experts/rank; observed EP worlds `[2]`, invalid IDs 0.
- Full MMStar/ChartQA was not repeated; the frozen foundation and later physical path proofs were reused.
- Instrumented routing and clean B8 timing were separate. No latency simulator is used.
- Decode-step audit: `{"contract": "unique M-RoPE decode position; repeated external-DP padding forwards excluded", "max_sampled_decode_step": 240, "requests": 200, "status": "PASS", "violations": []}`. An initial 50-request partial trace that counted external-DP padding forwards was explicitly excluded; v2 keys steps by unique M-RoPE position.
