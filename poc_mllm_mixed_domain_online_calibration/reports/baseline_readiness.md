# Baseline readiness

Status: **READY_REUSED_AND_RECONFIRMED**.

- Model: `Qwen/Qwen3-VL-30B-A3B-Instruct` at frozen revision `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`, BF16, d=2048, m=768, 48 sparse MoE layers, 128 routed experts, top-k 8.
- Frozen Part-1 bundle `cdf380d9...f66d3` reports quality/actual-EP2/actual-EP4 ready and was reused; full MMStar/ChartQA was not repeated.
- TRUE EP4 trace: TP2-DP2-EP4-PP1, DeepEP high-throughput, linear 32 experts/rank; 160 requests, observed EP worlds `[4]`, invalid expert IDs 0, silent fallback `False`.
- TRUE EP2 trace: TP1-DP2-EP2, DeepEP high-throughput, linear 64 experts/rank; 48 requests, observed EP worlds `[2]`, invalid expert IDs 0, silent fallback `False`.
- Physical GPU UUIDs: 4 `GPU-6076e2f2-5b63-3761-5586-56ceb7df8139`; 5 `GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730`; 6 `GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28`; 7 `GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b`.
- Tracing is an instrumented evidence pass and is not treated as clean latency timing.
- EP8 is structural replay only; latency/TPOT/E2E/speedup are prohibited.
