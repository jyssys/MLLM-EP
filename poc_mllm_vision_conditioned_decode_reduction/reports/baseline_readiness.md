# Baseline readiness

Status: **READY_REUSED_AND_CAPTURE_VERIFIED**

The frozen Part-1 bundle passed its recorded integrity checks (30 digest checks,
six large anchors checked by size, no failures). The model contract remains
Qwen3-VL-30B-A3B-Instruct revision
`9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`, BF16, hidden 2048,
intermediate 768, 48 sparse MoE layers, 128 experts, and Top-8 routing. The
validated physical paths are TRUE TP1-DP2-EP2 (64 experts/rank) and TRUE
TP2-DP2-EP4 (32 experts/rank), both using DeepEP sparse dispatch/combine.

The prior public quality evaluation was reused rather than rerun. Its frozen
status is quality-ready. This project did not use the unvalidated latency
simulator.

## Long-reasoning evidence

- frozen pool: 20 requests (12 discovery, 8 held-out), image-hash disjoint;
- natural lengths: 130–237 tokens, median 186.5;
- datasets: MathVision 3, MathVista 7, We-Math 10;
- route trace: all 48 layers at 22 checkpoints through step 128;
- exact condition-A TP0 compact export: 21,120 rows;
- bounded vector capture: four routing-blind discovery requests, six layers,
  four 16-step windows, 1,530 complete real decode token/layer groups;
- all four 128-token causal replays matched the frozen reference token IDs.

The vector capture ran on physical GPUs 4,5,6,7 through TRUE EP4. It is an
instrumented diagnostic pass; its wall time is not clean latency evidence.
Idle-DP dummy rows and one 200-row chunked-prefill call were identified by
contract and excluded. No full hidden-state stream was stored.

The preferred 24-request pool was short by four. This does not affect the
discovery stop because both future-truth Stage-A gates failed by large margins;
held-out capture was therefore not opened.
