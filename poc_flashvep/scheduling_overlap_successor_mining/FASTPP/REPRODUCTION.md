# Native reproduction summary

Official pin `cb3ce5b39c085a671b2c79477ac6d673b55f8eac`, isolated SGLang0.4.1
fork with torch2.5.1-cu124, vLLM0.6.4.post1 and transformers4.51.2. Physical
GPUs 4–7, BF16 PP4/TP1; no native expert parallel claim. Transport controls use
NCCL_IB_DISABLE=1, and the later H100 comparisons keep P2P enabled.

Dense Qwen2.5-32B: 18 independent engines / 14,400 measured requests across
PP-only, dynamic greedy, ALP, ALP+rebalancing and transport controls. Steady,
bursty, short and official Azure-conversation input families are retained.
This verifies original mechanism direction on a different hardware platform,
not exact reproduction of the paper's A100 PCIe headline numbers.

Native Qwen3-30B-A3B: checkpoint
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, 48 layers,128 experts,top8.
Nine broad-screen engines / 7,200 measured requests. A separate actual-ALP
instrumented run records 3,950 fully joined PP invocations; clean performance
runs omit these hooks. Six full-warmup static-partition engines / 576 requests
test equal versus measured-cost placement. The final five-existing-knob
full-workload-warmup screen completed fifteen engines / 1,440 measured requests.
All fifteen short-answer checks pass. Existing PP-only/chunk2048 wins both
workloads; additional finite E2E envelope is zero.

Every completed native short-answer sanity passes after the common one-line
official EOS compatibility correction. Uneven PP additionally needs an opt-in
KV-layer-count correction and common memory capacity. Neither repair is a
successor method. Long free continuations vary also under identical-policy
restarts; correctness-uncertain gains are descriptive and cannot promote a
candidate. See CORRECTNESS_DIAGNOSIS.md and ../QUALITY_CORRECTNESS.md.

Raw request timelines, server logs, exact command lists, startup checks and
paired analyses are preserved under `fastpp_runs/` in the common result root.
