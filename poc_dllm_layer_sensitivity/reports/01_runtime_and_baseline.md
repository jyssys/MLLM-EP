# Runtime and baseline audit

## Fixed substrate

- Model: local `inclusionAI/LLaDA2.0-flash` revision
  `LLaDA2.0-flash-744c3f8`, BF16, 32 transformer layers, hidden size 4096,
  256 routed experts/top-8 plus one shared expert.
- Runtime: dInfer on SGLang 0.5.3.post1, DeepEP normal/HT-style dispatch,
  owner-local fused expert execution, and reverse combine.
- Topology: dense TP4 plus routed EP4, DP1. Each physical rank owns 64
  complete routed experts. Nonzero remote traffic and the dispatch/expert/
  combine timeline were already proven in the immediately preceding true-EP4
  substrate audit and were re-used without changing topology.
- GPUs: only physical H100 0--3. UUIDs and all NV18 peer links are recorded in
  the preceding substrate report. Model-loaded HBM was approximately
  74.9--75.5 GiB/rank.
- Best-static setting: submitted batch 32, `mini_batch_size=32`, generation
  budget 32, diffusion block 32, threshold 0.9, config 42. No weak-default
  comparison was used.

The project worktree started from MLLM-EP commit `43184dbea1b0`; the external
dInfer instrumentation started from true-EP4 trace commit `9132ce9b2580`.

## Clean deterministic baseline

| task | clean request median | NFE | throughput | bounded quality |
|---|---:|---:|---:|---:|
| GSM8K | 6.075 s | 66 | 438.0 token/s | 5/32 exact answer |
| HumanEval | 7.343 s | 86 | 450.6 token/s | 6/32 pass@1 |

These are three-restart medians from the immediately preceding identical
best-static substrate. A fresh GSM8K run measured 5.884 s/NFE66. Fresh causal
baseline policies reproduced all 32 prior answers and NFE exactly on both
tasks. The low absolute score is a bounded 32-token-generation substrate
anchor, not a claim about full-benchmark model quality.

Low-overhead component attribution gives the following clean-E2E upper shares:

| task | routed router+dispatch+expert+combine | whole measured MoE | attention proxy |
|---|---:|---:|---:|
| GSM8K | 54.37% | 59.52% | 27.10% |
| HumanEval | 56.76% | 62.81% | 29.84% |

Component shares are Amdahl upper attributions and are not jointly removable.
The layer-level MLP bracket includes asynchronous waits and sums above clean
E2E; it was never used additively. Full-layer oracles instead normalize all
non-MoE request time across measured non-MoE layer stages, deliberately
favoring the layer-skipping hypothesis while capping the total at 100%.

## Observer boundary

- GSM8K stability trace: 45.31 s versus 6.075 s clean median.
- HumanEval stability trace: 46.98 s versus 7.343 s clean median.
- The tensor-norm stability trace is therefore representation evidence only.
- Request performance and cost mapping use clean baseline medians and the
  earlier low-overhead CUDA-event trace. Causal policy elapsed time is also
  excluded because captures, Python logging, and policy ordering perturb it.

The exact run inventory and 4-GPU wall accounting are in `GPU_TIME_LOG.csv`.
