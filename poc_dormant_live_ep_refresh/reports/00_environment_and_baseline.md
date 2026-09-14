# Environment and strongest baseline

## Hardware and isolation

Every task-owned launch used `CUDA_VISIBLE_DEVICES=4,5,6,7`.  Runtime logical
devices 0--3 therefore map to physical devices 4--7, in that order.  The four
devices are H100 80GB GPUs with UUIDs:

| physical | UUID |
|---:|---|
| 4 | `GPU-6076e2f2-5b63-3761-5586-56ceb7df8139` |
| 5 | `GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730` |
| 6 | `GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28` |
| 7 | `GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b` |

All six pairs report NV18.  The task-owned utilization process was stopped
before measurement and no non-owned process was terminated.  GPU 0--3 were not
used.

## Model/runtime truth

- Checkpoint: `/home/esjung/models/LLaDA2.0-flash-744c3f8`.
- Model: BF16, 32 layers, hidden size 4096, 256 routed experts, top-8, one
  shared expert, MoE intermediate size 1024.
- Runtime: dense TP4, routed EP4, DP1; 64 contiguous routed experts per rank.
- Sparse path: DeepEP normal dispatch, owner-rank Triton fused experts, reverse
  DeepEP combine.  The trace contains nonzero remote assignments on every rank.
- Workload: submitted batch 32, `mini_batch_size=32`, generation/block length
  32, threshold 0.9, config 42, deterministic decoding, CUDA graphs disabled
  for the DeepEP-normal path.

The preceding five-restart static sweep established mini32 as the strongest
feasible EP4 point.  This campaign did not compare against mini4 or another weak
default.

The causal/trace instrumentation lives in an isolated dInfer worktree at local
commit `2fe1d59`; its portable format-patch is included with this report.

## Clean request anchor

| task | independent restarts (s) | median (s) | NFE |
|---|---|---:|---:|
| GSM8K bounded-32 | 8.839, 5.833, 5.925 | 5.925 | 66 |
| HumanEval bounded-32 | 7.261, 7.193, 7.367 | 7.261 | 86 |

The first GSM8K launch is a cold outlier and remains visible rather than being
silently deleted; the median is the primary statistic.  Observer-heavy timing,
shape, and causal runs took 22--39 seconds and are never used as E2E speedup
evidence.  Component mass is normalized to an identical-substrate low-overhead
trace: router+dispatch+expert+combine is 54.37%/56.76% of clean E2E on
GSM8K/HumanEval; the primary fresh-router candidate can remove at most the
dispatch+expert+combine portion, 47.94%/49.54%.

DeepEP warns that this installation uses 20 communication SMs and a default
local E=64 fused-expert configuration.  These are part of the already selected
best-static substrate, not candidate-specific penalties.  A previous matched
run measured roughly 76--79GB peak HBM per rank; this campaign avoided an
observer in the clean path and therefore does not claim a fresh memory delta.
