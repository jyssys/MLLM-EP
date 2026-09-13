# Experiment log

## 2026-09-13 — setup

- Created isolated project branch and worktree from validated layer-sensitivity
  commit `0f9298b3173e53d812a9e436db5edd816f853abb`.
- Created isolated dInfer worktree from instrumentation commit
  `8c9561f5badf185b0ddcf38fc4753e3b2a49af88`.
- Confirmed the attached 550-line working contract by SHA-256.
- Confirmed the repository-owned utilization process on physical GPUs 0--3;
  it remains active during CPU-only audit and will be stopped before live
  measurement.
- 13:25--13:46 KST: audited the dInfer/SGLang/DeepEP execution contract. No
  historical measurement is relabeled as fresh evidence.
- Analytical screen from prior clean runs: shared-expert total ceiling is 2.98%
  GSM8K / 3.50% HumanEval. Dispatch (12.1--13.3%) and expert (29.5%) are the
  only individual MoE stages with enough clean-E2E mass for the 8% gate.
- Verified GPU UUID/NVLink topology and exact burn ownership. No task
  measurement was launched while burn remained active.
- Added read-only real-input overlap replay for full shared/attention units,
  local-vs-remote routing, and complete-wave dispatch/expert/combine pairs.
  Production generation output is untouched.

## 2026-09-13 — live measurements

- Stopped only the verified repository-owned burn on physical GPUs 0--3 and
  launched every task process with `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- Reproduced the strongest static EP4 substrate on 32 bounded requests with
  `mini_batch_size=32`, generation/block length 32, BF16, and threshold 0.9.
  Three independent clean restarts gave 5.845 s GSM8K and 7.262 s HumanEval
  medians. NFE was 66/86 and all answer hashes matched within each dataset.
- The first diagnostic smoke exposed misuse of DeepEP `EventOverlap`; changed
  the wait to its public `current_stream_wait()` interface and reran a passing
  smoke. The failure contributes to GPU accounting but not evidence.
- Captured real layer-16 tensors/routes/weights at GSM8K waves 0/30/60 and
  HumanEval waves 0/40/80. Each legal pair used 5 warmups and at least 30
  measured CUDA-event repetitions on every rank.
- `remote dispatch || local source expert` and `dispatch || attention` were
  consistently slower than serial execution. DeepEP communication consumed
  sufficient SM/memory-system resources to defeat the nominal complementarity.
- `dispatch(next) || expert(current) || combine(previous)` was the strongest
  exact diagnostic: 0.106--0.186 ms per sampled layer, but it requires three
  independent waves. Sample-median service upper bounds were 4.35% GSM8K and
  4.59% HumanEval; the deliberately extreme best-saving-at-every-wave bound was
  4.84%/6.73%. Neither is direct single-request latency.
- A first cross-state launch omitted the DeepEP capacity environment and was
  discarded. The corrected GSM8K/HumanEval runs measured all ordered pairs
  among three captured physical states. The best two-stage savings were 0.120
  and 0.109 ms, but no phase-conditioned result beat the generic three-stage
  pipeline.
- Every completed diagnostic preserved the production generation output and
  expert replay: minimum expert cosine 0.99999988, maximum relative L2 0.
- Total task-owned live GPU wall was 1,268.477 s including two diagnosed setup
  failures, equal to 1.4094 four-GPU hours. Passing evidence runs account for
  1,140.562 s / 1.2673 four-GPU hours.

## Decision

- Strongest credible direct-request exact oracle: 3.50%, the full HumanEval
  shared-expert contribution ceiling.
- Strongest independent-wave service upper: 4.59% using sampled medians and
  6.73% only under the extreme every-wave-best assumption.
- Refinement phase changed communication fraction modestly but did not create a
  demonstrated increment over generic overlap.
- Final label: `NO-OVERLAP-SIGNAL`.
