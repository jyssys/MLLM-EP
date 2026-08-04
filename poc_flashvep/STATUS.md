# FlashVEP PoC Status

Last updated: 2026-08-03 16:55 KST

## Current TP4 / Effective-EP4 Result

- User-authorized devices: physical GPUs `4,5,6,7` only; logical ranks map to
  those devices in order. GPU 0 and GPUs 1-3 were not exposed.
- Configuration: BF16, TP=4, effective EP=4, DP=1, PP=1, 32 local experts per
  rank, linear placement, fixed batch size 1, prefill, `max_tokens=1`.
- Exact checkpoint snapshot:
  `/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`.
- Phase 0: completed for TP4/EP4. The 224x224 smoke generated token 1986
  (`This`) and captured routes with shape `[79,48,8]`.
- Phase 1: completed with 5 warm-ups and 20 fixed-input measurements per pass.
- Phase 2: **not started** because the mandatory Phase 1 numeric gate failed.
  No tiling, replay, scheduler, kernel, or overlap implementation exists.

## Baseline And Input

The canonical current command is `poc_flashvep/baseline_command_tp4.sh`; the
old `baseline_command.sh` remains the unmodified TP7 failure reproducer. The
representative case is one deterministic 896x896 gray image plus the exact
text `Describe this image briefly.` Prefix caching is disabled.

- Total prompt tokens: 799.
- Exact image placeholder ID: 151655, derived from the current processor.
- Visual indices: contiguous `[4, 787]`, 784 tokens.
- Vision boundary indices: start ID 151652 at 3; end ID 151653 at 788.
- Text and special tokens: 15. Classification covers all 799 indices.
- All 5 warm-ups and 20 measured iterations produced token 1986 (`This`) and
  routed-expert shape `[799,48,8]`, IDs 0-127.

The non-profiled request wall time was median 75.881 ms, p90 87.258 ms, mean
77.752 ms, and standard deviation 4.568 ms.

## Runtime Path And Measurement Boundary

vLLM selected FlashAttention 3 for attention. Its automatic MoE selection
chose FlashInfer CUTLASS Unquantized, but TP4/EP4 stopped making progress in
the initial dummy language-model forward after weight loading. The same stop
persisted with CUDA graph capture off, multimodal startup profiling skipped,
custom all-reduce disabled, and NCCL fallback. Explicit
`moe_backend=triton` completed and is the measured baseline; this backend
change is explicit and is not presented as equivalent to the unusable auto
backend.

Installed vLLM reports `MoEPrepareAndFinalizeNoDPEPModular`. With TP4/EP4 and
DP1 each TP rank sees the full token sequence, owns 32 of 128 experts, and the
MoE result is combined by a tensor-parallel all-reduce. There is no dispatch
All-to-All in this layout. Therefore `T_dispatch=0` is a structural fact, not
an invented duration, and `T_combine` measures TP all-reduce rather than a
combine All-to-All.

The opt-in profiler is disabled unless `FLASHVEP_PROFILE_JSONL` is set. It
uses NVTX and CUDA events, records GPU-relative start/end times and durations,
and synchronizes pending events only during worker shutdown. Startup call
sequence was measured as two max-token dummy forwards plus one one-token dummy
forward; measured prefill calls therefore start at call index 8 after five
warm-ups.

## Phase 1 Results

Raw detailed coverage is complete: 20 iterations x 48 layers x 4 ranks x 14
stages = 53,760 JSONL records, with no error/null duration and exact 6,392
route assignments per rank/layer/request. A separate lean pass contains 7,680
`decoder_layer`/`router_topk` records.

Across all 960 iteration-layer samples in the internally consistent detailed
pass, medians are:

| Value | Median |
|---|---:|
| `T_layer` | 1.998 ms |
| `T_attention` | 0.908 ms |
| `T_norm_router` | 0.212 ms |
| `T_dispatch` | 0.000 ms, structurally absent |
| `T_expert_max` (max-rank fused 32-local-expert span) | 0.447 ms |
| `T_combine` | 0.808 ms |
| complete fused MoE call | 1.057 ms |
| expert fraction | 22.08% |
| exposed non-expert fraction | 56.17% |
| `T_optimistic` | 1.925 ms |
| oracle speedup | 1.035x |

For the per-request sum across all 48 layers, median `T_layer` is 96.930 ms,
median `T_optimistic` is 93.159 ms, median oracle speedup is 1.0349x, and p90
oracle speedup is 1.0785x. None of the 48 layers has a per-layer median oracle
speedup at or above 1.15x. Layer 0 has the highest median detailed `T_layer`
(2.444 ms). Logical rank 2 / physical GPU 6 is the most frequent layer
critical rank (297 of 960 samples); logical rank 3 / physical GPU 7 is the
most frequent expert critical rank (492 of 960 samples).

These are estimates from measured baseline components, not achieved overlap.
No first-tile latency was measured or estimated as an achieved result.
`T_expert_max` is not a per-expert latency: the Triton boundary fuses all 32
local experts on a rank. The individual slowest-expert duration is unavailable.

## Profiling Overhead

- Detailed 14-stage pass: median 121.152 ms, +59.66% over non-profiled.
- Lean two-stage pass: median 95.173 ms, +25.42% over non-profiled.

Mixing lean `T_layer` with detailed-pass components yields a nonphysical
0.792x ratio, so it is not used as the oracle result. The reported 1.0349x is
the scale-consistent detailed-pass optimistic bound; its already-high outer
layer overhead biases the numerator favorably, yet it still fails 1.15x.

## Phase 2 Gate And Decision

- Required oracle speedup >=1.15x: **FAIL** (1.0349x median; 1.0785x p90).
- Required exposed non-expert fraction >=15%: **PASS** (55.92% on the
  all-layer/request aggregate).
- Combined gate: **FAIL**.

Decision: **NO-GO / HOLD for Phase 2 on this TP4/EP4 stack.** The current
runtime also lacks the dispatch collective that FlashVEP is intended to hide.
Per the specification, no trace-driven tile simulator or offline replay was
implemented after the failed gate.

## Current Artifacts And Modified Files

- Modified minimally: `scripts/vllm_ep_sanity.py`, `sitecustomize.py`.
- Added profiler/analysis: `poc_flashvep/flashvep/instrumentation.py`,
  `poc_flashvep/scripts/profile_tp4.py`,
  `poc_flashvep/scripts/analyze_tp4_profile.py`.
- Added reproducers: `poc_flashvep/baseline_command_tp4.sh`,
  `poc_flashvep/scripts/run_tp4_phase1_profile.sh`.
- Raw results: `poc_flashvep/results/tp4_phase1_vision896/`.
- Stable result copies: `poc_flashvep/results/baseline/summary_tp4_ep4_vision896.csv`,
  `layer_breakdown_tp4_ep4_vision896.csv`, and
  `gate_tp4_ep4_vision896.json`. Existing TP7 placeholder/result files were
  not overwritten.
- Debate reports: `docs/flashvep_poc/tp4_backend_blocker_20260803.md` and
  `docs/flashvep_poc/tp4_phase1_gate_20260803.md`.

## TP4 Commands Actually Executed

The exact model value in every command was the local snapshot listed above.
All Python commands used
`/home/esjung/anaconda3/envs/flashvep-poc/bin/python`, offline model flags,
`VLLM_WORKER_MULTIPROC_METHOD=spawn`, `PYTHONPATH=.`, and
`CUDA_VISIBLE_DEVICES=4,5,6,7`.

```bash
# Minimum successful smoke (after auto-backend diagnosis)
python scripts/vllm_ep_sanity.py \
  --model-path "$MODEL" --tensor-parallel-size 4 \
  --kv-cache-memory-bytes 1073741824 \
  --max-model-len 512 --max-num-batched-tokens 512 --max-num-seqs 1 \
  --skip-mm-profiling --moe-backend triton \
  --output poc_flashvep/results/baseline/smoke_tp4_gpu4567_triton.json

# Non-profiled vision-heavy baseline
python poc_flashvep/scripts/profile_tp4.py \
  --model-path "$MODEL" \
  --output poc_flashvep/results/tp4_phase1_vision896/baseline_requests.json \
  --warmups 5 --iterations 20 --tensor-parallel-size 4 \
  --moe-backend triton --image-size 896 \
  --max-model-len 1024 --max-num-batched-tokens 1024

# Detailed pass: same command and settings, with
FLASHVEP_PROFILE_JSONL=poc_flashvep/results/tp4_phase1_vision896/stages.jsonl \
FLASHVEP_RUN_ID=tp4_phase1_vision896 \
FLASHVEP_SKIP_LAYER_CALLS=8 FLASHVEP_MEASURE_LAYER_CALLS=20 \
FLASHVEP_PHYSICAL_GPUS=4,5,6,7 \
python poc_flashvep/scripts/profile_tp4.py \
  --model-path "$MODEL" \
  --output poc_flashvep/results/tp4_phase1_vision896/profile_requests.json \
  --warmups 5 --iterations 20 --tensor-parallel-size 4 \
  --moe-backend triton --image-size 896 \
  --max-model-len 1024 --max-num-batched-tokens 1024

# Lean pass added FLASHVEP_PROFILE_STAGES=decoder_layer,router_topk and wrote
# lean_stages.jsonl / lean_requests.json with otherwise identical settings.

python poc_flashvep/scripts/analyze_tp4_profile.py \
  --stages poc_flashvep/results/tp4_phase1_vision896/stages.jsonl \
  --requests poc_flashvep/results/tp4_phase1_vision896/profile_requests.json \
  --output-dir poc_flashvep/results/tp4_phase1_vision896/detailed_only_analysis
```

The reusable wrapper `poc_flashvep/scripts/run_tp4_phase1_profile.sh` creates a
new timestamped directory and refuses to overwrite results.

## Remaining Blockers And Next Debate

1. Auto/strong FlashInfer CUTLASS MoE does not complete its initial TP4/EP4
   dummy forward; Triton has no tuned H100 config for E=32,N=768 and warns
   that its default may be suboptimal.
2. TP-derived EP with DP1 has no dispatch All-to-All, so it does not exercise
   the principal FlashVEP communication pipeline.
3. Fine-grained Python/CUDA-event annotation has material overhead. A future
   Phase 1 refinement should use lower-overhead CUPTI/Nsight collection or
   selected-layer passes before trusting smaller differences.
4. The archive has no Git metadata or original environment; current commit
   and worktree status remain unavailable.

Recommended next debate is not Phase 2 implementation. First decide whether
to (a) diagnose/fix the FlashInfer CUTLASS TP4 hang and repeat Phase 1, or (b)
use a DP-based EP layout that actually invokes dispatch/combine collectives,
with its changed request semantics explicitly accepted.

<details>
<summary>Archived TP7 Phase 0/1 record from earlier on 2026-08-03</summary>

## Scope Gate

- This session stopped after Phase 0 and the achievable part of Phase 1.
- No tile replay, tile execution/scheduling, live overlap, custom CUDA/Triton
  kernel, placement/replication, token merging, quantization, or fine-tuning was
  implemented.
- Physical GPU 0 was excluded from every PoC CUDA command. Only physical GPUs
  1-7 were made visible.

## Final State

- Phase 0: **completed with a structural baseline blocker**.
- Phase 1: **blocked before model loading; no latency measurement exists**.
- Phase 2 decision for the requested seven-GPU configuration: **NO-GO / HOLD**.
  The required speedup and exposed-time thresholds were not measurable and
  therefore were not demonstrated.

## Repository And Environment

- Working/archive root: `/home/esjung/MLLM-EP`.
- This directory is not a Git repository. Commit, branch, repository root, and
  working-tree status are unavailable. The original path recorded in the
  backup is `/home/work/euisoo.jung/mllm-moe-ep`, which is absent.
- Host: `cloud-0n58xq`, eight NVIDIA H100 80GB HBM3 GPUs, NV18 between every
  GPU pair. GPU 0 is occupied by an unrelated process and was not used.
- Isolated environment created at
  `/home/esjung/anaconda3/envs/flashvep-poc` with Python 3.12.13.
- Runtime: vLLM 0.20.0+cu129, PyTorch 2.11.0+cu129, CUDA runtime 12.9,
  NCCL 2.28.9, Triton 3.6.0, Transformers 5.14.1,
  flashinfer-python 0.6.8.post1. `flash-attn` is not installed. Driver is
  570.211.01; `nvidia-smi` reports CUDA 12.8. Nsight Systems is 2024.6.2.
- Model weights and benchmark data are absent. The Hugging Face model config
  was fetched without downloading weights.

## Model And Parallel Facts

- Model: `Qwen/Qwen3-VL-30B-A3B-Instruct`;
  `Qwen3VLMoeForConditionalGeneration`.
- Config: 48 decoder layers, 32 attention heads, 4 KV heads, 128 routed
  experts, top-k 8, hidden size 2048, expert intermediate size 768.
- Historical baseline: BF16, TP=8, EP enabled, DP=1, PP=1, linear placement,
  `allgather_reducescatter`, EPLB disabled.
- Requested seven-device attempt: physical GPUs 1-7, TP=7, EP enabled, DP=1,
  PP=1. vLLM rejected it because 32 attention heads are not divisible by TP=7.
- In vLLM 0.20, EP size is derived from TP x DP x PCP, not PP. The other
  seven-device factorizations do not preserve the requested exact single-
  request baseline: DP=7 distributes/duplicates request slots across ranks,
  PCP=7 is unsupported by the installed attention backend, and PP=7 leaves
  EP=1. No alternative layout was silently substituted.

## Baseline Command

The archived eight-GPU command is preserved verbatim as a comment in
`poc_flashvep/baseline_command.sh` but is not runnable under the GPU policy
because it exposes GPU 0. The executable portion reproduces the current
seven-GPU failure and writes only to a timestamped path under
`poc_flashvep/results/baseline/` if it ever reaches output generation.

## Commands Actually Executed

Key commands were:

```bash
pwd
git rev-parse --show-toplevel
git status --short --branch
find /home -maxdepth 5 -type d -name .git -prune -print
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version,pci.bus_id --format=csv,noheader
nvidia-smi topo -m
python3 --version
conda info --envs
nsys --version

conda create -y -n flashvep-poc python=3.12
conda run -n flashvep-poc python -m pip install uv
conda run -n flashvep-poc uv pip install vllm==0.20.0
conda run -n flashvep-poc uv pip install --reinstall --no-deps \
  'https://github.com/vllm-project/vllm/releases/download/v0.20.0/vllm-0.20.0%2Bcu129-cp38-abi3-manylinux_2_31_x86_64.whl'

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONPATH=. \
conda run -n flashvep-poc python scripts/vllm_ep_sanity.py \
  --model-path Qwen/Qwen3-VL-30B-A3B-Instruct \
  --tensor-parallel-size 7 \
  --kv-cache-memory-bytes 1073741824 \
  --max-model-len 512 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 1 \
  --output poc_flashvep/results/baseline/smoke_tp7.json
```

The first plain PyPI vLLM wheel resolved to a CUDA 13 binary and failed to
import due to missing `libcudart.so.13`; it was replaced with the official
v0.20.0 CUDA 12.9 wheel before the smoke attempt.

## Smoke Result

- Status: **failed at configuration validation**, exit code 1.
- Exact error: `Total number of attention heads (32) must be divisible by
  tensor parallel size (7).`
- The architecture/config was resolved, but weight loading and GPU worker
  launch did not start. No smoke output JSON was produced and no historical
  result was overwritten.
- GPUs 1-7 remained at 0 MiB used after the attempt. The process on GPU 0 was
  unrelated and pre-existing.

## Phase 1 Profiling Result

- Warm-ups completed: 0 of 5 required.
- Measured iterations: 0 of 20 required.
- Profiling overhead: not measurable; profiling was never activated.
- QKV, attention core, output projection, residual/RMSNorm, router, dispatch,
  local expert, combine, and whole MoE-layer latency: **not measured**.
- Layer/rank breakdown, critical layer/rank, expert load, local expert batches,
  and layer wall-clock critical path: **not measured**.
- Vision metadata: model special IDs were verified (`image_token_id=151655`,
  `video_token_id=151656`), but no current request ran, so current vision/text
  token counts and index ranges are unavailable.
- The archived TP=8 `FusedMoE.forward` timings are not reused as seven-GPU
  Phase 1 measurements. They do not split dispatch/expert/combine or provide
  full decoder-layer wall time.

## Headroom And Decision

`T_layer`, `T_attention`, `T_norm_router`, `T_dispatch`, `T_expert_max`,
`T_combine`, `T_optimistic`, expert fraction, exposed non-expert fraction, and
oracle speedup are all **not measured**. No zero, estimate, or inherited TP=8
number was inserted.

Consequently the Phase 2 gates (oracle layer speedup >=1.15x and exposed
non-expert fraction >=15% on a representative vision-heavy case) are not
satisfied by evidence. The current decision is **NO-GO / HOLD**, caused by an
invalid seven-GPU baseline rather than a measured lack of FlashVEP headroom.

## Files Created

- `poc_flashvep/README.md`
- `poc_flashvep/.gitignore`
- `poc_flashvep/STATUS.md`
- `poc_flashvep/repo_audit.md`
- `poc_flashvep/env_snapshot.txt`
- `poc_flashvep/baseline_command.sh`
- `poc_flashvep/scripts/run_baseline_profile.sh`
- `poc_flashvep/results/baseline/smoke_failure.json`
- `poc_flashvep/results/baseline/summary.csv`
- `poc_flashvep/results/baseline/layer_breakdown.csv`
- `poc_flashvep/reports/phase1_profile.md`
- `docs/flashvep_poc/phase0_audit.md`
- `docs/flashvep_poc/phase1_profile.md`

No archived script, model/checkpoint, or existing output file was modified.

## Blockers And Debate Options

1. Re-authorize GPU 0 and restore the historical TP=8/EP=8 baseline.
2. Explicitly approve a changed TP=4/EP=4 experiment on an allowed subset of
   GPUs, accepting that it is not the seven-GPU baseline.
3. Approve a separate runtime/parallel-design investigation that can express
   exact seven-rank EP while preserving a single request.

Any option is a new decision. Phase 2 has not started.

</details>

## Phase 1b TP2/DP2/EP4 revalidation (2026-08-03)

- Scope stopped at Phase 1b. Phase 2 was not started.
- Physical GPUs 4-7 ran BF16 TP=2, DP=2, EP=4, PP=1 against the same exact
  checkpoint and fixed 896x896/799-token request. All 20 measured outputs were
  token 1986 (`This`).
- vLLM selected `MoEPrepareAndFinalizeNaiveDPEPModular`,
  `AgRsAll2AllManager`, sequence-parallel MoE, and `TritonExperts` under
  `moe_backend=auto`. Every rank owns 32 experts.
- Actual DPEP was proven on all four ranks. Dispatch all-gatherv used
  `ncclDevKernel_Broadcast_RING_LL`; combine reduce-scatterv used
  `ncclDevKernel_Reduce_Sum_bf16_RING_LL`; final TP sequence combine used
  `ncclDevKernel_AllGather_RING_LL`.
- The single real request is on DP0. DP1 participates through vLLM's native
  `START_DP_WAVE` idle path. Per layer, the real request contributes 6,392
  assignments, TP padding contributes 8, and idle DP dummy tokens contribute
  16; all are separately accounted.
- Selected layers `[0,12,24,36,47]` produced 4,800 complete CUDA-event stage
  records with no null/error duration. Aggregate medians were
  `T_layer=2.613 ms`, `T_attention=0.824 ms`, `T_norm_router=0.188 ms`,
  `T_dispatch=1.663 ms`, `T_expert_max=0.449 ms`,
  `T_combine drain=0.773 ms`, and `T_full_moe=2.385 ms`.
- The prior 0.447 ms expert result is plausible but measures only the fused
  local-expert kernel boundary. Current in-path `T_expert_max` is 0.4487 ms
  median; the warm isolated layer-24 microbenchmark is 0.2473-0.2604 ms across
  ranks.
- Same-period no-profiler/profile medians were 2,413.84/2,627.68 ms, for
  +8.86% selected-layer profiling overhead. An earlier 3,825.62 ms baseline
  under changing concurrent system load is preserved but not used as the
  overhead denominator.
- Timestamp oracle retains a nonzero 0.1014 ms first-tile fill and the complete
  DPEP-combine-through-TP-all-gather drain. The five-layer sum is 14.3296 ms
  current versus 13.3218 ms optimistic: **1.07465x median**, 1.08558x p90,
  and 1.09963x maximum.
- Final Phase 1b gate: **NO-GO**. Median oracle is below 1.10x, no selected
  layer median reaches 1.15x, and the combine drain is longer than the expert
  window. No FlashVEP tiling/replay/scheduler/kernel was implemented.
- Single recommended next task: study and reduce the AgRs DPEP dispatch/combine
  cost before reconsidering FlashVEP Phase 2.

Canonical Phase 1b artifacts:

- `poc_flashvep/results/phase1b_tp2dp2_vision896/`
- `poc_flashvep/results/phase1b_tp2dp2_nsys_224/`
- `poc_flashvep/reports/phase1b_tp2dp2_profile.md`
- `poc_flashvep/reports/phase1b_expert_timing_audit.md`
- `poc_flashvep/results/baseline/gate_phase1b_tp2dp2_vision896.json`
- `poc_flashvep/scripts/run_phase1b_tp2dp2.sh`

## Batch 16/32 Quick PoC (2026-08-04)

- Reused BF16 TP=2/DP=2/EP=4 on physical GPUs 4-7 with the fixed
  896x896/799-token prompt. Batch 16 split 8/8 and Batch 32 split 16/16 across
  DP0/DP1; every output remained token 1986.
- Both batches executed `MoEPrepareAndFinalizeNaiveDPEPModular` through
  `AgRsAll2AllManager`, with real dispatch all-gatherv and combine
  reduce-scatterv collectives on all four ranks.
- Valid Batch 16 medians over layers 12/24/36 were `T_layer=9.617 ms`,
  `T_dispatch=3.494 ms`, `T_expert_max=1.995 ms`, and combine drain
  `2.791 ms`. Expert fraction was 21.55%, communication/expert fell to 3.116x,
  and the selected-layer oracle was 1.175x (extended 1.355x).
- Batch 16 profiler overhead was 13.56%. Batch 32 ran without OOM, but its
  61.81% overhead crossed the immediate-stop gate; its stage values are kept
  only as diagnostics and no alternate Batch 32 run was attempted.
- Final Quick PoC gate: **HOLD**. Batch 16 is in the 20-25% expert-fraction
  band and the oracle gain has little margin over profiling uncertainty;
  Batch 32 cannot resolve the decision until measured below 20% overhead.
- Phase 2A and live overlap were not started. The single next task is one
  same-configuration, active-offset-only Batch 32 revalidation with overhead
  below 20%.

Canonical Quick PoC artifacts:

- `poc_flashvep/results/batch16_32_quick_poc_20260804_131743/`
- `poc_flashvep/reports/batch16_32_quick_poc.md`
- `poc_flashvep/results/baseline/gate_batch16_32_quick_poc.json`
- `poc_flashvep/scripts/run_batch16_32_quick_poc.sh`
- `poc_flashvep/scripts/analyze_batch16_32_quick_poc.py`
- `docs/prompt/debate_agent_flashvep_batch16_32_review_prompt_ko.txt`
