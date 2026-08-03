# Repository Audit

## TP4/EP4 Re-audit (2026-08-03 14:57 KST)

The user subsequently authorized physical GPUs 4-7 only and TP=4/effective
EP=4. The archive and Git facts below are unchanged: `/home/esjung/MLLM-EP`
is still not a Git repository, so no commit or working-tree status exists.

The exact 13-shard, 57.87 GiB checkpoint is now present at
`/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`.
TP4 is valid for 32 attention heads and maps 128 experts evenly to 32 experts
per rank. Logical ranks 0-3 map to physical GPUs 4-7. Current runtime versions
remain those captured in `env_snapshot.txt`.

The observed current flow is:

```text
profile_tp4.py -> fixed multimodal request -> Qwen3-VL vision encoder
  -> 48 Qwen3-MoE decoder layers
       residual/RMSNorm
       QKV -> Q/K RMSNorm + RoPE -> FlashAttention 3 -> output projection
       residual/RMSNorm
       internal router, top-k 8
       NoDPEP local prepare (no dispatch collective)
       Triton local experts (32 experts/rank)
       local finalize -> TP all-reduce combine
  -> greedy max_tokens=1 -> routed expert capture and JSON
```

Automatic backend selection chooses FlashInfer CUTLASS Unquantized and stalls
in the first dummy forward. Explicit Triton is the only completed TP4 backend
tested and warns that the installed package lacks a tuned H100 config for
`E=32,N=768`. The baseline command records that choice rather than hiding it.

The new opt-in instrumentation is isolated in
`poc_flashvep/flashvep/instrumentation.py` and activated through
`FLASHVEP_PROFILE_JSONL`. Project `sitecustomize.py` only installs it when that
variable is present and PyTorch is importable. Existing baseline defaults are
otherwise unchanged. Stage hooks target the current installed vLLM classes;
no installed package or checkpoint was edited.

The representative processor output exactly identifies visual token indices
4-787, vision boundaries at 3 and 788, and 15 text/special positions. This is
current metadata, not an inference from the archived TP8 result.

The earlier seven-GPU audit is preserved below as provenance.

Audit date: 2026-08-03 (Asia/Seoul)

## Location And Git State

- Current working directory/archive root: `/home/esjung/MLLM-EP`.
- `git rev-parse --show-toplevel` and `git status`: failed with
  `not a git repository`.
- No project `.git` directory exists in the archive or the searched `/home`
  project paths. An unrelated Codex plugin-cache repository was ignored.
- Original repository path from `PROJECT_BACKUP_SUMMARY.md`:
  `/home/work/euisoo.jung/mllm-moe-ep` (absent).
- Git root, commit, branch, and working-tree status are unavailable. They are
  not inferred.
- Root README is absent. The audit used `PROJECT_BACKUP_SUMMARY.md`, the full
  `docs/flashvep_poc_spec.md`, archived reports, scripts, and preserved results.

## Archive Scope

Present: research Python code, tests, launch/analysis scripts, reports/specs,
small output artifacts, and selected external source snapshots.

Absent by backup policy: approximately 58 GB of Qwen weights, approximately
15 GB of data/Hugging Face cache, the original Python environment, and Git
metadata.

## Baseline Execution Flow

```text
scripts/vllm_ep_sanity.py
  -> AutoProcessor for Qwen3-VL-30B-A3B-Instruct
  -> fixed 224x224 gray image + fixed prompt
  -> vLLM.LLM(BF16, TP=8, EP enabled, static linear placement)
  -> multimodal preprocessing and vision encoder
  -> 48 Qwen3-MoE decoder layers
       input RMSNorm
       QKV projection -> Q/K norm + RoPE -> attention -> output projection
       post-attention residual + RMSNorm
       sparse MoE router -> EP dispatch -> local experts -> EP combine
  -> greedy generation with max_tokens=1
  -> prompt ids, output token, routed expert ids, GPU memory JSON
```

The historical output records 79 prompt tokens, 64 image tokens, routed shape
`[79, 48, 8]`, expert IDs 0-127, and about 11.8 GiB per GPU. These are prior
TP=8 results, not a current measurement.

## Model And Parallel Configuration

- Identifier: `Qwen/Qwen3-VL-30B-A3B-Instruct`.
- Historical local path: `models/Qwen3-VL-30B-A3B-Instruct` (absent).
- Current config fetch, without weights: architecture
  `Qwen3VLMoeForConditionalGeneration`, model type `qwen3_vl_moe`, 48 layers,
  hidden size 2048, 32 attention heads, 4 KV heads, 128 experts, top-k 8,
  expert intermediate size 768.
- Historical parallelism: TP=8, effective EP=8, DP=1, PP=1. BF16, linear
  expert placement, `allgather_reducescatter`, routed-expert capture on, EPLB
  off, and EP weight filtering on.
- Current requested policy: use all and only physical GPUs 1-7.

## Seven-GPU Compatibility Gate

The actual vLLM 0.20.0 source and model-config validator were inspected.

- TP=7 fails because 32 attention heads are not divisible by 7. This was
  reproduced through the existing baseline entry point before weight loading.
- vLLM derives MoE EP size from TP x DP x PCP when EP is enabled; PP is not
  part of the EP group.
- Uneven 128/7 expert ownership itself is supported by the expert-map helper
  (19 experts on two ranks and 18 on five), so it is not reported as the
  primary blocker.
- DP=7/EP=7 offline inference partitions requests by DP rank and uses
  placeholder prompts for ranks without local work. A single real vision
  request therefore does not remain the exact batch-size-1 baseline.
- PCP=7 is rejected because the installed attention implementations do not
  advertise PCP support.
- PP=7 does not create EP=7; EP remains 1 when TP=DP=PCP=1.

Seven is prime, so there is no other seven-device factorization in this stack.
Changing to TP=4 or using GPU 0 for TP=8 would be a materially different user
decision and was not assumed.

## Existing Profiling Code And Reuse Boundary

- `scripts/vllm_phase2a_profile.py` performs fixed benchmark selection,
  prefill-oriented generation with `max_tokens=1`, captures routed expert IDs,
  and aggregates expert/rank assignments. Its expert-to-rank mapping assumes
  the historical eight-rank layout.
- It classifies exact prompt token IDs 151655 and 151656 as image and video
  placeholders. The downloaded current model config confirms those IDs, but a
  current prompt was never tokenized because baseline validation failed.
- `vllm_moe_timing.py` opt-in wraps `FusedMoE.forward` in CUDA events and emits
  layer/rank intervals. It measures a whole fused MoE call; it cannot separate
  router, dispatch, local expert execution, or combine, and it does not measure
  the whole decoder-layer wall-clock critical path.
- `scripts/summarize_moe_cuda_timing.py` aggregates those event records.
- `scripts/vllm_phase2b1_compare.py` has warm-up and wall timing, but also
  changes expert placement and removes selected target timing/audit files. It
  is not safe as an unchanged Phase 1 baseline driver.

## Installed vLLM 0.20.0 Stage Locations

All paths below are under
`/home/esjung/anaconda3/envs/flashvep-poc/lib/python3.12/site-packages/vllm/`.

- QKV projection, attention core, output projection, residual/RMSNorm and MoE
  call: `model_executor/models/qwen3_moe.py`.
- Fused MoE orchestration and internal gate/router:
  `model_executor/layers/fused_moe/runner/moe_runner.py`.
- `allgather_reducescatter` prepare/finalize boundaries:
  `model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py`.
- Concrete all-gather dispatch and reduce-scatter combine:
  `distributed/device_communicators/all2all.py`, class
  `AgRsAll2AllManager`.
- EP group wrappers: `distributed/parallel_state.py`.

The configured backend name is historical vLLM terminology: this path uses
all-gather for dispatch and reduce-scatter for combine, not a literal pair of
token-routed All-to-All calls. The exact local expert kernel backend is runtime
selected. Since the model never loaded, no backend was observed and none is
claimed.

These call sites would permit opt-in NVTX/CUDA-event boundaries after a valid
baseline exists. Patching them before a runnable baseline would produce
untestable, backend-dependent instrumentation, so no installed package or
existing project file was modified.

## Input And Vision Metadata

- The minimal smoke input is generated locally: one fixed 224x224 gray image
  plus fixed text, so it does not require the absent benchmark datasets.
- Benchmark inputs expected by the archived profiling script are absent.
- Current config confirms `vision_start_token_id=151652`,
  `vision_end_token_id=151653`, `image_token_id=151655`, and
  `video_token_id=151656`.
- Exact current request vision/text token counts and index ranges are not known:
  config validation failed before tokenization/inference. Historical counts
  are preserved but not substituted.

## Baseline Preservation

All archived `scripts/`, root Python files, `outputs/`, weights, and checkpoint
references remain untouched. New files are isolated under `poc_flashvep/` and
`docs/flashvep_poc/`. New result names do not overlap old results. Every CUDA
command used `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7`.
