# FlashVEP Debate Note: Phase 0 Audit

> Current TP4/effective-EP4 re-audit:
> `docs/flashvep_poc/tp4_phase0_reaudit_20260803.md`. The remainder preserves
> the earlier TP7 blocker audit.

Date: 2026-08-03
Decision: **baseline blocked; do not advance automatically**

## Confirmed Facts

- `/home/esjung/MLLM-EP` is the lightweight archive described by
  `PROJECT_BACKUP_SUMMARY.md`, not a Git repository. Original Git metadata,
  model weights, benchmark data, and runtime were not backed up.
- An isolated Python 3.12 environment was restored with the official vLLM
  0.20.0 CUDA 12.9 wheel and PyTorch 2.11.0+cu129.
- The host has eight H100 80GB GPUs. Every PoC CUDA command exposed only
  physical GPUs 1-7; GPU 0 was never used.
- Current model config confirms 32 attention heads, 128 experts, top-k 8, and
  48 decoder layers.
- The archived baseline is TP=8/effective EP=8. The current required TP=7
  attempt fails before weights load because 32 is not divisible by 7.

## Reproduced Failure

```bash
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

Result: exit code 1, `Total number of attention heads (32) must be divisible
by tensor parallel size (7).` No GPU workers, weights, or output JSON were
created.

## Seven-GPU Interpretation

Uneven 128/7 expert ownership is supported, so expert divisibility is not the
blocker. The blocker is preserving a valid single-request outer parallel
layout. In installed vLLM 0.20, DP=7 changes request semantics, PCP=7 is
unsupported, and PP=7 does not produce EP=7. Because 7 is prime, there is no
other factorization in this runtime.

## Debate Position

Phase 0 is complete with a blocker. Phase 1 cannot generate trustworthy timing
until a valid baseline exists. Do not interpret archived TP=8 timing as a
current seven-GPU result and do not silently switch to four GPUs.

Decision options:

1. Re-authorize physical GPU 0 and restore TP=8/EP=8.
2. Approve a changed TP=4/EP=4 comparison on allowed GPUs.
3. Approve a separate exact-seven-rank runtime/parallelism investigation.

Evidence details: `poc_flashvep/repo_audit.md`,
`poc_flashvep/env_snapshot.txt`, and `poc_flashvep/STATUS.md`.
