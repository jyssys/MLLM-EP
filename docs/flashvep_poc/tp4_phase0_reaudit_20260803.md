# Debate Note: TP4/EP4 Phase 0 Re-audit

Date: 2026-08-03 (Asia/Seoul)

## Repository

- Current directory and archive root: `/home/esjung/MLLM-EP`.
- It is not a Git repository. Repository root, commit, branch, and working-tree
  status are unavailable and were not inferred.
- The original root named in `PROJECT_BACKUP_SUMMARY.md`,
  `/home/work/euisoo.jung/mllm-moe-ep`, is absent.
- The archive contains the entry scripts, prior profiling code/reports, and
  small outputs, but not its original Git metadata or environment.

## Environment

- Host: `cloud-0n58xq`; eight NVIDIA H100 80GB HBM3 GPUs; NV18 links between
  GPU pairs; one NUMA node.
- This experiment exposes only physical GPUs 4,5,6,7. Logical ranks 0-3 map
  in that order. GPU 0 remains occupied by an unrelated process.
- Python environment: `/home/esjung/anaconda3/envs/flashvep-poc`, Python
  3.12.13.
- vLLM 0.20.0+cu129, PyTorch 2.11.0+cu129, CUDA runtime 12.9, NCCL 2.28.9,
  Triton 3.6.0, Transformers 5.14.1, flashinfer 0.6.8.post1. vLLM uses
  FlashAttention 3; the standalone `flash-attn` package is absent.
- Driver 570.211.01; `nvidia-smi` reports CUDA 12.8; Nsight Systems 2024.6.2.

## Model And Parallelism

- Identifier: `Qwen/Qwen3-VL-30B-A3B-Instruct`.
- Exact local snapshot: `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`,
  13 safetensor shards, 57.87 GiB.
- Architecture: 48 decoder layers, hidden size 2048, 32 attention heads,
  4 KV heads, 128 routed experts, top-k 8, expert intermediate size 768.
- Current setting: BF16, TP=4, effective EP=4, DP=1, PP=1, linear placement,
  32 local experts per rank, expert weight filtering enabled.

## Smoke

The auto-selected FlashInfer CUTLASS MoE path did not complete startup; see
`tp4_backend_blocker_20260803.md`. Explicit Triton completed the unchanged
224x224 smoke request on GPUs 4-7:

- 79 prompt tokens, including 64 visual tokens;
- greedy output token 1986, text `This`;
- routed expert tensor `[79,48,8]`, IDs 0-127;
- approximately 18.95 GiB used on each selected GPU.

The prior TP7 command and evidence remain untouched. The current runnable
baseline is a new script, `poc_flashvep/baseline_command_tp4.sh`, so old result
files and commands were not overwritten.

## Phase 0 Conclusion

TP4/EP4 is runnable only with the explicit Triton backend in the tested stack.
This satisfies the configuration/smoke prerequisite for Phase 1, with the
backend substitution and missing dispatch collective carried forward as
blockers rather than hidden assumptions.

