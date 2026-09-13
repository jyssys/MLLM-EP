# Environment and substrate audit

Snapshot: 2026-09-13 KST.

## Repository isolation

- Project worktree: `/home/esjung/MLLM-EP-exact-overlap`
- Branch: `flashvep/dllm-moe-ep-exact-overlap-discovery-poc`
- Base commit: `0f9298b3173e53d812a9e436db5edd816f853abb`
- Runtime worktree: `/home/esjung/external/dinfer-llada2-exact-overlap`
- Runtime base commit: `8c9561f5badf185b0ddcf38fc4753e3b2a49af88`
- Model revision: `/home/esjung/models/LLaDA2.0-flash-744c3f8`
- Python environment: `/home/esjung/.venvs/llada2-flash-sglang-053`
- Torch/CUDA/SGLang: 2.8.0+cu128 / 12.8 / 0.5.3.post1

The user's dirty primary checkout is not modified.

## Physical GPU contract

Only physical GPU 0--3 are task-scoped. The mapping at the audit point was:

| physical | UUID | device | memory |
|---:|---|---|---:|
| 0 | `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d` | H100 80GB HBM3 | 81,559 MiB |
| 1 | `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb` | H100 80GB HBM3 | 81,559 MiB |
| 2 | `GPU-24200107-8a7f-de46-1bc8-b81f8d3af13e` | H100 80GB HBM3 | 81,559 MiB |
| 3 | `GPU-17488c15-2d4c-5d9e-d503-29b0d959a8a8` | H100 80GB HBM3 | 81,559 MiB |

Every pair among GPUs 0--3 is `NV18` in `nvidia-smi topo -m`. The initial
occupants were verified as the task-owned
`/home/esjung/vllm-ep/utilize.py --gpus 0,1,2,3` tree (parent PID 990612), not
an unknown user's workload. It is stopped only immediately before measurement.

## Validated execution contract

- Dense path: TP4, replicated logical token matrix.
- Routed path: true EP4, 256 routed experts, 64 contiguous experts per rank.
- DeepEP normal dispatch sends routed rows to owner ranks.
- The owner executes the resident BF16 Triton fused expert.
- DeepEP reverse combine returns routed contributions to source rows.
- The shared expert is rank-local and exact.
- A final TP-group all-gather reconstructs the replicated dense-layer input.

The overlap probe replays real captured inputs and resident weights only after
the normal layer returns; it does not alter generation output.

## Profiler boundary

Nsight Systems 2024.6.2 is installed, but GPU hardware-metric collection is
blocked by `ERR_NVGPUCTRPERM`; Nsight Compute is not installed. Quantitative
TensorCore/HBM/NVLink counters are therefore marked unavailable rather than
estimated. Resource classification uses (1) the kernels and tensor movement in
source, (2) same-rank CUDA-event durations, and (3) observed slowdown under real
concurrent execution. This limitation does not affect dependency or elapsed-time
measurements, but it prevents a counter-level utilization claim.

## Measurement isolation

- Clean request wall is measured without diagnostic replay.
- Pairwise replay uses five warmups and 30 measured repetitions per captured
  state, same-rank CUDA events, and an offline maximum across ranks.
- No cross-rank absolute CUDA timestamps are subtracted.
- The observer-heavy generation wall is never used as a speedup claim.
- GPU 4--7 were neither selected nor controlled by this task.
