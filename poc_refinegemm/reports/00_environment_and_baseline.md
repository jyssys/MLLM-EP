# Environment and baseline

## Frozen substrate

- Date: 2026-09-15 KST.
- Project branch: `flashvep/refinegemm-refinement-adaptive-expert-execution-poc`,
  base `6d11373e04973f409564441b49e8ff0f06c3d431`.
- dInfer worktree: commit `9132ce9b2580ac2e64bcaaf975b863a5005c7739`.
- DeepEP source: commit `73b6ea4a439ba03a695563f9fd242c8e4b02b37c`.
- Model: `/home/esjung/models/LLaDA2.0-flash-744c3f8`; config SHA-256
  `ac35e9dc8313f49b2600da2a8b3ec31c53c01d22a5943ec26c8e21f8e208ab07`.
- Config: BF16, hidden 4096, 32 layers, 256 routed experts, top-8, one shared
  expert, routed intermediate 1024, eight routing groups/top-four groups.
- Software: PyTorch 2.8.0+cu128, CUDA 12.8, Triton 3.4.0, vLLM 0.10.2,
  SGLang 0.5.3.post1, Transformers 4.57.0.

The validated execution is dense TP4 plus routed EP4, DP1: each rank owns 64
complete routed experts, DeepEP normal dispatches remote branches to the owner,
the SGLang/vLLM BF16 fused expert executes owner-local packed rows, and DeepEP
reverse-combines them. The shared expert is replicated. The route traces have
non-zero remote traffic and exact owner-local expert ranges, so this is not TP4
mislabelled as EP4.

## GPU boundary

All new GPU benchmarks and the compatibility clean run used
`CUDA_VISIBLE_DEVICES=4,5,6,7`. Logical CUDA device 0 was checked against the
physical GPU 4 UUID before every kernel process.

| physical | UUID | device | peer topology |
|---:|---|---|---|
| 4 | `GPU-6076e2f2-5b63-3761-5586-56ceb7df8139` | H100 80GB HBM3 | NV18 |
| 5 | `GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730` | H100 80GB HBM3 | NV18 |
| 6 | `GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28` | H100 80GB HBM3 | NV18 |
| 7 | `GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b` | H100 80GB HBM3 | NV18 |

GPU 0--3 were never launched or stopped by this PoC. New kernel measurements
used physical GPU 4 because expert execution is owner-local; GPUs 5--7 are
needed only for the EP4 clean compatibility check. The six clean kernel series
used three independent processes, warmup 5 and 30 randomized repetitions per
backend/case. [GPU_TIME_LOG.csv](../GPU_TIME_LOG.csv) records their UUIDs.

## Strongest request baseline

The frozen full-model setting is submitted batch 32, mini-batch 32, generation
32, block 32, threshold 0.9 and config 42. The preceding five-run sweep selected
mini32 over mini16; no weak default is used. The prior three clean restarts were:

| task | request wall median | range | NFE | bounded score | prior peak HBM/rank |
|---|---:|---:|---:|---:|---:|
| GSM8K | 6.075 s | 5.794--6.715 s | 66 | 5/32 | about 57.0 GiB |
| HumanEval | 7.343 s | 7.211--7.721 s | 86 | 6/32 | about 57.0 GiB |

The fresh GPU4--7 compatibility output is retained under
`results/clean_gpu4_7/`. It completed with NFE66 and all 32 answer fields match
the prior anchor exactly. Its 9.060 s forward timing is excluded from baseline
statistics because it is a single warm-state compatibility run; the three prior
matched restarts remain the statistical request baseline. The launch-to-exit
sampler observed up to 75,327 MiB (73.6 GiB) including weight loading and runtime
initialization; it is not substituted for the prior steady-state HBM statistic.

## Evidence boundary

Clean runs support request latency/NFE/output claims. Route and CUDA-event
traces are observer-heavy diagnostics. The post-compaction corpus retains only
future-known live rows and is an Epoch-like sensitivity, not measured Epoch.
Owner-local replay excludes DeepEP dispatch/combine and therefore supports
expert-operator claims, not measured request speedup.
