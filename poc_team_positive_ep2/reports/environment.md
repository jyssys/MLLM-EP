# Environment and fidelity audit

## Reproducible pins

| Component | Pin / value |
|---|---|
| Project base HEAD | `f0dc8372ec0d1d2e10347c82e630b84430c5dc5f` |
| TEAM repository | `e9c502e5753ce79f660371e2fb4a8666f66cae75` |
| SDAR repository | `4c2749ba103448f45520e8411533710a1e66574d` |
| SDAR checkpoint revision | `c351bbc37d240aa6871f167e8f92d694281b0c22` |
| Python | 3.11.15 |
| PyTorch / CUDA / NCCL | 2.8.0+cu128 / 12.8 / 2.27.3 |
| Transformers | 4.52.4 |
| dtype | FP16, matching released TEAM configuration |
| GPU 6 | H100 80GB, `GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28` |
| GPU 7 | H100 80GB, `GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b` |
| Driver | 570.211.01 |
| GPU 6↔7 topology | NV18 NVLink |

The checkpoint is 57 GiB on disk and contains 128 routed experts, top-8
routing, 48 MoE decoder layers, hidden size 2048, and expert intermediate size
768.  Single-GPU peak allocation was approximately 61.8 GB, so no quantization,
CPU offload, or multi-GPU adaptation was needed for Stage A.

## Fidelity boundary

The released normal baseline/TEAM modeling files and released generation
function bodies were used.  The repository's bundled OpenCompass tree cannot
be imported cleanly in the isolated modern environment because unrelated
optional backend requirements are absent.  The harness therefore extracts the
unchanged released generation helpers by AST and evaluates the same checkpoint,
prompts, FP16 dtype, block length 32, 32 denoising steps, threshold 0.95, and
greedy top-k 1.

This is a faithful bounded positive-control reproduction, not a full four-task
official leaderboard reproduction.  It covers four GSM8K and four HumanEval
samples, two 256-token truncation controls, and independent two-request
restarts.  MATH and MBPP were not run after the qualitative gate was decisively
passed; GPU time was redirected to true EP2 and residual profiling.

The official TEAM repository reports 1.94x average and up to 2.2x on
HumanEval.^1 The local claim is deliberately narrower: a median 1.832x clean
speedup across three bounded restart pairs, with work reduction in the same
direction.

## GPU safety

All experiment runners hard-fail unless Stage A sees exactly physical GPU 6 or
Stage B sees exactly `CUDA_VISIBLE_DEVICES=6,7`.  The recorded UUIDs are checked
inside the runner.  No task experiment exposed GPUs 0–5.  Utilization runs are
excluded from all experiment-time and performance statistics.

## Source

1. PKU-SEC-Lab, “[TEAM-MoE-dLLM official repository](https://github.com/PKU-SEC-Lab/TEAM-MoE-dLLM),” accessed September 11, 2026.
