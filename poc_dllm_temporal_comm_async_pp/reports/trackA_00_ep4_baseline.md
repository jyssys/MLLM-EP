# Track A0 — true-EP4 baseline

## Runtime truth

- Project base: `jyssys/MLLM-EP` commit
  `fee91a74d353c4f3a9efc17f1be9d045d777bdd8`.
- dInfer substrate base: commit
  `d5b8e66074ce6367ab8a66b3c2b85bc8c3e272e9`.
- Model: local `inclusionAI/LLaDA2.0-flash` revision directory
  `LLaDA2.0-flash-744c3f8`, BF16, 32 layers, H=4096, 256 routed
  experts, top-8, one shared expert, MoE intermediate 1024.
- Software: PyTorch 2.8.0+cu128, NCCL 2.27.3, Transformers 4.57.0,
  SGLang 0.5.3.post1.
- Physical GPUs were exactly 4/5/6/7.  Dense collectives use TP4; routed
  experts use EP4 with 64 experts/rank, DeepEP dispatch, owner-rank fused
  expert execution, and reverse combine.  The runtime logs independently
  report EP size 4 on all ranks.

The strongest previously selected static setting was reproduced: submitted
batch 32, mini-batch 32, generation 32, block length 32, threshold 0.9,
configuration 42, DeepEP normal/HT path, CUDA graph off.

## Clean request results

| Task | Restarts (s) | Median (s) | NFE | Throughput, requests/s |
|---|---:|---:|---:|---:|
| GSM8K-32 | 5.932, 6.711, 5.932 | **5.932** | 66 | 5.394 |
| HumanEval-32 | 11.481, 7.214, 7.361 | **7.361** | 86 | 4.347 |

The first HumanEval restart is a cold outlier; the median is retained.  Clean
runs contain no tensor capture.  The model occupies roughly 70 GiB/rank in the
normal full-model topology; observer policies reached about 75 GiB/rank and
are excluded from clean timing.

Fresh payload calibration spans 4--4096 global physical token rows (32--32768
top-k assignments).  DeepEP p50 dispatch ranges from 0.100 to 0.255 ms and
combine from 0.153 to 0.264 ms per MoE invocation.  This flat curve is the key
economic fact for Track A: the path is mostly startup/shape dominated at these
messages.

Evidence: `logs/clean_*`, `trackA/deepep_payload_sweep/summary.csv`, and the
fresh topology records under the result root.
