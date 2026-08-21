# Routing-preserving ragged Grouped GEMM PoC

This directory contains the bounded SonicMoE/QuACK and Qwen3 EP experiments
for the routing-preserving tail-kernel question.  The external SonicMoE checkout
is intentionally kept at `/home/esjung/external/sonic-moe` and is never
vendored here.

The benchmark fixes routing assignments, weights, dtype, GPU, total assignment
count (`N`), and active expert count (`G`) within each causal comparison.  Only
the per-expert histogram changes.  GPU commands must be launched with physical
GPUs 4--7 explicitly selected; the driver scripts refuse other visibility.

```bash
CUDA_VISIBLE_DEVICES=4 python benchmark_sonic.py \
  --output results/sonic_ep_poc_<run_id>/synthetic.json
```

`run_poc.sh` records the environment, runs the synthetic experiment, and then
invokes the analysis.  Live Qwen3 capture is a separate stage because it needs
the 61 GB BF16 checkpoint and four GPUs.
