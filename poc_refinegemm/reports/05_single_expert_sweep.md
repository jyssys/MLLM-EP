# Single-expert and homogeneous sweep

The exact BF16 contract uses `X[M_e,4096]`, gate/up weights
`[2*1024,4096]`, SwiGLU and down weights `[4096,1024]`.

An isolated one-expert call strongly favors PyTorch grouped in this installation
at every M_e, but that E=1 setup is not representative of a 64-expert owner
invocation. The decisive homogeneous 64-expert control shows the actual
crossover:

| M_e on every local expert | vLLM fused (ms) | torch grouped (ms) | grouped gain |
|---:|---:|---:|---:|
| 1 | **0.698** | 0.736 | -5.17% |
| 2 | 0.854 | **0.707** | 17.03% |
| 4 | 0.858 | **0.707** | 17.57% |
| 16 | 0.915 | **0.715** | 21.67% |
| 64 | 1.128 | **0.787** | 30.25% |
| 256 | 2.188 | **1.091** | 50.16% |

Thus there is a different tiny winner only at exactly `M_e=1`; there is no
broad tiny/decode range in which the production kernel wins. This narrow
crossover motivated the measured `fused(M_e=1)+grouped(M_e>=2)` subgroup test.

Raw medians: [SINGLE_EXPERT_SWEEP.csv](../SINGLE_EXPERT_SWEEP.csv). Figures:
[08 latency](../figures/08_single_expert_latency_vs_me.png),
[09 TFLOP/s](../figures/09_single_expert_tflops_vs_me.png), and
[10 crossover](../figures/10_grouped_tiny_kernel_crossover.png).

