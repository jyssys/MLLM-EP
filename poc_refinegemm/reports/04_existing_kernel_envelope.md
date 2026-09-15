# Existing expert-kernel envelope

## Valid backends

The current production path calls vLLM/SGLang `fused_experts` on already
owner-packed rows. The installed vLLM has no tuned H100 config for
`E=64,N=1024` and reports its default configuration at runtime. PyTorch 2.8
`torch._grouped_mm` supports BF16 jagged-M grouped multiplication and was tested
with the same packed rows, expert weights, SwiGLU math and route-weight multiply.
It is the strongest valid owner-local kernel in this environment.

SonicMoE was audited but not installed: its current official prerequisites call
for CUDA 12.9+ and recommend Python 3.12, while this validated runtime is CUDA
12.8/Python 3.11. MonoMoE is a quantized decode-oriented reference rather than
the same BF16 contract. Neither is treated as a measured backend.

## Real replay result

Across 54 dense and 48 compacted real case-restarts (30 repetitions each),
PyTorch grouped wins **102/102** against the production fused kernel. Median
owner-local improvements are:

| scope/task | early | middle | late | all phases |
|---|---:|---:|---:|---:|
| dense GSM8K | 21.67% | 23.57% | 18.81% | **22.08%** |
| dense HumanEval | 18.42% | 23.89% | 17.40% | **20.52%** |
| compacted GSM8K | 20.26% | 22.64% | 34.76% | 22.39% |
| compacted HumanEval | 24.18% | 25.17% | 46.12% | 26.05% |

This exposes a practical *backend tuning/replacement* opportunity, projected at
6.51% GSM8K and 6.06% HumanEval request E2E against the current untuned
production expert kernel. It is not measured E2E and is not RefineGEMM's novel
heterogeneous scheduling opportunity.

Machine evidence: [EXISTING_EXPERT_KERNELS.csv](../EXISTING_EXPERT_KERNELS.csv)
and [REFINEGEMM_MICROBENCH.csv](../REFINEGEMM_MICROBENCH.csv).

