# Prior art and novelty attack

| work/system | directly relevant capability | implication |
|---|---|---|
| [PyTorch grouped_mm](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grouped_mm.html) | CUDA BF16 grouped multiplication with jagged per-group M | already supplies the strongest tested whole-invocation primitive |
| [CUTLASS grouped GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/operators/tutorials/005_grouped_gemm_contiguous_offset.html) | grouped problems, persistent scheduling and contiguous offsets | generic expert-problem scheduling is established |
| [SonicMoE](https://github.com/Dao-AILab/sonic-moe) / [paper](https://arxiv.org/abs/2512.14080) | H100 BF16 tile/IO-aware fused MoE kernels | close collision with generic heterogeneous tile scheduling; not measured here due runtime prerequisites |
| [MonoMoE](https://arxiv.org/abs/2609.04244) / [implementation](https://github.com/flashinfer-ai/flashinfer/tree/main/csrc/fused_moe/monomoe) | weight-major persistent quantized MoE optimized for decode | directly occupies the tiny-expert strategy space, though not this BF16 contract |
| [DA-MoE](https://arxiv.org/abs/2607.23099) | routing-distribution-aware fused-kernel selection | direct adjacency to histogram/skew-aware whole-kernel dispatch |
| [MegaBlocks](https://github.com/databricks/megablocks) | block-sparse expert packing and grouped execution | “pack expert rows” alone is not novel |

The proposed distinction would have been one persistent BF16 kernel applying
different tile strategies to different experts within the same dLLM refinement
invocation. That is narrower than whole-kernel dispatch and could still be
technically distinct from MonoMoE. But SonicMoE/CUTLASS/DA-MoE make the space
crowded, and novelty cannot rescue a sub-5% perfect request oracle.

The measured positive is instead mundane but useful: this installed vLLM has no
tuned H100 configuration for `E=64,N=1024`, and PyTorch grouped is substantially
faster. Integrating/tuning that existing primitive may improve engineering
performance, but it is not a paper-level RefineGEMM contribution.

