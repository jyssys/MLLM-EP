# Strongest backend and prior-art collision

| Candidate mechanism | Existing system/source-level status | Collision / remaining claim |
|---|---|---|
| BF16 jagged `M_e`, zero-length full groups | [PyTorch grouped_mm](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grouped_mm.html), fastest tested exact BF16 owner-local replay; 102/102 previous real-case/restart wins over installed default fused vLLM | Already handles inactive zero groups with very little measurable tax. New per-expert strategy not justified. |
| Resident BF16 vLLM fused | Installed vLLM `fused_experts`, source `deepep_ht_prepare_finalize.py` and `modular_kernel.py` | Remains production integrated comparison, but default H100 E64/N1024 reports missing tuned config. We do **not** claim that weak default establishes new-kernel novelty. |
| Hopper gate/up–SwiGLU epilogue | [SonicMoE paper](https://arxiv.org/html/2512.14080v2) Section 4.1.2 and [official repo](https://github.com/Dao-AILab/sonic-moe) | Already fuses forward SwiGLU into up-GEMM epilogue, can gather activations and reduces separate IO/launch; activated intermediate for down GEMM remains. Just gate/up+SwiGLU fusion is not novel. |
| Descriptor table/packed offsets | [CUTLASS grouped GEMM contiguous offsets](https://docs.nvidia.com/cutlass/latest/media/docs/operators/tutorials/005_grouped_gemm_contiguous_offset.html); [NVIDIA cuDNN MoE grouped matmul](https://docs.nvidia.com/deeplearning/cudnn/v1.23.0/operations/MoeGroupedMatmul.html) uses `FirstTokenOffset` | Generic active group compaction and offset interfaces established. |
| GPU DeepEP receive prefix and direct metadata use | [DeepEP V2 elastic Buffer implementation](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/buffers/elastic.py) exposes GPU `psum_num_recv_tokens_per_expert`; installed legacy DeepEP/vLLM handoff returns CPU list + GPU metadata conversion | Legacy conversion is an engineering opportunity, but **GPU offsets themselves already exist upstream**. Any new contribution must show additional request-critical benefit and work beyond normal metadata wiring. |
| Persistent/fused full MoE + communication | [FlashMoE official paper/artifact](https://flash-moe.github.io/) and [UniEP](https://arxiv.org/abs/2604.19241) (training scope) | Fused expert/EP execution and persistent scheduling already a crowded generic area; our proposed exact BF16 path must beat *strongest valid expert operator*, not just naive A2A. |
| Decode-oriented fused forward | [MonoMoE paper](https://arxiv.org/abs/2609.04244) and [FlashInfer official implementation](https://github.com/flashinfer-ai/flashinfer/tree/main/csrc/fused_moe/monomoe) | Quantized H200/decode-oriented, not our H100 BF16 contract; still occupies small expert MLP fusion territory. |

SonicMoE official requirements at audit time include CUDA 12.9+,
PyTorch 2.11+ and recommended Python 3.12+. Here CUDA 12.8,
PyTorch 2.8, Python 3.11. Installing/patching SonicMoE in the
validated LLaDA2 EP4 environment would compromise the frozen runtime.
Its BF16 H100 gated-MLP ability was therefore **source audited**, not
reported as a measured local benchmark. An isolated compatible
container/worktree with imported exact expert weights is the right
follow-up **only if** a new and larger oracle is subsequently found.

SonicMoE is not a claim that its entire gate/up→down MLP stays on chip:
the paper's forward uses up/SwiGLU, down, and aggregation, so down
still consumes activated intermediate. Weight epilogue / aggregation
is also not identical to the real DeepEP combine transaction, which
already applies top-k reduction before communication. Combining
SonicMoE's available activation fusion and DeepEP V2's GPU prefix
narrows the unique overlap of the proposed
`DeepEP-Native Active-Support-Aware Fused Expert MLP` substantially.

For a paper-level method we would need (i) proven **request-critical**
inactive group handling that `torch._grouped_mm` does not already
discard; (ii) quantitatively significant benefit **beyond**
SonicMoE/cuDNN's epilogue and DeepEP V2's GPU counts; and (iii)
quality-equivalent full EP4 integrated E2E. Current O4 optimistic
request bound <1% fails (i) and (ii) before any implementation.
