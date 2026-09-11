# Environment snapshot

Captured on 2026-09-11 (Asia/Seoul). No GPU was used while preparing this
document; the hardware fields below come from the saved live-run audit.

| Item | Value |
|---|---|
| Parent repository HEAD | `f0dc8372ec0d1d2e10347c82e630b84430c5dc5f` |
| PoC branch | `flashvep/dllm-moe-ep-temporal-poc` |
| dInfer upstream HEAD | `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f` |
| Isolated environment | `/home/esjung/anaconda3/envs/dinfer-ep-poc` |
| Python | 3.11.15 |
| dInfer | 0.1, editable from the pinned checkout |
| vLLM | 0.10.2 |
| PyTorch | 2.8.0+cu128 |
| Transformers | 4.55.2 |
| CUDA build | 12.8 |
| NCCL | 2.27.3 |
| Driver | 570.211.01 |
| GPUs | 4× NVIDIA H100 80GB HBM3 |
| Visible physical GPUs | 4, 5, 6, 7 only |
| Model | `inclusionAI/LLaDA-MoE-7B-A1B-Instruct`, fused checkpoint |
| Runtime dtype | BF16 |
| Architecture | 16 MoE layers, 64 experts, top-k 8, hidden size 2048 |
| Decoder | threshold decoder, threshold 0.8 |
| Generation | block 64, budget 64, cache off, early stop on |
| Measured input | fixed prompt length 64; one request per DP rank |

Physical UUID mapping recorded before launch:

| Physical GPU | UUID |
|---:|---|
| 4 | `GPU-6076e2f2-5b63-3761-5586-56ceb7df8139` |
| 5 | `GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730` |
| 6 | `GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28` |
| 7 | `GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b` |

The model config advertises `torch_dtype=float32`; all measured runs explicitly
loaded weights and executed the PoC path as BF16. The checked-in config records
both the measured workload and the larger workload points that were planned but
not run after the GPUs were returned.
