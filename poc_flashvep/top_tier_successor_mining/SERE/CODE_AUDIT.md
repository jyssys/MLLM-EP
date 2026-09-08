# SERE — official source audit

Repository `JL-Cheng/SERE`, commit
`8512f39b108cd1fa04e9261a5b6846fb173989e4`.

| Source | Audited behavior |
|---|---|
| `calibration/cal_expert_similarity.py` | Text parquet; truncates sampled sequences; one full forward; saves matrices in checkpoint and separate file. |
| `calibration/adapted_modeling_qwen3_moe.py:312` | Runs **all** experts on identical hidden states, computes similarity, then stock routed output continues. |
| `calibration/utils.py` | Frobenius distance normalized by maximum pairwise distance; cosine normalized to [0,1]; CKA alternatives. |
| `vllm/SERE_vllm/sere_qwen3_moe.py` | Stock fused_topk then custom rerouting; no explicit prefill/decode gating. |
| `rerouting_cuda_ops/rerouting_kernel.cu` | Primary mask is per-call token union; cutoff applies only if rho>0; IDs mutated in-place; weights unchanged; current stream used. |
| `rerouting_cuda_ops/__init__.py` | Converts IDs to int64; calls extension; returns unmodified weights. |

Official environment is vLLM .8.4/V0, not installed .20/V1. Keep its algorithm
and official CUDA rerouting kernel; use an isolated, documented boundary port.
Duplicate IDs after substitution are permitted by the algorithm. Do not silently
merge them in the baseline or count them as removed GEMM assignments.

Required sanity: reference versus official kernel exact IDs/weights; rho=1 no-op;
S=K no-op; end-to-end no-op logits/generation; real text sanity before MLLM failure.
