# RLT patch attribution

The patch modifies vllm-rlt revision `4a350344027960a48384626aa646f1765fda6d4f`.
The upstream repository is https://github.com/InferMatrixLab/vllm-rlt.
It uses Apache License 2.0. The accompanying license text is `Apache-2.0.txt`.

The 2026-10-22 modifications add a bounded prefill group option and scheduler progression.
They do not change model weights, model operators, attention kernels, or exit policy.

Upstream NOTICE identifies ByteDance/Ouro-1.4B revision
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1` as the model architecture source.
Its upstream model/configuration attribution includes:
Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
All rights reserved.

The standalone RLT implementation and these patches are not an upstream vLLM plugin.
Preserve upstream licenses and notices when applying or distributing the patch.
