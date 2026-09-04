#!/usr/bin/env python3
"""Record the exact DeepSeek-V2-Lite configuration and runtime assumptions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    fields = {
        k: getattr(cfg, k, None)
        for k in (
            "model_type", "architectures", "hidden_size", "num_hidden_layers",
            "n_routed_experts", "num_experts", "num_experts_per_tok", "top_k",
            "n_shared_experts", "moe_layer_freq", "first_k_dense_replace",
            "moe_intermediate_size", "intermediate_size", "torch_dtype",
        )
    }
    fields["model"] = args.model
    fields["ep_size_requested"] = 4
    fields["physical_gpus"] = [1, 2, 3, 4]
    fields["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    fields["note"] = (
        "DeepSeek-V2-Lite-Chat config: 64 routed experts, top-6, 2 shared "
        "experts; first_k_dense_replace=1 means layer 0 is dense and layers "
        "1-26 are routed MoE."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fields, indent=2, default=str) + "\n")
    print(json.dumps(fields, indent=2, default=str))


if __name__ == "__main__":
    main()
