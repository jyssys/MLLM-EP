#!/usr/bin/env python3
"""Record the Qwen3-30B-A3B model configuration used by the PoC."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    fields = {k: getattr(cfg, k, None) for k in (
        "model_type", "architectures", "hidden_size", "num_hidden_layers",
        "num_experts", "num_experts_per_tok", "moe_intermediate_size",
        "intermediate_size", "torch_dtype")}
    fields.update({
        "model": args.model, "ep_size_requested": 8,
        "experts_per_ep_rank": (int(fields["num_experts"]) // 8
                                 if fields.get("num_experts") else None),
        "physical_gpus": list(range(8)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "note": "Qwen3-30B-A3B: 128 routed experts, top-8, 16 experts/GPU at EP8.",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fields, indent=2, default=str) + "\n")
    print(json.dumps(fields, indent=2, default=str))


if __name__ == "__main__":
    main()
