#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from poc_mllm_ep_communication_motivation.contract import (
    ALLOWED_GPUS,
    MODEL,
    MODEL_REVISION,
    TOPOLOGY,
    atomic_json,
)


def output(command: list[str]) -> str:
    return subprocess.run(
        command, text=True, capture_output=True, check=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3,6,7":
        raise RuntimeError("audit requires CUDA_VISIBLE_DEVICES=2,3,6,7")
    import torch
    import transformers
    import vllm
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    text = config.text_config
    shapes = {
        "hidden_size": int(text.hidden_size),
        "moe_intermediate_size": int(text.moe_intermediate_size),
        "num_experts": int(text.num_experts),
        "num_experts_per_tok": int(text.num_experts_per_tok),
        "num_hidden_layers": int(text.num_hidden_layers),
    }
    expected = {
        "hidden_size": 2048,
        "moe_intermediate_size": 768,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 48,
    }
    if shapes != expected:
        raise RuntimeError((shapes, expected))
    gpu_rows = []
    for line in output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader",
        ]
    ).splitlines():
        fields = [part.strip() for part in line.split(",")]
        if int(fields[0]) in ALLOWED_GPUS:
            gpu_rows.append(
                {
                    "physical_index": int(fields[0]),
                    "uuid": fields[1],
                    "name": fields[2],
                    "memory_total": fields[3],
                }
            )
    atomic_json(
        args.output,
        {
            "status": "PASS",
            "git_sha": output(["git", "rev-parse", "HEAD"]),
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "dtype": "bfloat16",
            "shapes": shapes,
            "physical_gpus": gpu_rows,
            "topology": TOPOLOGY,
            "runtime": {
                "vllm": vllm.__version__,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda": torch.version.cuda,
            },
            "primary_engine_contract": {
                "tensor_parallel_size": 2,
                "data_parallel_size": 2,
                "enable_expert_parallel": True,
                "all2all_backend": "deepep_high_throughput",
                "deep_ep_mode": "high_throughput",
                "expert_placement_strategy": "linear",
                "enable_ep_weight_filter": True,
                "enable_dbo": False,
                "enable_prefix_caching": False,
                "enable_chunked_prefill": False,
                "max_num_batched_tokens": 20000,
                "max_num_seqs_per_dp": 5,
                "eplb": False,
                "redundant_experts": 0,
                "enforce_eager": True,
            },
            "communication_accounting": {
                "dispatch_contract": "DeepEP unique token-destination rows; destination dedup",
                "payload": "BF16 hidden activations only; excludes metadata/control overhead",
                "physical_latency": "CUDA events around installed DeepEP prepare/finalize",
            },
        },
    )


if __name__ == "__main__":
    main()
