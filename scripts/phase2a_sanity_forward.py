"""Sanity-check Qwen3-VL-MoE multimodal prefill on 8 GPUs.

This script is intentionally measurement-only. It does not patch placement,
merge tokens, or alter MoE dispatch.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _init_dist() -> None:
    if _world_size() > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def _build_dummy_inputs(processor: AutoProcessor) -> dict[str, torch.Tensor]:
    image = Image.new("RGB", (224, 224), (128, 128, 128))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image briefly."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[image], return_tensors="pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--output", default="outputs/phase2a_sanity.json")
    parser.add_argument("--use-deepspeed", action="store_true")
    args = parser.parse_args()

    _init_dist()
    rank = _rank()
    local_rank = _local_rank()
    world_size = _world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    inputs = _build_dummy_inputs(processor)
    mm_token_type_ids = inputs["mm_token_type_ids"]
    num_vision = int((mm_token_type_ids == 1).sum().item())
    num_text = int((mm_token_type_ids == 0).sum().item())
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }

    torch.cuda.reset_peak_memory_stats(device)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    engine_class = None
    if args.use_deepspeed:
        import deepspeed

        model = deepspeed.init_inference(
            model,
            dtype=torch.bfloat16,
            replace_with_kernel_inject=False,
            ep_size=world_size,
            moe_experts=[128],
            moe_type="standard",
        )
        engine_class = type(model).__name__
        forward_model = model.module if hasattr(model, "module") else model
    else:
        forward_model = model

    with torch.inference_mode():
        outputs = forward_model(**inputs, use_cache=False, return_dict=True)
        logits_shape = tuple(outputs.logits.shape)

    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    result = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": torch.cuda.get_device_name(device),
        "deepspeed_enabled": bool(args.use_deepspeed),
        "deepspeed_engine_class": engine_class,
        "logits_shape": logits_shape,
        "num_text_tokens": num_text,
        "num_vision_tokens": num_vision,
        "peak_memory_gb": peak_gb,
    }
    gathered = [None for _ in range(world_size)]
    if dist.is_initialized():
        dist.all_gather_object(gathered, result)
        dist.barrier()
    else:
        gathered = [result]

    if rank == 0:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(gathered, indent=2), encoding="utf-8")
        print(json.dumps(gathered, indent=2))

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
