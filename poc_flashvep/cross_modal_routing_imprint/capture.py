"""Capture full-token Qwen3-VL MoE routes for a fixed-image-grid probe suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import socket
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from poc_flashvep.prerouter_visual_signal.run_capture import MODEL, _base_suite
from poc_flashvep.vision_tile_motivation.profile_vision_tile_motivation import _load_direct_trace

FIXED_EDGE = 448
FIXED_PROMPT = "Describe the image briefly."
LAYERS = 48
TOPK = 8


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _prepare(processor: Any, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from PIL import Image

    source = Path(row["image_paths"][0])
    original = Image.open(source).convert("RGB")
    image = original.resize((FIXED_EDGE, FIXED_EDGE))
    content = [
        {"type": "image", "image": image},
        {"type": "text", "text": FIXED_PROMPT},
    ]
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True,
    )
    processed = processor(text=[prompt], images=[image], return_tensors="pt")
    ids = processed["input_ids"][0].tolist()
    image_id = int(processor.tokenizer.convert_tokens_to_ids(processor.image_token))
    positions = [index for index, token in enumerate(ids) if token == image_id]
    if not positions or positions != list(range(positions[0], positions[-1] + 1)):
        raise AssertionError(f"{row['sample_id']}: non-contiguous visual-token span")
    grid = [int(value) for value in processed["image_grid_thw"][0].tolist()]
    merge = int(processor.image_processor.merge_size)
    expected = grid[0] * grid[1] * grid[2] // (merge * merge)
    if expected != len(positions):
        raise AssertionError(f"{row['sample_id']}: grid/token mismatch")
    metadata = {
        "sample_id": row["sample_id"], "category": row["category"],
        "source_path": str(source), "source_sha256": _sha(source),
        "original_size": list(original.size), "input_size": list(image.size),
        "fixed_prompt": FIXED_PROMPT, "prompt_tokens": len(ids),
        "vision_tokens": len(positions), "visual_span": [positions[0], positions[-1] + 1],
        "post_visual_span": [positions[-1] + 1, len(ids)],
        "image_token_id": image_id, "image_grid_thw": grid, "merge_size": merge,
    }
    return {"prompt": prompt, "multi_modal_data": {"image": image}}, metadata


def _run_rank(rank: int, port: int, args: argparse.Namespace,
              partition: list[tuple[dict[str, Any], dict[str, Any]]], barrier: Any) -> None:
    output = args.output_dir / f"capture.dp{rank}.json"
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": "2", "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_DIRECT_ROUTING_DIR": str((args.output_dir / "direct_routes").resolve()),
        })
        from vllm import LLM, SamplingParams
        from vllm.outputs import RequestOutput

        llm = LLM(
            model=args.model_path, dtype="bfloat16", tensor_parallel_size=2,
            enable_expert_parallel=True, expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput", enable_dbo=False,
            enable_return_routed_experts=True, enable_ep_weight_filter=True,
            trust_remote_code=True, gpu_memory_utilization=.90,
            kv_cache_memory_bytes=1 << 30, max_model_len=1024,
            max_num_batched_tokens=8192, max_num_seqs=24,
            limit_mm_per_prompt={"image": 1}, skip_mm_profiling=True,
            enable_prefix_caching=False, enable_flashinfer_autotune=False,
            enforce_eager=True,
        )
        prompts = [item[0] for item in partition]
        metadata = [item[1] for item in partition]
        barrier.wait(timeout=900)
        submitted = llm._add_completion_requests(
            prompts, SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)
        barrier.wait(timeout=900)
        lengths = [len(item.prompt_token_ids or []) for item in outputs]
        routes, call_groups = _load_direct_trace(args.output_dir / "direct_routes", rank, lengths)
        cursor = 0
        records = []
        for request_id, meta, result in zip(submitted, metadata, outputs, strict=True):
            token_ids = np.asarray(result.prompt_token_ids or [], dtype=np.int64)
            local = routes[cursor:cursor + len(token_ids)]
            cursor += len(token_ids)
            if local.shape != (len(token_ids), LAYERS, TOPK):
                raise AssertionError(f"{meta['sample_id']}: route shape {local.shape}")
            file_name = f"routing.dp{rank}.{meta['sample_id']}.npz"
            np.savez_compressed(args.output_dir / file_name,
                                routed_experts=local, prompt_token_ids=token_ids)
            records.append({**meta, "dp_rank": rank, "array_file": file_name,
                            "submitted_request_id": request_id,
                            "returned_request_id": result.request_id,
                            "output_token_ids": list(result.outputs[0].token_ids),
                            "model_call_groups": call_groups})
        _write_json(output, {"ok": True, "records": records})
    except BaseException:
        _write_json(output, {"ok": False, "traceback": traceback.format_exc()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", default=MODEL)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    prepared = [_prepare(processor, row) for row in _base_suite()]
    # Exact 24/24 split: fixed 276-token requests fit the preregistered 8192-token cap.
    partitions = [prepared[::2], prepared[1::2]]
    manifest = {
        "model": args.model_path,
        "configuration": {"dtype": "BF16", "tp": 2, "dp": 2, "ep": 4,
                          "pp": 1, "all2all": "deepep_high_throughput",
                          "physical_gpus": [1, 2, 3, 4], "dbo": False},
        "fixed_edge": FIXED_EDGE, "fixed_prompt": FIXED_PROMPT,
        "samples": [item[1] for item in prepared],
        "partition": [[item[1]["sample_id"] for item in part] for part in partitions],
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    port = _port()
    processes = [ctx.Process(target=_run_rank, args=(rank, port, args, partitions[rank], barrier))
                 for rank in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    codes = [process.exitcode for process in processes]
    if codes != [0, 0]:
        raise RuntimeError(f"capture failed: {codes}")
    print(args.output_dir)


if __name__ == "__main__":
    main()
