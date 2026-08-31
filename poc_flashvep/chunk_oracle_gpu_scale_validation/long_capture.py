"""Bounded long multi-image Qwen3-VL route capture for Stage 2.

This is read-only routing capture.  It reuses the validated TP2/DP2/EP4
vLLM path and the existing direct router hook; no route, placement, or model
code is changed.  Four requests with 6/8/10/12 distinct local images provide
multiple long-sequence scales without downloading data.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import socket
import traceback
from pathlib import Path
from typing import Any

from poc_flashvep.vision_tile_motivation.profile_vision_tile_motivation import (
    _balanced_partition,
    _load_direct_trace,
    _prepare_sample,
    expanded_sample_suite,
    sample_suite,
)


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows() -> list[dict[str, Any]]:
    available = {row["sample_id"]: row for row in sample_suite() + expanded_sample_suite()}
    groups = [
        ("long_6img_natural_fine", ["astronaut", "motorcycle", "coffee", "cat", "retina", "histology"]),
        ("long_8img_mixed", ["deep_field", "rocket", "grass", "gravel", "fast_gptq", "method", "model_card", "bit_allocate"]),
        ("long_10img_chart_mixed", ["tui_main", "tui_log", "tui_model_selection", "moon", "camera", "horse", "coins", "brick", "phantom", "cell"]),
        ("long_12img_broad", ["astronaut", "motorcycle", "coffee", "cat", "retina", "histology", "fast_gptq", "method", "model_card", "bit_allocate", "moon", "camera"]),
    ]
    rows = []
    for sample_id, ids in groups:
        missing = [item for item in ids if item not in available]
        if missing:
            raise FileNotFoundError(f"missing local sample IDs for {sample_id}: {missing}")
        paths = [path for item in ids for path in available[item]["image_paths"]]
        rows.append({
            "category": "multi_image",
            "sample_id": sample_id,
            "image_paths": paths,
            "question": "Compare the images and summarize their visible content and common patterns briefly.",
            "suite_origin": "bounded_long_scale_validation",
        })
    return rows


def _run_rank(rank: int, port: int, args: argparse.Namespace,
              barrier: Any, partition: list[tuple[dict[str, Any], dict[str, Any]]]) -> None:
    out = args.output_dir
    path = out / f"profile.dp_rank{rank}.json"
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": "2", "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_DIRECT_ROUTING_DIR": str((out / "direct_router_capture").resolve()),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false",
        })
        from vllm import LLM, SamplingParams
        from vllm.outputs import RequestOutput

        llm = LLM(
            model=args.model_path, dtype="bfloat16", tensor_parallel_size=2,
            enable_expert_parallel=True, expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput", enable_dbo=False,
            enable_return_routed_experts=True, enable_ep_weight_filter=True,
            trust_remote_code=True, gpu_memory_utilization=0.90,
            kv_cache_memory_bytes=1 << 30, max_model_len=16384,
            max_num_batched_tokens=16384, max_num_seqs=4,
            limit_mm_per_prompt={"image": 16}, skip_mm_profiling=True,
            enable_prefix_caching=False, enable_flashinfer_autotune=False,
            enforce_eager=True, disable_log_stats=True,
        )
        prompts = [item[0] for item in partition]
        metadata = [item[1] for item in partition]
        barrier.wait(timeout=1800)
        submitted = llm._add_completion_requests(
            prompts, SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False
        )
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)
        barrier.wait(timeout=1800)
        if len(outputs) != len(metadata):
            raise AssertionError(f"DP{rank}: output count {len(outputs)} != {len(metadata)}")
        prompt_lengths = [len(output.prompt_token_ids or []) for output in outputs]
        routed, call_groups = _load_direct_trace(out / "direct_router_capture", rank, prompt_lengths)
        if routed.shape != (sum(prompt_lengths), 48, 8):
            raise AssertionError(f"DP{rank}: unexpected route shape {routed.shape}")
        records = []
        offset = 0
        for submitted_id, meta, request, length in zip(submitted, metadata, outputs, prompt_lengths, strict=True):
            token_ids = list(request.prompt_token_ids or [])
            local = routed[offset:offset + length]
            offset += length
            if len(token_ids) != length:
                raise AssertionError("prompt token length mismatch")
            image_id = int(meta["image_token_id"])
            if token_ids.count(image_id) != int(meta["processor_vision_tokens"]):
                raise AssertionError(f"{meta['sample_id']}: processor/vLLM image-token mismatch")
            route_name = f"routing.{meta['sample_id']}.npz"
            import numpy as np
            np.savez_compressed(out / route_name,
                                routed_experts=local.astype(np.int16),
                                prompt_token_ids=np.asarray(token_ids, dtype=np.int64))
            records.append({
                **meta, "dp_rank": rank, "submitted_request_id": submitted_id,
                "returned_request_id": str(request.request_id), "route_file": route_name,
                "model_call_groups": call_groups, "prompt_tokens": int(length),
                "vision_tokens": int(meta["processor_vision_tokens"]),
                "routed_shape": list(local.shape),
                "output_token_ids": list(request.outputs[0].token_ids),
            })
        _json(path, {"ok": True, "dp_rank": rank, "records": records})
    except BaseException:
        _json(path, {"ok": False, "dp_rank": rank, "traceback": traceback.format_exc()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = _rows()
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    prepared = [_prepare_sample(processor, row) for row in rows]
    partitions = _balanced_partition(prepared)
    manifest = {
        "model": args.model_path,
        "configuration": {"dtype": "BF16", "tp": 2, "dp": 2, "ep": 4,
                           "pp": 1, "all2all": "deepep_high_throughput", "dbo": False,
                           "max_model_len": 16384, "physical_gpus": [1, 2, 3, 4]},
        "samples": [], "partition": [[x[1]["sample_id"] for x in p] for p in partitions],
        "source_sha256": {path: _sha(Path(path)) for row in rows for path in row["image_paths"]},
    }
    for _, metadata in prepared:
        manifest["samples"].append(metadata)
    _json(args.output_dir / "sample_manifest.json", manifest)
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    # Both DP processes must share the same rendezvous port.
    port = _port()
    processes = [context.Process(target=_run_rank, args=(rank, port, args, barrier, partitions[rank])) for rank in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(1800)
    codes = [process.exitcode for process in processes]
    if codes != [0, 0]:
        raise RuntimeError(f"long capture failed: {codes}")
    _json(args.output_dir / "capture_summary.json", {
        "status": "ok", "requests": len(rows), "prompt_tokens": [m[1]["processor_prompt_tokens"] for m in prepared],
        "vision_tokens": [m[1]["processor_vision_tokens"] for m in prepared],
        "physical_gpus": [1, 2, 3, 4], "source": "existing local image assets only",
    })


if __name__ == "__main__":
    main()
