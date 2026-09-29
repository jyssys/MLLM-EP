#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
import queue
import socket
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from poc_mllm_ep_communication_motivation.contract import MODEL, atomic_json, read_json


def port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def image_prompt(processor: Any, row: dict[str, Any]) -> dict[str, Any]:
    with Image.open(row["image_path"]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": row["question"]},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt, "multi_modal_data": {"image": image}}


def wave(llm: Any, prompts: list[Any], params: list[Any], barrier: Any, wave_id: int):
    from vllm.outputs import RequestOutput
    from vllm.v1.engine import EngineCoreRequestType

    barrier.wait(timeout=1800)
    if prompts:
        llm._add_completion_requests(prompts, params, use_tqdm=False)
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)
    else:
        llm.llm_engine.engine_core._send_input(
            EngineCoreRequestType.START_DP_WAVE, (wave_id, -1)
        )
        outputs = []
    barrier.wait(timeout=1800)
    return outputs


def worker(
    rank: int,
    master_port: int,
    args: argparse.Namespace,
    barrier: Any,
    progress: Any,
    schedule: list[dict[str, Any]],
) -> None:
    llm = None
    try:
        os.environ.update(
            {
                "VLLM_DP_RANK": str(rank),
                "VLLM_DP_RANK_LOCAL": str(rank),
                "VLLM_DP_SIZE": "2",
                "VLLM_DP_MASTER_IP": "127.0.0.1",
                "VLLM_DP_MASTER_PORT": str(master_port),
                "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
                "FLASHVEP_CONFIGURED_DBO": "false",
            }
        )
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
        special = list(map(int, processor.tokenizer.all_special_ids))
        llm = LLM(
            model=MODEL,
            dtype="bfloat16",
            trust_remote_code=True,
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput",
            enable_ep_weight_filter=True,
            enable_dbo=False,
            gpu_memory_utilization=0.90,
            kv_cache_memory_bytes=2 << 30,
            max_model_len=20000,
            max_num_batched_tokens=20000,
            max_num_seqs=1,
            enable_chunked_prefill=False,
            limit_mm_per_prompt={"image": 1},
            skip_mm_profiling=True,
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            enable_flashinfer_autotune=False,
            enforce_eager=True,
            disable_log_stats=True,
        )
        progress.put((rank, -1, "MODEL_READY"))
        records = []
        for wave_id, entry in enumerate(schedule):
            if rank == 0:
                atomic_json(
                    args.control,
                    {
                        **entry,
                        "wave": wave_id,
                        "special_token_ids": special,
                        "expected_phase": "prefill",
                    },
                )
            barrier.wait(timeout=1800)
            prompts = [image_prompt(processor, entry["request"])] if rank == 0 else []
            params = (
                [SamplingParams(max_tokens=1, temperature=0.0, seed=3407)]
                if rank == 0
                else []
            )
            begin = time.perf_counter_ns()
            outputs = wave(
                llm, [copy.deepcopy(p) for p in prompts], params, barrier, wave_id
            )
            wall_ms = (time.perf_counter_ns() - begin) / 1e6
            records.append(
                {
                    "wave": wave_id,
                    "condition": entry["condition"],
                    "case_key": entry["case_key"],
                    "replicate": entry["replicate"],
                    "wall_ms": wall_ms,
                    "output_count": len(outputs),
                }
            )
            progress.put((rank, wave_id, "COMPLETE"))
        from poc_mllm_ep_communication_motivation.instrumentation import _drain

        _drain()
        atomic_json(
            args.output / f"worker.dp{rank}.json", {"ok": True, "waves": records}
        )
    except BaseException:
        atomic_json(
            args.output / f"worker.dp{rank}.json",
            {"ok": False, "traceback": traceback.format_exc()},
        )
        progress.put((rank, -2, "FAILED"))
        raise
    finally:
        if llm is not None and callable(getattr(llm, "shutdown", None)):
            try:
                llm.shutdown()
            except BaseException:
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--watchdog-sec", type=int, default=3600)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3,6,7":
        raise RuntimeError("requires CUDA_VISIBLE_DEVICES=2,3,6,7")
    if os.environ.get("MLLM_COMM_MOTIVATION_HOOK_IMPORTED") != "1":
        raise RuntimeError("startup hook was not imported")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    rows = read_json(args.manifest)["q2_visual_sweep"]
    schedule = []
    latin = [
        ["FULL_REPLAY", "VISUAL_ONLY", "TEXT_ONLY"],
        ["VISUAL_ONLY", "TEXT_ONLY", "FULL_REPLAY"],
        ["TEXT_ONLY", "FULL_REPLAY", "VISUAL_ONLY"],
    ]
    for row in rows:
        key = f"q2::{row['size_class']}::{row['request_id']}"
        schedule.append(
            {
                "condition": "FULL_SEED",
                "case_key": key,
                "replicate": -1,
                "capture_stage": False,
                "route_stats": False,
                "request": row,
            }
        )
        for rep in range(args.repeats):
            for condition in latin[rep % len(latin)]:
                schedule.append(
                    {
                        "condition": condition,
                        "case_key": key,
                        "replicate": rep,
                        "capture_stage": True,
                        "route_stats": True,
                        "request": row,
                    }
                )
    atomic_json(
        args.output / "manifest.json",
        {
            "status": "FROZEN",
            "topology": "TP2-DP2-EP4-PP1",
            "physical_gpus": [2, 3, 6, 7],
            "chunked_prefill": False,
            "max_num_batched_tokens": 20000,
            "conditions": ["FULL_REPLAY", "VISUAL_ONLY", "TEXT_ONLY"],
            "repeats": args.repeats,
            "requests": rows,
            "route_contract": "same FULL_SEED IDs/weights per layer; non-selected modality rows set to -1",
        },
    )
    atomic_json(args.control, {"condition": "STARTUP", "capture_stage": False})
    context = mp.get_context("spawn")
    barrier, progress, master_port = context.Barrier(2), context.Queue(), port()
    workers = [
        context.Process(
            target=worker, args=(rank, master_port, args, barrier, progress, schedule)
        )
        for rank in (0, 1)
    ]
    for process in workers:
        process.start()
    last = time.monotonic()
    failure = None
    while any(process.is_alive() for process in workers):
        try:
            rank, _, status = progress.get(timeout=5)
            last = time.monotonic()
            if status == "FAILED":
                failure = f"worker {rank} failed"
                break
        except queue.Empty:
            pass
        if time.monotonic() - last > args.watchdog_sec:
            failure = "TIMEOUT: no worker progress"
            break
    if failure:
        for process in workers:
            if process.is_alive():
                process.terminate()
        for process in workers:
            process.join(timeout=30)
        atomic_json(
            args.output / "failure.json",
            {"status": "RUNTIME_UNAVAILABLE", "reason": failure},
        )
        raise RuntimeError(failure)
    for process in workers:
        process.join()
    if any(process.exitcode != 0 for process in workers):
        raise RuntimeError([process.exitcode for process in workers])
    atomic_json(
        args.output / "run.json", {"status": "COMPLETE", "waves": len(schedule)}
    )


if __name__ == "__main__":
    main()
