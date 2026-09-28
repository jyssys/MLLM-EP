#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import queue
import socket
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from poc_mllm_rank_aware_active_expert_reduction.contract import MODEL, atomic_json, read_json


def port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def prepare(processor: Any, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    with Image.open(row["original_image_path"]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": row["question"]},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processed = processor(text=[prompt], images=[image], return_tensors="pt")
    ids = processed["input_ids"][0].tolist()
    image_id = int(processor.tokenizer.convert_tokens_to_ids(processor.image_token))
    positions = [index for index, token in enumerate(ids) if token == image_id]
    if not positions or positions != list(range(positions[0], positions[-1] + 1)):
        raise RuntimeError((row["request_id"], "non-contiguous visual placeholder"))
    meta = {"request_id": row["request_id"], "domain": row["domain"],
            "prompt_token_count": len(ids), "visual_span": [positions[0], positions[-1] + 1]}
    return {"prompt": prompt, "multi_modal_data": {"image": image}}, meta


def wave(llm: Any, prompts: list[dict[str, Any]], params: list[Any], barrier: Any, wave_id: int):
    from vllm.outputs import RequestOutput
    from vllm.v1.engine import EngineCoreRequestType
    barrier.wait(timeout=1800)
    if prompts:
        llm._add_completion_requests(prompts, params, use_tqdm=False)
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)
    else:
        llm.llm_engine.engine_core._send_input(EngineCoreRequestType.START_DP_WAVE, (wave_id, -1))
        outputs = []
    barrier.wait(timeout=1800)
    return outputs


def worker(rank: int, master_port: int, args: argparse.Namespace, barrier: Any,
           progress: Any, rows: list[dict[str, Any]]) -> None:
    llm = None
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": "2", "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(master_port),
            "MLLM_RANK_VDS_ENABLE": "1",
            "MLLM_RANK_VDS_CONTROL": str(args.control.resolve()),
            "MLLM_RANK_VDS_OUTPUT": str((args.output / "raw_vds").resolve()),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false",
        })
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
        llm = LLM(
            model=MODEL, dtype="bfloat16", trust_remote_code=True,
            tensor_parallel_size=2, enable_expert_parallel=True,
            expert_placement_strategy="linear", all2all_backend="deepep_high_throughput",
            enable_ep_weight_filter=True, enable_dbo=False, gpu_memory_utilization=.90,
            kv_cache_memory_bytes=2 << 30, max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_model_len, max_num_seqs=1,
            limit_mm_per_prompt={"image": 1}, skip_mm_profiling=True,
            enable_prefix_caching=False, mm_processor_cache_gb=0,
            enable_flashinfer_autotune=False, enforce_eager=True, disable_log_stats=True,
            logits_processors=[
                "poc_mllm_vision_dominant_captioning.force_tokens:ForceTokensLogitsProcessor"
            ],
        )
        progress.put((rank, -1, "MODEL_READY"))
        local = rows[rank::2]
        total_waves = max((len(rows) + 1) // 2, 1)
        for wave_id in range(total_waves):
            row = local[wave_id] if wave_id < len(local) else None
            prepared = None if row is None else (row, *prepare(processor, row))
            gathered = [None, None]
            if prepared is not None:
                gathered[rank] = prepared[2]
            # Exchange small metadata through files/barriers, avoiding Manager complexity.
            meta_path = args.output / f"wave{wave_id}.dp{rank}.json"
            atomic_json(meta_path, {} if prepared is None else prepared[2])
            barrier.wait(timeout=1800)
            if rank == 0:
                by_dp = {str(dp): read_json(args.output / f"wave{wave_id}.dp{dp}.json")
                         for dp in (0, 1)}
                by_dp = {key: value for key, value in by_dp.items() if value}
                atomic_json(args.control, {"capture": True, "wave": wave_id,
                                           "layers": args.layers, "steps": args.steps,
                                           "by_dp": by_dp})
            barrier.wait(timeout=1800)
            if prepared is None:
                prompts, params = [], []
            else:
                row, prompt, _ = prepared
                forced = [int(value) for value in row["forced_token_ids"]]
                prompts = [copy.deepcopy(prompt)]
                params = [SamplingParams(max_tokens=len(forced), temperature=0.0, seed=3407,
                                         extra_args={"mllm_vds_forced_token_ids": forced})]
            outputs = wave(llm, prompts, params, barrier, wave_id)
            if prepared is not None:
                expected = prepared[0]["forced_token_ids"]
                actual = list(outputs[0].outputs[0].token_ids)
                if actual != expected:
                    raise RuntimeError((prepared[0]["request_id"], "forced-token mismatch"))
            progress.put((rank, wave_id, "COMPLETE"))
        atomic_json(args.output / f"worker.dp{rank}.json", {"ok": True})
    except BaseException:
        atomic_json(args.output / f"worker.dp{rank}.json",
                    {"ok": False, "traceback": traceback.format_exc()})
        progress.put((rank, -2, "FAILED"))
        raise
    finally:
        if llm is not None:
            shutdown = getattr(llm, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except BaseException:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 12, 24, 36, 44, 47])
    parser.add_argument("--steps", type=int, nargs="+", default=[8, 16, 32, 48, 64, 96, 112])
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--watchdog-sec", type=int, default=3600)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3,6,7":
        raise RuntimeError("Q2 TRUE EP4 requires CUDA_VISIBLE_DEVICES=2,3,6,7")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    payload = read_json(args.manifest)
    rows = payload["anchors"][:args.limit]
    atomic_json(args.output / "manifest.json", {
        "status": "FROZEN", "request_count": len(rows),
        "request_ids": [row["request_id"] for row in rows],
        "layers": args.layers, "steps": args.steps,
        "topology": "TRUE_TP2_DP2_EP4_PP1", "physical_gpus": [2, 3, 6, 7],
        "evidence": "INSTRUMENTED_DIAGNOSTIC_NOT_CLEAN_TIMING",
    })
    context = mp.get_context("spawn")
    barrier = context.Barrier(2); progress = context.Queue(); master_port = port()
    workers = [context.Process(target=worker, args=(rank, master_port, args, barrier, progress, rows))
               for rank in (0, 1)]
    for process in workers: process.start()
    last = time.monotonic(); failure = None
    while any(process.is_alive() for process in workers):
        try:
            rank, wave_id, status = progress.get(timeout=5)
            last = time.monotonic()
            if status == "FAILED": failure = f"worker {rank} failed"; break
        except queue.Empty:
            pass
        if time.monotonic() - last > args.watchdog_sec:
            failure = "TIMEOUT: no progress"; break
    if failure:
        for process in workers:
            if process.is_alive(): process.terminate()
        for process in workers: process.join(timeout=30)
        atomic_json(args.output / "failure.json", {"status": "RUNTIME_UNAVAILABLE", "reason": failure})
        raise RuntimeError(failure)
    for process in workers: process.join()
    if any(process.exitcode != 0 for process in workers):
        raise RuntimeError([process.exitcode for process in workers])
    atomic_json(args.output / "run.json", {"status": "COMPLETE", "request_count": len(rows)})


if __name__ == "__main__":
    main()
