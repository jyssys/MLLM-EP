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

from poc_mllm_rank_aware_active_expert_reduction.contract import MODEL, atomic_json, read_json


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0)); return int(sock.getsockname()[1])


def _prepare(processor: Any, row: dict[str, Any]) -> dict[str, Any]:
    with Image.open(row["image_path"]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": row["question"]},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": prompt, "multi_modal_data": {"image": image}}


def _wave(llm: Any, prompts: list[dict[str, Any]], params: list[Any], barrier: Any, wave_id: int):
    from vllm.outputs import RequestOutput
    from vllm.v1.engine import EngineCoreRequestType
    barrier.wait(timeout=1800)
    if prompts:
        llm._add_completion_requests(prompts, params, use_tqdm=False)
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)
    else:
        llm.llm_engine.engine_core._send_input(EngineCoreRequestType.START_DP_WAVE, (wave_id, -1))
        outputs = []
    barrier.wait(timeout=1800); return outputs


def _worker(rank: int, master_port: int, args: argparse.Namespace, barrier: Any,
            progress: Any, schedule: list[dict[str, Any]]) -> None:
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": "2", "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(master_port), "MLLM_RANK_STAGE_ENABLE": "1",
            "MLLM_RANK_STAGE_CONTROL": str(args.control.resolve()),
            "MLLM_RANK_STAGE_OUTPUT": str((args.output / "raw_stage").resolve()),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false",
        })
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
        tp = 1 if args.topology == "ep2" else 2
        llm = LLM(
            model=MODEL, dtype="bfloat16", trust_remote_code=True,
            tensor_parallel_size=tp, enable_expert_parallel=True,
            expert_placement_strategy="linear", all2all_backend="deepep_high_throughput",
            enable_ep_weight_filter=True, enable_dbo=False, gpu_memory_utilization=.90,
            kv_cache_memory_bytes=2 << 30, max_model_len=4096, max_num_batched_tokens=4096,
            max_num_seqs=4, limit_mm_per_prompt={"image": 1}, skip_mm_profiling=True,
            enable_prefix_caching=False, mm_processor_cache_gb=0,
            enable_flashinfer_autotune=False, enforce_eager=True, disable_log_stats=True,
            logits_processors=[
                "poc_mllm_vision_dominant_captioning.force_tokens:ForceTokensLogitsProcessor"
            ],
        )
        progress.put((rank, -1, "MODEL_READY"))
        from poc_mllm_rank_aware_active_expert_reduction.stage_instrumentation import _flush
        for wave_id, entry in enumerate(schedule):
            local_rows = entry["by_dp"][rank]
            if rank == 0:
                atomic_json(args.control, {"wave": wave_id, "condition": entry["condition"],
                                           "capture": entry["capture"], "steps": args.steps})
            barrier.wait(timeout=1800)
            prompts = [_prepare(processor, row) for row in local_rows]
            params = []
            for row in local_rows:
                forced = [int(value) for value in row["forced_token_ids"][:128]]
                params.append(SamplingParams(max_tokens=len(forced), temperature=0.0, seed=3407,
                    extra_args={"mllm_vds_forced_token_ids": forced}))
            outputs = _wave(llm, prompts, params, barrier, wave_id)
            for row, output in zip(local_rows, outputs):
                if list(output.outputs[0].token_ids) != [int(v) for v in row["forced_token_ids"][:128]]:
                    raise RuntimeError((row["request_id"], "parity"))
            progress.put((rank, wave_id, "COMPLETE"))
        _flush()
        atomic_json(args.output / f"worker.dp{rank}.json", {"ok": True})
    except BaseException:
        atomic_json(args.output / f"worker.dp{rank}.json",
                    {"ok": False, "traceback": traceback.format_exc()})
        progress.put((rank, -2, "FAILED")); raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", choices=("ep2", "ep4"), required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 16, 32, 64, 96, 128])
    parser.add_argument("--watchdog-sec", type=int, default=3600)
    args = parser.parse_args()
    expected = "2,3" if args.topology == "ep2" else "2,3,6,7"
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected:
        raise RuntimeError(f"{args.topology} requires {expected}")
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    rows = read_json(args.pool)["requests"]
    if len(rows) < 13: raise RuntimeError("need 13 frozen fixed128 requests")
    cursor = 0; schedule = []; request_executions = 0
    for global_batch in (1, 4, 8):
        per_dp = [global_batch // 2, global_batch - global_batch // 2]
        selected = rows[cursor:cursor + global_batch]; cursor += global_batch
        boundary = per_dp[0]
        frozen_by_dp = [selected[:boundary], selected[boundary:]]
        for capture in (False, True):
            by_dp = [list(frozen_by_dp[0]), list(frozen_by_dp[1])]
            request_executions += global_batch
            schedule.append({"condition": f"B{global_batch}_{'trace' if capture else 'warmup'}",
                             "capture": capture, "by_dp": by_dp})
    atomic_json(args.output / "manifest.json", {
        "status": "FROZEN", "topology": args.topology, "physical_gpus": list(map(int, expected.split(','))),
        "schedule": [{"condition": e["condition"], "capture": e["capture"],
                      "request_ids": [[r["request_id"] for r in rows] for rows in e["by_dp"]]}
                     for e in schedule],
        "steps": args.steps, "request_executions": request_executions,
        "evidence": "INSTRUMENTED_STAGE_TRACE_NOT_CLEAN_E2E_TIMING",
    })
    context = mp.get_context("spawn"); barrier = context.Barrier(2); progress = context.Queue()
    master_port = _port()
    workers = [context.Process(target=_worker, args=(rank, master_port, args, barrier, progress, schedule))
               for rank in (0, 1)]
    for process in workers: process.start()
    last = time.monotonic(); failure = None
    while any(process.is_alive() for process in workers):
        try:
            rank, wave_id, status = progress.get(timeout=5); last = time.monotonic()
            if status == "FAILED": failure = f"worker {rank} failed"; break
        except queue.Empty: pass
        if time.monotonic() - last > args.watchdog_sec:
            failure = "TIMEOUT: no worker progress"; break
    if failure:
        for process in workers:
            if process.is_alive(): process.terminate()
        for process in workers: process.join(timeout=30)
        atomic_json(args.output / "failure.json", {"status": "RUNTIME_UNAVAILABLE", "reason": failure})
        raise RuntimeError(failure)
    for process in workers: process.join()
    if any(process.exitcode != 0 for process in workers): raise RuntimeError([p.exitcode for p in workers])
    atomic_json(args.output / "run.json", {"status": "COMPLETE", "request_executions": request_executions})


if __name__ == "__main__": main()
