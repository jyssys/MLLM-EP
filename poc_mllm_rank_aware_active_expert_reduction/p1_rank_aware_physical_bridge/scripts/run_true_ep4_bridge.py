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


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _prepare(processor: Any, row: dict[str, Any]) -> dict[str, Any]:
    with Image.open(row["image_path"]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": row["question"]},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": prompt, "multi_modal_data": {"image": image}}


def _wave(llm: Any, prompts: list[dict[str, Any]], params: list[Any], barrier: Any, wave: int):
    from vllm.outputs import RequestOutput
    from vllm.v1.engine import EngineCoreRequestType

    barrier.wait(timeout=1800)
    if prompts:
        llm._add_completion_requests(prompts, params, use_tqdm=False)
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)
    else:
        llm.llm_engine.engine_core._send_input(EngineCoreRequestType.START_DP_WAVE, (wave, -1))
        outputs = []
    barrier.wait(timeout=1800)
    return outputs


def _schedule(rows: list[dict[str, Any]], repeats: int, steps: list[int]) -> list[dict[str, Any]]:
    if len(rows) < 8:
        raise ValueError("physical bridge requires eight frozen requests")
    selected = rows[:8]
    by_dp = [selected[:4], selected[4:8]]
    result = [{
        "condition": "SEED", "capture": False, "replicate": -1,
        "pair_key": "frozen_b8_seed", "by_dp": by_dp,
    }]
    latin = [
        ["NATIVE_K8", "NATIVE_TAIL_K6", "RANK_AWARE_K6", "SAME_PAIR_COLLAPSE25"],
        ["NATIVE_TAIL_K6", "SAME_PAIR_COLLAPSE25", "NATIVE_K8", "RANK_AWARE_K6"],
        ["RANK_AWARE_K6", "NATIVE_K8", "SAME_PAIR_COLLAPSE25", "NATIVE_TAIL_K6"],
        ["SAME_PAIR_COLLAPSE25", "RANK_AWARE_K6", "NATIVE_TAIL_K6", "NATIVE_K8"],
    ]
    for replicate in range(repeats):
        for condition in latin[replicate % len(latin)]:
            result.append({
                "condition": condition, "capture": True, "replicate": replicate,
                "pair_key": "frozen_b8_seed", "by_dp": by_dp, "steps": steps,
            })
    return result


def _worker(rank: int, port: int, args: argparse.Namespace, barrier: Any,
            progress: Any, schedule: list[dict[str, Any]]) -> None:
    llm = None
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": "2", "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false",
        })
        hook = str((Path(__file__).parents[1] / "hooks").resolve())
        os.environ["PYTHONPATH"] = f"{hook}:{os.environ.get('PYTHONPATH', '')}"
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
        llm = LLM(
            model=MODEL, dtype="bfloat16", trust_remote_code=True,
            tensor_parallel_size=2, enable_expert_parallel=True,
            expert_placement_strategy="linear", all2all_backend="deepep_high_throughput",
            enable_ep_weight_filter=True, enable_dbo=False, gpu_memory_utilization=.90,
            # Keep the complete distinct-image B8 prefill in one scheduled
            # chunk.  The installed external-DP/DeepEP path has a validated
            # B8 contract at this bound; a 4k aggregate chunk can create
            # unequal DP chunks before decode and is not the target regime.
            kv_cache_memory_bytes=4 << 30, max_model_len=20000,
            max_num_batched_tokens=20000, max_num_seqs=4,
            limit_mm_per_prompt={"image": 1}, skip_mm_profiling=True,
            enable_prefix_caching=False, mm_processor_cache_gb=0,
            enable_flashinfer_autotune=False, enforce_eager=True, disable_log_stats=True,
            logits_processors=[
                "poc_mllm_vision_dominant_captioning.force_tokens:ForceTokensLogitsProcessor"
            ],
        )
        progress.put((rank, -1, "MODEL_READY"))
        records = []
        for wave_id, entry in enumerate(schedule):
            if rank == 0:
                atomic_json(args.control, {
                    "wave": wave_id, "condition": entry["condition"],
                    "capture": entry["capture"], "replicate": entry["replicate"],
                    "pair_key": entry["pair_key"], "steps": args.steps,
                })
            barrier.wait(timeout=1800)
            local_rows = entry["by_dp"][rank]
            prompts = [_prepare(processor, row) for row in local_rows]
            params = []
            for row in local_rows:
                forced = [int(value) for value in row["forced_token_ids"][:128]]
                params.append(SamplingParams(
                    max_tokens=len(forced), temperature=0.0, seed=3407,
                    extra_args={"mllm_vds_forced_token_ids": forced},
                ))
            begin = time.perf_counter_ns()
            outputs = _wave(llm, [copy.deepcopy(item) for item in prompts], params, barrier, wave_id)
            wall_ms = (time.perf_counter_ns() - begin) / 1e6
            for row, output in zip(local_rows, outputs):
                actual = [int(value) for value in output.outputs[0].token_ids]
                expected = [int(value) for value in row["forced_token_ids"][:128]]
                if actual != expected:
                    raise RuntimeError((row["request_id"], "forced-token parity"))
            records.append({
                "wave": wave_id, "condition": entry["condition"],
                "replicate": entry["replicate"], "wall_ms": wall_ms,
                "requests": [row["request_id"] for row in local_rows],
            })
            progress.put((rank, wave_id, "COMPLETE"))
        atomic_json(args.output / f"worker.dp{rank}.json", {"ok": True, "waves": records})
    except BaseException:
        atomic_json(args.output / f"worker.dp{rank}.json", {
            "ok": False, "traceback": traceback.format_exc(),
        })
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
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, nargs="+", default=[16, 32, 64, 96, 128])
    parser.add_argument("--watchdog-sec", type=int, default=3600)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3,6,7":
        raise RuntimeError("PoC-1 requires CUDA_VISIBLE_DEVICES=2,3,6,7")
    if os.environ.get("MLLM_P1_BRIDGE_HOOK_IMPORTED") != "1":
        raise RuntimeError("PoC-1 startup hook was not imported")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    rows = read_json(args.pool)["requests"]
    schedule = _schedule(rows, args.repeats, args.steps)
    atomic_json(args.output / "manifest.json", {
        "schema_version": "rank_bridge_run.v1",
        "physical_gpus": [2, 3, 6, 7], "topology": "TP2-DP2-EP4-PP1",
        "backend": "deepep_high_throughput", "global_batch": 8,
        "fixed_decode_tokens": 128, "steps": args.steps,
        "schedule": [{
            "condition": row["condition"], "replicate": row["replicate"],
            "capture": row["capture"],
        } for row in schedule],
        "request_executions": sum(len(part) for row in schedule for part in row["by_dp"]),
        "route_contract": "SAME_PROMPTS_AND_FORCED_TOKENS; POLICY_TRANSFORMS_CURRENT_NATIVE_ROUTE",
    })
    atomic_json(args.control, {"condition": "STARTUP", "capture": False})
    context = mp.get_context("spawn")
    barrier, progress, port = context.Barrier(2), context.Queue(), _port()
    workers = [context.Process(
        target=_worker, args=(rank, port, args, barrier, progress, schedule)
    ) for rank in (0, 1)]
    for worker in workers:
        worker.start()
    last = time.monotonic()
    failure = None
    while any(worker.is_alive() for worker in workers):
        try:
            rank, _, status = progress.get(timeout=5)
            last = time.monotonic()
            if status == "FAILED":
                failure = f"worker {rank} failed"
                break
        except queue.Empty:
            pass
        if time.monotonic() - last > args.watchdog_sec:
            failure = "TIMEOUT: no progress"
            break
    if failure:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join(timeout=30)
        atomic_json(args.output / "failure.json", {"status": "RUNTIME_UNAVAILABLE", "reason": failure})
        raise RuntimeError(failure)
    for worker in workers:
        worker.join()
    if any(worker.exitcode != 0 for worker in workers):
        raise RuntimeError([worker.exitcode for worker in workers])
    atomic_json(args.output / "run.json", {"status": "COMPLETE", "waves": len(schedule)})


if __name__ == "__main__":
    main()
