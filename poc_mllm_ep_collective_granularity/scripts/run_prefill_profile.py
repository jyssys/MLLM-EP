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

from poc_mllm_ep_collective_granularity.contract import (
    MODEL,
    atomic_json,
    chunk_label,
    read_json,
)
from poc_mllm_ep_communication_motivation.scripts.run_q3_interference import (
    image_prompt,
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def worker(
    rank: int,
    master_port: int,
    args: argparse.Namespace,
    barrier: Any,
    progress: Any,
    request: dict[str, Any],
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
        budget = 20_000 if args.chunk_size == 0 else args.chunk_size
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
            max_model_len=20_000,
            max_num_batched_tokens=budget,
            max_num_seqs=1,
            enable_chunked_prefill=args.chunk_size > 0,
            limit_mm_per_prompt={"image": 1},
            skip_mm_profiling=True,
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            enable_flashinfer_autotune=False,
            enforce_eager=True,
            disable_log_stats=True,
        )
        progress.put((rank, -1, "MODEL_READY"))
        prompt = image_prompt(processor, request)
        wave_records = []
        for wave_id, replicate in enumerate(range(-1, args.repeats)):
            capture = replicate >= 0
            base_control = {
                "condition": "PREFILL_PROFILE",
                "replicate": replicate,
                "wave": wave_id,
                "special_token_ids": special,
                "expected_phase": "prefill",
                "injection_request_ids": [],
                "capture_stage": capture,
                "route_stats": capture,
                "chunk_label": chunk_label(args.chunk_size),
                "chunk_size": args.chunk_size,
            }
            if rank == 0:
                atomic_json(args.control, base_control)
            barrier.wait(timeout=1800)
            request_ids = []
            if rank == 0:
                request_ids = llm._add_completion_requests(
                    [copy.deepcopy(prompt)],
                    [SamplingParams(max_tokens=1, temperature=0.0, seed=3407)],
                    use_tqdm=False,
                )
            steps = []
            begin_wave_ns = time.perf_counter_ns()
            outer_iteration = 0
            while llm.llm_engine.has_unfinished_requests():
                control = {**base_control, "outer_iteration": outer_iteration}
                if rank == 0:
                    atomic_json(args.control, control)
                barrier.wait(timeout=1800)
                begin_ns = time.perf_counter_ns()
                outputs = llm.llm_engine.step()
                elapsed_ms = (time.perf_counter_ns() - begin_ns) / 1e6
                barrier.wait(timeout=1800)
                steps.append(
                    {
                        "outer_iteration": outer_iteration,
                        "step_wall_ms": elapsed_ms,
                        "finished_outputs": sum(output.finished for output in outputs),
                    }
                )
                outer_iteration += 1
                if outer_iteration > 32:
                    raise RuntimeError("prefill profile exceeded 32 bounded iterations")
            wave_records.append(
                {
                    "wave": wave_id,
                    "replicate": replicate,
                    "request_ids": [str(value) for value in request_ids],
                    "wall_ms": (time.perf_counter_ns() - begin_wave_ns) / 1e6,
                    "steps": steps,
                }
            )
            progress.put((rank, wave_id, "COMPLETE"))
        from poc_mllm_ep_collective_granularity.instrumentation import _drain_events
        from poc_mllm_ep_communication_motivation.instrumentation import _drain

        _drain()
        _drain_events()
        atomic_json(
            args.output / f"worker.dp{rank}.json", {"ok": True, "waves": wave_records}
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
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--watchdog-sec", type=int, default=3600)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3,6,7":
        raise RuntimeError("requires CUDA_VISIBLE_DEVICES=2,3,6,7")
    if os.environ.get("MLLM_COMM_MOTIVATION_HOOK_IMPORTED") != "1":
        raise RuntimeError("startup hook was not imported")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    request = read_json(args.manifest)["q3_large_visual"]
    atomic_json(
        args.output / "manifest.json",
        {
            "status": "FROZEN",
            "physical_gpus": [2, 3, 6, 7],
            "topology": "TP2-DP2-EP4-PP1",
            "chunk_size_requested": args.chunk_size,
            "chunk_label": chunk_label(args.chunk_size),
            "enable_chunked_prefill": args.chunk_size > 0,
            "max_num_batched_tokens": 20_000
            if args.chunk_size == 0
            else args.chunk_size,
            "max_num_seqs_per_dp": 1,
            "repeats": args.repeats,
            "request": request,
        },
    )
    atomic_json(args.control, {"condition": "STARTUP", "capture_stage": False})
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    progress = context.Queue()
    master_port = free_port()
    workers = [
        context.Process(
            target=worker,
            args=(rank, master_port, args, barrier, progress, request),
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
        args.output / "run.json", {"status": "COMPLETE", "waves": args.repeats + 1}
    )


if __name__ == "__main__":
    main()
