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

from poc_mllm_ep_collective_granularity.contract import MODEL, atomic_json, read_json
from poc_mllm_ep_collective_granularity.scripts.run_serving import make_injection


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
    waves: list[dict[str, Any]],
    manifest: dict[str, Any],
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
        from vllm.outputs import RequestOutput
        from vllm.v1.engine import EngineCoreRequestType

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
            max_model_len=20_000,
            max_num_batched_tokens=20_000,
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
        for wave_id, entry in enumerate(waves):
            prompt, token_count, preprocess = make_injection(
                processor, entry["condition"], manifest["q3_large_visual"]
            )
            control = {
                **entry,
                "wave": wave_id,
                "outer_iteration": 0,
                "special_token_ids": special,
                "expected_phase": "prefill",
                "capture_stage": entry["replicate"] >= 0,
                "route_stats": entry["replicate"] >= 0,
                "moe_only_replay": entry["replicate"] >= 0,
            }
            if rank == 0:
                atomic_json(args.control, control)
            barrier.wait(timeout=1800)
            begin = time.perf_counter_ns()
            if rank == 0:
                llm._add_completion_requests(
                    [copy.deepcopy(prompt)],
                    [SamplingParams(max_tokens=1, temperature=0.0, seed=3407)],
                    use_tqdm=False,
                )
                outputs = llm._run_engine(RequestOutput, use_tqdm=False)
            else:
                llm.llm_engine.engine_core._send_input(
                    EngineCoreRequestType.START_DP_WAVE, (wave_id, -1)
                )
                outputs = []
            barrier.wait(timeout=1800)
            records.append(
                {
                    "wave": wave_id,
                    "condition": entry["condition"],
                    "replicate": entry["replicate"],
                    "prompt_tokens": token_count,
                    "wall_ms": (time.perf_counter_ns() - begin) / 1e6,
                    "output_count": len(outputs),
                    **preprocess,
                }
            )
            progress.put((rank, wave_id, "COMPLETE"))
        from poc_mllm_ep_collective_granularity.instrumentation import _drain_events
        from poc_mllm_ep_communication_motivation.instrumentation import _drain

        _drain()
        _drain_events()
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
    conditions = ["VISUAL_PREFILL", "TOKEN_MATCHED_TEXT"]
    waves = [{"condition": condition, "replicate": -1} for condition in conditions]
    for replicate in range(args.repeats):
        offset = replicate % len(conditions)
        for condition in conditions[offset:] + conditions[:offset]:
            waves.append({"condition": condition, "replicate": replicate})
    manifest = read_json(args.manifest)
    atomic_json(
        args.output / "manifest.json",
        {
            "status": "FROZEN",
            "physical_gpus": [2, 3, 6, 7],
            "topology": "TP2-DP2-EP4-PP1",
            "conditions": conditions,
            "repeats": args.repeats,
            "contract": (
                "exact per-layer hidden/top-k IDs/top-k weights are cloned before "
                "the native MoE call and physically replayed once through the same "
                "TRUE-EP4 DeepEP kernel; replay output is discarded"
            ),
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
            args=(rank, master_port, args, barrier, progress, waves, manifest),
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
    atomic_json(args.output / "run.json", {"status": "COMPLETE", "waves": len(waves)})


if __name__ == "__main__":
    main()
