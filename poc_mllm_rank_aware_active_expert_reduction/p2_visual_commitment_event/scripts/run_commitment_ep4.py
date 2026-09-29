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
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _prepare(processor: Any, row: dict[str, Any], condition: str) -> dict[str, Any]:
    image_path = row["original_image_path"] if condition == "A" else row["counterfactual_image_path"]
    with Image.open(image_path) as source:
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


def _schedule(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(pairs)
    if len(rows) % 2:
        rows.append({})
    schedule = []
    for offset in range(0, len(rows), 2):
        group = rows[offset : offset + 2]
        for condition in ("A", "B"):
            schedule.append({
                "condition": condition,
                "by_dp": [[group[0]] if group[0] else [], [group[1]] if group[1] else []],
            })
    return schedule


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
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
        llm = LLM(
            model=MODEL, dtype="bfloat16", trust_remote_code=True,
            tensor_parallel_size=2, enable_expert_parallel=True,
            expert_placement_strategy="linear", all2all_backend="deepep_high_throughput",
            enable_ep_weight_filter=True, enable_dbo=False, gpu_memory_utilization=.90,
            kv_cache_memory_bytes=2 << 30, max_model_len=4096,
            max_num_batched_tokens=4096, max_num_seqs=1,
            limit_mm_per_prompt={"image": 1}, skip_mm_profiling=True,
            enable_prefix_caching=False, mm_processor_cache_gb=0,
            enable_flashinfer_autotune=False, enforce_eager=True, disable_log_stats=True,
            logits_processors=[
                "poc_mllm_rank_aware_active_expert_reduction.p2_visual_commitment_event.logits_processor:CommitmentLogitsProcessor"
            ],
        )
        progress.put((rank, -1, "MODEL_READY"))
        records = []
        for wave_id, entry in enumerate(schedule):
            rows = entry["by_dp"][rank]
            condition = entry["condition"]
            prompts = [_prepare(processor, row, condition) for row in rows]
            params = []
            for row in rows:
                forced = [int(value) for value in row["forced_token_ids"][:128]]
                params.append(SamplingParams(
                    max_tokens=len(forced), temperature=0.0, seed=3407,
                    extra_args={
                        "commit_forced_token_ids": forced,
                        "commit_pair_id": row["pair_id"],
                        "commit_condition": condition,
                        "commit_positions": args.positions,
                        "commit_split": row["split"],
                        "commit_domain": row["domain"],
                    },
                ))
            begin = time.perf_counter_ns()
            outputs = _wave(llm, [copy.deepcopy(item) for item in prompts], params, barrier, wave_id)
            wall_ms = (time.perf_counter_ns() - begin) / 1e6
            for row, output in zip(rows, outputs):
                expected = [int(value) for value in row["forced_token_ids"][:128]]
                actual = [int(value) for value in output.outputs[0].token_ids]
                if actual != expected:
                    raise RuntimeError((row["pair_id"], condition, "forced-token parity"))
                records.append({
                    "pair_id": row["pair_id"], "condition": condition,
                    "wave": wave_id, "wall_ms": wall_ms, "forced_exact": True,
                })
            progress.put((rank, wave_id, "COMPLETE"))
        atomic_json(args.output / f"worker.dp{rank}.json", {"ok": True, "records": records})
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
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--positions", type=int, nargs="+",
                        default=list(range(1, 33)) + list(range(36, 129, 4)))
    parser.add_argument("--watchdog-sec", type=int, default=3600)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3,6,7":
        raise RuntimeError("PoC-2 requires CUDA_VISIBLE_DEVICES=2,3,6,7")
    if args.output.exists() or args.raw_output.exists():
        raise FileExistsError((args.output, args.raw_output))
    args.output.mkdir(parents=True)
    args.raw_output.mkdir(parents=True)
    os.environ["MLLM_P2_COMMIT_OUTPUT"] = str(args.raw_output.resolve())
    cohort = read_json(args.cohort)
    schedule = _schedule(cohort["pairs"])
    atomic_json(args.output / "manifest.json", {
        "schema_version": "visual_commitment_run.v1",
        "physical_gpus": [2, 3, 6, 7], "topology": "TP2-DP2-EP4-PP1",
        "backend": "deepep_high_throughput", "max_forced_tokens": 128,
        "positions": args.positions, "pair_count": len(cohort["pairs"]),
        "request_executions": 2 * len(cohort["pairs"]),
        "evidence": "FULL_VOCAB_SAME_PREFIX_SAME_DOMAIN_IMAGE_INTERVENTION",
    })
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
