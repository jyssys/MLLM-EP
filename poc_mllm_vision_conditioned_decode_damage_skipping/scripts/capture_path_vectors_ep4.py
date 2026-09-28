#!/usr/bin/env python3
"""Run the shared exact selected-branch capture on the frozen damage cohort."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import socket
import time
from pathlib import Path

from poc_mllm_vision_conditioned_decode_damage_skipping.contract import atomic_json, read_json
from poc_mllm_vision_conditioned_decode_reduction.scripts.capture_decode_vectors_ep4 import _worker


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--model-path", default=(
        "/home/esjung/.cache/huggingface/hub/"
        "models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/"
        "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
    ))
    parser.add_argument("--layers", default="4,12,24,36,44,47")
    parser.add_argument("--steps", default="8,16,32,48,64,96,112")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--kv-cache-gib", type=float, default=2.0)
    parser.add_argument("--wave-timeout-sec", type=int, default=3600)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4,5,6,7":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 4,5,6,7")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    args.control.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.control, {"capture": False, "capture_id": "startup", "source_dp_rank": -1})
    payload = read_json(args.manifest)
    requests = [{
        "pair_id": row["pair_id"],
        "request_id": f"{row['pair_id']}::damage_vector_capture",
        "condition": "A",
        "question": row["question"],
        "analysis_split": row["damage_split"],
        "image_path": row["original_image_path"],
        "image_sha256": row["original_image_sha256"],
        "forced_token_ids": row["forced_token_ids"],
        "decode_cap": len(row["forced_token_ids"]),
        "run_mode": "damage_vector_capture",
        "phase": row["damage_split"],
    } for row in payload["anchors"]][args.start_index:]
    atomic_json(args.output / "manifest.json", {
        "request_count": len(requests),
        "request_executions": len(requests),
        "request_ids": [row["pair_id"] for row in requests],
        "start_index": args.start_index,
        "layers": [int(value) for value in args.layers.split(",")],
        "steps": [int(value) for value in args.steps.split(",")],
        "topology": "TRUE_TP2_DP2_EP4_PP1",
        "evidence": "INSTRUMENTED_DIAGNOSTIC_NOT_CLEAN_TIMING",
    })
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    progress = context.Queue()
    port = _port()
    workers = [
        context.Process(target=_worker, args=(rank, port, args, barrier, progress, requests))
        for rank in range(2)
    ]
    for worker in workers:
        worker.start()
    last = time.monotonic()
    failure = None
    while any(worker.is_alive() for worker in workers):
        try:
            rank, wave, status = progress.get(timeout=5)
            last = time.monotonic()
            if status == "FAILED":
                failure = f"worker {rank} failed"
                break
        except queue.Empty:
            pass
        if time.monotonic() - last > args.wave_timeout_sec:
            failure = "TIMEOUT: no worker progress"
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
    atomic_json(args.output / "run.json", {"status": "COMPLETE", "requests": len(requests)})
    print(json.dumps({"status": "COMPLETE", "requests": len(requests)}, indent=2))


if __name__ == "__main__":
    main()
