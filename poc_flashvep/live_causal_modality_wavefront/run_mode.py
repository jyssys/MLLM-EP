"""Run stock or live-wavefront mode over the fixed 24-image workload."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import socket
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from poc_flashvep.live_prefill_execution_regime.run_live import _requests


IMAGE_TOKEN_ID = 151655


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _schedule(previous: Path, warmups: int, iterations: int) -> list[dict[str, Any]]:
    manifest = json.loads((previous / "workload_manifest.json").read_text())
    rows = []
    for pair in manifest["pairs"]:
        item = pair["vision"]
        route = np.load(previous / item["route_file"])
        token_ids = route["prompt_token_ids"].astype(np.int64)
        image_positions = np.flatnonzero(token_ids == IMAGE_TOKEN_ID)
        if image_positions.size == 0:
            raise AssertionError(item["request_id"])
        rows.append(
            {
                "request_id": item["request_id"],
                "pair_id": int(pair["pair_id"]),
                "token_bucket": pair["token_bucket"],
                "prompt_tokens": int(token_ids.size),
                "prefix_tokens": int(image_positions[-1]) + 1,
                "tail_tokens": int(token_ids.size - image_positions[-1] - 1),
            }
        )
    schedule = []
    for row in rows:
        for iteration in range(warmups):
            schedule.append(
                {**row, "phase": "warmup", "measured": False, "iteration": iteration}
            )
        for iteration in range(iterations):
            schedule.append(
                {**row, "phase": "measured", "measured": True, "iteration": iteration}
            )
        schedule.append(
            {**row, "phase": "correctness", "measured": False, "iteration": 0}
        )
    for wave, row in enumerate(schedule):
        row["wave"] = wave
        row["timeline"] = (
            row["request_id"] == "histology"
            and row["phase"] == "measured"
            and row["iteration"] == 0
        )
    return schedule


def _generate(llm: Any, prompt: dict[str, Any], sampling: Any, barrier: Any) -> Any:
    from vllm.outputs import RequestOutput

    barrier.wait(timeout=900)
    llm._add_completion_requests([copy.deepcopy(prompt)], sampling, use_tqdm=False)
    output = llm._run_engine(RequestOutput, use_tqdm=False)
    barrier.wait(timeout=900)
    return output


def _run_rank(
    rank: int,
    port: int,
    args: argparse.Namespace,
    barrier: Any,
    schedule: list[dict[str, Any]],
) -> None:
    output_path = args.output_dir / f"driver.dp_rank{rank}.json"
    try:
        os.environ.update(
            {
                "VLLM_DP_RANK": str(rank),
                "VLLM_DP_RANK_LOCAL": str(rank),
                "VLLM_DP_SIZE": "2",
                "VLLM_DP_MASTER_IP": "127.0.0.1",
                "VLLM_DP_MASTER_PORT": str(port),
                "FLASHVEP_LIVE_WAVEFRONT_CONTROL": str(
                    (args.output_dir / "control.json").resolve()
                ),
                "FLASHVEP_LIVE_WAVEFRONT_RAW": str((args.output_dir / "raw").resolve()),
                "FLASHVEP_LIVE_WAVEFRONT_MODE": args.mode,
                "FLASHVEP_DEEPEP_PROOF_DIR": str(
                    (args.output_dir / "backend_proof").resolve()
                ),
                "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
                "FLASHVEP_CONFIGURED_DBO": str(args.mode == "wavefront").lower(),
            }
        )
        from vllm import LLM, SamplingParams

        requests = _requests(args.previous, args.model_path)
        wavefront = args.mode == "wavefront"
        llm = LLM(
            model=args.model_path,
            dtype="bfloat16",
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput",
            enable_dbo=wavefront,
            dbo_prefill_token_threshold=1,
            dbo_decode_token_threshold=32,
            enable_return_routed_experts=False,
            enable_ep_weight_filter=True,
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
            kv_cache_memory_bytes=1 << 30,
            max_model_len=4096,
            max_num_batched_tokens=16384,
            max_num_seqs=2,
            skip_mm_profiling=True,
            enable_prefix_caching=False,
            enable_flashinfer_autotune=False,
            enforce_eager=True,
        )
        sampling = SamplingParams(max_tokens=1, temperature=0.0)
        records = []
        for entry in schedule:
            if rank == 0:
                temporary = args.output_dir / "control.tmp.json"
                _json(temporary, entry)
                temporary.replace(args.output_dir / "control.json")
            barrier.wait(timeout=900)
            start = time.perf_counter_ns()
            outputs = _generate(llm, requests[entry["request_id"]], sampling, barrier)
            wall_ms = (time.perf_counter_ns() - start) / 1_000_000
            tokens = [int(token) for token in outputs[0].outputs[0].token_ids]
            records.append(
                {
                    **entry,
                    "driver_dp_rank": rank,
                    "ttft_wall_ms": wall_ms,
                    "output_tokens": tokens,
                }
            )
        # Preserve all requested measurements before the instrumentation-flush
        # request.  A flush failure must not discard an otherwise complete run.
        _json(output_path, {"ok": True, "mode": args.mode, "records": records})
        flush = {
            **schedule[-1],
            "wave": len(schedule),
            "phase": "flush",
            "measured": False,
            "timeline": False,
            "iteration": 0,
        }
        if rank == 0:
            temporary = args.output_dir / "control.tmp.json"
            _json(temporary, flush)
            temporary.replace(args.output_dir / "control.json")
        barrier.wait(timeout=900)
        _generate(llm, requests[flush["request_id"]], sampling, barrier)
    except BaseException:
        _json(
            output_path,
            {
                "ok": False,
                "mode": args.mode,
                "records": locals().get("records", []),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("stock", "wavefront"), required=True)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--max-requests", type=int, default=24)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule = _schedule(args.previous, args.warmups, args.iterations)
    request_ids = list(dict.fromkeys(row["request_id"] for row in schedule))[
        : args.max_requests
    ]
    schedule = [row for row in schedule if row["request_id"] in request_ids]
    for wave, row in enumerate(schedule):
        row["wave"] = wave
    _json(args.output_dir / "schedule.json", schedule)
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    port = _port()
    processes = [
        context.Process(target=_run_rank, args=(rank, port, args, barrier, schedule))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    exits = [process.exitcode for process in processes]
    if exits != [0, 0]:
        raise RuntimeError(f"{args.mode} failed: {exits}")
    print(args.output_dir)


if __name__ == "__main__":
    main()
