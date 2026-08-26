"""Run preregistered stock A or sequential split S with DBO disabled."""

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
REQUEST_IDS = ("coins", "histology", "method")
WARMUPS = 3
ITERATIONS = 10


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _schedule(previous: Path, variant: str) -> list[dict[str, Any]]:
    manifest = json.loads((previous / "workload_manifest.json").read_text())
    by_id = {pair["vision"]["request_id"]: pair for pair in manifest["pairs"]}
    schedule = []
    for request_id in REQUEST_IDS:
        pair = by_id[request_id]
        route = np.load(previous / pair["vision"]["route_file"])
        token_ids = route["prompt_token_ids"].astype(np.int64)
        image_positions = np.flatnonzero(token_ids == IMAGE_TOKEN_ID)
        prefix = int(image_positions[-1]) + 1
        base = {
            "request_id": request_id,
            "variant": variant,
            "prompt_tokens": int(token_ids.size),
            "prefix_tokens": prefix,
            "tail_tokens": int(token_ids.size - prefix),
        }
        for iteration in range(WARMUPS):
            schedule.append(
                base
                | {
                    "phase": "warmup",
                    "iteration": iteration,
                    "measured": False,
                    "stage_profile": False,
                    "flush_after": False,
                }
            )
        for iteration in range(ITERATIONS):
            schedule.append(
                base
                | {
                    "phase": "measured",
                    "iteration": iteration,
                    "measured": True,
                    "stage_profile": iteration == 0,
                    "flush_after": False,
                }
            )
        schedule.append(
            base
            | {
                "phase": "correctness",
                "iteration": 0,
                "measured": False,
                "stage_profile": False,
                "flush_after": False,
            }
        )
    schedule[-1]["flush_after"] = True
    for wave, entry in enumerate(schedule):
        entry["wave"] = wave
    return schedule


def _set_control(directory: Path, entry: dict[str, Any]) -> None:
    temporary = directory / "control.tmp.json"
    _json(temporary, entry)
    temporary.replace(directory / "control.json")


def _generate(llm: Any, prompt: dict[str, Any], sampling: Any, barrier: Any) -> Any:
    from vllm.outputs import RequestOutput

    barrier.wait(timeout=900)
    llm._add_completion_requests([copy.deepcopy(prompt)], sampling, use_tqdm=False)
    output = llm._run_engine(RequestOutput, use_tqdm=False)
    barrier.wait(timeout=900)
    return output


def _rank(
    rank: int,
    port: int,
    args: argparse.Namespace,
    barrier: Any,
    schedule: list[dict[str, Any]],
) -> None:
    output = args.output_dir / f"driver.dp_rank{rank}.json"
    records = []
    try:
        os.environ.update(
            {
                "VLLM_DP_RANK": str(rank),
                "VLLM_DP_RANK_LOCAL": str(rank),
                "VLLM_DP_SIZE": "2",
                "VLLM_DP_MASTER_IP": "127.0.0.1",
                "VLLM_DP_MASTER_PORT": str(port),
                "FLASHVEP_NON_DBO_WAVEFRONT_ENABLE": "1",
                "FLASHVEP_NON_DBO_WAVEFRONT_CONTROL": str(
                    (args.output_dir / "control.json").resolve()
                ),
                "FLASHVEP_NON_DBO_WAVEFRONT_RAW": str(
                    (args.output_dir / "raw").resolve()
                ),
                "FLASHVEP_NON_DBO_WAVEFRONT_VARIANT": args.variant,
                "FLASHVEP_NON_DBO_WAVEFRONT_CODE_SHA": args.code_sha,
            }
        )
        from vllm import LLM, SamplingParams

        requests = _requests(args.previous, args.model_path)
        llm = LLM(
            model=args.model_path,
            dtype="bfloat16",
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput",
            enable_dbo=False,
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
        for entry in schedule:
            if rank == 0:
                _set_control(args.output_dir, entry)
            barrier.wait(timeout=900)
            start = time.perf_counter_ns()
            outputs = _generate(llm, requests[entry["request_id"]], sampling, barrier)
            records.append(
                entry
                | {
                    "driver_dp_rank": rank,
                    "wall_ms": (time.perf_counter_ns() - start) / 1_000_000,
                    "output_tokens": [
                        int(token) for token in outputs[0].outputs[0].token_ids
                    ],
                }
            )
        _json(
            output,
            {
                "ok": True,
                "variant": args.variant,
                "code_sha": args.code_sha,
                "records": records,
            },
        )
    except BaseException:
        _json(
            output,
            {
                "ok": False,
                "variant": args.variant,
                "code_sha": args.code_sha,
                "records": records,
                "traceback": traceback.format_exc(),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("A", "S"))
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-sha", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule = _schedule(args.previous, args.variant)
    _json(
        args.output_dir / "experiment_contract.json",
        {
            "variant": args.variant,
            "code_sha": args.code_sha,
            "dbo": False,
            "host_owner_threads": 1,
            "requests": list(REQUEST_IDS),
            "warmups": WARMUPS,
            "iterations": ITERATIONS,
            "schedule": schedule,
        },
    )
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    port = _port()
    processes = [
        context.Process(target=_rank, args=(rank, port, args, barrier, schedule))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    exits = [process.exitcode for process in processes]
    if exits != [0, 0]:
        raise RuntimeError(f"{args.variant} failed: {exits}")


if __name__ == "__main__":
    main()
