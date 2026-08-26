"""Run one preregistered A0/A1/A2/C live-forensics variant."""

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
STAGE_REQUEST_ID = "histology"
WARMUPS = 3
ITERATIONS = 10


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request_rows(previous: Path) -> list[dict[str, Any]]:
    manifest = json.loads((previous / "workload_manifest.json").read_text())
    by_id = {pair["vision"]["request_id"]: pair for pair in manifest["pairs"]}
    rows = []
    for request_id in REQUEST_IDS:
        pair = by_id[request_id]
        item = pair["vision"]
        route = np.load(previous / item["route_file"])
        token_ids = route["prompt_token_ids"].astype(np.int64)
        image_positions = np.flatnonzero(token_ids == IMAGE_TOKEN_ID)
        if image_positions.size == 0:
            raise AssertionError(request_id)
        prefix_tokens = int(image_positions[-1]) + 1
        rows.append(
            {
                "request_id": request_id,
                "pair_id": int(pair["pair_id"]),
                "token_bucket": pair["token_bucket"],
                "prompt_tokens": int(token_ids.size),
                "prefix_tokens": prefix_tokens,
                "tail_tokens": int(token_ids.size - prefix_tokens),
            }
        )
    return rows


def _schedule(
    previous: Path, variant: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    schedule = []
    rows = _request_rows(previous)
    for row in rows:
        for iteration in range(WARMUPS):
            schedule.append(
                {
                    **row,
                    "variant": variant,
                    "phase": "warmup",
                    "measured": False,
                    "iteration": iteration,
                    "stage_profile": False,
                    "torch_profile": False,
                    "flush_after": False,
                }
            )
        for iteration in range(ITERATIONS):
            schedule.append(
                {
                    **row,
                    "variant": variant,
                    "phase": "measured",
                    "measured": True,
                    "iteration": iteration,
                    "stage_profile": request_id_is_stage(row, iteration),
                    "torch_profile": False,
                    "flush_after": False,
                }
            )
        schedule.append(
            {
                **row,
                "variant": variant,
                "phase": "correctness",
                "measured": False,
                "iteration": 0,
                "stage_profile": False,
                "torch_profile": False,
                "flush_after": False,
            }
        )
    schedule[-1]["flush_after"] = True
    for wave, entry in enumerate(schedule):
        entry["wave"] = wave
    profile_row = next(row for row in rows if row["request_id"] == STAGE_REQUEST_ID)
    profile_entry = {
        **profile_row,
        "variant": variant,
        "phase": "torch_profile",
        "measured": False,
        "iteration": 0,
        "stage_profile": False,
        "torch_profile": True,
        "flush_after": False,
        "wave": len(schedule),
    }
    return schedule, profile_entry


def request_id_is_stage(row: dict[str, Any], iteration: int) -> bool:
    return row["request_id"] == STAGE_REQUEST_ID and iteration == 0


def _set_control(output_dir: Path, entry: dict[str, Any]) -> None:
    temporary = output_dir / "control.tmp.json"
    _json(temporary, entry)
    temporary.replace(output_dir / "control.json")


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
    profile_entry: dict[str, Any],
) -> None:
    output_path = args.output_dir / f"driver.dp_rank{rank}.json"
    records: list[dict[str, Any]] = []
    profile_result: dict[str, Any] = {"attempted": False, "completed": False}
    try:
        os.environ.update(
            {
                "VLLM_DP_RANK": str(rank),
                "VLLM_DP_RANK_LOCAL": str(rank),
                "VLLM_DP_SIZE": "2",
                "VLLM_DP_MASTER_IP": "127.0.0.1",
                "VLLM_DP_MASTER_PORT": str(port),
                "FLASHVEP_WAVEFRONT_FORENSICS_ENABLE": "1",
                "FLASHVEP_WAVEFRONT_FORENSICS_CONTROL": str(
                    (args.output_dir / "control.json").resolve()
                ),
                "FLASHVEP_WAVEFRONT_FORENSICS_RAW": str(
                    (args.output_dir / "raw").resolve()
                ),
                "FLASHVEP_WAVEFRONT_FORENSICS_VARIANT": args.variant,
                "FLASHVEP_WAVEFRONT_FORENSICS_CODE_SHA": args.code_sha,
                "FLASHVEP_DEEPEP_PROOF_DIR": str(
                    (args.output_dir / "backend_proof").resolve()
                ),
                "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
                "FLASHVEP_CONFIGURED_DBO": str(args.variant != "A0").lower(),
            }
        )
        from vllm import LLM, SamplingParams

        requests = _requests(args.previous, args.model_path)
        dbo = args.variant != "A0"
        llm = LLM(
            model=args.model_path,
            dtype="bfloat16",
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput",
            enable_dbo=dbo,
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
        for entry in schedule:
            if rank == 0:
                _set_control(args.output_dir, entry)
            barrier.wait(timeout=900)
            start = time.perf_counter_ns()
            outputs = _generate(llm, requests[entry["request_id"]], sampling, barrier)
            wall_ms = (time.perf_counter_ns() - start) / 1_000_000
            records.append(
                {
                    **entry,
                    "driver_dp_rank": rank,
                    "ttft_wall_ms": wall_ms,
                    "output_tokens": [
                        int(token) for token in outputs[0].outputs[0].token_ids
                    ],
                }
            )
        _json(
            output_path,
            {
                "ok": True,
                "variant": args.variant,
                "code_sha": args.code_sha,
                "records": records,
                "torch_profile": profile_result,
            },
        )

        profile_result["attempted"] = True
        if rank == 0:
            _set_control(args.output_dir, profile_entry)
        barrier.wait(timeout=180)
        try:
            start = time.perf_counter_ns()
            _generate(
                llm,
                requests[profile_entry["request_id"]],
                sampling,
                barrier,
            )
            profile_result.update(
                {
                    "completed": True,
                    "wall_ms": (time.perf_counter_ns() - start) / 1_000_000,
                }
            )
        except BaseException:
            profile_result["traceback"] = traceback.format_exc()
        _json(
            output_path,
            {
                "ok": True,
                "variant": args.variant,
                "code_sha": args.code_sha,
                "records": records,
                "torch_profile": profile_result,
            },
        )
    except BaseException:
        _json(
            output_path,
            {
                "ok": False,
                "variant": args.variant,
                "code_sha": args.code_sha,
                "records": records,
                "torch_profile": profile_result,
                "traceback": traceback.format_exc(),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("A0", "A1", "A2", "C"), required=True)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-sha", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule, profile_entry = _schedule(args.previous, args.variant)
    _json(
        args.output_dir / "experiment_contract.json",
        {
            "variant": args.variant,
            "code_sha": args.code_sha,
            "requests": list(REQUEST_IDS),
            "stage_request": STAGE_REQUEST_ID,
            "warmups": WARMUPS,
            "iterations": ITERATIONS,
            "schedule": schedule,
            "torch_profile_entry": profile_entry,
        },
    )
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    port = _port()
    processes = [
        context.Process(
            target=_run_rank,
            args=(rank, port, args, barrier, schedule, profile_entry),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    exits = [process.exitcode for process in processes]
    if exits != [0, 0]:
        raise RuntimeError(f"{args.variant} failed: {exits}")
    print(args.output_dir)


if __name__ == "__main__":
    main()
