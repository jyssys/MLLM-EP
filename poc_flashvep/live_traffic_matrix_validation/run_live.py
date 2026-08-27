"""Run a bounded subset of the validated real-image Qwen3-VL workload."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import shutil
import socket
import time
import traceback
from pathlib import Path
from typing import Any

from poc_flashvep.live_prefill_execution_regime.run_live import (
    _generate,
    _requests,
    _source_dp,
)


REQUEST_PAIRS = (0, 1, 4, 8, 9, 12, 16, 20)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _schedule(previous: Path, warmups: int, iterations: int) -> list[dict[str, Any]]:
    manifest = json.loads((previous / "workload_manifest.json").read_text())
    captures: dict[str, int] = {}
    for rank in range(2):
        payload = json.loads((previous / f"capture.dp_rank{rank}.json").read_text())
        captures.update({row["request_id"]: rank for row in payload["records"]})
    selected = [pair for pair in manifest["pairs"] if int(pair["pair_id"]) in REQUEST_PAIRS]
    if len(selected) != len(REQUEST_PAIRS):
        raise AssertionError("validated request subset is incomplete")
    rows = []
    for pair in selected:
        item = pair["vision"]
        rows.append({
            "request_id": item["request_id"], "modality": "vision",
            "pair_id": int(pair["pair_id"]), "token_bucket": pair["token_bucket"],
            "prompt_tokens": int(item["prompt_tokens"]),
            "source_dp_rank": _source_dp(pair, "vision", captures),
        })
    schedule: list[dict[str, Any]] = []
    for row in rows:
        for iteration in range(warmups + iterations):
            schedule.append({
                **row, "phase": "main", "instrument": True,
                "measured": iteration >= warmups, "iteration": iteration - warmups,
            })
    for wave, row in enumerate(schedule):
        row["wave"] = wave
    return schedule


def _run_rank(rank: int, port: int, args: argparse.Namespace,
              barrier: Any, schedule: list[dict[str, Any]]) -> None:
    output = args.output_dir / f"driver.dp_rank{rank}.json"
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": "2", "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_MATRIX_CONTROL": str((args.output_dir / "control.json").resolve()),
            "FLASHVEP_MATRIX_RAW_DIR": str((args.output_dir / "raw_live").resolve()),
            "FLASHVEP_DEEPEP_PROOF_DIR": str((args.output_dir / "backend_proof").resolve()),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false",
        })
        if args.instrument:
            os.environ["FLASHVEP_MATRIX_ENABLE"] = "1"
        from vllm import LLM, SamplingParams

        requests = _requests(args.previous, args.model_path)
        llm = LLM(
            model=args.model_path, dtype="bfloat16", tensor_parallel_size=2,
            enable_expert_parallel=True, expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput", enable_dbo=False,
            enable_return_routed_experts=False, enable_ep_weight_filter=True,
            trust_remote_code=True, gpu_memory_utilization=0.90,
            kv_cache_memory_bytes=1 << 30, max_model_len=4096,
            max_num_batched_tokens=16384, max_num_seqs=2,
            skip_mm_profiling=True, enable_prefix_caching=False,
            enable_flashinfer_autotune=False, enforce_eager=True,
        )
        sampling = SamplingParams(max_tokens=1, temperature=0.0)
        records = []
        for entry in schedule:
            if rank == 0:
                tmp = args.output_dir / "control.tmp.json"
                _write(tmp, entry); tmp.replace(args.output_dir / "control.json")
            barrier.wait(timeout=900)
            prompt = ([copy.deepcopy(requests[entry["request_id"]])]
                      if rank == entry["source_dp_rank"] else [])
            start = time.perf_counter_ns()
            outputs = _generate(llm, prompt, sampling, barrier, int(entry["wave"]))
            wall = (time.perf_counter_ns() - start) / 1_000_000
            tokens = [int(t) for out in outputs for t in out.outputs[0].token_ids]
            records.append({**entry, "driver_dp_rank": rank, "wall_ms": wall,
                            "output_tokens": tokens})
        flush = {**schedule[-1], "wave": len(schedule), "flush": True,
                 "instrument": False, "measured": False}
        if rank == 0:
            tmp = args.output_dir / "control.tmp.json"
            _write(tmp, flush); tmp.replace(args.output_dir / "control.json")
        barrier.wait(timeout=900)
        prompt = ([copy.deepcopy(requests[flush["request_id"]])]
                  if rank == flush["source_dp_rank"] else [])
        _generate(llm, prompt, sampling, barrier, int(flush["wave"]))
        _write(output, {"ok": True, "records": records})
    except BaseException:
        _write(output, {"ok": False, "traceback": traceback.format_exc()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--instrument", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("workload_manifest.json", "text_prompts.json"):
        shutil.copy2(args.previous / name, args.output_dir / name)
    schedule = _schedule(args.previous, args.warmups, args.iterations)
    _write(args.output_dir / "schedule.json", schedule)
    manifest = json.loads((args.previous / "workload_manifest.json").read_text())
    _write(args.output_dir / "run_metadata.json", {
        "source_result": str(args.previous), "request_pairs": list(REQUEST_PAIRS),
        "requests": len(REQUEST_PAIRS), "warmups": args.warmups,
        "iterations": args.iterations, "instrument": args.instrument,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_path": args.model_path, "configuration": manifest.get("configuration", {}),
    })
    context = mp.get_context("spawn")
    barrier = context.Barrier(2); port = _port()
    processes = [context.Process(target=_run_rank, args=(rank, port, args, barrier, schedule))
                 for rank in range(2)]
    for process in processes: process.start()
    for process in processes: process.join()
    codes = [process.exitcode for process in processes]
    if codes != [0, 0]:
        raise RuntimeError(f"live run failed: {codes}")
    print(args.output_dir)


if __name__ == "__main__":
    main()
