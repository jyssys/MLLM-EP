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

from ..contract import LAYERS, MODEL_PATH, PHYSICAL_GPUS, STEPS, TOPOLOGY, atomic_json, read_json
from poc_mllm_vision_conditioned_decode_reduction.scripts.capture_decode_vectors_ep4 import (
    _ep_wave,
    _model_kwargs,
    _prepare,
)


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def conditions(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    rows = []
    mapping = (
        ("original", "original_image_path", "original_image_sha256"),
        ("same_domain_swap", "counterfactual_image_path", "counterfactual_image_sha256"),
        ("semantic_neutral", "neutral_image_path", None),
    )
    for case in manifest["cases"]:
        if case["split"] != split:
            continue
        for condition, path_key, hash_key in mapping:
            row = dict(case)
            row.update({
                "condition": condition,
                "capture_id": f"{case['case_id']}.{condition}",
                "image_path": case[path_key],
                "image_sha256": None if hash_key is None else case[hash_key],
            })
            rows.append(row)
    return rows


def _sampling(row: dict[str, Any]):
    from vllm import SamplingParams
    forced = [int(value) for value in row["forced_token_ids"]]
    return SamplingParams(
        max_tokens=len(forced), temperature=0.0, seed=3407,
        extra_args={"vcdr_forced_token_ids": forced},
    )


def _worker(rank: int, port: int, args: argparse.Namespace, barrier: Any,
            progress: Any, rows: list[dict[str, Any]]) -> None:
    llm = None
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank), "VLLM_DP_SIZE": "2",
            "VLLM_DP_MASTER_IP": "127.0.0.1", "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false",
            "MLLM_DMC_TRACE_ENABLE": "1",
            "MLLM_DMC_TRACE_CONTROL": str(args.control.resolve()),
            "MLLM_DMC_TRACE_RAW": str((args.output / "raw").resolve()),
        })
        hook = str((Path(__file__).parents[1] / "hooks").resolve())
        os.environ["PYTHONPATH"] = f"{hook}:{os.environ.get('PYTHONPATH', '')}"
        from transformers import AutoProcessor
        from vllm import LLM
        from ..rich_trace import install

        install()
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        kwargs = _model_kwargs(args.max_model_len, args.kv_cache_gib)
        kwargs.update({"model": MODEL_PATH, "max_num_seqs": 1})
        kwargs.pop("revision", None)
        llm = LLM(**kwargs)
        progress.put((rank, -1, "MODEL_READY"))
        records = []
        for wave, row in enumerate(rows):
            source = wave % 2
            if rank == 0:
                atomic_json(args.control, {
                    "capture": True,
                    "capture_id": row["capture_id"],
                    "source_dp_rank": source,
                    "layers": list(LAYERS),
                    "steps": list(STEPS),
                })
            barrier.wait(timeout=args.wave_timeout_sec)
            prepared = _prepare(processor, row) if rank == source else None
            prompts = [copy.deepcopy(prepared[0])] if prepared else []
            params = [_sampling(row)] if prepared else []
            started = time.perf_counter_ns()
            outputs = _ep_wave(llm, prompts, params, barrier, wave)
            wall_ms = (time.perf_counter_ns() - started) / 1e6
            if rank == source:
                actual = [int(value) for value in outputs[0].outputs[0].token_ids]
                if actual != row["forced_token_ids"]:
                    raise AssertionError(f"forced mismatch: {row['capture_id']}")
                records.append({
                    "capture_id": row["capture_id"], "source_dp_rank": source,
                    "forced_exact": True, "wall_ms_instrumented": wall_ms,
                })
            progress.put((rank, wave, "COMPLETE"))
        atomic_json(args.output / f"worker.dp{rank}.json", {"ok": True, "records": records})
    except BaseException:
        atomic_json(args.output / f"worker.dp{rank}.json", {"ok": False, "traceback": traceback.format_exc()})
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--max-model-len", type=int, default=20000)
    parser.add_argument("--kv-cache-gib", type=float, default=2.0)
    parser.add_argument("--wave-timeout-sec", type=int, default=1800)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3,6,7":
        raise RuntimeError("campaign TRUE EP4 requires CUDA_VISIBLE_DEVICES=2,3,6,7")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    (args.output / "raw").mkdir()
    rows = conditions(read_json(args.manifest), args.split)
    if args.limit_cases is not None:
        selected = {row["case_id"] for row in rows[::3][:args.limit_cases]}
        rows = [row for row in rows if row["case_id"] in selected]
    atomic_json(args.output / "manifest.json", {
        "schema_version": 1, "split": args.split, "condition_executions": len(rows),
        "case_count": len(rows) // 3, "layers": list(LAYERS), "steps": list(STEPS),
        "topology": TOPOLOGY, "physical_gpus": list(PHYSICAL_GPUS),
        "evidence": "MEASURED_INSTRUMENTED_TRUE_EP4_NOT_CLEAN_TIMING",
    })
    atomic_json(args.control, {"capture": False, "capture_id": "startup"})
    context = mp.get_context("spawn")
    barrier, progress, port = context.Barrier(2), context.Queue(), _port()
    workers = [context.Process(target=_worker, args=(rank, port, args, barrier, progress, rows)) for rank in range(2)]
    for worker in workers:
        worker.start()
    last, failure = time.monotonic(), None
    while any(worker.is_alive() for worker in workers):
        try:
            rank, _, status = progress.get(timeout=5)
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
    actual_sites = len(list((args.output / "raw").glob("site.*.npz")))
    expected_sites = len(rows) * len(LAYERS) * len(STEPS)
    if actual_sites != expected_sites:
        raise RuntimeError(f"site count {actual_sites} != {expected_sites}")
    atomic_json(args.output / "run.json", {
        "status": "COMPLETE", "condition_executions": len(rows),
        "actual_sites": actual_sites, "expected_sites": expected_sites,
    })
    print(json.dumps({"status": "COMPLETE", "conditions": len(rows), "sites": actual_sites}, indent=2))


if __name__ == "__main__":
    main()
