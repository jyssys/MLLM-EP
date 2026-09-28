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

from poc_mllm_rank_aware_active_expert_reduction.q2_2_causal_visual_path.contract import (
    MODEL_PATH, TOPOLOGY, atomic_json, read_json,
)
from poc_mllm_vision_conditioned_decode_reduction.scripts.capture_decode_vectors_ep4 import (
    _ep_wave, _model_kwargs, _prepare,
)


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _rows(payload: dict[str, Any], limit: int | None) -> list[dict[str, Any]]:
    cases = payload["cases"][:limit] if limit else payload["cases"]
    rows = []
    for case in cases:
        forced = [int(value) for value in case["forced_token_ids"]]
        for condition, path, digest in (
            ("orig", case["original_image_path"], case["original_image_sha256"]),
            ("cf", case["counterfactual_image_path"], case["counterfactual_image_sha256"]),
        ):
            rows.append({
                "capture_id": f"{case['case_id']}.{condition}",
                "case_id": case["case_id"],
                "condition": condition,
                "request_id": case["request_id"],
                "domain": case["domain"],
                "split": case["split"],
                "question": case["question"],
                "image_path": path,
                "image_sha256": digest,
                "forced_token_ids": forced,
                "probe_step": int(case["probe_step"]),
                "target_layer": int(case["target_layer"]),
                "forced_prefix_sha256": case["forced_prefix_sha256"],
            })
    return rows


def _sampling(row: dict[str, Any]):
    from vllm import SamplingParams
    forced = row["forced_token_ids"]
    return SamplingParams(
        max_tokens=len(forced), temperature=0.0, seed=3407,
        extra_args={"vcdr_forced_token_ids": forced},
    )


def _worker(rank: int, port: int, args: argparse.Namespace, barrier: Any, progress: Any, rows: list[dict[str, Any]]) -> None:
    llm = None
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(rank),
            "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": "2",
            "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false",
            "MLLM_Q22_CAPTURE_ENABLE": "1",
            "MLLM_Q22_CAPTURE_CONTROL": str(args.control.resolve()),
            "MLLM_Q22_CAPTURE_RAW": str((args.output / "raw").resolve()),
        })
        hook = str((Path(__file__).parents[1] / "hooks").resolve())
        os.environ["PYTHONPATH"] = f"{hook}:{os.environ.get('PYTHONPATH', '')}"
        from transformers import AutoProcessor
        from vllm import LLM
        from poc_mllm_rank_aware_active_expert_reduction.q2_2_causal_visual_path.causal_vector_capture import install

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
                    "target_layer": row["target_layer"],
                    "target_step": row["probe_step"],
                })
            barrier.wait(timeout=args.wave_timeout_sec)
            prepared = _prepare(processor, row) if rank == source else None
            prompts = [copy.deepcopy(prepared[0])] if prepared else []
            params = [_sampling(row)] if prepared else []
            started = time.perf_counter_ns()
            outputs = _ep_wave(llm, prompts, params, barrier, wave)
            wall_ms = (time.perf_counter_ns() - started) / 1e6
            if rank == source:
                if len(outputs) != 1:
                    raise AssertionError((wave, source, len(outputs)))
                actual = [int(value) for value in outputs[0].outputs[0].token_ids]
                if actual != row["forced_token_ids"]:
                    raise AssertionError(f"forced mismatch: {row['capture_id']}")
                records.append({
                    "capture_id": row["capture_id"],
                    "case_id": row["case_id"],
                    "condition": row["condition"],
                    "source_dp_rank": source,
                    "forced_exact": True,
                    "token_count": len(actual),
                    "wall_ms_instrumented": wall_ms,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--max-model-len", type=int, default=20000)
    parser.add_argument("--kv-cache-gib", type=float, default=2.0)
    parser.add_argument("--wave-timeout-sec", type=int, default=1800)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3,6,7":
        raise RuntimeError("Q2-2 TRUE EP4 requires CUDA_VISIBLE_DEVICES=2,3,6,7")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    args.control.parent.mkdir(parents=True, exist_ok=True)
    rows = _rows(read_json(args.manifest), args.limit_cases)
    atomic_json(args.control, {"capture": False, "capture_id": "startup"})
    atomic_json(args.output / "manifest.json", {
        "schema_version": 1,
        "condition_executions": len(rows),
        "case_count": len(rows) // 2,
        "topology": TOPOLOGY,
        "physical_gpus": [2, 3, 6, 7],
        "evidence": "INSTRUMENTED_CAUSAL_VECTOR_CAPTURE_NOT_CLEAN_TIMING",
    })
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    progress = context.Queue()
    port = _port()
    workers = [context.Process(target=_worker, args=(rank, port, args, barrier, progress, rows)) for rank in range(2)]
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
    atomic_json(args.output / "run.json", {"status": "COMPLETE", "condition_executions": len(rows)})
    print(json.dumps({"status": "COMPLETE", "conditions": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
