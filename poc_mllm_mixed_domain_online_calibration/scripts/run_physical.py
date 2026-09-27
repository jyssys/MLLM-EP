from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import queue
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from poc_mllm_mixed_domain_online_calibration.contract import (
    DTYPE,
    MODEL,
    MODEL_REVISION,
    atomic_json,
    append_jsonl,
    read_json,
    read_jsonl,
)


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ensure_tools() -> None:
    if not shutil.which("ninja"):
        candidate = Path(sys.executable).parent / "ninja"
        if candidate.is_file():
            os.environ["PATH"] = f"{candidate.parent}:{os.environ.get('PATH', '')}"
    if not shutil.which("ninja"):
        raise RuntimeError("ninja unavailable")


def _prepare(processor: Any, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    with Image.open(row["image_path"]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": row["question"]},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processed = processor(text=[prompt], images=[image], return_tensors="pt")
    grid = [int(value) for value in processed["image_grid_thw"][0].tolist()]
    merge = int(processor.image_processor.merge_size)
    vision_tokens = grid[0] * grid[1] * grid[2] // (merge * merge)
    return ({"prompt": prompt, "multi_modal_data": {"image": image}}, {
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "image_grid_thw": grid,
        "vision_token_count": vision_tokens,
        "prompt_token_count": int(processed["input_ids"].shape[1]),
        "image_size_rgb": list(image.size),
    })


def _model_kwargs(topology: str, max_model_len: int, kv_cache_gib: float) -> dict[str, Any]:
    tp = 1 if topology == "ep2" else 2
    return {
        "model": MODEL,
        "revision": MODEL_REVISION,
        "dtype": DTYPE,
        "trust_remote_code": True,
        "tensor_parallel_size": tp,
        "enable_expert_parallel": True,
        "expert_placement_strategy": "linear",
        "all2all_backend": "deepep_high_throughput",
        "enable_ep_weight_filter": True,
        "enable_return_routed_experts": False,
        "enable_dbo": False,
        "gpu_memory_utilization": 0.90,
        "kv_cache_memory_bytes": int(kv_cache_gib * (1 << 30)),
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_model_len,
        "max_num_seqs": 1,
        "limit_mm_per_prompt": {"image": 1},
        "skip_mm_profiling": True,
        "enable_prefix_caching": False,
        "mm_processor_cache_gb": 0,
        "enable_flashinfer_autotune": False,
        "enforce_eager": True,
        "disable_log_stats": True,
    }


def _sampling(max_tokens: int):
    from vllm import SamplingParams
    return SamplingParams(max_tokens=max_tokens, min_tokens=0, temperature=0.0, seed=3407)


def _wave(llm: Any, prompts: list[dict[str, Any]], params: list[Any], barrier: Any, wave: int):
    from vllm.outputs import RequestOutput
    from vllm.v1.engine import EngineCoreRequestType
    barrier.wait(timeout=1800)
    if prompts:
        llm._add_completion_requests(prompts, params, use_tqdm=False)
        output = llm._run_engine(RequestOutput, use_tqdm=False)
    else:
        llm.llm_engine.engine_core._send_input(EngineCoreRequestType.START_DP_WAVE, (wave, -1))
        output = []
    barrier.wait(timeout=1800)
    return output


def _control(path: Path, wave: int, batches: list[list[dict[str, Any]]], topology: str) -> None:
    atomic_json(path, {
        "wave": wave,
        "topology": topology,
        "by_dp": {
            str(rank): {
                "request_id": batch[0]["request_id"] if batch else f"idle_dp{rank}",
                "domain": batch[0].get("domain") if batch else None,
                "analysis_split": batch[0].get("analysis_split") if batch else None,
            }
            for rank, batch in enumerate(batches)
        },
    })


def _record(row: dict[str, Any], meta: dict[str, Any], output: Any, wall_ms: float, rank: int, wave: int, topology: str) -> dict[str, Any]:
    completion = output.outputs[0]
    ids = [int(value) for value in completion.token_ids]
    metrics = output.metrics
    ttft = None if metrics is None else float(metrics.first_token_latency * 1000.0)
    tpot = None
    if metrics is not None and len(ids) > 1:
        tpot = float((metrics.last_token_ts - metrics.first_token_ts) * 1000.0 / (len(ids) - 1))
    return {
        **row,
        **meta,
        "dp_rank": rank,
        "wave": wave,
        "topology": topology,
        "wall_ms": wall_ms,
        "ttft_ms": ttft,
        "mean_tpot_ms": tpot,
        "output_text": completion.text,
        "output_token_ids": ids,
        "output_token_count": len(ids),
        "finish_reason": completion.finish_reason,
        "stop_reason": str(completion.stop_reason),
        "evidence": f"MEASURED_FULL_MODEL_TRUE_{topology.upper()}",
    }


def _worker(rank: int, port: int, args: argparse.Namespace, barrier: Any, progress: Any, waves: list[list[list[dict[str, Any]]]]) -> None:
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(rank),
            "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": "2",
            "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false",
        })
        from transformers import AutoProcessor
        from vllm import LLM
        processor = AutoProcessor.from_pretrained(MODEL, revision=MODEL_REVISION, trust_remote_code=True)
        llm = LLM(**_model_kwargs(args.topology, args.max_model_len, args.kv_cache_gib))
        progress.put((rank, -1, "MODEL_READY"))
        for wave_id, wave in enumerate(waves):
            if rank == 0:
                _control(args.control, wave_id, wave, args.topology)
            batch = wave[rank]
            prepared = [(row, *_prepare(processor, row)) for row in batch]
            prompts = [copy.deepcopy(prompt) for _, prompt, _ in prepared]
            params = [_sampling(args.max_tokens) for _ in prepared]
            begin = time.perf_counter_ns()
            outputs = _wave(llm, prompts, params, barrier, wave_id)
            wall_ms = (time.perf_counter_ns() - begin) / 1e6
            records = [
                _record(row, meta, output, wall_ms, rank, wave_id, args.topology)
                for (row, _, meta), output in zip(prepared, outputs)
            ]
            if records:
                append_jsonl(args.output / f"results.dp{rank}.jsonl", records)
            progress.put((rank, wave_id, "COMPLETE"))
        atomic_json(args.output / f"worker.dp{rank}.json", {"ok": True})
    except BaseException:
        atomic_json(args.output / f"worker.dp{rank}.json", {"ok": False, "traceback": traceback.format_exc()})
        progress.put((rank, -2, "FAILED"))
        raise


def _ordered_requests(pool: Path, stream: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = {row["request_id"]: row for row in read_json(pool)["requests"]}
    stream_payload = read_json(stream)
    ids = stream_payload.get("request_ids")
    if ids is None:
        ids = [event["request_id"] for event in stream_payload["events"]]
    result = [copy.deepcopy(rows[request]) for request in ids]
    return result[:limit] if limit else result


def _waves(requests: list[dict[str, Any]]) -> list[list[list[dict[str, Any]]]]:
    rows = list(requests)
    if len(rows) % 2:
        rows.append({})
    return [[[row] if row else [] for row in rows[offset : offset + 2]] for offset in range(0, len(rows), 2)]


def _run(args: argparse.Namespace, requests: list[dict[str, Any]]) -> None:
    waves = _waves(requests)
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    progress = context.Queue()
    port = _port()
    workers = [context.Process(target=_worker, args=(rank, port, args, barrier, progress, waves)) for rank in range(2)]
    for worker in workers:
        worker.start()
    completed = {0: set(), 1: set()}
    last = time.monotonic()
    failure = None
    while any(worker.is_alive() for worker in workers):
        try:
            rank, wave, status = progress.get(timeout=5)
            last = time.monotonic()
            if status == "COMPLETE":
                completed[rank].add(wave)
            elif status == "FAILED":
                failure = f"worker {rank} failed"
                break
        except queue.Empty:
            pass
        if time.monotonic() - last > args.wave_timeout_sec:
            failure = f"TIMEOUT after {args.wave_timeout_sec}s without progress"
            break
    if failure:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join(timeout=30)
        atomic_json(args.output / "failure.json", {"status": "RUNTIME_UNAVAILABLE", "reason": failure, "completed": {str(k): sorted(v) for k, v in completed.items()}})
        raise RuntimeError(failure)
    for worker in workers:
        worker.join()
    if any(worker.exitcode != 0 for worker in workers):
        raise RuntimeError([worker.exitcode for worker in workers])
    combined = []
    for rank in range(2):
        combined.extend(read_jsonl(args.output / f"results.dp{rank}.jsonl"))
    combined.sort(key=lambda row: row["request_id"])
    append_jsonl(args.output / "results.jsonl", combined)


def main() -> None:
    _ensure_tools()
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", choices=("ep2", "ep4"), required=True)
    parser.add_argument("--physical-gpus", required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--kv-cache-gib", type=float, default=4.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--wave-timeout-sec", type=int, default=1800)
    args = parser.parse_args()
    expected = "4,5" if args.topology == "ep2" else "4,5,6,7"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.physical_gpus != expected or visible != expected:
        raise RuntimeError(f"{args.topology} requires CUDA_VISIBLE_DEVICES={expected}; got {visible}")
    if not 1 <= args.max_tokens <= 192:
        raise ValueError("main generation must remain within 1..192")
    if os.environ.get("MLLM_MDOC_HOOK_IMPORTED") != "1":
        raise RuntimeError("trace hook was not imported")
    os.environ["MLLM_MDOC_TRACE_DIR"] = str(args.trace_dir.resolve())
    os.environ["MLLM_MDOC_CONTROL"] = str(args.control.resolve())
    requests = _ordered_requests(args.pool, args.stream, args.limit)
    prior = sum(int(row.get("request_executions", 0)) for row in read_jsonl(args.ledger) if row.get("status") == "COMPLETE")
    if prior + len(requests) > 400:
        raise RuntimeError(f"execution budget {prior}+{len(requests)}>400")
    args.output.mkdir(parents=True, exist_ok=True)
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.topology}-{int(time.time())}"
    atomic_json(args.output / "invocation.json", {
        "run_id": run_id,
        "topology": "TP1-DP2-EP2" if args.topology == "ep2" else "TP2-DP2-EP4-PP1",
        "physical_gpus": args.physical_gpus,
        "model": MODEL,
        "revision": MODEL_REVISION,
        "dtype": DTYPE,
        "backend": "deepep_high_throughput",
        "requests": len(requests),
        "max_tokens": args.max_tokens,
        "evidence": "INSTRUMENTED_PHYSICAL_ROUTING_NOT_CLEAN_TIMING",
    })
    try:
        _run(args, requests)
    except BaseException:
        append_jsonl(args.ledger, [{"run_id": run_id, "request_executions": 0, "status": "FAILED"}])
        raise
    append_jsonl(args.ledger, [{"run_id": run_id, "request_executions": len(requests), "status": "COMPLETE"}])
    atomic_json(args.output / "run.json", {"status": "COMPLETE", "run_id": run_id, "records": len(requests)})


if __name__ == "__main__":
    main()
