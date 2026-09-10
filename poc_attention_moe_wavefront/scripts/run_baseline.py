#!/usr/bin/env python3
"""Run clean/interleaved instrumented Phase-0 Qwen3-VL prefills."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import random
import socket
import time
import traceback
from pathlib import Path
from typing import Any


PHYSICAL_GPUS = "4,5,6,7"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_workloads(model: str, previous: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    from transformers import AutoProcessor
    from poc_flashvep.vision_tile_motivation.profile_vision_tile_motivation import _prepare_sample

    manifest = json.loads((previous / "workload_manifest.json").read_text())
    vision_source = Path(manifest["vision_source"])
    samples = {row["sample_id"]: row for row in json.loads((vision_source / "sample_manifest.json").read_text())["samples"]}
    texts = {row["request_id"]: row for row in json.loads((previous / "text_prompts.json").read_text())}
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    requests: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}

    text_id = "text_23_method"
    text_prompt = texts[text_id]["prompt"]
    text_tokens = processor.tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
    requests["W1_text"] = {"prompt": text_prompt}
    metadata["W1_text"] = {
        "workload_id": "W1_text", "kind": "text-only control",
        "processor_prompt_tokens": len(text_tokens), "processor_vision_tokens": 0,
        "modality_boundaries": [], "source": text_id,
    }

    for workload_id, sample_id, kind in (
        ("W2_standard", "coffee", "standard single-image"),
        ("W3_vision_heavy", "method", "high-resolution vision-heavy"),
        ("W4_multi_image", "coffee_rocket", "multi-image multimodal"),
    ):
        request, meta = _prepare_sample(processor, samples[sample_id])
        requests[workload_id] = request
        spans = [image["token_span"] for image in meta["images"]]
        boundaries = sorted({int(value) for span in spans for value in span})
        metadata[workload_id] = {
            "workload_id": workload_id, "kind": kind,
            "processor_prompt_tokens": int(meta["processor_prompt_tokens"]),
            "processor_vision_tokens": int(meta["processor_vision_tokens"]),
            "modality_boundaries": boundaries, "image_spans": spans,
            "source": sample_id,
        }
    return requests, metadata


def schedule(metadata: dict[str, dict[str, Any]], warmups: int, repetitions: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for workload_id in metadata:
        for iteration in range(warmups):
            rows.append({"workload_id": workload_id, "request_id": workload_id,
                         "iteration": iteration, "warmup": True, "instrumented": False})
    measured = []
    for workload_id in metadata:
        for iteration in range(repetitions):
            for instrumented in (False, True):
                measured.append({"workload_id": workload_id, "request_id": workload_id,
                                 "iteration": iteration, "warmup": False,
                                 "instrumented": instrumented})
    random.Random(20260910).shuffle(measured)
    rows.extend(measured)
    for wave, row in enumerate(rows):
        row["wave"] = wave
    rows[-1]["flush"] = True
    return rows


def set_control(directory: Path, entry: dict[str, Any]) -> None:
    temporary = directory / "control.tmp.json"
    write_json(temporary, entry)
    temporary.replace(directory / "control.json")


def generate(llm: Any, prompt: dict[str, Any] | None, sampling: Any, barrier: Any, wave: int) -> list[Any]:
    from vllm.outputs import RequestOutput
    from vllm.v1.engine import EngineCoreRequestType

    if prompt is not None:
        barrier.wait(timeout=1200)
        llm._add_completion_requests([copy.deepcopy(prompt)], sampling, use_tqdm=False)
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)
    else:
        llm.llm_engine.engine_core._send_input(EngineCoreRequestType.START_DP_WAVE, (wave, -1))
        barrier.wait(timeout=1200)
        outputs = []
    barrier.wait(timeout=1200)
    return outputs


def worker(rank: int, args: argparse.Namespace, port: int, barrier: Any,
           rows: list[dict[str, Any]]) -> None:
    output = args.output_dir / f"driver.dp_rank{rank}.json"
    records = []
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != PHYSICAL_GPUS:
            raise RuntimeError(f"refusing GPU mapping {os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
        os.environ.update({
            "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank), "VLLM_DP_SIZE": "2",
            "VLLM_DP_MASTER_IP": "127.0.0.1", "VLLM_DP_MASTER_PORT": str(port),
            "WAVEFRONT_ENABLE": "1", "WAVEFRONT_CONTROL": str((args.output_dir / "control.json").resolve()),
            "WAVEFRONT_RAW": str((args.output_dir / "raw").resolve()), "WAVEFRONT_RUN_ID": args.run_id,
        })
        from vllm import LLM, SamplingParams

        requests, metadata = build_workloads(args.model, args.previous)
        if rank == 0:
            write_json(args.output_dir / "workload_manifest.json", metadata)
        barrier.wait(timeout=1200)
        llm = LLM(
            model=args.model, dtype="bfloat16", tensor_parallel_size=2,
            enable_expert_parallel=True, expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput", enable_dbo=False,
            enable_ep_weight_filter=True, trust_remote_code=True,
            gpu_memory_utilization=0.90, kv_cache_memory_bytes=1 << 30,
            max_model_len=4096, max_num_batched_tokens=16384, max_num_seqs=2,
            limit_mm_per_prompt={"image": 2}, skip_mm_profiling=True,
            enable_prefix_caching=False, enable_flashinfer_autotune=False,
            enforce_eager=True, disable_log_stats=False,
        )
        pc = llm.llm_engine.vllm_config.parallel_config
        runtime = {
            "tp": int(pc.tensor_parallel_size), "dp": int(pc.data_parallel_size),
            "ep": int(pc.data_parallel_size * pc.tensor_parallel_size),
            "enable_expert_parallel": bool(pc.enable_expert_parallel),
            "all2all_backend": str(pc.all2all_backend), "dbo": bool(pc.use_ubatching),
        }
        if rank == 0:
            write_json(args.output_dir / "runtime_config.json", runtime)
        sampling = SamplingParams(max_tokens=1, temperature=0.0)
        for entry in rows:
            if rank == 0:
                set_control(args.output_dir, entry)
            barrier.wait(timeout=1200)
            start = time.perf_counter_ns()
            outputs = generate(llm, requests[entry["workload_id"]] if rank == 0 else None,
                               sampling, barrier, int(entry["wave"]))
            wall_ms = (time.perf_counter_ns() - start) / 1e6
            request_row: dict[str, Any] = {}
            if outputs:
                result = outputs[0]
                metrics = result.metrics
                request_row = {
                    "prompt_tokens": len(result.prompt_token_ids or []),
                    "output_tokens": [int(token) for token in result.outputs[0].token_ids],
                    "arrival_time": getattr(metrics, "arrival_time", None),
                    "queued_ts": getattr(metrics, "queued_ts", None),
                    "scheduled_ts": getattr(metrics, "scheduled_ts", None),
                    "first_token_ts": getattr(metrics, "first_token_ts", None),
                    "first_token_latency_s": getattr(metrics, "first_token_latency", None),
                }
                if request_row["scheduled_ts"] is not None and request_row["first_token_ts"] is not None:
                    request_row["scheduled_to_first_token_ms"] = 1000 * (request_row["first_token_ts"] - request_row["scheduled_ts"])
            records.append({**entry, "driver_dp_rank": rank, "driver_wall_ms": wall_ms, **request_row})
        write_json(output, {"ok": True, "runtime": runtime, "records": records})
        llm.llm_engine.engine_core.shutdown()
    except BaseException:
        write_json(output, {"ok": False, "records": records, "traceback": traceback.format_exc()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != PHYSICAL_GPUS:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES must be {PHYSICAL_GPUS}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Parent builds metadata without touching CUDA so the exact schedule is persisted.
    _, metadata = build_workloads(args.model, args.previous)
    rows = schedule(metadata, args.warmups, args.repetitions)
    write_json(args.output_dir / "schedule.json", rows)
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    port = free_port()
    processes = [context.Process(target=worker, args=(rank, args, port, barrier, rows)) for rank in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    exits = [process.exitcode for process in processes]
    if exits != [0, 0]:
        raise RuntimeError(f"baseline failed: {exits}")


if __name__ == "__main__":
    main()
