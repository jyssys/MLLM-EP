#!/usr/bin/env python3
"""Capture bounded selected-branch decode vectors on TRUE TP2-DP2-EP4."""

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

from PIL import Image, ImageOps

MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
MODEL_REVISION = "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _prepare(processor: Any, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    with Image.open(row["image_path"]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": str(row["question"])},
        ]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return {"prompt": prompt, "multi_modal_data": {"image": image}}, {}


def _model_kwargs(max_model_len: int, kv_cache_gib: float) -> dict[str, Any]:
    return {
        "model": MODEL,
        "revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "tensor_parallel_size": 2,
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
        "logits_processors": [
            "poc_mllm_vision_conditioned_decode_reduction.force_tokens:ForceTokensLogitsProcessor"
        ],
    }


def _sampling(row: dict[str, Any], max_tokens: int):
    from vllm import SamplingParams

    forced = list(row["forced_token_ids"])
    return SamplingParams(
        max_tokens=len(forced),
        temperature=0.0,
        seed=3407,
        extra_args={"vcdr_forced_token_ids": forced},
    )


def _ep_wave(llm: Any, prompts: list[dict[str, Any]], params: list[Any], barrier: Any, wave: int):
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


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _reference(path: Path) -> dict[str, list[int]]:
    return {str(row["pair_id"]): [int(x) for x in row["output_token_ids"]] for row in read_jsonl(path)}


def _requests(manifest: Path, reference_results: Path | None, limit: int) -> list[dict[str, Any]]:
    anchors = json.loads(manifest.read_text(encoding="utf-8"))["anchors"]
    tokens = _reference(reference_results) if reference_results is not None else {}
    for row in anchors:
        if row.get("forced_token_ids"):
            tokens[str(row["pair_id"])] = [int(value) for value in row["forced_token_ids"]]
    eligible = [
        row for row in anchors
        if row["analysis_split"] == "discovery" and len(tokens.get(row["pair_id"], [])) >= 128
    ][:limit]
    if len(eligible) < limit:
        raise RuntimeError(f"only {len(eligible)} discovery anchors have >=128 natural tokens")
    requests = []
    for row in eligible:
        requests.append({
            "pair_id": row["pair_id"],
            "request_id": f"{row['pair_id']}::A::vector_capture128",
            "condition": "A",
            "question": row["question"],
            "analysis_split": "discovery",
            "image_path": row["original_image_path"],
            "image_sha256": row["original_image_sha256"],
            "forced_token_ids": tokens[row["pair_id"]][:128],
            "decode_cap": 128,
            "run_mode": "vector_capture",
            "phase": "oracle_discovery",
        })
    return requests


def _worker(
    rank: int,
    port: int,
    args: argparse.Namespace,
    barrier: Any,
    progress: Any,
    requests: list[dict[str, Any]],
) -> None:
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(rank),
            "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": "2",
            "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false",
            "MLLM_VCDR_CAPTURE_CONTROL": str(args.control.resolve()),
            "MLLM_VCDR_CAPTURE_RAW": str((args.output / "raw").resolve()),
            "MLLM_VCDR_CAPTURE_LAYERS": args.layers,
            "MLLM_VCDR_CAPTURE_STEPS": args.steps,
            "MLLM_VCDR_CAPTURE_ENABLE": "1",
        })
        hook_dir = str((Path(__file__).parents[1] / "hooks").resolve())
        os.environ["PYTHONPATH"] = f"{hook_dir}:{os.environ.get('PYTHONPATH', '')}"
        from transformers import AutoProcessor
        from vllm import LLM
        from poc_mllm_vision_conditioned_decode_reduction.decode_vector_capture import install

        install()
        processor = AutoProcessor.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        kwargs = _model_kwargs(args.max_model_len, args.kv_cache_gib)
        kwargs["model"] = args.model_path
        kwargs.pop("revision", None)
        kwargs["max_num_seqs"] = 1
        llm = LLM(**kwargs)
        progress.put((rank, -1, "MODEL_READY"))
        records = []
        for wave, row in enumerate(requests):
            source = wave % 2
            if rank == 0:
                atomic_json(args.control, {
                    "wave": wave,
                    "capture": True,
                    "capture_id": row["pair_id"],
                    "source_dp_rank": source,
                })
            barrier.wait(timeout=args.wave_timeout_sec)
            prepared = _prepare(processor, row) if rank == source else None
            prompts = [copy.deepcopy(prepared[0])] if prepared else []
            params = [_sampling(row, 128)] if prepared else []
            started = time.perf_counter_ns()
            outputs = _ep_wave(llm, prompts, params, barrier, wave)
            wall_ms = (time.perf_counter_ns() - started) / 1e6
            if rank == source:
                if len(outputs) != 1:
                    raise AssertionError((wave, source, len(outputs)))
                actual = [int(x) for x in outputs[0].outputs[0].token_ids]
                if actual != row["forced_token_ids"]:
                    raise AssertionError(f"forced token mismatch for {row['pair_id']}")
                records.append({
                    "pair_id": row["pair_id"],
                    "wave": wave,
                    "source_dp_rank": source,
                    "wall_ms_instrumented": wall_ms,
                    "token_count": len(actual),
                    "forced_exact": True,
                })
            progress.put((rank, wave, "COMPLETE"))
        atomic_json(args.output / f"worker.dp{rank}.json", {"ok": True, "records": records})
    except BaseException:
        atomic_json(
            args.output / f"worker.dp{rank}.json",
            {"ok": False, "traceback": traceback.format_exc()},
        )
        progress.put((rank, -2, "FAILED"))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path)
    parser.add_argument(
        "--model-path",
        default=(
            "/home/esjung/.cache/huggingface/hub/"
            "models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/"
            "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--layers", default="4,12,24,36,44,47")
    parser.add_argument(
        "--steps",
        default=",".join(str(step) for start in (1, 33, 65, 97) for step in range(start, start + 16)),
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--kv-cache-gib", type=float, default=2.0)
    parser.add_argument("--wave-timeout-sec", type=int, default=3600)
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != "4,5,6,7":
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be 4,5,6,7, got {visible!r}")
    args.output.mkdir(parents=True, exist_ok=False)
    args.control.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.control, {"capture": False, "capture_id": "startup", "source_dp_rank": -1})
    requests = _requests(args.manifest, args.reference_results, args.limit)
    atomic_json(args.output / "manifest.json", {
        "requests": [{k: v for k, v in row.items() if k != "forced_token_ids"} for row in requests],
        "request_executions": len(requests),
        "layers": [int(x) for x in args.layers.split(",")],
        "steps": [int(x) for x in args.steps.split(",")],
        "topology": "TRUE_TP2_DP2_EP4_PP1",
        "evidence": "INSTRUMENTED_DIAGNOSTIC_NOT_CLEAN_TIMING",
    })
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    progress = context.Queue()
    rendezvous = _port()
    workers = [
        context.Process(target=_worker, args=(rank, rendezvous, args, barrier, progress, requests))
        for rank in range(2)
    ]
    for worker in workers:
        worker.start()
    last = time.monotonic()
    failed = None
    while any(worker.is_alive() for worker in workers):
        try:
            rank, wave, status = progress.get(timeout=5)
            last = time.monotonic()
            if status == "FAILED":
                failed = f"worker {rank} failed"
                break
        except queue.Empty:
            pass
        if time.monotonic() - last > args.wave_timeout_sec:
            failed = "TIMEOUT: no worker progress"
            break
    if failed:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join(timeout=30)
        atomic_json(args.output / "failure.json", {"status": failed})
        raise RuntimeError(failed)
    for worker in workers:
        worker.join()
    if any(worker.exitcode != 0 for worker in workers):
        raise RuntimeError([worker.exitcode for worker in workers])
    atomic_json(args.output / "run.json", {"status": "COMPLETE", "requests": len(requests)})


if __name__ == "__main__":
    main()
