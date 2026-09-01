#!/usr/bin/env python3
"""Run a bounded, real-vLLM concurrent-serving straggler experiment.

Requests are submitted as lists to the vLLM engine, so the normal v1
scheduler performs the co-batching.  No route, placement, or model code is
changed.  A single process per DP rank owns the engine; no Python worker
threads execute model forwards.
"""

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


PREVIOUS = Path(
    "poc_flashvep/deepep_revalidation/results/"
    "live_prefill_execution_regime_20260821_111609"
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_schedule(warmups: int, iterations: int) -> list[dict[str, Any]]:
    """Preregistered condition/order; no latency-based selection."""
    entries: list[dict[str, Any]] = []
    # Short and medium real-image requests cover the usual low-load path.
    vision_short = ["coins", "coffee", "histology", "coffee_rocket"]
    text_short = ["text_00_coins", "text_05_coffee", "text_08_histology", "text_14_coffee_rocket"]
    # Repeated lists create scheduler concurrency while preserving each exact
    # prompt.  The cycle is fixed before the run.
    conditions = [
        ("text_only", "text", 1, text_short, "short"),
        ("vision_single", "vision", 1, vision_short, "short"),
    ]
    for concurrency in (1, 2, 4, 8, 16):
        conditions.append(("vision_heavy", "vision", concurrency, vision_short, "concurrent"))
        conditions.append(("text_control", "text", concurrency, text_short, "concurrent"))
    # Long local multi-image requests are bounded to 1/2/4 concurrent due to
    # the fixed KV budget; they are still submitted through the same scheduler.
    for concurrency in (1, 2, 4):
        conditions.append(("long_multi_image", "vision", concurrency,
                           ["long_6img_natural_fine", "long_10img_chart_mixed"], "long"))

    wave = 0
    for condition, modality, concurrency, request_ids, phase in conditions:
        for iteration in range(warmups + iterations):
            entries.append({
                "wave": wave,
                "batch_id": f"{condition}_c{concurrency}_i{iteration}",
                "condition": condition,
                "modality": modality,
                "concurrency": int(concurrency),
                "request_ids": list(request_ids),
                "phase": phase,
                "instrument": True,
                "measured": iteration >= warmups,
                "iteration": iteration - warmups,
                "source_dp_rank": 0,
            })
            wave += 1
    return entries


def _prepare_requests(model_path: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    from transformers import AutoProcessor
    from poc_flashvep.live_prefill_execution_regime.run_live import _requests
    from poc_flashvep.vision_tile_motivation.profile_vision_tile_motivation import _prepare_sample
    from poc_flashvep.chunk_oracle_gpu_scale_validation.long_capture import _rows as long_rows

    requests = _requests(PREVIOUS, model_path)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    metadata: dict[str, Any] = {}
    manifest = json.loads((PREVIOUS / "workload_manifest.json").read_text())
    for pair in manifest["pairs"]:
        for modality in ("vision", "text"):
            item = pair[modality]
            rid = str(item["request_id"])
            metadata[rid] = {
                "request_id": rid,
                "modality": modality,
                "prompt_tokens": int(item["prompt_tokens"]),
                "image_count": len(item.get("image_paths", [])) if modality == "vision" else 0,
                "category": item.get("category", "text_only"),
            }
    for row in long_rows():
        prepared, meta = _prepare_sample(processor, row)
        rid = str(meta["sample_id"])
        requests[rid] = prepared
        metadata[rid] = {**meta, "request_id": rid, "modality": "vision",
                         "condition": "long_multi_image"}
    # Stable aliases used by the fixed schedule.
    alias = {
        "coffee": "coffee",
        "histology": "histology",
        "coffee_rocket": "coffee_rocket",
        "text_05_coffee": "text_05_coffee",
        "text_08_histology": "text_08_histology",
        "text_14_coffee_rocket": "text_14_coffee_rocket",
    }
    missing = [rid for rid in alias.values() if rid not in requests]
    if missing:
        raise AssertionError(f"missing schedule requests: {missing}")
    return requests, metadata


def _generate(llm: Any, prompts: list[dict[str, Any]], sampling: Any,
              barrier: Any, wave: int) -> list[Any]:
    from vllm.outputs import RequestOutput
    from vllm.v1.engine import EngineCoreRequestType

    if prompts:
        barrier.wait(timeout=1800)
        llm._add_completion_requests(prompts, sampling, use_tqdm=False)
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)
    else:
        llm.llm_engine.engine_core._send_input(
            EngineCoreRequestType.START_DP_WAVE, (wave, -1)
        )
        barrier.wait(timeout=1800)
        outputs = []
    barrier.wait(timeout=1800)
    return outputs


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
            "FLASHVEP_SCHEDULER_TRACE_DIR": str((args.output_dir / "scheduler_trace").resolve()),
            "FLASHVEP_DEEPEP_PROOF_DIR": str((args.output_dir / "backend_proof").resolve()),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false",
            "FLASHVEP_SERVING_PROBE": "1",
        })
        # Explicit installation makes the hook reliable even when Python has
        # already initialized sitecustomize before this child sets its env.
        from poc_flashvep.deepep_revalidation.backend_probe import install_backend_probe
        from poc_flashvep.ep4_serving_straggler_regime.live_instrumentation import install
        from poc_flashvep.ep4_serving_straggler_regime.serving_probe import install as install_scheduler
        install_backend_probe(); install(); install_scheduler()

        from vllm import LLM, SamplingParams
        requests, metadata = _prepare_requests(args.model_path)
        llm = LLM(
            model=args.model_path, dtype="bfloat16", tensor_parallel_size=2,
            enable_expert_parallel=True, expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput", enable_dbo=False,
            enable_return_routed_experts=False, enable_ep_weight_filter=True,
            trust_remote_code=True, gpu_memory_utilization=0.90,
            kv_cache_memory_bytes=8 << 30, max_model_len=16384,
            max_num_batched_tokens=int(args.max_num_batched_tokens), max_num_seqs=16,
            limit_mm_per_prompt={"image": 16}, skip_mm_profiling=True,
            enable_prefix_caching=False, enable_flashinfer_autotune=False,
            enforce_eager=True, disable_log_stats=True,
        )
        sampling = SamplingParams(max_tokens=1, temperature=0.0)
        records: list[dict[str, Any]] = []
        for entry in schedule:
            if rank == 0:
                _write(args.output_dir / "control.tmp.json", entry)
                (args.output_dir / "control.tmp.json").replace(args.output_dir / "control.json")
            barrier.wait(timeout=1800)
            if rank == int(entry["source_dp_rank"]):
                ids = entry["request_ids"]
                prompts = [copy.deepcopy(requests[ids[i % len(ids)]])
                           for i in range(int(entry["concurrency"]))]
            else:
                prompts = []
            started = time.perf_counter_ns()
            outputs = _generate(llm, prompts, sampling, barrier, int(entry["wave"]))
            wall_ms = (time.perf_counter_ns() - started) / 1e6
            records.append({
                **entry,
                "driver_dp_rank": rank,
                "wall_ms": wall_ms,
                "output_count": len(outputs),
                "output_tokens": [
                    [int(t) for t in out.outputs[0].token_ids]
                    for out in outputs
                ],
                "prompt_metadata": [metadata[entry["request_ids"][i % len(entry["request_ids"])] ]
                                    for i in range(int(entry["concurrency"]))]
                if rank == 0 else [],
            })
        flush = {
            **schedule[-1], "wave": len(schedule), "batch_id": "flush",
            "flush": True, "instrument": False, "measured": False,
        }
        if rank == 0:
            _write(args.output_dir / "control.tmp.json", flush)
            (args.output_dir / "control.tmp.json").replace(args.output_dir / "control.json")
        barrier.wait(timeout=1800)
        flush_prompts = [copy.deepcopy(requests["coins"])] if rank == 0 else []
        _generate(llm, flush_prompts, sampling, barrier, int(flush["wave"]))
        _write(output, {"ok": True, "records": records})
    except BaseException:
        _write(output, {"ok": False, "traceback": traceback.format_exc()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule = _build_schedule(args.warmups, args.iterations)
    _write(args.output_dir / "schedule.json", schedule)
    _write(args.output_dir / "run_metadata.json", {
        "model_path": args.model_path,
        "configuration": {
            "dtype": "BF16", "tp": 2, "dp": 2, "ep": 4, "pp": 1,
            "all2all": "deepep_high_throughput", "dbo": False,
            "prefix_cache": False, "expert_placement": "linear",
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_model_len": 16384, "max_num_seqs": 16,
            "physical_gpus": [1, 2, 3, 4],
        },
        "warmups": args.warmups, "iterations": args.iterations,
        "scheduling": "real vLLM v1 scheduler via batched _add_completion_requests",
        "workload_source": str(PREVIOUS),
    })
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    port = _port()
    processes = [context.Process(target=_run_rank,
                                 args=(rank, port, args, barrier, schedule))
                 for rank in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(5400)
    codes = [process.exitcode for process in processes]
    if codes != [0, 0]:
        raise RuntimeError(f"serving run failed: {codes}")
    print(args.output_dir)


if __name__ == "__main__":
    main()
