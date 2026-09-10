#!/usr/bin/env python3
"""Launch one matched text or vision Qwen3-VL pivot run on physical GPUs 4--7."""

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
MODEL = "/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
VISION_MANIFEST = Path("/home/esjung/MLLM-EP-github/poc_flashvep/deepep_revalidation/results/tile_slack_mechanism_20260820_150852/stage_a/sample_manifest.json")
TEXT_PROMPTS = Path("/home/esjung/MLLM-EP-github/poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609/text_prompts.json")
CAPTURE = Path("/home/esjung/MLLM-EP/poc_flashvep/results/offline_wavefront_quick_poc_20260805_130322/layer24_capture.pt")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_request(model: str, modality: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from transformers import AutoProcessor
    from poc_flashvep.vision_tile_motivation.profile_vision_tile_motivation import _prepare_sample

    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    if modality == "text":
        row = next(item for item in json.loads(TEXT_PROMPTS.read_text()) if item["request_id"] == "text_23_method")
        return {"prompt": row["prompt"]}, {
            "modality": "text", "source": row["request_id"],
            "processor_prompt_tokens": int(row["prompt_tokens"]), "vision_tokens": 0,
        }
    manifest = json.loads(VISION_MANIFEST.read_text())
    sample = next(item for item in manifest["samples"] if item["sample_id"] == "method")
    request, meta = _prepare_sample(processor, sample)
    return request, {
        "modality": "vision", "source": sample["sample_id"],
        "processor_prompt_tokens": int(meta["processor_prompt_tokens"]),
        "vision_tokens": int(meta["processor_vision_tokens"]),
        "image_paths": list(sample["image_paths"]),
    }


def build_schedule(modality: str, repetitions: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    wave = 0
    for iteration in range(2):
        rows.append({"wave": wave, "mode": "warmup", "modality": modality,
                     "iteration": iteration, "capture_logits": False}); wave += 1
    paired = []
    for iteration in range(repetitions):
        for mode in ("clean", "instrumented"):
            paired.append({"mode": mode, "modality": modality, "iteration": iteration,
                           "capture_logits": iteration == 0})
    random.Random(20260911 + (0 if modality == "text" else 1)).shuffle(paired)
    for row in paired:
        rows.append({"wave": wave, **row}); wave += 1
    rows.append({"wave": wave, "mode": "pairwise", "modality": modality,
                 "iteration": 0, "capture_logits": True}); wave += 1
    # A fresh clean request after all diagnostic replays is the correctness control.
    rows.append({"wave": wave, "mode": "correctness", "modality": modality,
                 "iteration": 0, "capture_logits": True}); wave += 1
    # Decode policy capture. Prefill does not trigger because the hook requires <=4 local tokens.
    rows.append({"wave": wave, "mode": "policy_decode", "modality": modality,
                 "iteration": 0, "capture_logits": True, "max_tokens": 2}); wave += 1
    rows.append({"wave": wave, "mode": "flush", "modality": modality,
                 "iteration": 0, "capture_logits": False, "flush": True})
    return rows


def set_control(root: Path, row: dict[str, Any]) -> None:
    temp = root / "control.tmp.json"; write_json(temp, row); temp.replace(root / "control.json")


def worker(rank: int, port: int, args: argparse.Namespace, barrier: Any,
           schedule: list[dict[str, Any]]) -> None:
    output = args.output / f"driver.dp{rank}.json"
    records: list[dict[str, Any]] = []
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != PHYSICAL_GPUS:
            raise RuntimeError(f"refusing mapping {os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
        os.environ.update({
            "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank), "VLLM_DP_SIZE": "2",
            "VLLM_DP_MASTER_IP": "127.0.0.1", "VLLM_DP_MASTER_PORT": str(port),
            "MODPHASE_ENABLE": "1", "MODPHASE_CONTROL": str((args.output / "control.json").resolve()),
            "MODPHASE_RAW": str((args.output / "raw").resolve()), "MODPHASE_LAYER": str(args.layer),
            "MODPHASE_CAPTURE": str(CAPTURE), "MODPHASE_FIXED_REPEAT": str(args.fixed_repeat),
            "MODPHASE_PAIR_WARMUPS": str(args.pair_warmups),
            "MODPHASE_PAIR_ITERATIONS": str(args.pair_iterations),
            "MODPHASE_POLICY_WARMUPS": str(args.policy_warmups),
            "MODPHASE_POLICY_ITERATIONS": str(args.policy_iterations),
        })
        from vllm import LLM, SamplingParams

        request, metadata = build_request(args.model, args.modality)
        llm = LLM(
            model=args.model, dtype="bfloat16", tensor_parallel_size=2,
            enable_expert_parallel=True, expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput", enable_dbo=False,
            enable_ep_weight_filter=True, trust_remote_code=True,
            gpu_memory_utilization=0.90, kv_cache_memory_bytes=1 << 30,
            max_model_len=4096, max_num_batched_tokens=16384, max_num_seqs=4,
            limit_mm_per_prompt={"image": 2}, mm_processor_cache_gb=0,
            skip_mm_profiling=True, enable_prefix_caching=False,
            enable_flashinfer_autotune=False, enforce_eager=True, disable_log_stats=False,
        )
        pc = llm.llm_engine.vllm_config.parallel_config
        runtime = {
            "tp": int(pc.tensor_parallel_size), "dp": int(pc.data_parallel_size),
            "ep": int(pc.tensor_parallel_size * pc.data_parallel_size),
            "enable_ep": bool(pc.enable_expert_parallel), "all2all": str(pc.all2all_backend),
            "dbo": bool(pc.use_ubatching), "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        }
        if rank == 0:
            write_json(args.output / "workload.json", metadata)
            write_json(args.output / "runtime.json", runtime)
        from vllm.outputs import RequestOutput
        sampling1 = SamplingParams(max_tokens=1, temperature=0.0)
        sampling2 = SamplingParams(max_tokens=2, temperature=0.0)
        for row in schedule:
            if rank == 0:
                set_control(args.output, row)
            barrier.wait(timeout=1800)
            start = time.perf_counter_ns()
            if row["mode"] != "flush":
                llm._add_completion_requests([copy.deepcopy(request)],
                                             sampling2 if row.get("max_tokens") == 2 else sampling1,
                                             use_tqdm=False)
                outputs = llm._run_engine(RequestOutput, use_tqdm=False)
            else:
                # A normal final request makes every worker observe the flush control.
                llm._add_completion_requests([copy.deepcopy(request)], sampling1, use_tqdm=False)
                outputs = llm._run_engine(RequestOutput, use_tqdm=False)
            wall_ms = (time.perf_counter_ns() - start) / 1e6
            record: dict[str, Any] = {**row, "driver_rank": rank, "wall_ms": wall_ms}
            if outputs:
                result = outputs[0]; metrics = result.metrics
                record.update({
                    "prompt_tokens": len(result.prompt_token_ids or []),
                    "output_tokens": [int(x) for x in result.outputs[0].token_ids],
                    "scheduled_ts": getattr(metrics, "scheduled_ts", None),
                    "first_token_ts": getattr(metrics, "first_token_ts", None),
                })
                if record["scheduled_ts"] is not None and record["first_token_ts"] is not None:
                    record["ttft_ms"] = 1000 * (record["first_token_ts"] - record["scheduled_ts"])
            records.append(record)
            barrier.wait(timeout=1800)
        write_json(output, {"ok": True, "runtime": runtime, "records": records})
        llm.llm_engine.engine_core.shutdown()
    except BaseException:
        write_json(output, {"ok": False, "records": records, "traceback": traceback.format_exc()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--modality", choices=("text", "vision"), required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--pair-warmups", type=int, default=4)
    parser.add_argument("--pair-iterations", type=int, default=12)
    parser.add_argument("--policy-warmups", type=int, default=3)
    parser.add_argument("--policy-iterations", type=int, default=10)
    parser.add_argument("--fixed-repeat", type=int, default=3)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != PHYSICAL_GPUS:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES must be exactly {PHYSICAL_GPUS}")
    # The spawned DP driver interpreters import sitecustomize before entering
    # worker(), so enable the hook in their inherited environment here.
    os.environ["MODPHASE_ENABLE"] = "1"
    args.output.mkdir(parents=True, exist_ok=False)
    request, metadata = build_request(args.model, args.modality)
    schedule = build_schedule(args.modality, args.repetitions)
    write_json(args.output / "schedule.json", schedule)
    write_json(args.output / "launch.json", {
        "model": args.model, "capture": str(CAPTURE), "layer": args.layer,
        "modality": args.modality, "workload": metadata,
        "physical_gpus": [4, 5, 6, 7], "fixed_repeat": args.fixed_repeat,
    })
    context = mp.get_context("spawn"); barrier = context.Barrier(2); port = free_port()
    processes = [context.Process(target=worker, args=(rank, port, args, barrier, schedule)) for rank in range(2)]
    for process in processes: process.start()
    for process in processes: process.join()
    exits = [process.exitcode for process in processes]
    if exits != [0, 0]:
        raise RuntimeError(f"run failed: {exits}")


if __name__ == "__main__":
    main()
