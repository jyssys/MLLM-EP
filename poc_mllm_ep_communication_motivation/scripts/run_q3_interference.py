#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from collections import Counter
import multiprocessing as mp
import os
import queue
import socket
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from poc_mllm_ep_communication_motivation.contract import MODEL, atomic_json, read_json


def port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def image_prompt(processor: Any, row: dict[str, Any]) -> dict[str, Any]:
    with Image.open(row["image_path"]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": row["question"]},
            ],
        }
    ]
    return {
        "prompt": processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ),
        "multi_modal_data": {"image": image},
    }


def text_prompt(processor: Any, text: str) -> str:
    return processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": text}]}],
        tokenize=False,
        add_generation_prompt=True,
    )


def matched_text(processor: Any, target: int) -> tuple[str, int]:
    unit = (
        "Describe the observed context carefully using only the provided information. "
    )
    low, high = 1, max(2, target // 4)
    best = None
    while low <= high:
        middle = (low + high) // 2
        prompt = text_prompt(processor, unit * middle)
        count = len(processor.tokenizer.encode(prompt, add_special_tokens=False))
        candidate = (abs(count - target), prompt, count)
        if best is None or candidate[0] < best[0]:
            best = candidate
        if count < target:
            low = middle + 1
        else:
            high = middle - 1
    assert best is not None
    return best[1], best[2]


def injection(
    processor: Any, condition: str, visual: dict[str, Any]
) -> tuple[Any | None, int | None]:
    if condition in {"DECODE_ONLY", "AUDIT_BYPASS"}:
        return None, None
    if condition == "VISUAL_PREFILL":
        return image_prompt(processor, visual), int(visual["prompt_token_count"])
    if condition == "TEXT_PREFILL":
        prompt = text_prompt(
            processor,
            "Summarize the general considerations for describing an image clearly and accurately.",
        )
        return prompt, len(processor.tokenizer.encode(prompt, add_special_tokens=False))
    if condition == "TOKEN_MATCHED_TEXT":
        return matched_text(processor, int(visual["prompt_token_count"]))
    raise ValueError(condition)


def worker(
    rank: int,
    master_port: int,
    args: argparse.Namespace,
    barrier: Any,
    progress: Any,
    schedule: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    llm = None
    try:
        os.environ.update(
            {
                "VLLM_DP_RANK": str(rank),
                "VLLM_DP_RANK_LOCAL": str(rank),
                "VLLM_DP_SIZE": "2",
                "VLLM_DP_MASTER_IP": "127.0.0.1",
                "VLLM_DP_MASTER_PORT": str(master_port),
                "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
                "FLASHVEP_CONFIGURED_DBO": "false",
            }
        )
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
        special = list(map(int, processor.tokenizer.all_special_ids))
        llm = LLM(
            model=MODEL,
            dtype="bfloat16",
            trust_remote_code=True,
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput",
            enable_ep_weight_filter=True,
            enable_dbo=False,
            gpu_memory_utilization=0.90,
            kv_cache_memory_bytes=4 << 30,
            max_model_len=20000,
            max_num_batched_tokens=20000,
            max_num_seqs=5,
            enable_chunked_prefill=False,
            limit_mm_per_prompt={"image": 1},
            skip_mm_profiling=True,
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            enable_flashinfer_autotune=False,
            enforce_eager=True,
            disable_log_stats=True,
            logits_processors=[
                "poc_mllm_vision_dominant_captioning.force_tokens:ForceTokensLogitsProcessor"
            ],
        )
        progress.put((rank, -1, "MODEL_READY"))
        decode_rows = manifest["q3_decode_pool"][rank * 4 : (rank + 1) * 4]
        decode_prompts = [image_prompt(processor, row) for row in decode_rows]
        all_wave_records = []
        for wave_id, entry in enumerate(schedule):
            control = {
                **entry,
                "wave": wave_id,
                "special_token_ids": special,
                "expected_phase": "mixed",
                "injection_request_ids": [],
            }
            if rank == 0:
                atomic_json(args.control, control)
            barrier.wait(timeout=1800)
            params = []
            for row in decode_rows:
                forced = [
                    int(value)
                    for value in row["forced_token_ids"][: args.decode_tokens]
                ]
                params.append(
                    SamplingParams(
                        max_tokens=len(forced),
                        temperature=0.0,
                        seed=3407,
                        extra_args={"mllm_vds_forced_token_ids": forced},
                    )
                )
            llm._add_completion_requests(
                [copy.deepcopy(value) for value in decode_prompts],
                params,
                use_tqdm=False,
            )
            expected_sequences = Counter(
                tuple(int(v) for v in row["forced_token_ids"][: args.decode_tokens])
                for row in decode_rows
            )
            injection_prompt, injection_tokens = injection(
                processor, entry["condition"], manifest["q3_large_visual"]
            )
            records, completed, injected_ids = [], {}, []
            iteration = 0
            while llm.llm_engine.has_unfinished_requests():
                if (
                    iteration == args.injection_iteration
                    and injection_prompt is not None
                    and rank == 0
                ):
                    injected_ids = llm._add_completion_requests(
                        [copy.deepcopy(injection_prompt)],
                        [SamplingParams(max_tokens=1, temperature=0.0, seed=3407)],
                        use_tqdm=False,
                    )
                    control["injection_request_ids"] = [
                        str(value) for value in injected_ids
                    ]
                    atomic_json(args.control, control)
                barrier.wait(timeout=1800)
                begin = time.perf_counter_ns()
                outputs = llm.llm_engine.step()
                elapsed = (time.perf_counter_ns() - begin) / 1e6
                barrier.wait(timeout=1800)
                for output in outputs:
                    if output.finished:
                        completed[str(output.request_id)] = output
                records.append(
                    {
                        "iteration": iteration,
                        "step_wall_ms": elapsed,
                        "local_unfinished_after": llm.llm_engine.get_num_unfinished_requests(),
                        "injection_iteration": iteration == args.injection_iteration,
                        "injection_prompt_tokens": injection_tokens,
                    }
                )
                iteration += 1
                if iteration > args.decode_tokens + 8:
                    raise RuntimeError(f"wave exceeded bounded iterations: {iteration}")
            actual_sequences = Counter(
                tuple(int(v) for v in output.outputs[0].token_ids)
                for output in completed.values()
                if len(output.outputs[0].token_ids) == args.decode_tokens
            )
            if actual_sequences != expected_sequences:
                raise RuntimeError(
                    (
                        "forced-token parity",
                        actual_sequences,
                        expected_sequences,
                        "completed_ids",
                        sorted(completed),
                    )
                )
            all_wave_records.append(
                {
                    "wave": wave_id,
                    "condition": entry["condition"],
                    "replicate": entry["replicate"],
                    "trace": entry["trace"],
                    "decode_request_ids": [row["request_id"] for row in decode_rows],
                    "injection_request_ids": injected_ids,
                    "injection_prompt_tokens": injection_tokens,
                    "steps": records,
                }
            )
            progress.put((rank, wave_id, "COMPLETE"))
        from poc_mllm_ep_communication_motivation.instrumentation import _drain

        _drain()
        atomic_json(
            args.output / f"worker.dp{rank}.json",
            {"ok": True, "waves": all_wave_records},
        )
    except BaseException:
        atomic_json(
            args.output / f"worker.dp{rank}.json",
            {"ok": False, "traceback": traceback.format_exc()},
        )
        progress.put((rank, -2, "FAILED"))
        raise
    finally:
        if llm is not None and callable(getattr(llm, "shutdown", None)):
            try:
                llm.shutdown()
            except BaseException:
                pass


def schedule(
    repeats: int,
    trace_only: bool = False,
    decode_only_audit: bool = False,
    paired_overhead_audit: bool = False,
) -> list[dict[str, Any]]:
    conditions = ["DECODE_ONLY", "TEXT_PREFILL", "VISUAL_PREFILL", "TOKEN_MATCHED_TEXT"]
    result = []
    if paired_overhead_audit:
        result.append(
            {
                "condition": "DECODE_ONLY",
                "replicate": -1,
                "trace": False,
                "capture_stage": False,
                "route_stats": False,
                "instrumentation_bypass": False,
            }
        )
        for replicate in range(repeats):
            order = (
                ("AUDIT_BYPASS", "DECODE_ONLY")
                if replicate % 2 == 0
                else ("DECODE_ONLY", "AUDIT_BYPASS")
            )
            for condition in order:
                result.append(
                    {
                        "condition": condition,
                        "replicate": replicate,
                        "trace": False,
                        "capture_stage": False,
                        "route_stats": False,
                        "instrumentation_bypass": condition == "AUDIT_BYPASS",
                    }
                )
        return result
    if decode_only_audit:
        return [
            {
                "condition": "DECODE_ONLY",
                "replicate": replicate,
                "trace": False,
                "capture_stage": False,
                "route_stats": False,
            }
            for replicate in range(-1, repeats)
        ]
    if trace_only:
        return [
            {
                "condition": condition,
                "replicate": 1,
                "trace": True,
                "capture_stage": True,
                "route_stats": True,
                "capture_iterations": list(range(15, 33)),
            }
            for condition in conditions
        ]
    # Bounded thermal warm-up, excluded from primary estimates.
    for condition in ("DECODE_ONLY", "VISUAL_PREFILL"):
        result.append(
            {
                "condition": condition,
                "replicate": -1,
                "trace": False,
                "capture_stage": False,
                "route_stats": False,
            }
        )
    latin = [
        conditions,
        conditions[1:] + conditions[:1],
        conditions[2:] + conditions[:2],
    ]
    for replicate in range(repeats):
        for condition in latin[replicate % len(latin)]:
            result.append(
                {
                    "condition": condition,
                    "replicate": replicate,
                    "trace": False,
                    "capture_stage": False,
                    "route_stats": False,
                }
            )
    for condition in conditions:
        result.append(
            {
                "condition": condition,
                "replicate": 0,
                "trace": True,
                "capture_stage": True,
                "route_stats": True,
                "capture_iterations": list(range(15, 33)),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--injection-iteration", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--decode-only-audit", action="store_true")
    parser.add_argument("--no-hook", action="store_true")
    parser.add_argument("--paired-overhead-audit", action="store_true")
    parser.add_argument("--watchdog-sec", type=int, default=3600)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3,6,7":
        raise RuntimeError("requires CUDA_VISIBLE_DEVICES=2,3,6,7")
    if not args.no_hook and os.environ.get("MLLM_COMM_MOTIVATION_HOOK_IMPORTED") != "1":
        raise RuntimeError("startup hook was not imported")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    manifest = read_json(args.manifest)
    waves = schedule(
        args.repeats,
        args.trace_only,
        args.decode_only_audit,
        args.paired_overhead_audit,
    )
    atomic_json(
        args.output / "manifest.json",
        {
            "status": "FROZEN",
            "physical_gpus": [2, 3, 6, 7],
            "topology": "TP2-DP2-EP4-PP1",
            "global_decode_batch": 8,
            "fixed_decode_tokens": args.decode_tokens,
            "injection_iteration": args.injection_iteration,
            "chunked_prefill": False,
            "max_num_batched_tokens": 20000,
            "max_num_seqs_per_dp": 5,
            "schedule": waves,
            "instrumentation_hook_enabled": not args.no_hook,
            "request_executions": len(waves) * 8
            + sum(w["condition"] != "DECODE_ONLY" for w in waves),
        },
    )
    atomic_json(args.control, {"condition": "STARTUP", "capture_stage": False})
    context = mp.get_context("spawn")
    barrier, progress, master_port = context.Barrier(2), context.Queue(), port()
    workers = [
        context.Process(
            target=worker,
            args=(rank, master_port, args, barrier, progress, waves, manifest),
        )
        for rank in (0, 1)
    ]
    for process in workers:
        process.start()
    last = time.monotonic()
    failure = None
    while any(process.is_alive() for process in workers):
        try:
            rank, _, status = progress.get(timeout=5)
            last = time.monotonic()
            if status == "FAILED":
                failure = f"worker {rank} failed"
                break
        except queue.Empty:
            pass
        if time.monotonic() - last > args.watchdog_sec:
            failure = "TIMEOUT: no worker progress"
            break
    if failure:
        for process in workers:
            if process.is_alive():
                process.terminate()
        for process in workers:
            process.join(timeout=30)
        atomic_json(
            args.output / "failure.json",
            {"status": "RUNTIME_UNAVAILABLE", "reason": failure},
        )
        raise RuntimeError(failure)
    for process in workers:
        process.join()
    if any(process.exitcode != 0 for process in workers):
        raise RuntimeError([process.exitcode for process in workers])
    atomic_json(args.output / "run.json", {"status": "COMPLETE", "waves": len(waves)})


if __name__ == "__main__":
    main()
