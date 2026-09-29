#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
import queue
import socket
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from poc_mllm_ep_collective_granularity.contract import (
    MODEL,
    atomic_json,
    chunk_label,
    engine_token_budget,
    read_json,
)
from poc_mllm_ep_communication_motivation.scripts.run_q3_interference import (
    image_prompt,
    matched_text,
    text_prompt,
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_injection(
    processor: Any, condition: str, visual: dict[str, Any]
) -> tuple[Any | None, int | None, dict[str, int]]:
    begin = time.perf_counter_ns()
    if condition == "DECODE_ONLY":
        return None, None, {"preprocess_begin_ns": begin, "preprocess_end_ns": begin}
    if condition == "VISUAL_PREFILL":
        prompt = image_prompt(processor, visual)
        tokens = int(visual["prompt_token_count"])
    elif condition == "TOKEN_MATCHED_TEXT":
        prompt, tokens = matched_text(processor, int(visual["prompt_token_count"]))
    elif condition == "TEXT_PREFILL":
        prompt = text_prompt(
            processor,
            "Summarize the general considerations for describing an image clearly and accurately.",
        )
        tokens = len(processor.tokenizer.encode(prompt, add_special_tokens=False))
    else:
        raise ValueError(condition)
    return (
        prompt,
        tokens,
        {
            "preprocess_begin_ns": begin,
            "preprocess_end_ns": time.perf_counter_ns(),
        },
    )


def schedule(conditions: list[str], repeats: int, trace: bool) -> list[dict[str, Any]]:
    result = [
        {
            "condition": "DECODE_ONLY",
            "replicate": -1,
            "trace": False,
        }
    ]
    if "VISUAL_PREFILL" in conditions:
        result.append({"condition": "VISUAL_PREFILL", "replicate": -1, "trace": False})
    for replicate in range(repeats):
        offset = replicate % len(conditions)
        for condition in conditions[offset:] + conditions[:offset]:
            result.append(
                {"condition": condition, "replicate": replicate, "trace": trace}
            )
    return result


def worker(
    rank: int,
    master_port: int,
    args: argparse.Namespace,
    barrier: Any,
    progress: Any,
    waves: list[dict[str, Any]],
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
            max_model_len=20_000,
            max_num_batched_tokens=engine_token_budget(args.chunk_size),
            max_num_seqs=5,
            enable_chunked_prefill=args.chunk_size > 0,
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
        # Keep the two external-DP schedulers iteration-aligned under chunked
        # prefill.  The four frozen victims are duplicated across the DP
        # replicas; each copy is still an independent serving request.  Using
        # different visual lengths per DP can make one manually stepped engine
        # retire early and is not a valid collective comparison.
        decode_rows = manifest["q3_decode_pool"][:4]
        decode_prompts = [image_prompt(processor, row) for row in decode_rows]
        all_wave_records = []
        for wave_id, entry in enumerate(waves):
            base_control = {
                **entry,
                "wave": wave_id,
                "special_token_ids": special,
                "expected_phase": "unknown",
                "injection_request_ids": [],
                "capture_stage": False,
                "route_stats": False,
                "chunk_label": chunk_label(args.chunk_size),
                "chunk_size": args.chunk_size,
            }
            if rank == 0:
                atomic_json(args.control, base_control)
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
            request_arrival_ns = time.perf_counter_ns()
            submitted_decode_ids = llm._add_completion_requests(
                [copy.deepcopy(value) for value in decode_prompts],
                params,
                use_tqdm=False,
            )
            expected_sequences = Counter(
                tuple(
                    int(value)
                    for value in row["forced_token_ids"][: args.decode_tokens]
                )
                for row in decode_rows
            )
            injection_prompt, injection_tokens, preprocess = make_injection(
                processor, entry["condition"], manifest["q3_large_visual"]
            )
            completed: dict[str, Any] = {}
            injection_ids: list[Any] = []
            # Only DP0 admits the injected prefill.  DP1 still participates in
            # the physical collective but has no local injected request whose
            # completion it can observe.
            injection_finished = injection_prompt is None or rank != 0
            injection_submit_iteration = None
            injection_submit_ns = None
            injection_finish_ns = None
            records = []
            outer_iteration = 0
            while llm.llm_engine.has_unfinished_requests():
                if (
                    injection_prompt is not None
                    and not injection_ids
                    and outer_iteration == args.injection_progress
                    and rank == 0
                ):
                    injection_ids = llm._add_completion_requests(
                        [copy.deepcopy(injection_prompt)],
                        [SamplingParams(max_tokens=1, temperature=0.0, seed=3407)],
                        use_tqdm=False,
                    )
                    injection_submit_iteration = outer_iteration
                    injection_submit_ns = time.perf_counter_ns()
                capture = bool(entry["trace"]) and outer_iteration >= max(
                    0, args.injection_progress - 4
                )
                if (
                    capture
                    and outer_iteration > args.injection_progress + 8
                    and injection_finished
                    and not args.trace_through_decode
                ):
                    capture = False
                control = {
                    **base_control,
                    "outer_iteration": outer_iteration,
                    "decode_progress": outer_iteration,
                    "capture_stage": capture,
                    "route_stats": capture,
                    "injection_request_ids": [str(value) for value in injection_ids],
                }
                if rank == 0:
                    atomic_json(args.control, control)
                barrier.wait(timeout=1800)
                begin_ns = time.perf_counter_ns()
                outputs = llm.llm_engine.step()
                elapsed_ms = (time.perf_counter_ns() - begin_ns) / 1e6
                barrier.wait(timeout=1800)
                for output in outputs:
                    request_id = str(output.request_id)
                    if output.finished:
                        completed[request_id] = output
                        # External-DP caller IDs and RequestOutput IDs may use
                        # different namespaces.  The bounded injected request
                        # emits exactly one token, whereas every resident emits
                        # ``decode_tokens`` forced tokens.
                        if (
                            injection_ids
                            and output.outputs
                            and len(output.outputs[0].token_ids) == 1
                        ):
                            injection_finished = True
                            injection_finish_ns = time.perf_counter_ns()
                records.append(
                    {
                        "outer_iteration": outer_iteration,
                        "step_wall_ms": elapsed_ms,
                        "decode_progress_before": outer_iteration,
                        "decode_progress_after": outer_iteration + 1,
                        "injection_submitted": injection_submit_iteration
                        == outer_iteration,
                        "injection_finished": injection_finished,
                    }
                )
                outer_iteration += 1
                if outer_iteration > args.decode_tokens + 64:
                    raise RuntimeError(
                        f"wave exceeded bounded iterations: {outer_iteration}"
                    )
            actual_sequences = Counter(
                tuple(int(value) for value in output.outputs[0].token_ids)
                for output in completed.values()
                if len(output.outputs[0].token_ids) == args.decode_tokens
            )
            if actual_sequences != expected_sequences:
                raise RuntimeError(
                    ("forced-token parity", actual_sequences, expected_sequences)
                )
            all_wave_records.append(
                {
                    "wave": wave_id,
                    "condition": entry["condition"],
                    "replicate": entry["replicate"],
                    "trace": entry["trace"],
                    "decode_request_ids": [row["request_id"] for row in decode_rows],
                    "submitted_decode_ids": [
                        str(value) for value in submitted_decode_ids
                    ],
                    "injection_request_ids": [str(value) for value in injection_ids],
                    "injection_prompt_tokens": injection_tokens,
                    "request_arrival_ns": request_arrival_ns,
                    "injection_submit_iteration": injection_submit_iteration,
                    "injection_submit_ns": injection_submit_ns,
                    "injection_finish_ns": injection_finish_ns,
                    **preprocess,
                    "steps": records,
                }
            )
            progress.put((rank, wave_id, "COMPLETE"))
        from poc_mllm_ep_collective_granularity.instrumentation import _drain_events
        from poc_mllm_ep_communication_motivation.instrumentation import _drain

        _drain()
        _drain_events()
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument(
        "--conditions",
        default="DECODE_ONLY,VISUAL_PREFILL,TOKEN_MATCHED_TEXT",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace-through-decode", action="store_true")
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--injection-progress", type=int, default=16)
    parser.add_argument("--watchdog-sec", type=int, default=3600)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3,6,7":
        raise RuntimeError("requires CUDA_VISIBLE_DEVICES=2,3,6,7")
    if os.environ.get("MLLM_COMM_MOTIVATION_HOOK_IMPORTED") != "1":
        raise RuntimeError("startup hook was not imported")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    conditions = args.conditions.split(",")
    waves = schedule(conditions, args.repeats, args.trace)
    manifest = read_json(args.manifest)
    atomic_json(
        args.output / "manifest.json",
        {
            "status": "FROZEN",
            "physical_gpus": [2, 3, 6, 7],
            "topology": "TP2-DP2-EP4-PP1",
            "chunk_size_requested": args.chunk_size,
            "chunk_label": chunk_label(args.chunk_size),
            "enable_chunked_prefill": args.chunk_size > 0,
            "max_num_batched_tokens": engine_token_budget(args.chunk_size),
            "max_num_seqs_per_dp": 5,
            "global_decode_batch": 8,
            "decode_tokens": args.decode_tokens,
            "injection_progress": args.injection_progress,
            "conditions": conditions,
            "repeats": args.repeats,
            "trace": args.trace,
            "trace_through_decode": args.trace_through_decode,
            "waves": waves,
            "large_visual_request": manifest["q3_large_visual"],
        },
    )
    atomic_json(args.control, {"condition": "STARTUP", "capture_stage": False})
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    progress = context.Queue()
    master_port = free_port()
    workers = [
        context.Process(
            target=worker,
            args=(
                rank,
                master_port,
                args,
                barrier,
                progress,
                waves,
                manifest,
            ),
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
