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

from poc_mllm_ep_collective_granularity.scripts.run_serving import make_injection
from poc_mllm_ep_static4_comm_only.contract import (
    MODEL,
    TOPOLOGY,
    atomic_json,
    read_json,
    require_gpu_contract,
)
from poc_mllm_ep_communication_motivation.scripts.run_q3_interference import (
    image_prompt,
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def schedule(repeats: int) -> list[dict[str, Any]]:
    waves = [
        {"policy": "NATIVE_FULL", "replicate": -1},
        {"policy": "STATIC4", "replicate": -1},
    ]
    policies = ["NATIVE_FULL", "STATIC4"]
    for replicate in range(repeats):
        offset = replicate % 2
        for policy in policies[offset:] + policies[:offset]:
            waves.append({"policy": policy, "replicate": replicate})
    return waves


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
            max_num_batched_tokens=20_000,
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
        decode_rows = manifest["q3_decode_pool"][:4]
        decode_prompts = [image_prompt(processor, row) for row in decode_rows]
        all_wave_records = []
        for wave_id, entry in enumerate(waves):
            if args.mode == "correctness" and entry["policy"] == "NATIVE_FULL":
                # Static correctness waves already execute an internal native
                # shadow, so separate native serving waves add no information.
                continue
            base_control = {
                **entry,
                "wave": wave_id,
                "condition": "VISUAL_PREFILL",
                "special_token_ids": special,
                "expected_phase": "unknown",
                "injection_request_ids": [],
                "capture_stage": False,
                "route_stats": False,
                "static4_enabled": entry["policy"] == "STATIC4",
                "static4_correctness": args.mode == "correctness",
                "static4_layers": list(range(48)),
                "static4_search_start": args.injection_progress,
                # External-DP async admission can lag host submission by more
                # than eight outer polls.  Keep the bounded detector live only
                # through this fixed decode wave so the real mixed iteration
                # cannot silently fall outside the capture window.
                "static4_search_end": args.injection_progress
                + args.decode_tokens,
            }
            if rank == 0:
                atomic_json(args.control, base_control)
            barrier.wait(timeout=1800)
            params = []
            for row in decode_rows:
                forced = [
                    int(value) for value in row["forced_token_ids"][: args.decode_tokens]
                ]
                params.append(
                    SamplingParams(
                        max_tokens=len(forced),
                        temperature=0.0,
                        seed=3407,
                        extra_args={"mllm_vds_forced_token_ids": forced},
                    )
                )
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
                processor, "VISUAL_PREFILL", manifest["q3_large_visual"]
            )
            completed: dict[str, Any] = {}
            injection_ids: list[Any] = []
            injection_finished = rank != 0
            injection_submit_ns = None
            injection_finish_ns = None
            injection_output_tokens: list[int] = []
            records = []
            outer_iteration = 0
            while llm.llm_engine.has_unfinished_requests():
                if (
                    not injection_ids
                    and outer_iteration == args.injection_progress
                    and rank == 0
                ):
                    injection_ids = llm._add_completion_requests(
                        [copy.deepcopy(injection_prompt)],
                        [SamplingParams(max_tokens=1, temperature=0.0, seed=3407)],
                        use_tqdm=False,
                    )
                    injection_submit_ns = time.perf_counter_ns()
                capture = entry["replicate"] >= 0 and (
                    args.injection_progress - 2
                    <= outer_iteration
                    <= args.injection_progress + args.decode_tokens
                )
                native_diagnostic = not (
                    args.clean_native and entry["policy"] == "NATIVE_FULL"
                )
                control = {
                    **base_control,
                    "outer_iteration": outer_iteration,
                    "decode_progress": outer_iteration,
                    "capture_stage": capture
                    and entry["policy"] == "NATIVE_FULL"
                    and native_diagnostic,
                    "route_stats": capture and native_diagnostic,
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
                        if output.outputs and len(output.outputs[0].token_ids) == 1:
                            injection_finished = True
                            injection_finish_ns = time.perf_counter_ns()
                            injection_output_tokens = [
                                int(value) for value in output.outputs[0].token_ids
                            ]
                records.append(
                    {
                        "outer_iteration": outer_iteration,
                        "step_wall_ms": elapsed_ms,
                        "injection_submitted": outer_iteration
                        == args.injection_progress,
                        "injection_finished": injection_finished,
                    }
                )
                outer_iteration += 1
                if outer_iteration > args.decode_tokens + 64:
                    raise RuntimeError("bounded wave iteration limit exceeded")
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
                    "policy": entry["policy"],
                    "replicate": entry["replicate"],
                    "mode": args.mode,
                    "decode_request_ids": [row["request_id"] for row in decode_rows],
                    "submitted_decode_ids": [str(value) for value in submitted_decode_ids],
                    "injection_request_ids": [str(value) for value in injection_ids],
                    "injection_prompt_tokens": injection_tokens,
                    "injection_submit_ns": injection_submit_ns,
                    "injection_finish_ns": injection_finish_ns,
                    "injection_output_tokens": injection_output_tokens,
                    "injection_wall_ms": (
                        None
                        if injection_submit_ns is None or injection_finish_ns is None
                        else (injection_finish_ns - injection_submit_ns) / 1e6
                    ),
                    **preprocess,
                    "steps": records,
                }
            )
            progress.put((rank, wave_id, "COMPLETE"))
        from poc_mllm_ep_collective_granularity import instrumentation as prior
        from poc_mllm_ep_communication_motivation import instrumentation as base
        from poc_mllm_ep_static4_comm_only.instrumentation import _drain_static4

        _drain_static4()
        prior._drain_events()
        base._drain()
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
    parser.add_argument("--mode", choices=("correctness", "timing"), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--injection-progress", type=int, default=16)
    parser.add_argument("--watchdog-sec", type=int, default=3600)
    parser.add_argument(
        "--clean-native",
        action="store_true",
        help="leave NATIVE_FULL waves free of diagnostic route/stage capture",
    )
    args = parser.parse_args()
    require_gpu_contract()
    if os.environ.get("MLLM_STATIC4_HOOK_IMPORTED") != "1":
        raise RuntimeError("Static4 startup hook was not imported")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    waves = schedule(args.repeats)
    manifest = read_json(args.manifest)
    atomic_json(
        args.output / "manifest.json",
        {
            "status": "FROZEN",
            "physical_gpus": [2, 3, 6, 7],
            "topology": TOPOLOGY,
            "implementation": "STOCK_DEEPEP_PYTORCH_COMPOSITION",
            "combine_mode": "PROTO_PER_SLICE_COMBINE",
            "slices": 4,
            "decode_priority_after_slice": 2,
            "expert_invocations_per_prefill_layer": 1,
            "mode": args.mode,
            "repeats": args.repeats,
            "decode_tokens": args.decode_tokens,
            "injection_progress": args.injection_progress,
            "clean_native": args.clean_native,
            "waves": waves,
            "large_visual_request": manifest["q3_large_visual"],
        },
    )
    atomic_json(args.control, {"condition": "STARTUP", "static4_enabled": False})
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    progress = context.Queue()
    master_port = free_port()
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
