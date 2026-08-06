"""Run one stock vLLM backend configuration over controlled request counts."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import socket
import statistics
import time
import traceback
from pathlib import Path
from typing import Any


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        weight = position - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    return {
        "median": float(statistics.median(values)),
        "p90": float(percentile(0.9)),
        "mean": float(statistics.fmean(values)),
        "stdev": float(statistics.stdev(values) if len(values) > 1 else 0.0),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _prompt(model_path: str, image_size: int) -> tuple[dict[str, Any], int]:
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    image = Image.new("RGB", (image_size, image_size), (128, 128, 128))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image briefly."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_token_id = int(
        processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    )
    return {"prompt": text, "multi_modal_data": {"image": image}}, image_token_id


def _generate(
    llm: Any,
    prompts: list[dict[str, Any]],
    sampling: Any,
    barrier: Any,
    wave: int,
) -> list[Any]:
    from vllm.outputs import RequestOutput
    from vllm.v1.engine import EngineCoreRequestType

    if prompts:
        barrier.wait(timeout=600)
        llm._add_completion_requests(prompts, sampling, use_tqdm=False)
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)
    else:
        llm.llm_engine.engine_core._send_input(
            EngineCoreRequestType.START_DP_WAVE, (wave, -1)
        )
        barrier.wait(timeout=600)
        outputs = []
    barrier.wait(timeout=600)
    return outputs


def _rank_path(output: Path, dp_rank: int) -> Path:
    return output.with_name(f"{output.stem}.dp_rank{dp_rank}{output.suffix}")


def _run_rank(
    dp_rank: int,
    port: int,
    args: argparse.Namespace,
    barrier: Any,
) -> None:
    path = _rank_path(args.output, dp_rank)
    try:
        os.environ.update(
            {
                "VLLM_DP_RANK": str(dp_rank),
                "VLLM_DP_RANK_LOCAL": str(dp_rank),
                "VLLM_DP_SIZE": "2",
                "VLLM_DP_MASTER_IP": "127.0.0.1",
                "VLLM_DP_MASTER_PORT": str(port),
                "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": args.all2all_backend,
                "FLASHVEP_CONFIGURED_DBO": str(args.enable_dbo).lower(),
            }
        )
        from vllm import LLM, SamplingParams

        prompt, image_token_id = _prompt(args.model_path, args.image_size)
        llm = LLM(
            model=args.model_path,
            dtype="bfloat16",
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            expert_placement_strategy="linear",
            all2all_backend=args.all2all_backend,
            enable_dbo=args.enable_dbo,
            dbo_prefill_token_threshold=512,
            enable_return_routed_experts=False,
            enable_ep_weight_filter=True,
            trust_remote_code=True,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            max_model_len=1024,
            max_num_batched_tokens=8192,
            max_num_seqs=8,
            skip_mm_profiling=True,
            enable_prefix_caching=False,
            enable_flashinfer_autotune=False,
            moe_backend="auto",
            enforce_eager=True,
            disable_log_stats=False,
        )
        parallel = llm.llm_engine.vllm_config.parallel_config
        sampling = SamplingParams(max_tokens=1, temperature=0.0)
        wave = 0
        batch_results = []
        all_correct = True
        for global_batch in args.request_counts:
            local_batch = 1 if global_batch == 1 and dp_rank == 0 else (
                0 if global_batch == 1 else global_batch // 2
            )
            prompts = [copy.deepcopy(prompt) for _ in range(local_batch)]
            for _ in range(args.warmups):
                outputs = _generate(llm, prompts, sampling, barrier, wave)
                wave += 1
                if len(outputs) != local_batch:
                    raise AssertionError("warmup output count mismatch")
            samples = []
            output_tokens: list[list[int]] = []
            prompt_count = None
            image_count = None
            for _ in range(args.iterations):
                start = time.perf_counter_ns()
                outputs = _generate(llm, prompts, sampling, barrier, wave)
                end = time.perf_counter_ns()
                wave += 1
                samples.append((end - start) / 1_000_000)
                if len(outputs) != local_batch:
                    raise AssertionError("measured output count mismatch")
                if outputs:
                    ids = [
                        [int(value) for value in request.outputs[0].token_ids]
                        for request in outputs
                    ]
                    output_tokens.extend(ids)
                    prompt_ids = [int(value) for value in outputs[0].prompt_token_ids]
                    prompt_count = len(prompt_ids)
                    image_count = sum(value == image_token_id for value in prompt_ids)
                    if any(values != [args.expected_output_token] for values in ids):
                        all_correct = False
            batch_results.append(
                {
                    "global_request_count": global_batch,
                    "local_request_count": local_batch,
                    "wall_ms": samples,
                    "wall_ms_stats": _stats(samples),
                    "prompt_tokens_per_request": prompt_count,
                    "vision_tokens_per_request": image_count,
                    "output_token_ids": output_tokens,
                    "correctness": all(
                        values == [args.expected_output_token]
                        for values in output_tokens
                    ),
                }
            )
        result = {
            "status": "ok" if all_correct else "correctness_failed",
            "dp_rank": dp_rank,
            "settings": {
                "physical_gpus": [4, 5, 6, 7],
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "model_path": args.model_path,
                "dtype": "bfloat16",
                "tp": int(parallel.tensor_parallel_size),
                "dp": int(parallel.data_parallel_size),
                "ep": 4,
                "pp": int(parallel.pipeline_parallel_size),
                "all2all_backend": str(parallel.all2all_backend),
                "enable_dbo": bool(parallel.enable_dbo),
                "num_ubatches": int(parallel.num_ubatches),
                "warmups": args.warmups,
                "iterations": args.iterations,
            },
            "batches": batch_results,
        }
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except BaseException as exc:
        path.write_text(
            json.dumps(
                {
                    "status": "error",
                    "dp_rank": dp_rank,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all2all-backend", required=True)
    parser.add_argument("--enable-dbo", action="store_true")
    parser.add_argument("--request-counts", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=896)
    parser.add_argument("--expected-output-token", type=int, default=1986)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1073741824)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    if any(count < 1 or (count > 1 and count % 2) for count in args.request_counts):
        raise ValueError("counts above one must divide evenly over DP2")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or any(_rank_path(args.output, rank).exists() for rank in range(2)):
        raise FileExistsError("refusing to overwrite backend result")

    port = _open_port()
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    processes = [
        context.Process(target=_run_rank, args=(rank, port, args, barrier))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + args.timeout_seconds
    for process in processes:
        process.join(max(0, deadline - time.monotonic()))
    timed_out = [process for process in processes if process.is_alive()]
    for process in timed_out:
        process.terminate()
        process.join(15)
    rows = [
        json.loads(_rank_path(args.output, rank).read_text())
        for rank in range(2)
        if _rank_path(args.output, rank).exists()
    ]
    status = (
        "ok"
        if not timed_out
        and [process.exitcode for process in processes] == [0, 0]
        and len(rows) == 2
        and all(row.get("status") == "ok" for row in rows)
        else "error"
    )
    aggregate = {
        "status": status,
        "all2all_backend": args.all2all_backend,
        "enable_dbo": args.enable_dbo,
        "master_port": port,
        "exit_codes": [process.exitcode for process in processes],
        "timed_out": bool(timed_out),
        "rank_results": rows,
    }
    args.output.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "exit_codes": aggregate["exit_codes"]}, indent=2))
    if status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
