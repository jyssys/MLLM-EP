"""Run one real multimodal request through vLLM TP2/DP2 coupled DPEP."""

from __future__ import annotations

import argparse
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


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p90": _percentile(values, 0.9),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _prompt(model_path: str, image_size: int) -> tuple[dict[str, Any], dict[str, int]]:
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
    tokenizer = processor.tokenizer
    ids = {
        "image_token_id": int(tokenizer.convert_tokens_to_ids(processor.image_token)),
        "vision_start_token_id": int(
            tokenizer.convert_tokens_to_ids("<|vision_start|>")
        ),
        "vision_end_token_id": int(tokenizer.convert_tokens_to_ids("<|vision_end|>")),
    }
    return {"prompt": text, "multi_modal_data": {"image": image}}, ids


def _rank_output_path(output: Path, dp_rank: int) -> Path:
    return output.with_name(f"{output.stem}.dp_rank{dp_rank}{output.suffix}")


def _generate_synchronized(
    llm: Any,
    local_prompts: list[dict[str, Any]],
    sampling: Any,
    dp_barrier: Any,
    wave: int,
) -> list[Any]:
    """Start the same native DP wave on the real and idle engine cores."""
    from vllm.v1.engine import EngineCoreRequestType
    from vllm.outputs import RequestOutput

    if local_prompts:
        dp_barrier.wait(timeout=180)
        llm._add_completion_requests(local_prompts, sampling, use_tqdm=False)
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)
    else:
        llm.llm_engine.engine_core._send_input(
            EngineCoreRequestType.START_DP_WAVE,
            (wave, -1),
        )
        dp_barrier.wait(timeout=180)
        outputs = []
    dp_barrier.wait(timeout=180)
    return outputs


def _run_dp_rank(
    dp_rank: int,
    dp_size: int,
    master_port: int,
    args: argparse.Namespace,
    dp_barrier: Any,
) -> None:
    rank_output = _rank_output_path(Path(args.output), dp_rank)
    try:
        os.environ["VLLM_DP_RANK"] = str(dp_rank)
        os.environ["VLLM_DP_RANK_LOCAL"] = str(dp_rank)
        os.environ["VLLM_DP_SIZE"] = str(dp_size)
        os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
        os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
        os.environ["FLASHVEP_PHASE1B_MOE_BACKEND"] = args.moe_backend

        from vllm import LLM, SamplingParams

        prompt, token_ids = _prompt(args.model_path, args.image_size)
        llm = LLM(
            model=args.model_path,
            dtype="bfloat16",
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            expert_placement_strategy="linear",
            all2all_backend="allgather_reducescatter",
            enable_return_routed_experts=False,
            enable_ep_weight_filter=True,
            trust_remote_code=True,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=1,
            skip_mm_profiling=True,
            enable_prefix_caching=False,
            enable_flashinfer_autotune=False,
            moe_backend=args.moe_backend,
            enforce_eager=True,
            disable_log_stats=False,
        )
        parallel = llm.llm_engine.vllm_config.parallel_config
        sampling = SamplingParams(max_tokens=1, temperature=0.0)
        local_prompts = [prompt] if dp_rank == 0 else []
        execution_wave = 0

        warmup_tokens: list[list[int]] = []
        for _ in range(args.warmups):
            outputs = _generate_synchronized(
                llm, local_prompts, sampling, dp_barrier, execution_wave
            )
            execution_wave += 1
            if dp_rank == 0:
                warmup_tokens.append(
                    [int(value) for value in outputs[0].outputs[0].token_ids]
                )
            elif outputs:
                raise AssertionError("idle DP rank unexpectedly returned an output")

        rows: list[dict[str, Any]] = []
        reference_prompt_ids: list[int] | None = None
        for iteration in range(args.iterations):
            start_ns = time.perf_counter_ns()
            outputs = _generate_synchronized(
                llm, local_prompts, sampling, dp_barrier, execution_wave
            )
            execution_wave += 1
            end_ns = time.perf_counter_ns()
            row: dict[str, Any] = {
                "iteration_id": iteration,
                "wall_ms": (end_ns - start_ns) / 1_000_000,
                "real_request_count": len(local_prompts),
            }
            if dp_rank == 0:
                request = outputs[0]
                completion = request.outputs[0]
                prompt_ids = [int(value) for value in (request.prompt_token_ids or [])]
                if reference_prompt_ids is None:
                    reference_prompt_ids = prompt_ids
                elif prompt_ids != reference_prompt_ids:
                    raise AssertionError("fixed prompt tokenization changed")
                row.update(
                    {
                        "prompt_token_count": len(prompt_ids),
                        "image_token_count": sum(
                            value == token_ids["image_token_id"] for value in prompt_ids
                        ),
                        "text_and_special_token_count": sum(
                            value != token_ids["image_token_id"] for value in prompt_ids
                        ),
                        "vision_start_indices": [
                            index
                            for index, value in enumerate(prompt_ids)
                            if value == token_ids["vision_start_token_id"]
                        ],
                        "vision_end_indices": [
                            index
                            for index, value in enumerate(prompt_ids)
                            if value == token_ids["vision_end_token_id"]
                        ],
                        "output_token_ids": [
                            int(value) for value in completion.token_ids
                        ],
                        "output_text": completion.text,
                        "routed_experts_return_capture": False,
                    }
                )
            elif outputs:
                raise AssertionError("idle DP rank unexpectedly returned an output")
            rows.append(row)

        microbenchmark_output: dict[str, Any] | None = None
        if args.microbenchmark:
            outputs = _generate_synchronized(
                llm, local_prompts, sampling, dp_barrier, execution_wave
            )
            execution_wave += 1
            if dp_rank == 0:
                completion = outputs[0].outputs[0]
                microbenchmark_output = {
                    "output_token_ids": [int(value) for value in completion.token_ids],
                    "output_text": completion.text,
                }

        result = {
            "status": "ok",
            "dp_rank": dp_rank,
            "settings": {
                "model_path": args.model_path,
                "physical_gpus": [4, 5, 6, 7],
                "tensor_parallel_size": int(parallel.tensor_parallel_size),
                "data_parallel_size": int(parallel.data_parallel_size),
                "pipeline_parallel_size": int(parallel.pipeline_parallel_size),
                "expected_expert_parallel_size": 4,
                "dtype": "bfloat16",
                "all2all_backend": str(parallel.all2all_backend),
                "moe_backend": args.moe_backend,
                "prefix_caching": False,
                "flashinfer_autotune": False,
                "idle_dp_execution": "EngineCoreRequestType.START_DP_WAVE",
                "return_routed_experts": False,
                "warmups": args.warmups,
                "iterations": args.iterations,
                "image_size": [args.image_size, args.image_size],
                "real_request_on_this_dp_rank": dp_rank == 0,
            },
            "token_ids": token_ids,
            "warmup_output_token_ids": warmup_tokens,
            "prompt_token_ids": reference_prompt_ids,
            "iterations": rows,
            "request_wall_ms": _stats([float(row["wall_ms"]) for row in rows]),
            "microbenchmark_request_output": microbenchmark_output,
        }
        rank_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except BaseException as exc:
        failure = {
            "status": "error",
            "dp_rank": dp_rank,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        rank_output.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--moe-backend", default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1073741824)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--microbenchmark", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    rank_outputs = [_rank_output_path(output, rank) for rank in range(2)]
    existing = [path for path in [output, *rank_outputs] if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite: {existing}")
    output.parent.mkdir(parents=True, exist_ok=True)

    master_port = _open_port()
    context = mp.get_context("spawn")
    dp_barrier = context.Barrier(2)
    processes = [
        context.Process(
            target=_run_dp_rank,
            args=(rank, 2, master_port, args, dp_barrier),
            name=f"flashvep-phase1b-dp{rank}",
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()

    deadline = time.monotonic() + args.timeout_seconds
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    timed_out = [process for process in processes if process.is_alive()]
    for process in timed_out:
        process.terminate()
    for process in timed_out:
        process.join(15)
        if process.is_alive():
            process.kill()
            process.join(5)

    rank_results = []
    for path in rank_outputs:
        if path.exists():
            rank_results.append(json.loads(path.read_text(encoding="utf-8")))
    exit_codes = [process.exitcode for process in processes]
    status = (
        "ok"
        if not timed_out
        and exit_codes == [0, 0]
        and len(rank_results) == 2
        and all(result.get("status") == "ok" for result in rank_results)
        else "error"
    )
    aggregate = {
        "status": status,
        "run_id": output.parent.name,
        "master_port": master_port,
        "timed_out_dp_ranks": [
            index for index, process in enumerate(processes) if process in timed_out
        ],
        "exit_codes": exit_codes,
        "rank_results": rank_results,
    }
    output.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "exit_codes": exit_codes}, indent=2))
    if status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
