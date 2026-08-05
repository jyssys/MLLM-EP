"""Run exactly one opt-in vLLM capture request on TP2/DP2/EP4."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import time
import traceback
from pathlib import Path
from typing import Any


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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
    return (
        {"prompt": text, "multi_modal_data": {"image": image}},
        {
            "image_token_id": int(
                tokenizer.convert_tokens_to_ids(processor.image_token)
            ),
            "vision_start_token_id": int(
                tokenizer.convert_tokens_to_ids("<|vision_start|>")
            ),
            "vision_end_token_id": int(
                tokenizer.convert_tokens_to_ids("<|vision_end|>")
            ),
        },
    )


def _rank_path(output: Path, dp_rank: int) -> Path:
    return output.with_name(f"{output.stem}.dp_rank{dp_rank}{output.suffix}")


def _run_dp_rank(
    dp_rank: int,
    master_port: int,
    args: argparse.Namespace,
    barrier: Any,
) -> None:
    rank_path = _rank_path(Path(args.output), dp_rank)
    try:
        os.environ["VLLM_DP_RANK"] = str(dp_rank)
        os.environ["VLLM_DP_RANK_LOCAL"] = str(dp_rank)
        os.environ["VLLM_DP_SIZE"] = "2"
        os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
        os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
        from vllm import LLM, SamplingParams
        from vllm.outputs import RequestOutput
        from vllm.v1.engine import EngineCoreRequestType

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
            max_model_len=1024,
            max_num_batched_tokens=1024,
            max_num_seqs=1,
            skip_mm_profiling=True,
            enable_prefix_caching=False,
            enable_flashinfer_autotune=False,
            moe_backend="auto",
            enforce_eager=True,
            disable_log_stats=False,
        )
        sampling = SamplingParams(max_tokens=1, temperature=0.0)
        barrier.wait(timeout=args.timeout_seconds)
        if dp_rank == 0:
            llm._add_completion_requests([prompt], sampling, use_tqdm=False)
            outputs = llm._run_engine(RequestOutput, use_tqdm=False)
        else:
            llm.llm_engine.engine_core._send_input(
                EngineCoreRequestType.START_DP_WAVE, (0, -1)
            )
            outputs = []
        barrier.wait(timeout=args.timeout_seconds)
        result: dict[str, Any] = {
            "status": "ok",
            "dp_rank": dp_rank,
            "real_requests": 1 if dp_rank == 0 else 0,
            "token_ids": token_ids,
        }
        if dp_rank == 0:
            if len(outputs) != 1:
                raise AssertionError(f"expected one output, received {len(outputs)}")
            request = outputs[0]
            prompt_ids = [int(value) for value in (request.prompt_token_ids or [])]
            output_ids = [int(value) for value in request.outputs[0].token_ids]
            if len(prompt_ids) != 799:
                raise AssertionError(f"expected 799 prompt tokens, got {len(prompt_ids)}")
            if output_ids != [args.expected_output_token]:
                raise AssertionError(
                    f"output token consistency failed: {output_ids} != "
                    f"[{args.expected_output_token}]"
                )
            result.update(
                {
                    "prompt_token_count": len(prompt_ids),
                    "image_token_count": sum(
                        value == token_ids["image_token_id"] for value in prompt_ids
                    ),
                    "output_token_ids": output_ids,
                    "output_text": request.outputs[0].text,
                }
            )
        rank_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except BaseException as exc:
        rank_path.write_text(
            json.dumps(
                {
                    "status": "error",
                    "dp_rank": dp_rank,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=896)
    parser.add_argument("--expected-output-token", type=int, default=1986)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1073741824)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    output = Path(args.output)
    rank_paths = [_rank_path(output, rank) for rank in range(2)]
    existing = [path for path in [output, *rank_paths] if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite: {existing}")
    output.parent.mkdir(parents=True, exist_ok=False) if not output.parent.exists() else None

    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    port = _open_port()
    processes = [
        context.Process(
            target=_run_dp_rank,
            args=(rank, port, args, barrier),
            name=f"offline-wavefront-dp{rank}",
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
        process.join(15)
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in rank_paths
        if path.exists()
    ]
    exit_codes = [process.exitcode for process in processes]
    status = (
        "ok"
        if not timed_out
        and exit_codes == [0, 0]
        and len(rows) == 2
        and all(row.get("status") == "ok" for row in rows)
        else "error"
    )
    output.write_text(
        json.dumps(
            {
                "status": status,
                "master_port": port,
                "exit_codes": exit_codes,
                "timed_out": bool(timed_out),
                "dp_results": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "exit_codes": exit_codes}, indent=2))
    if status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
