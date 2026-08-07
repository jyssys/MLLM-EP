"""Capture full raw logits for the bounded final DBO correctness gate."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import traceback
from pathlib import Path

import numpy as np

from poc_flashvep.deepep_revalidation.vllm_backend_matrix import (
    _generate,
    _open_port,
    _prompt,
    _rank_path,
)


def _dense_logits(step: dict[int, object]) -> np.ndarray:
    size = max(int(token_id) for token_id in step) + 1
    values = np.empty(size, dtype=np.float32)
    for token_id, item in step.items():
        values[int(token_id)] = float(item.logprob)
    return values


def _run_rank(rank: int, port: int, args: argparse.Namespace, barrier: object) -> None:
    output = _rank_path(args.output, rank)
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(rank),
            "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": "2",
            "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
        })
        from vllm import LLM, SamplingParams

        blue_prompt, _ = _prompt(args.model_path, 896, "text", 790, "blue")
        red_prompt, _ = _prompt(args.model_path, 896, "text", 790, "red")
        llm = LLM(
            model=args.model_path,
            dtype="bfloat16",
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput",
            enable_dbo=args.enable_dbo,
            dbo_prefill_token_threshold=512,
            enable_ep_weight_filter=True,
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
            kv_cache_memory_bytes=1073741824,
            max_model_len=1024,
            max_num_batched_tokens=8192,
            max_num_seqs=8,
            skip_mm_profiling=True,
            enable_prefix_caching=False,
            enable_flashinfer_autotune=False,
            moe_backend="auto",
            enforce_eager=True,
            logprobs_mode="raw_logits",
            max_logprobs=151936,
        )
        sampling = SamplingParams(
            max_tokens=4, temperature=0.0, logprobs=-1
        )
        arrays: dict[str, np.ndarray] = {}
        runs = []
        for repetition in range(3):
            outputs, submitted = _generate(
                llm, [blue_prompt, red_prompt], sampling, barrier, repetition
            )
            red_request_id = str(submitted[1]).split("-", 1)[0]
            red_output = next(
                output for output in outputs
                if str(output.request_id) == red_request_id
            )
            candidate = red_output.outputs[0]
            tokens = [int(value) for value in candidate.token_ids]
            if candidate.logprobs is None:
                raise RuntimeError("raw logits were not returned")
            dense = np.stack([_dense_logits(step) for step in candidate.logprobs])
            arrays[f"run_{repetition}"] = dense
            runs.append({
                "repetition": repetition,
                "submitted_request_id": submitted[1],
                "restored_request_id": str(red_output.request_id),
                "generated_token_ids": tokens,
                "logits_shape": list(dense.shape),
            })
        np.savez_compressed(output.with_suffix(".npz"), **arrays)
        output.write_text(json.dumps({
            "status": "ok",
            "dp_rank": rank,
            "enable_dbo": args.enable_dbo,
            "runs": runs,
        }, indent=2) + "\n")
    except BaseException as exc:
        output.write_text(json.dumps({
            "status": "error", "dp_rank": rank, "error": repr(exc),
            "traceback": traceback.format_exc(),
        }, indent=2) + "\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--enable-dbo", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    port = _open_port()
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    processes = [context.Process(target=_run_rank, args=(rank, port, args, barrier)) for rank in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(1800)
    if any(process.is_alive() or process.exitcode != 0 for process in processes):
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
