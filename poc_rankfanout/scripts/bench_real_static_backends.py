#!/usr/bin/env python3
"""Paired Qwen3-VL TP2/DP2/EP4 clean/profile backend runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import socket
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poc_rankfanout.rankfanout.routing import linear_expert_to_rank, summarize_route  # noqa: E402


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q)) if values else float("nan")


def make_text_ids(tokenizer, length: int, seed: int) -> list[int]:
    head = tokenizer.encode(f"Request {seed}. Analyze this distributed systems workload. ", add_special_tokens=False)
    fill = tokenizer.encode(
        "Sparse experts process routed tokens while collective communication and computation share the GPU. ",
        add_special_tokens=False,
    )
    while len(head) < length:
        head.extend(fill)
    return head[:length]


def make_image(size: int, seed: int):
    from PIL import Image

    y, x = np.mgrid[0:size, 0:size]
    pixels = np.empty((size, size, 3), dtype=np.uint8)
    pixels[..., 0] = (x + seed * 17) % 256
    pixels[..., 1] = (y * 3 + seed * 29) % 256
    pixels[..., 2] = ((x // 16 + y // 16 + seed) % 2) * 255
    return Image.fromarray(pixels, mode="RGB")


def make_input(processor, tokenizer, case: dict, seed: int):
    kind = str(case["kind"])
    if kind == "text":
        return {"prompt_token_ids": make_text_ids(tokenizer, int(case["length"]), seed)}
    if kind in {"image", "vision_heavy"}:
        image = make_image(int(case.get("image_size", 448)), seed)
        text = str(case.get("text", "Describe the image carefully and identify its spatial patterns."))
        if kind == "vision_heavy":
            text += " " + "Compare every region. " * int(case.get("text_repeats", 8))
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "multi_modal_data": {"image": image}}
    raise ValueError(f"unsupported workload kind {kind}")


def normalize_routes(routed) -> np.ndarray:
    array = np.asarray(routed, dtype=np.int64)
    if array.ndim != 3 or array.shape[-1] != 8:
        raise ValueError(f"unexpected routed_experts shape {array.shape}")
    if array.shape[1] == 48:
        return array
    if array.shape[0] == 48:
        return np.transpose(array, (1, 0, 2))
    raise ValueError(f"cannot locate 48-layer axis in {array.shape}")


def worker(dp_rank: int, args, barrier, queue) -> None:
    output = Path(args.output)
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "4,5,6,7":
            raise RuntimeError("GPU safety violation")
        context_file = output / f"context_dp{dp_rank}.json"
        os.environ.update(
            VLLM_DP_RANK=str(dp_rank),
            VLLM_DP_RANK_LOCAL=str(dp_rank),
            VLLM_DP_SIZE="2",
            VLLM_DP_MASTER_IP="127.0.0.1",
            VLLM_DP_MASTER_PORT=str(args.port),
            RANKFANOUT_CONTEXT_FILE=str(context_file),
        )
        if args.profile_moe:
            trace_dir = output / "profile"
            trace_dir.mkdir(parents=True, exist_ok=True)
            os.environ["RANKFANOUT_TRACE_DIR"] = str(trace_dir)
            hook = ROOT / "poc_rankfanout" / "hooks"
            os.environ["PYTHONPATH"] = str(hook) + os.pathsep + os.environ.get("PYTHONPATH", "")
            import importlib.util

            spec = importlib.util.spec_from_file_location(f"rankfanout_hook_dp{dp_rank}", hook / "sitecustomize.py")
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        from transformers import AutoProcessor, AutoTokenizer
        from vllm import LLM, SamplingParams

        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        llm = LLM(
            model=args.model,
            dtype="bfloat16",
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            expert_placement_strategy="linear",
            all2all_backend=args.backend,
            enable_dbo=False,
            enable_eplb=False,
            # Direct route capture is provided by the profiling hook.  The
            # v0.20 returned-route request-slot buffer is disabled because it
            # returned zeros in a dedicated DP2 eager smoke test.
            enable_return_routed_experts=False,
            enable_ep_weight_filter=True,
            trust_remote_code=True,
            kv_cache_memory_bytes=args.kv_cache_gb << 30,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            enable_prefix_caching=False,
            enable_flashinfer_autotune=False,
            moe_backend="triton",
            enforce_eager=True,
            disable_log_stats=False,
        )
        parallel = llm.llm_engine.vllm_config.parallel_config
        proof = {
            "backend_requested": args.backend,
            "backend_runtime": str(parallel.all2all_backend),
            "tp": int(parallel.tensor_parallel_size),
            "dp": int(parallel.data_parallel_size),
            "effective_ep": 4,
            "dbo": bool(parallel.enable_dbo),
            "eplb": False,
            "moe_backend": "triton",
            "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "capture_routes": args.capture_routes,
            "vllm_returned_routes": False,
            "profile_moe": args.profile_moe,
        }
        if args.backend not in proof["backend_runtime"]:
            raise RuntimeError(f"backend fallback: {proof}")

        cases = json.loads(args.cases)
        rows: list[dict] = []
        route_rows: list[dict] = []
        route_dir = output / "routes"
        route_dir.mkdir(parents=True, exist_ok=True)
        mapping = linear_expert_to_rank()
        for case_id, case in enumerate(cases):
            concurrency = int(case.get("concurrency", 2))
            if concurrency % 2:
                raise ValueError("concurrency must be even for DP2 matched runs")
            local_n = concurrency // 2
            requests = [make_input(processor, tokenizer, case, case_id * 1000 + dp_rank * 100 + j) for j in range(local_n)]
            params = SamplingParams(temperature=0, max_tokens=int(case.get("output", 1)), ignore_eos=True)
            for iteration in range(args.warmup + args.repetitions):
                warmup = iteration < args.warmup
                context_file.write_text(json.dumps({
                    "run_id": args.run_id,
                    "workload_id": str(case["name"]),
                    "case_id": case_id,
                    "iteration": iteration,
                    "warmup": warmup,
                    "backend": args.backend,
                    "driver_dp_rank": dp_rank,
                }))
                barrier.wait(timeout=3600)
                wall_start = time.perf_counter()
                outputs = llm.generate(requests, params, use_tqdm=False)
                wall_ms = (time.perf_counter() - wall_start) * 1000.0
                barrier.wait(timeout=3600)
                request_rows = []
                for local_index, request in enumerate(outputs):
                    completion = request.outputs[0]
                    metrics = request.metrics
                    token_ids = [int(value) for value in completion.token_ids]
                    first = float(metrics.first_token_ts)
                    last = float(metrics.last_token_ts)
                    row = {
                        "request_id": f"dp{dp_rank}-r{local_index}",
                        "request_local": local_index,
                        "prompt_tokens": len(request.prompt_token_ids or []),
                        "output_tokens": len(token_ids),
                        "ttft_ms": float(metrics.first_token_latency) * 1000.0,
                        "e2e_ms": (float(metrics.first_token_latency) + last - first) * 1000.0,
                        "tpot_ms": (last - first) * 1000.0 / max(1, len(token_ids) - 1),
                        "output_token_ids": token_ids,
                    }
                    if args.capture_routes:
                        # Use the observer's direct select_experts tensors.
                        # The stock v0.20 returned-route slot buffer is not
                        # trusted in DP2 eager mode (it returned all zeros in a
                        # dedicated smoke run).
                        direct = output / "profile" / "direct_routes"
                        layers = []
                        source_ep_ranks = (2 * dp_rank, 2 * dp_rank + 1)
                        per_ep_layers: dict[int, list[np.ndarray]] = {rank: [] for rank in source_ep_ranks}
                        for layer_id in range(48):
                            layer_parts = []
                            for source_ep_rank in source_ep_ranks:
                                source = direct / f"{args.run_id}_{case['name']}_it{iteration}_ep{source_ep_rank}_layer{layer_id}.npy"
                                if not source.exists():
                                    raise RuntimeError(f"missing direct route capture: {source}")
                                layer_route = np.load(source)
                                if layer_route.ndim != 2 or layer_route.shape[1] != 8:
                                    raise RuntimeError(f"bad direct route shape {layer_route.shape}: {source}")
                                per_ep_layers[source_ep_rank].append(layer_route)
                                layer_parts.append(layer_route)
                            layers.append(np.concatenate(layer_parts, axis=0))
                        token_counts = {layer.shape[0] for layer in layers}
                        if len(token_counts) != 1:
                            raise RuntimeError(f"layer route token-count mismatch: {sorted(token_counts)}")
                        routes = np.stack(layers, axis=1)
                        route_name = f"{args.run_id}_{case['name']}_it{iteration}_dp{dp_rank}_r{local_index}.npz"
                        np.savez_compressed(route_dir / route_name, routed_experts=routes.astype(np.int16))
                        row["route_file"] = str(Path("routes") / route_name)
                        for source_ep_rank in source_ep_ranks:
                            for layer_id, layer_route in enumerate(per_ep_layers[source_ep_rank]):
                                digest = hashlib.sha256(layer_route.astype(np.int16).tobytes()).hexdigest()
                                route_rows.append({
                                    "run_id": args.run_id,
                                    "backend": args.backend,
                                    "workload_id": str(case["name"]),
                                    "request_id": row["request_id"],
                                    "phase": "prefill",
                                    "step_id": iteration,
                                    "layer_id": layer_id,
                                    "source_ep_rank": source_ep_rank,
                                    "num_tokens": int(layer_route.shape[0]),
                                    "route_hash": digest,
                                    **summarize_route(layer_route, mapping, source_rank=source_ep_rank).to_dict(),
                                })
                    request_rows.append(row)
                rows.append({
                    "run_id": args.run_id,
                    "backend": args.backend,
                    "workload_id": str(case["name"]),
                    "case_id": case_id,
                    "iteration": iteration,
                    "warmup": warmup,
                    "kind": str(case["kind"]),
                    "concurrency": concurrency,
                    "output_budget": int(case.get("output", 1)),
                    "dp_rank": dp_rank,
                    "wall_ms": wall_ms,
                    "requests": request_rows,
                })
        (output / f"driver_dp{dp_rank}.json").write_text(json.dumps({"proof": proof, "rows": rows}, indent=2) + "\n")
        if route_rows:
            with (output / f"route_rows_dp{dp_rank}.jsonl").open("w", encoding="utf-8") as handle:
                for row in route_rows:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        llm.llm_engine.engine_core.shutdown()
        queue.put((dp_rank, True, ""))
    except BaseException:
        error = traceback.format_exc()
        output.mkdir(parents=True, exist_ok=True)
        (output / f"failure_dp{dp_rank}.txt").write_text(error)
        queue.put((dp_rank, False, error))
        raise


def aggregate(args) -> None:
    root = Path(args.output)
    payloads = [json.loads((root / f"driver_dp{rank}.json").read_text()) for rank in range(2)]
    grouped: dict[tuple[int, int], list[dict]] = {}
    for payload in payloads:
        for row in payload["rows"]:
            grouped.setdefault((row["case_id"], row["iteration"]), []).append(row)
    summary_rows = []
    for (case_id, iteration), parts in sorted(grouped.items()):
        requests = [request for part in parts for request in part["requests"]]
        ttft = [request["ttft_ms"] for request in requests]
        summary_rows.append({
            "run_id": args.run_id,
            "backend": args.backend,
            "workload_id": parts[0]["workload_id"],
            "case_id": case_id,
            "iteration": iteration,
            "warmup": parts[0]["warmup"],
            "kind": parts[0]["kind"],
            "concurrency": parts[0]["concurrency"],
            "fleet_wall_ms": max(part["wall_ms"] for part in parts),
            "ttft_p50_ms": statistics.median(ttft),
            "ttft_p90_ms": percentile(ttft, 0.9),
            "requests": requests,
        })
    (root / "summary.json").write_text(json.dumps({
        "proof": payloads[0]["proof"],
        "rows": summary_rows,
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=("allgather_reducescatter", "deepep_high_throughput"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--capture-routes", action="store_true")
    parser.add_argument("--profile-moe", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--kv-cache-gb", type=int, default=16)
    args = parser.parse_args()
    if args.profile_moe and not args.capture_routes:
        parser.error("--profile-moe requires --capture-routes for alignment")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4,5,6,7":
        raise SystemExit("GPU safety violation")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    args.output = str(root)
    args.port = free_port()
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    queue = context.Queue()
    processes = [context.Process(target=worker, args=(rank, args, barrier, queue)) for rank in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(7200)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(20)
    statuses = [queue.get(timeout=20) for _ in processes]
    (root / "driver_status.json").write_text(json.dumps({"statuses": statuses}, indent=2) + "\n")
    if not all(status[1] and process.exitcode == 0 for status, process in zip(statuses, processes)):
        raise SystemExit(1)
    aggregate(args)


if __name__ == "__main__":
    main()
