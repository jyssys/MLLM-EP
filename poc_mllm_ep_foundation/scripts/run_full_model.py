from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from poc_mllm_ep_foundation.constants import MODEL, OFFICIAL_GENERATION
from poc_mllm_ep_foundation.contract import parse_physical_gpus
from poc_mllm_ep_foundation.evaluation.workloads import load_manifest
from poc_mllm_ep_foundation.io import append_jsonl_atomic, atomic_json, completed_ids


def _ensure_runtime_tools() -> None:
    """Expose console scripts shipped beside the explicitly invoked Python.

    The isolated runtime is commonly invoked by absolute Python path rather
    than activation.  In that mode its packages are visible but its ``ninja``
    console script is not, causing Triton JIT to fail during vLLM warm-up.
    """
    if shutil.which("ninja"):
        return
    runtime_bin = Path(sys.executable).parent
    if (runtime_bin / "ninja").is_file():
        os.environ["PATH"] = f"{runtime_bin}:{os.environ.get('PATH', '')}"
    if not shutil.which("ninja"):
        raise RuntimeError(f"ninja missing from runtime and PATH; checked {runtime_bin}")


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _profile_control(args: argparse.Namespace, wave: int,
                     batches: list[list[dict[str, Any]]], *, flush: bool = False) -> None:
    if args.profile_dir is None:
        return
    by_dp = {}
    for rank, batch in enumerate(batches):
        ids = [str(row["request_id"]) for row in batch]
        by_dp[str(rank)] = {
            "request_id": "+".join(ids) if ids else f"idle_dp{rank}",
            "source_dp_rank": rank,
            "modality": "vision",
            "pair_id": wave,
            "token_bucket": "actual",
        }
    atomic_json(args.profile_dir / "control.json", {
        "wave": wave,
        "iteration": int(batches[0][0].get("anchor_iteration", wave)) if batches and batches[0] else wave,
        "instrument": not flush,
        "measured": bool(batches[0][0].get("anchor_measured", True)) if batches and batches[0] else True,
        "phase": "anchor", "flush": flush,
        "by_dp": by_dp,
    })


def _prepare(processor: Any, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(row["image_path"])
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": str(row["question"])},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processed = processor(text=[prompt], images=[image], return_tensors="pt")
    ids = processed["input_ids"][0].tolist()
    grid = [int(value) for value in processed["image_grid_thw"][0].tolist()]
    image_token = int(processor.tokenizer.convert_tokens_to_ids(processor.image_token))
    placeholder_count = sum(int(token == image_token) for token in ids)
    merge = int(processor.image_processor.merge_size)
    expected = grid[0] * grid[1] * grid[2] // (merge * merge)
    if placeholder_count != expected:
        raise AssertionError((placeholder_count, expected, grid))
    request = {"prompt": prompt, "multi_modal_data": {"image": image}}
    pixels = processed["pixel_values"].detach().contiguous().cpu().numpy().tobytes()
    meta = {
        "formatted_prompt": prompt,
        "prompt_token_count": len(ids),
        "image_grid_thw": grid,
        "vision_token_count": expected,
        "image_size_rgb": list(image.size),
        "processed_pixel_values_sha256": hashlib.sha256(pixels).hexdigest(),
        "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "attention_mask_token_count": int(processed["attention_mask"].sum().item()),
        "mrope_position_status": "GENERATED_BY_CURRENT_VLLM_QWEN3_VL_RUNTIME",
        "prompt_leak_check_prefix": prompt[-160:],
    }
    return request, meta


def _sampling(lane: str, max_tokens: int, seed: int, ignore_eos: bool = False):
    from vllm import SamplingParams
    if lane == "official":
        return SamplingParams(
            max_tokens=max_tokens, temperature=0.7, top_p=0.8, top_k=20,
            repetition_penalty=1.0, presence_penalty=1.5, seed=seed,
            ignore_eos=ignore_eos,
        )
    return SamplingParams(max_tokens=max_tokens, temperature=0.0, seed=seed,
                          ignore_eos=ignore_eos)


def _kwargs(model: str, physical_count: int, backend: str, max_model_len: int,
            kv_cache_gib: float, max_num_seqs: int) -> dict[str, Any]:
    ep = backend.startswith("physical_ep")
    # The installed vLLM only enables an all-to-all prepare/finalize when
    # data_parallel_size > 1.  ``physical_ep2`` therefore uses TP1/DP2 so it
    # is a real sparse dispatch/combine anchor.  TP2/DP1 is retained under an
    # explicit diagnostic name because it shards the expert weights but uses
    # the no-DP TP-reduction contract rather than DeepEP transport.
    tp_size = (
        physical_count if not ep else
        1 if backend == "physical_ep2" else
        2
    )
    return {
        "model": model, "dtype": "bfloat16", "trust_remote_code": True,
        "tensor_parallel_size": tp_size,
        "enable_expert_parallel": ep,
        "expert_placement_strategy": "linear",
        "all2all_backend": "deepep_high_throughput" if ep else "allgather_reducescatter",
        "enable_ep_weight_filter": ep,
        # Route capture is an independent anchor command.  Enabling vLLM's
        # capturer here makes its dummy warm-up assert on the deliberately
        # tiny per-DP shape and is unrelated to generation semantics.
        "enable_return_routed_experts": False,
        "enable_dbo": False,
        "gpu_memory_utilization": 0.90,
        "kv_cache_memory_bytes": int(kv_cache_gib * (1 << 30)),
        "max_model_len": max_model_len,
        "max_num_batched_tokens": min(max_model_len, 16384),
        "max_num_seqs": max_num_seqs,
        "limit_mm_per_prompt": {"image": 1},
        "skip_mm_profiling": True,
        "enable_prefix_caching": False,
        "mm_processor_cache_gb": 0,
        "enable_flashinfer_autotune": False,
        "enforce_eager": True,
        "disable_log_stats": True,
    }


def _record(row: dict[str, Any], meta: dict[str, Any], result: Any, wall_ms: float,
            lane: str, backend: str, dp_rank: int, batch_id: int,
            batch_size: int) -> dict[str, Any]:
    completion = result.outputs[0]
    text = completion.text
    leaked = bool(meta["formatted_prompt"] and meta["formatted_prompt"] in text)
    return {
        **{key: value for key, value in row.items() if key != "answer"},
        **meta,
        "lane": lane,
        "backend": backend,
        "dp_rank": dp_rank,
        "wall_ms": wall_ms,
        "wall_scope": "BATCH_MAKESPAN" if batch_size > 1 else "REQUEST_LATENCY",
        "batch_id": batch_id,
        "batch_size": batch_size,
        "output_text": text,
        "output_token_ids": [int(value) for value in completion.token_ids],
        "output_token_count": len(completion.token_ids),
        "finish_reason": completion.finish_reason,
        "stop_reason": str(completion.stop_reason),
        "prompt_in_output": leaked,
        "answer": row.get("answer"),
        "evidence": "MEASURED_FULL_MODEL_GENERATION",
    }


def _run_reference(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    from transformers import AutoProcessor
    from vllm import LLM
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(**_kwargs(args.model, len(args.gpus), "reference", args.max_model_len,
                        args.kv_cache_gib, args.max_num_seqs))
    records = []
    for batch_id, offset in enumerate(range(0, len(rows), args.batch_size)):
        batch = rows[offset: offset + args.batch_size]
        prepared = [(row, *_prepare(processor, row)) for row in batch]
        requests = [copy.deepcopy(request) for _, request, _ in prepared]
        sampling = [_sampling(args.lane, args.max_tokens, args.seed, args.ignore_eos)
                    for _ in requests]
        begin = time.perf_counter_ns()
        results = llm.generate(requests, sampling, use_tqdm=False)
        wall_ms = (time.perf_counter_ns() - begin) / 1e6
        batch_records = [
            _record(row, meta, result, wall_ms, args.lane, "reference", 0,
                    batch_id, len(batch))
            for (row, _, meta), result in zip(prepared, results)
        ]
        records.extend(batch_records)
        append_jsonl_atomic(args.output / "results.jsonl", batch_records)
    atomic_json(args.output / "run.json", {"status": "COMPLETE", "records": len(records),
                "backend": "reference", "physical_gpus": args.gpus})


def _ep_wave(llm: Any, prompts: list[dict[str, Any]], sampling: Any,
             barrier: Any, wave: int) -> list[Any]:
    from vllm.outputs import RequestOutput
    from vllm.v1.engine import EngineCoreRequestType
    barrier.wait(timeout=1800)
    if prompts:
        llm._add_completion_requests(prompts, sampling, use_tqdm=False)
        outputs = llm._run_engine(RequestOutput, use_tqdm=False)
    else:
        llm.llm_engine.engine_core._send_input(EngineCoreRequestType.START_DP_WAVE, (wave, -1))
        outputs = []
    barrier.wait(timeout=1800)
    return outputs


def _ep_worker(rank: int, port: int, args: argparse.Namespace, barrier: Any,
               waves: list[list[list[dict[str, Any]]]]) -> None:
    target = args.output / f"results.dp{rank}.jsonl"
    try:
        os.environ.update({
            "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": str(args.topology_dp), "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false",
        })
        from transformers import AutoProcessor
        from vllm import LLM
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        llm = LLM(**_kwargs(args.model, len(args.gpus), args.backend, args.max_model_len,
                            args.kv_cache_gib, args.max_num_seqs))
        for wave_id, wave in enumerate(waves):
            if rank == 0:
                _profile_control(args, wave_id, wave)
            batch = wave[rank]
            prepared = [(row, *_prepare(processor, row)) for row in batch]
            requests = [copy.deepcopy(request) for _, request, _ in prepared]
            sampling = [_sampling(args.lane, args.max_tokens, args.seed, args.ignore_eos)
                        for _ in requests]
            begin = time.perf_counter_ns()
            outputs = _ep_wave(llm, requests, sampling, barrier, wave_id)
            wall_ms = (time.perf_counter_ns() - begin) / 1e6
            records = [
                _record(row, meta, output, wall_ms, args.lane, args.backend, rank,
                        wave_id, len(batch))
                for (row, _, meta), output in zip(prepared, outputs)
            ]
            if records:
                append_jsonl_atomic(target, records)
                timing_records = []
                for record in records:
                    timing = dict(record)
                    timing["logical_request_id"] = str(record["request_id"])
                    timing["request_id"] = (
                        f"{record['request_id']}@wave{wave_id}@"
                        f"iteration{record.get('anchor_iteration', 0)}"
                    )
                    timing_records.append(timing)
                append_jsonl_atomic(
                    args.output / f"timings.dp{rank}.jsonl", timing_records
                )
        if args.profile_dir is not None:
            if rank == 0:
                _profile_control(args, len(waves), waves[-1], flush=True)
            flush_batch = waves[-1][rank]
            prepared = [(row, *_prepare(processor, row)) for row in flush_batch]
            requests = [copy.deepcopy(request) for _, request, _ in prepared]
            sampling = [_sampling(args.lane, 1, args.seed, False) for _ in requests]
            _ep_wave(llm, requests, sampling, barrier, len(waves))
        atomic_json(args.output / f"worker.dp{rank}.json", {"ok": True})
    except BaseException:
        atomic_json(args.output / f"worker.dp{rank}.json", {"ok": False, "traceback": traceback.format_exc()})
        raise


def _run_dp_ep(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    stride = args.topology_dp * args.batch_size
    waves: list[list[list[dict[str, Any]]]] = []
    base_waves = []
    for offset in range(0, len(rows), stride):
        base_waves.append([
            rows[offset + rank * args.batch_size:
                 offset + (rank + 1) * args.batch_size]
            for rank in range(args.topology_dp)
        ])
    for base in base_waves:
        for iteration in range(args.warmup_repeats + args.measured_repeats):
            waves.append([[
                {**row, "anchor_iteration": iteration - args.warmup_repeats,
                 "anchor_measured": iteration >= args.warmup_repeats}
                for row in batch
            ] for batch in base])
    context = mp.get_context("spawn")
    barrier = context.Barrier(args.topology_dp)
    port = _port()
    workers = [context.Process(target=_ep_worker, args=(rank, port, args, barrier, waves))
               for rank in range(args.topology_dp)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if [worker.exitcode for worker in workers] != [0] * args.topology_dp:
        raise RuntimeError([worker.exitcode for worker in workers])
    records = []
    for rank in range(args.topology_dp):
        path = args.output / f"results.dp{rank}.jsonl"
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    append_jsonl_atomic(args.output / "results.jsonl", records)
    atomic_json(args.output / "run.json", {"status": "COMPLETE", "records": len(records),
                "backend": args.backend,
                "parallelism": {"tp": args.topology_tp, "dp": args.topology_dp,
                                "sp": 1, "pp": 1, "ep": args.topology_ep},
                "transport_contract": "DEEPEP_SPARSE_DISPATCH_COMBINE",
                "physical_gpus": args.gpus,
                "warmup_repeats": args.warmup_repeats,
                "measured_repeats": args.measured_repeats})


def _run_ep2_tp2_diagnostic(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    from transformers import AutoProcessor
    from vllm import LLM
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(**_kwargs(args.model, 2, "physical_ep2_tp2", args.max_model_len,
                        args.kv_cache_gib, args.max_num_seqs))
    for batch_id, offset in enumerate(range(0, len(rows), args.batch_size)):
        batch = rows[offset: offset + args.batch_size]
        prepared = [(row, *_prepare(processor, row)) for row in batch]
        requests = [copy.deepcopy(request) for _, request, _ in prepared]
        sampling = [_sampling(args.lane, args.max_tokens, args.seed, args.ignore_eos)
                    for _ in requests]
        begin = time.perf_counter_ns()
        outputs = llm.generate(requests, sampling, use_tqdm=False)
        wall_ms = (time.perf_counter_ns() - begin) / 1e6
        records = [
            _record(row, meta, output, wall_ms, args.lane, "physical_ep2_tp2", 0,
                    batch_id, len(batch))
            for (row, _, meta), output in zip(prepared, outputs)
        ]
        append_jsonl_atomic(args.output / "results.jsonl", records)
    atomic_json(args.output / "run.json", {
        "status": "COMPLETE", "records": len(rows), "backend": "physical_ep2_tp2",
        "parallelism": {"tp": 2, "dp": 1, "sp": 1, "pp": 1, "ep": 2},
        "transport_contract": "TP_REDUCTION_NO_DEEPEP_DISPATCH_DIAGNOSTIC",
        "physical_gpus": args.gpus,
    })


def main() -> None:
    _ensure_runtime_tools()
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpus", default="4,5,6,7")
    parser.add_argument("--backend", choices=("reference", "physical_ep", "physical_ep2",
                                              "physical_ep2_tp2", "physical_ep4"), required=True)
    parser.add_argument("--lane", choices=("official", "deterministic"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=str(MODEL))
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--kv-cache-gib", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--ignore-eos", action="store_true",
                        help="system-control only: force exactly --max-tokens decode steps")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--warmup-repeats", type=int, default=0)
    parser.add_argument("--measured-repeats", type=int, default=1)
    args = parser.parse_args()
    args.gpus = parse_physical_gpus(args.physical_gpus)
    args.backend = "physical_ep4" if args.backend == "physical_ep" else args.backend
    topology = {
        "physical_ep2": (1, 2, 2),
        "physical_ep2_tp2": (2, 1, 2),
        "physical_ep4": (2, 2, 4),
    }
    args.topology_tp, args.topology_dp, args.topology_ep = topology.get(
        args.backend, (len(args.gpus), 1, 1)
    )
    if args.backend == "physical_ep4" and len(args.gpus) != 4:
        raise ValueError("physical_ep4 requires four allowed physical GPUs")
    if args.backend in ("physical_ep2", "physical_ep2_tp2") and len(args.gpus) != 2:
        raise ValueError("physical_ep2 requires two allowed physical GPUs")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.warmup_repeats < 0 or args.measured_repeats < 1:
        raise ValueError("invalid repeat counts")
    args.max_tokens = args.max_tokens or (32768 if args.lane == "official" else 128)
    args.max_model_len = args.max_model_len or (40000 if args.lane == "official" else 4096)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpus))
    if args.profile_dir is not None:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        os.environ.update({
            "FLASHVEP_MATRIX_CONTROL": str((args.profile_dir / "control.json").resolve()),
            "FLASHVEP_MATRIX_RAW_DIR": str((args.profile_dir / "raw_live").resolve()),
            "FLASHVEP_DEEPEP_PROOF_DIR": str((args.profile_dir / "backend_proof").resolve()),
        })
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    rows = manifest["requests"][: args.limit]
    if args.resume:
        done = completed_ids(args.output / "results.jsonl")
        # A server/process failure can occur after per-DP atomic checkpoints
        # have landed but before the final merged index is written.  Treat
        # those worker indexes as authoritative completed samples as well.
        for rank in range(args.topology_dp):
            done.update(completed_ids(args.output / f"results.dp{rank}.jsonl"))
        rows = [row for row in rows if str(row["request_id"]) not in done]
    if not rows:
        print("nothing to run")
        return
    atomic_json(args.output / "invocation.json", {
        "physical_gpus": args.gpus, "backend": args.backend, "lane": args.lane,
        "manifest": str(args.manifest.resolve()), "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len, "seed": args.seed, "batch_size": args.batch_size,
        "kv_cache_gib": args.kv_cache_gib, "max_num_seqs": args.max_num_seqs,
        "ignore_eos": args.ignore_eos,
        "official_protocol_deviation": None if args.lane != "official" or args.max_tokens == 32768 else "max_new_tokens capped for bounded run",
    })
    if args.backend == "reference":
        _run_reference(args, rows)
    elif args.backend == "physical_ep2_tp2":
        _run_ep2_tp2_diagnostic(args, rows)
    else:
        _run_dp_ep(args, rows)


if __name__ == "__main__":
    main()
