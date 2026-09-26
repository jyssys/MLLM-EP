from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from poc_mllm_ep_foundation.contract import parse_physical_gpus
from poc_mllm_ep_foundation.evaluation.workloads import load_manifest
from poc_mllm_ep_foundation.io import append_jsonl_atomic, atomic_json, completed_ids
from poc_mllm_ep_foundation.scripts.run_full_model import (
    _ensure_runtime_tools,
    _ep_wave,
    _kwargs,
    _port,
    _profile_control,
    _sampling,
)
from poc_mllm_resolution_expert_tradeoff.contract import (
    MODEL,
    REFERENCE_MIN_PIXELS,
    configured_max_pixels,
    verify_foundation_bundle,
)

_REFERENCE_AREA_CACHE: dict[str, int] = {}


def _processed(
    processor: Any,
    prompt: str,
    image: Image.Image,
    mm_kwargs: dict[str, int] | None,
) -> Any:
    kwargs: dict[str, Any] = {
        "text": [prompt], "images": [image], "return_tensors": "pt"
    }
    if mm_kwargs:
        kwargs["images_kwargs"] = mm_kwargs
    return processor(**kwargs)


def _prepare(processor: Any, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    begin = time.perf_counter_ns()
    path = Path(row["image_path"])
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": str(row["question"])},
    ]}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_hash = str(row["image_sha256"])
    ratio = float(row.get("resolution_ratio_target", 1.0))
    if image_hash not in _REFERENCE_AREA_CACHE:
        reference = _processed(processor, prompt, image, None)
        reference_grid = [int(value) for value in reference["image_grid_thw"][0]]
        _REFERENCE_AREA_CACHE[image_hash] = reference_grid[1] * 16 * reference_grid[2] * 16
    reference_area = _REFERENCE_AREA_CACHE[image_hash]
    max_pixels = configured_max_pixels(reference_area, ratio)
    mm_kwargs = None if max_pixels is None else {
        "min_pixels": min(REFERENCE_MIN_PIXELS, max_pixels),
        "max_pixels": max_pixels,
    }
    processed = _processed(processor, prompt, image, mm_kwargs)
    ids = processed["input_ids"][0].tolist()
    grid = [int(value) for value in processed["image_grid_thw"][0].tolist()]
    image_token = int(processor.tokenizer.convert_tokens_to_ids(processor.image_token))
    placeholder_count = sum(int(token == image_token) for token in ids)
    merge = int(processor.image_processor.merge_size)
    expected = grid[0] * grid[1] * grid[2] // (merge * merge)
    if placeholder_count != expected:
        raise AssertionError((placeholder_count, expected, grid))
    request: dict[str, Any] = {
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    }
    if mm_kwargs:
        request["mm_processor_kwargs"] = mm_kwargs
    pixels = processed["pixel_values"].detach().contiguous().cpu().numpy().tobytes()
    processor_ms = (time.perf_counter_ns() - begin) / 1e6
    meta = {
        "formatted_prompt": prompt,
        "prompt_token_count": len(ids),
        "image_grid_thw": grid,
        "vision_token_count": expected,
        "reference_processed_pixels": reference_area,
        "configured_max_pixels": max_pixels,
        "actual_visual_token_ratio": expected / max(1, reference_area / (16 * 16 * 4)),
        "duplicate_resolution_grid": ratio != 1.0 and expected == reference_area // (16 * 16 * 4),
        "image_size_rgb": list(image.size),
        "processed_pixel_values_sha256": hashlib.sha256(pixels).hexdigest(),
        "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "processor_ms": processor_ms,
    }
    return request, meta


def _metrics(output: Any) -> dict[str, float | None]:
    metrics = output.metrics
    if metrics is None:
        return {"ttft_ms": None, "request_engine_ms": None, "queue_ms": None}
    scheduled = float(metrics.scheduled_ts)
    last = float(metrics.last_token_ts)
    queued = float(metrics.queued_ts)
    return {
        "ttft_ms": 1000.0 * float(metrics.first_token_latency),
        "request_engine_ms": 1000.0 * (last - scheduled) if last and scheduled else None,
        "queue_ms": 1000.0 * (scheduled - queued) if scheduled and queued else None,
    }


def _record(
    row: dict[str, Any], meta: dict[str, Any], output: Any, batch_wall_ms: float,
    rank: int, wave: int, batch_size: int, lane: str,
) -> dict[str, Any]:
    completion = output.outputs[0]
    return {
        **{key: value for key, value in row.items() if key != "answer"},
        **meta,
        **_metrics(output),
        "answer": row.get("answer"),
        "backend": "physical_ep4",
        "lane": lane,
        "dp_rank": rank,
        "batch_id": wave,
        "batch_size_per_dp": batch_size,
        "global_batch_size": batch_size * 2,
        "batch_makespan_ms": batch_wall_ms,
        "output_text": completion.text,
        "output_token_ids": [int(value) for value in completion.token_ids],
        "output_token_count": len(completion.token_ids),
        "finish_reason": completion.finish_reason,
        "stop_reason": str(completion.stop_reason),
        "actual_executed_k_contract": "VISUAL_RAGGED_INVALID_ID_MINUS_ONE_TEXT_AND_DECODE_K8",
        "evidence": "MEASURED_TRUE_EP4_FULL_MODEL_GENERATION",
    }


def _worker(
    rank: int, port: int, args: argparse.Namespace, barrier: Any,
    waves: list[list[list[dict[str, Any]]]],
) -> None:
    target = args.output / f"results.dp{rank}.jsonl"
    try:
        if os.environ.get("MLLM_REX_HOOK_IMPORTED") != "1":
            raise RuntimeError(
                "Part 2 startup hook was not imported in the spawned DP worker; "
                "refusing to label a baseline top-8 execution as mixed-k"
            )
        os.environ.update({
            "VLLM_DP_RANK": str(rank),
            "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": "2",
            "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND": "deepep_high_throughput",
            "FLASHVEP_CONFIGURED_DBO": "false",
        })
        from transformers import AutoProcessor
        from vllm import LLM

        processor = AutoProcessor.from_pretrained(str(MODEL), trust_remote_code=True)
        kwargs = _kwargs(
            str(MODEL), 4, "physical_ep4", args.max_model_len,
            args.kv_cache_gib, args.max_num_seqs,
        )
        kwargs["disable_log_stats"] = False
        llm = LLM(**kwargs)
        for local_wave_id, wave in enumerate(waves):
            wave_id = args.wave_offset + local_wave_id
            print(f"[part2 dp{rank}] wave={wave_id} prepare begin rows={len(wave[rank])}",
                  flush=True)
            if rank == 0:
                _profile_control(args, wave_id, wave)
            batch = wave[rank]
            prepared = [(row, *_prepare(processor, row)) for row in batch]
            requests = [copy.deepcopy(request) for _, request, _ in prepared]
            sampling = [
                _sampling(args.lane, args.max_tokens, args.seed, args.ignore_eos)
                for _ in requests
            ]
            print(f"[part2 dp{rank}] wave={wave_id} ep begin", flush=True)
            begin = time.perf_counter_ns()
            outputs = _ep_wave(llm, requests, sampling, barrier, wave_id)
            batch_wall_ms = (time.perf_counter_ns() - begin) / 1e6
            print(f"[part2 dp{rank}] wave={wave_id} ep returned", flush=True)
            records = [
                _record(
                    row, meta, output, batch_wall_ms, rank, wave_id, len(batch),
                    args.lane,
                )
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
                    timing["wall_ms"] = batch_wall_ms
                    timing["anchor_measured"] = bool(
                        record.get("anchor_measured", True)
                    )
                    append_jsonl_atomic(
                        args.output / f"timings.dp{rank}.jsonl", [timing]
                    )
            print(f"[part2 dp{rank}] wave={wave_id} checkpoint complete", flush=True)
        if args.profile_dir is not None:
            # vLLM worker processes are terminated by their engine parent, so
            # Python atexit handlers in those workers are not a reliable flush
            # boundary.  Trigger one explicitly unmeasured forward after the
            # timed ledger.  The control has instrument=False/flush=True: the
            # profiler resolves the already-recorded CUDA events before this
            # forward and records no event from the flush wave itself.
            flush_wave_id = args.wave_offset + len(waves)
            if rank == 0:
                _profile_control(args, flush_wave_id, waves[-1], flush=True)
            flush_batch = waves[-1][rank]
            prepared = [(row, *_prepare(processor, row)) for row in flush_batch]
            requests = [copy.deepcopy(request) for _, request, _ in prepared]
            sampling = [_sampling(args.lane, 1, args.seed, False) for _ in requests]
            _ep_wave(llm, requests, sampling, barrier, flush_wave_id)
        atomic_json(args.output / f"worker.dp{rank}.json", {"ok": True})
    except BaseException:
        atomic_json(args.output / f"worker.dp{rank}.json", {
            "ok": False, "traceback": traceback.format_exc()
        })
        raise


def main() -> None:
    _ensure_runtime_tools()
    verify_foundation_bundle()
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpus", default="4,5,6,7")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k-visual", type=int, choices=(8, 6, 4), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--single-active-dp",
        action="store_true",
        help="global batch one: DP0 carries one request while DP1 participates idle",
    )
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-model-len", type=int, default=40000)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--kv-cache-gib", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lane", choices=("official", "deterministic"), default="official")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--trace-save-ids", action="store_true")
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument(
        "--profile-mode", choices=("full", "ep", "both"), default="both"
    )
    parser.add_argument("--warmup-repeats", type=int, default=0)
    parser.add_argument("--measured-repeats", type=int, default=1)
    parser.add_argument(
        "--wave-offset", type=int, default=0,
        help="unique offset for repeat-preserving timing ledgers across restarts",
    )
    parser.add_argument(
        "--schedule",
        choices=("resolution_major", "manifest_order"),
        default="manifest_order",
        help="separate the three variants of one image across EP waves",
    )
    args = parser.parse_args()
    args.gpus = parse_physical_gpus(args.physical_gpus)
    if args.gpus != [4, 5, 6, 7]:
        raise ValueError(f"Part 2 TRUE EP4 requires physical GPUs 4,5,6,7: {args.gpus}")
    rows = load_manifest(args.manifest)["requests"]
    if args.schedule == "resolution_major":
        order = {"R0": 0, "R05": 1, "R025": 2}
        rows.sort(key=lambda row: (
            order[str(row["resolution_label"])], str(row["selection_key"])
        ))
    mismatched = [row["request_id"] for row in rows if int(row["k_visual"]) != args.k_visual]
    if mismatched:
        raise ValueError(f"manifest mixes k_visual; first mismatches: {mismatched[:3]}")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        done = completed_ids(args.output / "results.jsonl")
        done.update(completed_ids(args.output / "failures.jsonl"))
        for rank in range(2):
            done.update(completed_ids(args.output / f"results.dp{rank}.jsonl"))
        rows = [row for row in rows if str(row["request_id"]) not in done]
    pending_before_launch_limit = len(rows)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("nothing to run")
        return

    os.environ.update({
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, args.gpus)),
        "MLLM_REX_ENABLE": "1",
        "MLLM_REX_K_VISUAL": str(args.k_visual),
        "VLLM_NO_USAGE_STATS": "1",
    })
    # vLLM and the DP launcher create fresh Python grandchildren.  Put the
    # opt-in startup hook on their interpreter search path explicitly; cwd
    # alone is not preserved reliably by every spawn layer.
    package_root = Path(__file__).resolve().parents[1]
    repository_root = package_root.parent
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        path for path in (
            str(package_root / "hooks"),
            str(repository_root),
            inherited_pythonpath,
        )
        if path
    )
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MLLM_REX_TRACE_DIR"] = str(args.trace_dir.resolve())
        if args.trace_save_ids:
            os.environ["MLLM_REX_SAVE_IDS"] = "1"
    if args.profile_dir:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        os.environ.update({
            "FLASHVEP_MATRIX_CONTROL": str((args.profile_dir / "control.json").resolve()),
            "FLASHVEP_MATRIX_RAW_DIR": str((args.profile_dir / "raw_live").resolve()),
            "FLASHVEP_DEEPEP_PROOF_DIR": str((args.profile_dir / "backend_proof").resolve()),
        })
        os.environ[
            "MLLM_EP_FOUNDATION_PROFILE"
            if args.profile_mode == "both"
            else "MLLM_EP_FOUNDATION_PROFILE_FULL"
            if args.profile_mode == "full"
            else "MLLM_EP_FOUNDATION_PROFILE_EP"
        ] = "1"
    atomic_json(args.output / "invocation.json", {
        "physical_gpus": args.gpus,
        "physical_topology": {"tp": 2, "dp": 2, "sp": 1, "ep": 4, "pp": 1},
        "manifest": str(args.manifest.resolve()),
        "k_visual": args.k_visual,
        "k_text": 8,
        "k_decode": 8,
        "max_tokens": args.max_tokens,
        "lane": (
            "OFFICIAL_REPRODUCTION_PUBLIC_PROTOCOL_PARTIAL"
            if args.lane == "official"
            else "DETERMINISTIC_SYSTEM_CONTROL"
        ),
        "ignore_eos": args.ignore_eos,
        "seed": args.seed,
        "adapter": "OPT_IN_MINUS_ONE_RAGGED_VISUAL_K",
        "schedule": args.schedule,
        "wave_offset": args.wave_offset,
    })

    base_waves = []
    if args.single_active_dp:
        if args.batch_size != 1:
            raise ValueError("--single-active-dp requires --batch-size 1")
        base_waves = [[[row], []] for row in rows]
    else:
        stride = 2 * args.batch_size
        for offset in range(0, len(rows), stride):
            base_waves.append([
                rows[offset + rank * args.batch_size: offset + (rank + 1) * args.batch_size]
                for rank in range(2)
            ])
    waves = []
    for base in base_waves:
        for iteration in range(args.warmup_repeats + args.measured_repeats):
            waves.append([[
                {
                    **row,
                    "anchor_iteration": iteration - args.warmup_repeats,
                    "anchor_measured": iteration >= args.warmup_repeats,
                }
                for row in batch
            ] for batch in base])
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    port = _port()
    workers = [
        context.Process(target=_worker, args=(rank, port, args, barrier, waves))
        for rank in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if [worker.exitcode for worker in workers] != [0, 0]:
        raise RuntimeError([worker.exitcode for worker in workers])
    records = []
    for rank in range(2):
        path = args.output / f"results.dp{rank}.jsonl"
        if path.exists():
            records.extend(
                json.loads(line) for line in path.read_text().splitlines() if line.strip()
            )
    failure_path = args.output / "failures.jsonl"
    if failure_path.exists():
        records.extend(
            json.loads(line)
            for line in failure_path.read_text().splitlines()
            if line.strip()
        )
    # A resumed launch reads the complete per-DP ledgers.  Appending that
    # complete set to the combined ledger duplicates every prior checkpoint
    # on each restart.  Rebuild the convenience ledger atomically instead;
    # the per-DP ledgers remain the append-only recovery source of truth.
    by_request_id = {str(record["request_id"]): record for record in records}
    combined = args.output / "results.jsonl"
    combined_tmp = combined.with_suffix(".jsonl.tmp")
    combined_tmp.write_text(
        "".join(
            json.dumps(by_request_id[request_id], sort_keys=True) + "\n"
            for request_id in sorted(by_request_id)
        ),
        encoding="utf-8",
    )
    os.replace(combined_tmp, combined)
    atomic_json(args.output / "run.json", {
        "status": (
            "PARTIAL_CHECKPOINT"
            if len(rows) < pending_before_launch_limit
            else "COMPLETE"
        ),
        "records": len(by_request_id),
        "pending_before_launch_limit": pending_before_launch_limit,
        "launch_records": len(rows),
        "physical_topology": {"tp": 2, "dp": 2, "sp": 1, "ep": 4, "pp": 1},
        "physical_gpus": args.gpus,
        "k_visual": args.k_visual,
    })


if __name__ == "__main__":
    main()
