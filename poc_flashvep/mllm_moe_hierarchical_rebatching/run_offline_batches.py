#!/usr/bin/env python3
"""Run frozen offline cohorts on the validated Qwen3-VL TP2/DP2/EP4 path.

This is an execution-measurement harness, not a scheduler implementation.
Clean mode provides request/BCT evidence.  Observer mode provides diagnostic
same-device CUDA event and routing rows; its wall time is never used as clean
performance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import socket
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def render_request(llm, processor, row: dict, rid: str, max_images: int) -> tuple[dict, dict]:
    from PIL import Image

    started = time.monotonic()
    images = [Image.open(name).convert("RGB") for name in row["images"]]
    if len(images) > max_images:
        raise ValueError(f"{rid}: {len(images)} images exceeds {max_images}")
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": row["question"]})
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    item = {"prompt": prompt}
    if images:
        item["multi_modal_data"] = {"image": images[0] if len(images) == 1 else images}
        item["multi_modal_uuids"] = {"image": [f"{rid}:image{i}" for i in range(len(images))]}
    rendered = llm._preprocess_cmpl_one(item)
    rendered_at = time.monotonic()
    record = {
        "request_id": rid,
        "source_request_id": row["request_id"],
        "family": row["family"],
        "image_count": len(images),
        "total_pixels": row.get("total_pixels", 0),
        "question_words": row.get("question_words", len(row["question"].split())),
        "start_s": started,
        "rendered_s": rendered_at,
        "processor_s": rendered_at - started,
        "token_times": [],
        "frontend_token_times": [],
    }
    return rendered, record


def worker(rank: int, args, barrier) -> None:
    os.environ.update(
        VLLM_DP_RANK=str(rank), VLLM_DP_RANK_LOCAL=str(rank), VLLM_DP_SIZE="2",
        VLLM_DP_MASTER_IP="127.0.0.1", VLLM_DP_MASTER_PORT=str(args.port),
    )
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    pool = {row["request_id"]: row for row in read_jsonl(args.pool)}
    plan = json.loads(args.plan.read_text())
    llm = LLM(
        model=args.model, dtype="bfloat16", tensor_parallel_size=2,
        enable_expert_parallel=True, expert_placement_strategy="linear",
        all2all_backend="deepep_high_throughput", enable_dbo=False, enable_eplb=False,
        enable_ep_weight_filter=True, enable_return_routed_experts=False,
        kv_cache_memory_bytes=2 * 1024**3, max_model_len=8192,
        max_num_batched_tokens=8192, max_num_seqs=64, skip_mm_profiling=True,
        enable_prefix_caching=False, moe_backend="triton", enforce_eager=True,
        enable_flashinfer_autotune=False, mm_processor_kwargs={"max_pixels": args.max_pixels},
        limit_mm_per_prompt={"image": 4}, disable_log_stats=False,
        mm_processor_cache_gb=0,
    )
    engine = llm.llm_engine
    pc = engine.vllm_config.parallel_config
    assert (pc.tensor_parallel_size, pc.data_parallel_size, pc.enable_expert_parallel) == (2, 2, True)
    assert pc.all2all_backend == "deepep_high_throughput"
    config = {
        "tp": 2, "dp": 2, "ep": 4, "bf16": True,
        "backend": pc.all2all_backend, "dbo": pc.enable_dbo, "eplb": pc.enable_eplb,
        "visible_gpus": os.environ["CUDA_VISIBLE_DEVICES"], "observer": args.instrument,
        "scope": "offline closed-set cohorts; BCT is all-request completion time",
    }
    (args.out / f"runtime_dp{rank}.json").write_text(json.dumps(config, indent=2) + "\n")

    def run_cohort(rows: list[dict], label: str, warmup: bool) -> None:
        params = SamplingParams(
            temperature=0, max_tokens=args.output_tokens, ignore_eos=args.ignore_eos,
            output_kind=RequestOutputKind.CUMULATIVE,
        )
        barrier.wait(600)
        wave_submit = time.monotonic()
        records: dict[str, dict] = {}
        rendered_requests = []
        for position, row in enumerate(rows):
            rid = f"{args.run_id}:{label}:dp{rank}:p{position}:{row['request_id']}"
            rendered, record = render_request(llm, processor, row, rid, 4)
            record.update(label=label, dp_rank=rank, warmup=warmup,
                          output_budget=args.output_tokens, ignore_eos=args.ignore_eos,
                          instrumented=args.instrument, wave_submit_s=wave_submit)
            records[rid] = record
            rendered_requests.append((rid, rendered))
        # Closed-set semantics: every request in both DP shards is rendered
        # before any GPU submission.  This prevents request order from turning
        # CPU preprocessing overlap into an apparent batching-policy effect.
        barrier.wait(600)
        gpu_batch_start = time.monotonic()
        for rid, rendered in rendered_requests:
            submitted = time.monotonic()
            internal = engine.add_request(rid, rendered, params)
            records[rid].update(internal_request_id=internal, engine_submit_s=submitted,
                                gpu_batch_start_s=gpu_batch_start)
        while engine.get_num_unfinished_requests():
            outputs = engine.step()
            received = time.monotonic()
            for output in outputs:
                if output.request_id not in records or not output.outputs:
                    continue
                record = records[output.request_id]
                metrics = output.metrics
                assert metrics is not None and metrics.last_token_ts > 0
                tokens = list(output.outputs[0].token_ids)
                previous = len(record["token_times"])
                new = len(tokens) - previous
                assert new >= 0
                record["token_times"].extend([float(metrics.last_token_ts)] * new)
                record["frontend_token_times"].extend([received] * new)
                if previous == 0 and tokens:
                    record["token_times"][0] = float(metrics.first_token_ts)
                if output.finished:
                    times = record["token_times"]
                    record.update(
                        output_ids=tokens, output_text=output.outputs[0].text,
                        prompt_tokens=len(output.prompt_token_ids or []),
                        complete_s=float(metrics.last_token_ts),
                        e2e_s=float(metrics.last_token_ts) - record["start_s"],
                        engine_e2e_s=float(metrics.last_token_ts) - record["engine_submit_s"],
                        ttft_s=times[0] - record["start_s"],
                        tpot_s=0.0 if len(times) <= 1 else (times[-1] - times[0]) / (len(times) - 1),
                        frontend_e2e_s=received - record["start_s"],
                    )
        barrier.wait(600)
        wave_local_done = time.monotonic()
        assert all("complete_s" in row for row in records.values())
        with (args.out / f"requests_dp{rank}.jsonl").open("a") as sink:
            for record in records.values():
                record["wave_local_done_s"] = wave_local_done
                sink.write(json.dumps(record) + "\n")
        with (args.out / f"waves_dp{rank}.jsonl").open("a") as sink:
            sink.write(json.dumps({
                "label": label, "dp_rank": rank, "warmup": warmup,
                "wave_submit_s": wave_submit, "wave_done_s": wave_local_done,
                "gpu_batch_start_s": gpu_batch_start,
                "requests": len(rows), "request_ids": [row["request_id"] for row in rows],
            }) + "\n")
        barrier.wait(600)

    # Identical warmups for every restart. They are excluded from evidence.
    warm = list(pool.values())[:4]
    for warmup in range(2):
        run_cohort(warm[rank::2], f"warmup{warmup}", True)

    entries = list(plan["waves"])
    # Warm the actual large execution shape separately from the small generic
    # warmup.  These rows are labelled warmup and excluded by the reducer.
    for warmup in range(args.full_plan_warmups):
        entry = entries[0]
        if "dp_request_ids" in entry:
            local_ids = entry["dp_request_ids"][str(rank)]
            local_rows = [pool[request_id] for request_id in local_ids]
        else:
            global_rows = [pool[request_id] for request_id in entry["request_ids"]]
            local_rows = global_rows[rank::2]
        run_cohort(local_rows, f"full_plan_warmup{warmup}", True)
    random.Random(args.seed).shuffle(entries)
    for entry in entries:
        if "dp_request_ids" in entry:
            local_ids = entry["dp_request_ids"][str(rank)]
            local_rows = [pool[request_id] for request_id in local_ids]
        else:
            global_rows = [pool[request_id] for request_id in entry["request_ids"]]
            local_rows = global_rows[rank::2]
        run_cohort(local_rows, entry["label"], False)
    barrier.wait(600)
    engine.engine_core.shutdown()


def main() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4,5,6,7", os.environ.get("CUDA_VISIBLE_DEVICES")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--instrument", action="store_true")
    parser.add_argument("--observer-step-every", type=int, default=1)
    parser.add_argument("--observer-moe-every", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--max-pixels", type=int, default=512 * 512)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--full-plan-warmups", type=int, default=0)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT / "scheduling_overlap_successor_mining/FASTPP"))
    from run_configuration import assert_gpus_empty
    assert_gpus_empty()

    hook = ROOT / "scheduling_overlap_successor_mining/COMMON/ep_trace"
    os.environ.update(
        OMP_NUM_THREADS="4", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        TORCH_CUDA_ARCH_LIST="9.0", VLLM_WORKER_MULTIPROC_METHOD="spawn",
        SCHEDULING_EP_OBSERVE="1" if args.instrument else "0",
        SCHEDULING_EP_OBSERVER_MODE="full",
        SCHEDULING_EP_STEP_EVERY=str(args.observer_step_every),
        SCHEDULING_EP_MOE_CYCLE_EVERY=str(args.observer_moe_every),
        SCHEDULING_EP_TRACE_OUT=str(args.out / "operations"),
        PYTHONPATH=str(hook),
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        args.port = sock.getsockname()[1]
    sources = [Path(__file__), args.pool, args.plan, hook / "observer.py", hook / "sitecustomize.py"]
    manifest = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "start_unix": time.time(),
        "source_hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "evidence_contract": "clean wall-time and observer-heavy diagnostic runs are separate",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    processes = [context.Process(target=worker, args=(rank, args, barrier)) for rank in range(2)]
    for process in processes:
        process.start()
    import psutil
    owned = {}
    try:
        deadline = time.monotonic() + 5400
        while any(process.is_alive() for process in processes):
            for child in psutil.Process().children(recursive=True):
                owned[(child.pid, child.create_time())] = child
            if any(process.exitcode not in (None, 0) for process in processes):
                raise RuntimeError("An owned DP worker failed")
            if time.monotonic() > deadline:
                raise TimeoutError("Bounded offline-batch run exceeded 90 minutes")
            for process in processes:
                process.join(timeout=1)
    finally:
        for child in owned.values():
            try:
                if child.is_running():
                    child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, remaining = psutil.wait_procs(list(owned.values()), timeout=10)
        for child in remaining:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(10)
        manifest.update(end_unix=time.time(), exit_codes=[process.exitcode for process in processes])
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    assert all(process.exitcode == 0 for process in processes)


if __name__ == "__main__":
    main()
