"""Clean real-image EP4 transfer trace, not a port of any screened scheduler.

Uses the validated renderer and two-DP-process topology only. No SERE/MoDES or
previous experimental policy module is imported. Instrumented request times are
diagnostic; use the no-observer restart for clean serving latency.
"""
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

ROOT = Path(__file__).resolve().parent.parent


def warmup_pool(rows):
    preferred = [r for r in rows if r["family"] == "single_real_image"]
    if len(preferred) >= 4:
        return preferred
    # Preserve the legacy warmup when available. New trace families select a
    # deterministic real-image pool, identically for clean/observer restarts.
    for family in dict.fromkeys(r["family"] for r in rows if r["images"]):
        pool = [r for r in rows if r["family"] == family]
        if len(pool) >= 4:
            return pool
    if len(rows) >= 4:
        return rows
    raise ValueError("Need four warmup requests for two DP workers")


def worker(rank, args, barrier):
    os.environ.update(VLLM_DP_RANK=str(rank), VLLM_DP_RANK_LOCAL=str(rank), VLLM_DP_SIZE="2",
                      VLLM_DP_MASTER_IP="127.0.0.1", VLLM_DP_MASTER_PORT=str(args.port))
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    rows = [json.loads(s) for s in args.trace.read_text().splitlines()]
    llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=2,
              enable_expert_parallel=True, expert_placement_strategy="linear",
              all2all_backend="deepep_high_throughput", enable_dbo=False, enable_eplb=False,
              enable_ep_weight_filter=True, enable_return_routed_experts=False,
              kv_cache_memory_bytes=2 * 1024**3, max_model_len=8192,
              max_num_batched_tokens=8192, max_num_seqs=32, skip_mm_profiling=True,
              enable_prefix_caching=False, moe_backend="triton", enforce_eager=True,
              enable_flashinfer_autotune=False, mm_processor_kwargs={"max_pixels": args.max_pixels},
              limit_mm_per_prompt={"image": 4}, disable_log_stats=False)
    engine = llm.llm_engine
    pc = engine.vllm_config.parallel_config
    assert (pc.tensor_parallel_size, pc.data_parallel_size, pc.enable_expert_parallel) == (2, 2, True)
    (args.out / f"config_dp{rank}.json").write_text(json.dumps({
        "tp": 2, "dp": 2, "ep": 4, "bf16": True, "backend": pc.all2all_backend,
        "dbo": pc.enable_dbo, "eplb": pc.enable_eplb, "model_config": processor.tokenizer.name_or_path,
        "visible_gpus": os.environ["CUDA_VISIBLE_DEVICES"], "observer": args.instrument,
        "runtime_scope": "clean vLLM operation transfer trace, not Layered/FastPP/NanoFlow execution",
    }, indent=2))
    seen_hashes = set()

    def cohort(selected, label, warmup):
        barrier.wait(600)
        records = {}
        params = SamplingParams(temperature=0, max_tokens=args.output_tokens, ignore_eos=True,
                                output_kind=RequestOutputKind.CUMULATIVE)
        for row in selected:
            rid = f"{args.out.name}:{label}:dp{rank}:{row['request_id']}"
            started = time.monotonic()
            images = [Image.open(name).convert("RGB") for name in row["images"]]
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": row["question"]})
            prompt = processor.apply_chat_template([{"role": "user", "content": content}],
                                                    tokenize=False, add_generation_prompt=True)
            item = {"prompt": prompt}
            if images:
                item["multi_modal_data"] = {"image": images[0] if len(images) == 1 else images}
                item["multi_modal_uuids"] = {"image": [f"{rid}:image{i}" for i in range(len(images))]}
            rendered = llm._preprocess_cmpl_one(item)
            mm_hashes = [h for hs in rendered.get("mm_hashes", {}).values() for h in hs]
            if images:
                assert mm_hashes and not seen_hashes.intersection(mm_hashes)
                seen_hashes.update(mm_hashes)
            submitted = time.monotonic()
            internal = engine.add_request(rid, rendered, params)
            records[rid] = {"request_id": rid, "internal_request_id": internal,
                            "family": row["family"], "image_count": len(images),
                            "source_request_id": row["source_request_id"], "label": label,
                            "dp_rank": rank, "warmup": warmup, "start_s": started,
                            "engine_submit_s": submitted, "processor_s": submitted-started,
                            "mm_hashes": mm_hashes, "token_times": [], "frontend_token_times": [],
                            "instrumented": args.instrument, "coalesced_outputs": 0}
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
                ready = float(metrics.last_token_ts)
                assert record["start_s"] <= ready <= received
                record["token_times"].extend([ready] * new)
                record["frontend_token_times"].extend([received] * new)
                if not previous and tokens:
                    record["token_times"][0] = float(metrics.first_token_ts)
                record["coalesced_outputs"] += int(new > 1)
                if output.finished:
                    times = record["token_times"]
                    record.update(output_ids=tokens, output_text=output.outputs[0].text,
                                  prompt_tokens=len(output.prompt_token_ids or []),
                                  complete_s=ready, e2e_s=ready-record["start_s"],
                                  engine_e2e_s=ready-record["engine_submit_s"],
                                  ttft_s=times[0]-record["start_s"],
                                  tpot_s=(times[-1]-times[0])/(len(times)-1),
                                  frontend_e2e_s=received-record["start_s"],
                                  frontend_poll_delay_s=received-ready)
        assert all("complete_s" in row for row in records.values())
        with (args.out / f"requests_dp{rank}.jsonl").open("a") as sink:
            for row in records.values():
                sink.write(json.dumps(row) + "\n")
        print(json.dumps({"dp": rank, "label": label, "requests": len(records),
                          "mean_e2e_s": sum(r["e2e_s"] for r in records.values())/len(records)}), flush=True)
        barrier.wait(600)

    barrier.wait(600)
    single = warmup_pool(rows)
    for warmup in range(2):
        cohort(single[rank*2:(rank+1)*2], f"warmup{warmup}", True)
    families = sorted({r["family"] for r in rows})
    for repeat in range(args.repetitions):
        order = [(family, b) for family in families for b in args.batch_per_dp]
        random.Random(args.seed + repeat).shuffle(order)
        for family, b in order:
            pool = [r for r in rows if r["family"] == family]
            cohort(pool[rank*b:(rank+1)*b], f"r{repeat}:{family}:b{b}", False)
    barrier.wait(600)
    engine.engine_core.shutdown()


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4,5,6,7"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instrument", action="store_true")
    parser.add_argument("--observer-mode", choices=["full", "layer_sampled"], default="full")
    parser.add_argument("--observer-step-every", type=int, default=4)
    parser.add_argument("--observer-moe-every", type=int, default=4)
    parser.add_argument("--batch-per-dp", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=512*512)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    assert all(1 <= b <= 16 for b in args.batch_per_dp) and args.output_tokens >= 2
    assert args.observer_step_every >= 1 and args.observer_moe_every >= 1
    sys.path.insert(0, str(ROOT / "FASTPP"))
    from run_configuration import assert_gpus_empty
    assert_gpus_empty()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    hook = ROOT / "COMMON/ep_trace"
    os.environ.update(OMP_NUM_THREADS="4", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TORCH_CUDA_ARCH_LIST="9.0", VLLM_WORKER_MULTIPROC_METHOD="spawn",
                      SCHEDULING_EP_OBSERVE="1" if args.instrument else "0",
                      SCHEDULING_EP_OBSERVER_MODE=args.observer_mode,
                      SCHEDULING_EP_STEP_EVERY=str(args.observer_step_every),
                      SCHEDULING_EP_MOE_CYCLE_EVERY=str(args.observer_moe_every),
                      SCHEDULING_EP_TRACE_OUT=str(args.out / "operations"),
                      PYTHONPATH=str(hook))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        args.port = sock.getsockname()[1]
    manifest = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "start_unix": time.time(), "scope": "request-cohort operation transfer diagnostic",
                "source_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__), hook/"observer.py",
                                            hook/"observer_lite.py", hook/"sitecustomize.py", args.trace]}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    processes = [context.Process(target=worker, args=(rank, args, barrier)) for rank in range(2)]
    for process in processes:
        process.start()
    import psutil
    owned = {}
    try:
        deadline = time.monotonic() + 3600
        while any(p.is_alive() for p in processes):
            for child in psutil.Process().children(recursive=True):
                owned[(child.pid, child.create_time())] = child
            if any(p.exitcode not in (None, 0) for p in processes):
                raise RuntimeError("An owned DP worker failed")
            if time.monotonic() > deadline:
                raise TimeoutError("Bounded transfer trace timeout")
            for process in processes:
                process.join(timeout=1)
    finally:
        # Retain identities discovered while parents were alive, including an
        # engine core that becomes orphaned after its DP frontend fails.
        for child in owned.values():
            try:
                if child.is_running():
                    child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, remaining = psutil.wait_procs(list(owned.values()), timeout=8)
        for child in remaining:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(10)
        manifest.update(end_unix=time.time(), exit_codes=[p.exitcode for p in processes])
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    assert all(p.exitcode == 0 for p in processes)


if __name__ == "__main__":
    main()
