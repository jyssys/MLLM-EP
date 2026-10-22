"""One independent CUDA context. No distributed model execution."""

import argparse
import gc
import json
import os
import queue
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, OLD, MODEL, UUIDS, config, guard, write


def percentile(values, p):
    return float(np.percentile(values, p)) if values else None


def identity():
    gpu = guard()
    import subprocess

    actual = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    assert actual == UUIDS[gpu], (gpu, actual)
    assert torch.cuda.device_count() == 1
    return dict(
        physical_GPU=gpu,
        UUID=actual,
        CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],
        pid=os.getpid(),
        owner_uid=os.getuid(),
        CUDA_logical_device=0,
        CUDA_context_scope="Independent Single-GPU, no TP/PP/DP cluster benchmark",
        CPU_affinity=sorted(os.sched_getaffinity(0)),
    )


def load_model():
    from vllm_rlt.models import OuroForCausalLM

    torch.set_num_threads(1)
    return OuroForCausalLM.from_pretrained(MODEL, device="cuda:0", dtype=torch.bfloat16)


def new_engine(model, mode, chunk, width=8, budget=1024):
    from vllm_rlt.engine.stage_engine import StageEngine

    return StageEngine(model, **config(mode, chunk, width, token_budget=budget))


def close(engine):
    engine.close()
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def memory(engine):
    return dict(
        allocated=torch.cuda.memory_allocated(),
        reserved=torch.cuda.memory_reserved(),
        peak=torch.cuda.max_memory_allocated(),
        KV_pool_bytes=2
        * engine.cache_manager.key_cache.numel()
        * engine.cache_manager.key_cache.element_size(),
        staged_bytes=0
        if engine.model_runner.prefill_states is None
        else engine.model_runner.prefill_states.numel()
        * engine.model_runner.prefill_states.element_size(),
        graph_counts={
            k: getattr(engine.model_runner.graphs, k, 0)
            for k in ("captures", "replays", "fallbacks")
        }
        if engine.model_runner.graphs
        else {},
        graph_entries=len(engine.model_runner.graphs.entries)
        if engine.model_runner.graphs
        else 0,
    )


@torch.inference_mode()
def serve(engine, cases, cap, schedule, duration, drain=120, trace=False, observe=True):
    """Unbounded arrival queue. Producer never waits for model capacity."""
    from vllm_rlt import SamplingParams
    from poc_flashvep.ouro_chunk_loop_prefill.bench import Observe

    observer = Observe(engine, trace, observe)
    records = {}
    for i, (case, arrival) in enumerate(zip(cases, schedule)):
        records[str(i)] = dict(
            id=str(i),
            family=case["id"],
            input_tokens=len(case["input_ids"]),
            scheduled_offset_s=arrival,
            output_ids=[],
            token_times=[],
            depths=[],
            status="OFFERED",
            cap=cap,
        )
    inbox = queue.Queue()
    start = time.perf_counter() + 0.050
    cancelled = threading.Event()

    def arrive():
        for i, offset in enumerate(schedule):
            delay = start + offset - time.perf_counter()
            if delay > 0 and cancelled.wait(delay):
                return
            if cancelled.is_set():
                return
            inbox.put((str(i), time.perf_counter()))

    producer = threading.Thread(
        target=arrive, name="independent-open-loop-producer", daemon=True
    )
    producer.start()
    end_deadline = start + duration + drain
    queue_series = []
    first_submissions = {}
    last_tick = start
    params = SamplingParams(
        max_tokens=cap, max_loops=4, exit_threshold=1.0, temperature=0.0
    )
    while time.perf_counter() < end_deadline:
        now = time.perf_counter()
        while not inbox.empty():
            rid, sent = inbox.get_nowait()
            record = records[rid]
            record["actual_send"] = sent
            record["engine_enqueue"] = time.perf_counter()
            try:
                engine.add_request(rid, cases[int(rid)]["input_ids"], params)
                record["status"] = "ADMITTED_TO_ENGINE_QUEUE"
            except Exception as error:
                record.update(
                    status="ERROR", error=repr(error), completion=time.perf_counter()
                )
        if now - last_tick >= 0.050:
            outstanding = sum(
                r["scheduled_offset_s"] <= now - start
                and r["status"] not in ("COMPLETE", "ERROR", "TIMEOUT")
                for r in records.values()
            )
            queue_series.append(
                dict(
                    time=now - start,
                    outstanding=outstanding,
                    engine_requests=len(engine.scheduler.requests),
                    producer_queue=inbox.qsize(),
                    free_KV_blocks=engine.cache_manager.num_free_blocks
                    if hasattr(engine.cache_manager, "num_free_blocks")
                    else None,
                )
            )
            last_tick = now
        if engine.has_unfinished_requests():
            for out in engine.step():
                now = time.perf_counter()
                r = records[out.request_id]
                ids = list(out.token_ids)
                if len(ids) > len(r["output_ids"]):
                    assert len(ids) == len(r["output_ids"]) + 1, (
                        "No synthetic token timestamps"
                    )
                    r["token_times"].append(now)
                    r["output_ids"] = ids
                    r["depths"] = list(out.exit_depths)
                if out.finished:
                    r.update(
                        status="COMPLETE",
                        completion=now,
                        finish_reason=out.finish_reason,
                    )
                elif ids and ids[-1] in getattr(engine, "frontend_stop_token_ids", ()):
                    engine.abort_request(out.request_id)
                    r.update(
                        status="COMPLETE", completion=now, finish_reason="frontend_stop"
                    )
        elif (
            not producer.is_alive()
            and inbox.empty()
            and all(r["status"] in ("COMPLETE", "ERROR") for r in records.values())
        ):
            break
        else:
            time.sleep(0.0005)
    cancelled.set()
    producer.join(timeout=2)
    for rid, r in records.items():
        if r["status"] not in ("COMPLETE", "ERROR"):
            if rid in engine.scheduler.requests:
                engine.abort_request(rid)
            r.update(
                status="TIMEOUT",
                completion=time.perf_counter(),
                finish_reason="timeout",
            )
    engine.model_runner.synchronize()
    gpu = observer.finish()
    for q in observer.records:
        for span in q["spans"]:
            first_submissions.setdefault(span["id"], q["host_begin"])
    for r in records.values():
        r["scheduled_arrival"] = start + r["scheduled_offset_s"]
        r["first_GPU_submission"] = first_submissions.get(r["id"])
        if r["token_times"]:
            r["TTFT_s"] = r["token_times"][0] - r["scheduled_arrival"]
            r["ITLs_s"] = [
                b - a for a, b in zip(r["token_times"], r["token_times"][1:])
            ]
            r["TPOT_s"] = (
                (r["token_times"][-1] - r["token_times"][0])
                / (len(r["token_times"]) - 1)
                if len(r["token_times"]) > 1
                else None
            )
        r["E2E_s"] = r["completion"] - r["scheduled_arrival"]
    engine.executor.submit = observer.original_submit
    engine.model_runner.runtime.forward = observer.original_forward
    return dict(
        requests=list(records.values()),
        start=start,
        end=time.perf_counter(),
        arrival_window_s=duration,
        queue_series=queue_series,
        quantums=observer.records,
        gpu_quantums=gpu,
        clock=observer.clock,
        memory=memory(engine),
        trace=trace,
        observe=observe,
        evidence="LIVE_MEASURED_IN_PROCESS_OPEN_LOOP_NO_HTTP",
    )


def old_cases(trial, length, width):
    source = json.loads((OLD / "workload_manifest.json").read_text())["cases"]
    long = next(
        r for r in source if r["trial"] == trial and r.get("prompt_bin") == length
    )
    return [
        dict(id=f"cal_{trial}_{length}_{i}", input_ids=long["input_ids"])
        for i in range(width)
    ]


def calibration(model, job, target):
    results = []
    for length, width in job.get(
        "shapes", [(2048, 1), (2048, 4), (2048, 8), (8192, 1), (8192, 4), (8192, 8)]
    ):
        e = new_engine(model, "atomic", 1024, width)
        cases = old_cases(0, length, width)
        warm = serve(e, cases, 48, [0] * width, 0.001, trace=False, observe=False)
        write(target / f"warm_P{length}_N{width}.json", warm)
        for repetition in range(job.get("repetitions", 2)):
            data = serve(e, cases, 96, [0] * width, 0.001, trace=False, observe=True)
            ready = max(r["token_times"][0] for r in data["requests"])
            itls = [
                b - a
                for r in data["requests"]
                for a, b in zip(r["token_times"], r["token_times"][1:])
                if a >= ready
            ]
            data.update(
                prompt_bin=length,
                width=width,
                repetition=repetition,
                decode_mean_s=float(np.mean(itls)),
                decode_p50_s=percentile(itls, 50),
                decode_p95_s=percentile(itls, 95),
                decode_p99_s=percentile(itls, 99),
            )
            write(target / f"decode_P{length}_N{width}_r{repetition}.json", data)
            results.append(
                {
                    k: data[k]
                    for k in (
                        "prompt_bin",
                        "width",
                        "repetition",
                        "decode_mean_s",
                        "decode_p50_s",
                        "decode_p95_s",
                        "decode_p99_s",
                    )
                }
            )
        close(e)
        del e
        gc.collect()
        torch.cuda.empty_cache()
    for length in (2048, 8192):
        e = new_engine(model, "atomic", 1024, 1)
        cases = old_cases(0, length, 1)
        serve(e, cases, 1, [0], 0.001, observe=False)
        for repetition in range(3):
            data = serve(e, cases, 1, [0], 0.001, observe=True)
            data.update(prompt_bin=length, repetition=repetition)
            write(target / f"prefill_P{length}_r{repetition}.json", data)
        close(e)
        del e
        gc.collect()
        torch.cuda.empty_cache()
    return dict(calibration=results)


def grid(model, job, target):
    from poc_flashvep.ouro_chunk_loop_prefill.bench import run

    modes = list(job["cells"])
    for mode, chunk in modes:
        length, width = job["shape"]
        marker = target / f"{mode}_C{chunk}_complete.json"
        if marker.exists():
            continue
        e = new_engine(model, mode, chunk, width)
        run(model, mode, chunk, length, width, 0, False, True, False, e)
        for trial in job["trials"]:
            file = target / f"{mode}_C{chunk}_P{length}_N{width}_T{trial}.json"
            if file.exists():
                continue
            torch.cuda.reset_peak_memory_stats()
            data = run(
                model,
                mode,
                chunk,
                length,
                width,
                trial,
                job.get("trace", False),
                True,
                not job.get("observer_off", False),
                e,
            )
            data.update(
                GPU=guard(),
                job_id=job["id"],
                startup=job.get("startup"),
                parallelism="Independent Single-GPU worker",
                new_reserve_blocks=18432,
            )
            write(file, data)
            print(
                "GRID",
                job["id"],
                mode,
                chunk,
                trial,
                round(data["measured_cohort_s"], 3),
                flush=True,
            )
        write(marker, dict(status="COMPLETE"))
        close(e)
        del e
        gc.collect()
        torch.cuda.empty_cache()
    return dict(status="COMPLETE_GRID")


def observer_control(model, job, target):
    from poc_flashvep.ouro_chunk_loop_prefill.bench import run

    engine = new_engine(model, "atomic", 1024, 4)
    run(model, "atomic", 1024, 2048, 4, 0, False, True, False, engine)
    for i, trial in enumerate((2, 6, 9)):
        order = (True, False) if (i + guard()) % 2 else (False, True)
        for enabled in order:
            data = run(
                model, "atomic", 1024, 2048, 4, trial, False, True, enabled, engine
            )
            data.update(GPU=guard(), job_id=job["id"], observer_enabled=enabled)
            write(target / f"observer_{enabled}_T{trial}.json", data)
    close(engine)
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return dict(status="COMPLETE_PAIRED_OBSERVER_CONTROL")


def system_profile(model, job, target):
    from poc_flashvep.ouro_chunk_loop_prefill.bench import run

    mode, chunk = job["cell"]
    engine = new_engine(model, mode, chunk, 8)
    run(model, mode, chunk, 8192, 8, 0, False, True, False, engine)
    engine.model_runner.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    try:
        data = run(model, mode, chunk, 8192, 8, 3, True, True, True, engine)
        engine.model_runner.synchronize()
    finally:
        torch.cuda.cudart().cudaProfilerStop()
    data.update(GPU=guard(), job_id=job["id"], scope="HEAVY_NSYS_NOT_HEADLINE_E2E")
    write(target / "profile_case.json", data)
    close(engine)
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return dict(status="COMPLETE_NSYS_TARGET")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    a = parser.parse_args()
    job = json.loads(Path(a.job).read_text())
    target = ROOT / "raw" / job["id"]
    target.mkdir(parents=True, exist_ok=True)
    if (target / "completed.json").exists():
        print("SKIP_COMPLETED", job["id"], flush=True)
        return
    who = identity()
    write(target / "identity.json", dict(**who, job=job, start=time.time()))
    try:
        model = load_model()
        from collections import Counter

        tensors = list(model.parameters()) + list(model.buffers())
        storages = {
            (
                str(t.device),
                t.untyped_storage().data_ptr(),
            ): t.untyped_storage().nbytes()
            for t in tensors
        }
        write(
            target / "model_tensor_accounting.json",
            dict(
                physical_GPU=guard(),
                parameter_numel=sum(t.numel() for t in model.parameters()),
                unique_parameter_and_buffer_storage_bytes=sum(storages.values()),
                dtype_numel=dict(
                    Counter(
                        {
                            str(dtype): sum(
                                t.numel() for t in tensors if t.dtype == dtype
                            )
                            for dtype in {t.dtype for t in tensors}
                        }
                    )
                ),
                allocated_before_KV_engine=torch.cuda.memory_allocated(),
                reserved_before_KV_engine=torch.cuda.memory_reserved(),
                scope="Actual loaded parameter/buffer storage. Not HBM traffic, KV reserve, or graph workspace.",
            ),
        )
        del tensors, storages
        if job["kind"] == "calibration":
            summary = calibration(model, job, target)
        elif job["kind"] == "grid":
            summary = grid(model, job, target)
        elif job["kind"] == "traffic":
            from .traffic import main_job

            summary = main_job(model, job, target)
        elif job["kind"] == "numerical":
            from .numerical import main_job

            summary = main_job(model, job, target)
        elif job["kind"] == "geometry":
            from .numerical import geometry_job

            summary = geometry_job(model, job, target)
        elif job["kind"] == "observer":
            summary = observer_control(model, job, target)
        elif job["kind"] == "nsys":
            summary = system_profile(model, job, target)
        else:
            raise ValueError("Unknown job kind: " + job["kind"])
        write(target / "completed.json", dict(**who, **summary, end=time.time()))
    except Exception as error:
        write(
            target / "failure.json",
            dict(
                **who,
                error=repr(error),
                traceback=traceback.format_exc(),
                time=time.time(),
            ),
        )
        raise


if __name__ == "__main__":
    main()
