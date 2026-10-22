"""Resume-aware GPU work queue. Each child sees one allowed physical GPU."""

import argparse
import concurrent.futures
import json
import os
import queue
import subprocess
import threading
import time
import csv
from .common import ROOT, REPO, FORK, PYTHON, write


def launch(job, gpu):
    target = ROOT / "raw" / job["id"]
    if (target / "completed.json").exists():
        return dict(id=job["id"], GPU=gpu, status="SKIPPED_COMPLETE")
    if (target / "failure.json").exists():
        return dict(id=job["id"], GPU=gpu, status="PRESERVED_FAILURE_MANUAL_REVIEW")
    jobs = ROOT / "jobs"
    jobs.mkdir(exist_ok=True, parents=True)
    path = jobs / (job["id"] + ".json")
    if path.exists():
        assert json.loads(path.read_text()) == job, "Changed job manifest"
    else:
        write(path, job)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONPATH=str(FORK),
        OURO_PREFILL_BACKEND="flash_attn_4",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        CUDA_CACHE_PATH=f"/home/esjung/.cache/ouro-slo-worker-{gpu}/cuda",
        TRITON_CACHE_DIR=f"/home/esjung/.cache/ouro-slo-worker-{gpu}/triton",
        FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED="1",
        FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=f"/home/esjung/.cache/ouro-slo-worker-{gpu}/fa4",
    )
    (ROOT / "logs").mkdir(exist_ok=True)
    log = ROOT / "logs" / (job["id"] + f"_gpu{gpu}.log")
    with log.open("a") as stream:
        started = time.time()
        command = [
            PYTHON,
            "-u",
            "-m",
            "poc_flashvep.ouro_chunk_loop_slo.worker",
            "--job",
            str(path),
        ]
        if job["kind"] == "nsys":
            capture = ROOT / "private_profiles" / job["id"]
            capture.parent.mkdir(parents=True, exist_ok=True)
            command = [
                "nsys",
                "profile",
                "--sample=none",
                "--cpuctxsw=none",
                "--trace=cuda,nvtx,osrt",
                "--cuda-graph-trace=node",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "--output",
                str(capture),
            ] + command
        child = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        print("START", job["id"], "GPU", gpu, "PID", child.pid, flush=True)
        write(
            ROOT / "worker_state" / f"gpu{gpu}.json",
            dict(job=job["id"], pid=child.pid, status="RUNNING", started=started),
        )

        def monitor():
            target.mkdir(parents=True, exist_ok=True)
            with (target / "telemetry.csv").open("w") as output:
                writer = csv.writer(output)
                writer.writerow(
                    [
                        "wall_time",
                        "physical_GPU",
                        "uuid",
                        "utilization_gpu_percent",
                        "memory_used_MiB",
                        "power_W",
                        "SM_clock_MHz",
                        "HBM_clock_MHz",
                        "temperature_C",
                    ]
                )
                while child.poll() is None:
                    try:
                        values = (
                            subprocess.check_output(
                                [
                                    "nvidia-smi",
                                    "-i",
                                    str(gpu),
                                    "--query-gpu=uuid,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu",
                                    "--format=csv,noheader,nounits",
                                ],
                                text=True,
                                timeout=4,
                            )
                            .strip()
                            .split(", ")
                        )
                        writer.writerow([time.time(), gpu, *values])
                        output.flush()
                    except (subprocess.SubprocessError, OSError) as error:
                        writer.writerow(
                            [time.time(), gpu, "MONITOR_ERROR", repr(error)]
                        )
                    time.sleep(1)

        telemetry = threading.Thread(target=monitor, daemon=True)
        telemetry.start()
        code = child.wait()
        telemetry.join(timeout=5)
        if job["kind"] == "nsys" and code == 0:
            reports = "cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_trace,nvtx_pushpop_trace,cuda_gpu_kern_gb_sum"
            exported = subprocess.call(
                [
                    "nsys",
                    "stats",
                    "--report",
                    reports,
                    "--format",
                    "csv",
                    "--output",
                    str(target / "nsys"),
                    str(capture) + ".nsys-rep",
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            write(
                target / "nsys_export_status.json",
                dict(
                    return_code=exported,
                    reports=reports,
                    local_capture=str(capture) + ".nsys-rep",
                    scope="HEAVY_DIAGNOSTIC_NOT_CLEAN_TIMING",
                ),
            )
    result = dict(
        id=job["id"],
        GPU=gpu,
        exit_code=code,
        started=started,
        ended=time.time(),
        status="COMPLETE" if code == 0 else "FAILED",
    )
    write(ROOT / "worker_state" / f"gpu{gpu}.json", result)
    print("END", job["id"], gpu, result["status"], flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--gpus", default="4,5,6,7")
    a = p.parse_args()
    devices = list(map(int, a.gpus.split(",")))
    assert devices and set(devices) <= set((4, 5, 6, 7))
    jobs = json.loads(open(a.manifest).read())["jobs"]
    affinity_file = ROOT / "gpu_affinity_manifest.json"
    affinity = json.loads(affinity_file.read_text()) if affinity_file.exists() else {}
    assert all(affinity[j["id"]] in devices for j in jobs if j["id"] in affinity), (
        "Requested GPUs exclude a registered diagnostic device"
    )
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    results = []
    lock = threading.Lock()

    def worker(gpu):
        while True:
            job = None
            with lock:
                if pending.empty():
                    break
                for _ in range(pending.qsize()):
                    candidate = pending.get_nowait()
                    if affinity.get(candidate["id"], gpu) == gpu:
                        job = candidate
                        break
                    pending.put(candidate)
            if job is None:
                time.sleep(0.1)
                continue
            record = launch(job, gpu)
            with lock:
                results.append(record)
                write(ROOT / "dispatch_progress.json", results)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        list(pool.map(worker, devices))
    print("DISPATCH_COMPLETE", len(results), "jobs", flush=True)
    if any(
        r["status"] in ("FAILED", "PRESERVED_FAILURE_MANUAL_REVIEW") for r in results
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
