"""One isolated official engine restart, identical warmup and explicit traces.

This harness refuses occupied GPUs and never terminates unrelated processes.
Primary measurements leave the optional mechanism hook off.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import psutil

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "pp_only": [],
    "pp_chunk128": ["--chunked-prefill-size", "128"],
    "pp_chunk512": ["--chunked-prefill-size", "512"],
    "greedy": ["--chunked-prefill-size", "128", "--enable-dynamic-chunk",
               "--dynamic-chunk-strategy", "greedy"],
    "alp": ["--chunked-prefill-size", "128", "--enable-dynamic-chunk",
            "--dynamic-chunk-strategy", "alp"],
    "alp_rebalance": ["--chunked-prefill-size", "128", "--enable-dynamic-chunk",
                      "--dynamic-chunk-strategy", "alp", "--enable-batch-rebalancing"],
}


def assert_gpus_empty():
    lines = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    allowed = {uuid.strip() for line in lines.splitlines()
               for idx, uuid in [line.split(",", 1)] if int(idx) in {4, 5, 6, 7}}
    assert len(allowed) == 4
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    busy = [s for s in processes.splitlines() if s.split(",")[0].strip() in allowed]
    if busy:
        raise RuntimeError("Refusing occupied experiment GPUs: " + repr(busy))


def gpu_state():
    return subprocess.check_output([
        "nvidia-smi", "-i", "4,5,6,7",
        "--query-gpu=index,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu",
        "--format=csv"], text=True)


def cleanup(process):
    try:
        parent = psutil.Process(process.pid)
        assert parent.username() == psutil.Process().username()
        children = parent.children(recursive=True)
        for item in [parent] + children:
            try:
                item.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(children + [parent], timeout=8)
        for item in alive:
            try:
                item.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=5)
    except psutil.NoSuchProcess:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--trace", type=Path, action="append", required=True)
    p.add_argument("--warmup", type=Path, required=True)
    p.add_argument("--warmup-repeats", type=int, default=3)
    p.add_argument("--workload-warmup-repeats", type=int, default=0,
                   help="Full identical arrival trace immediately before each measurement")
    p.add_argument("--correctness-trace", type=Path,
                   default=ROOT / "COMMON/smoke_requests.jsonl")
    p.add_argument("--p2p-disable", choices=["0", "1"], default="1")
    p.add_argument("--startup-timeout", type=float, default=900)
    p.add_argument("--model")
    p.add_argument("--chunk", type=int)
    p.add_argument("--partition", help="Existing vLLM contiguous PP partition; requires fixed KV cap")
    p.add_argument("--max-total-tokens", type=int)
    p.add_argument("--mechanism", action="store_true")
    args = p.parse_args()
    if args.model and args.model.startswith("/"):
        assert (Path(args.model)/"config.json").is_file(), "Validate local snapshot before launching workers"
    assert_gpus_empty()
    args.out.mkdir(parents=True, exist_ok=False)
    run_id = args.out.name
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="4,5,6,7", NCCL_IB_DISABLE="1",
               SCHEDULING_NCCL_P2P_DISABLE=args.p2p_disable)
    # No inherited diagnostics or burn variables in performance runs.
    for key in ("SUCCESSOR_STARTUP_DIAGNOSTIC", "SUCCESSOR_FASTPP_TRACE"):
        env.pop(key, None)
    env["PYTHONPATH"] = str(ROOT / "FASTPP")
    env.pop("VLLM_PP_LAYER_PARTITION", None)
    if args.partition:
        counts = [int(x) for x in args.partition.split(",")]
        assert len(counts) == 4 and min(counts) > 0
        assert args.max_total_tokens, "Use same conservative KV capacity in all partition arms"
        env["VLLM_PP_LAYER_PARTITION"] = args.partition
    if args.mechanism:
        env["SUCCESSOR_FASTPP_TRACE"] = str(args.out / "mechanism")
    command = ["bash", str(ROOT / "FASTPP/launch_official.sh"), *VARIANTS[args.variant],
               "--dc-alp-model-path", str(args.out / "alp_model.json")]
    if args.model:
        command += ["--model-path", args.model]
    if args.chunk:
        command += ["--chunked-prefill-size", str(args.chunk)]
    if args.max_total_tokens:
        command += ["--max-total-tokens", str(args.max_total_tokens)]
    start = time.time()
    meta = {"run_id": run_id, "command": command, "start_unix": start,
            "gpus": [4, 5, 6, 7], "p2p_disable": args.p2p_disable,
            "ib_disable": "1", "variant": args.variant,
            "mechanism_instrumented": args.mechanism,
            "status": "STARTING", "request_runs": [],
            "gpu_state_start": gpu_state(), "warmup_repeats": args.warmup_repeats}
    meta["workload_warmup_repeats"] = args.workload_warmup_repeats
    meta["explicit_pp_partition"] = args.partition
    meta["fixed_max_total_tokens"] = args.max_total_tokens
    (args.out / "run.json").write_text(json.dumps(meta, indent=2))

    def client(trace, label):
        out = args.out / (label + ".jsonl")
        subprocess.run([sys.executable, str(ROOT / "COMMON/run_request_trace.py"),
                        "--trace", str(trace), "--out", str(out),
                        "--run-id", run_id + ":" + label, "--backend", "fastpp"],
                       check=True, env=env, timeout=1200)
        summary = json.loads(out.with_suffix(".summary.json").read_text())
        assert summary["valid"] == summary["requests"], "Incomplete/error requests"
        meta["request_runs"].append({"label": label, "trace": str(trace), **summary})
        (args.out / "run.json").write_text(json.dumps(meta, indent=2))
        return out

    with (args.out / "server.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, cwd=args.out, start_new_session=True)
        meta["engine_pid"] = process.pid
        try:
            ready = False
            while time.time() - start < args.startup_timeout:
                if process.poll() is not None:
                    raise RuntimeError("Server exited during startup")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:31800/health", timeout=2) as response:
                        ready = response.status == 200
                except Exception:
                    pass
                if ready:
                    break
                time.sleep(2)
            if not ready:
                raise TimeoutError("Official engine startup exceeded bounded timeout")
            meta["ready_unix"] = time.time()
            meta["status"] = "WARMUP"
            print(f"{run_id}: engine ready; same warmup", flush=True)
            for i in range(args.warmup_repeats):
                client(args.warmup, f"warmup{i}")
            smoke = client(args.correctness_trace, "correctness")
            subprocess.run([sys.executable, str(ROOT / "FASTPP/check_short_answers.py"),
                            str(smoke)], check=True)
            meta["status"] = "MEASURING"
            for i, trace in enumerate(args.trace):
                for warm in range(args.workload_warmup_repeats):
                    client(trace, f"workload_warmup{i}_{warm}_{trace.stem}")
                print(f"{run_id}: measured trace {trace.name}", flush=True)
                meta[f"gpu_state_before_measure{i}"] = gpu_state()
                client(trace, f"measure{i}_{trace.stem}")
                meta[f"gpu_state_after_measure{i}"] = gpu_state()
            meta["status"] = "REQUEST_COLLECTION_COMPLETE_CORRECTNESS_CROSS_CONFIG_PENDING"
        except BaseException as exc:
            meta.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            cleanup(process)
            meta["end_unix"] = time.time()
            meta["gpu_state_end"] = gpu_state()
            (args.out / "run.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
