"""Official TP2 Layered Prefill restart; request metrics before stage claims."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
# Reuse only process lifecycle/GPU guards, not FastPP runtime or algorithms.
sys.path.insert(0, str(ROOT / "FASTPP"))
from run_configuration import assert_gpus_empty, cleanup, gpu_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["chunked-prefill", "layered-prefill"], required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--stages", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=Path, required=True)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--correctness-trace", type=Path, required=True)
    parser.add_argument("--mechanism", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--workload-warmup-repeats", type=int, default=0)
    options = parser.parse_args()
    assert_gpus_empty()
    # Stock server unlinks this fixed name. Refuse any existing segment rather
    # than allowing it to remove another experiment's shared state.
    assert not Path("/dev/shm/nanovllm").exists(), "Inspect existing nanovllm SHM ownership first"
    options.out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="4,5,6,7",
               TORCH_CUDA_ARCH_LIST="9.0", PYTHONPATH=str(ROOT / "LAYERED_PREFILL"),
               SCHEDULING_LP_EAGER="0" if options.cuda_graph else "1")
    env.pop("SUCCESSOR_LP_TRACE", None)
    if options.mechanism:
        env["SUCCESSOR_LP_TRACE"] = str(options.out / "mechanism")
    command = ["bash", str(ROOT / "LAYERED_PREFILL/launch_official.sh"),
               "--schedule-mode", options.mode, "--max-num-batched-tokens", str(options.tokens),
               "--num-stages", str(options.stages)]
    meta = {"start_unix": time.time(), "command": command, "status": "STARTING",
            "physical_gpus": [4, 5], "visible_allowed_gpus": [4, 5, 6, 7],
            "topology": "TP2; all experts tensor sharded; NOT EP",
            "mode": options.mode, "stages": options.stages, "tokens": options.tokens,
            "cuda_graph": options.cuda_graph,
            "mechanism_instrumented": options.mechanism,
            "workload_warmup_repeats": options.workload_warmup_repeats,
            "request_runs": [], "gpu_state_start": gpu_state()}

    def save():
        (options.out / "run.json").write_text(json.dumps(meta, indent=2))

    def client(trace, label):
        out = options.out / (label + ".jsonl")
        subprocess.run([sys.executable, str(ROOT / "COMMON/run_request_trace.py"),
                        "--backend", "layered", "--trace", str(trace), "--out", str(out),
                        "--run-id", options.out.name + ":" + label],
                       check=True, env=env, timeout=1200)
        summary = json.loads(out.with_suffix(".summary.json").read_text())
        assert summary["valid"] == summary["requests"]
        meta["request_runs"].append({"label": label, "trace": str(trace), **summary})
        save()
        return out

    with (options.out / "server.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, cwd=options.out, start_new_session=True)
        meta["engine_pid"] = process.pid
        save()
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Official Layered engine exited")
                if time.time() - meta["start_unix"] > 1200:
                    raise TimeoutError("Bounded startup timeout")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:31800/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except Exception:
                    pass
                time.sleep(2)
            meta["ready_unix"] = time.time()
            for i in range(3):
                client(options.warmup, f"warmup{i}")
            smoke = client(options.correctness_trace, "correctness")
            subprocess.run([sys.executable, str(ROOT / "FASTPP/check_short_answers.py"), str(smoke)], check=True)
            for i, trace in enumerate(options.trace):
                for warm in range(options.workload_warmup_repeats):
                    client(trace, f"workload_warmup{i}_{warm}_{trace.stem}")
                meta[f"gpu_state_before_measure{i}"] = gpu_state()
                client(trace, f"measure{i}_{trace.stem}")
                meta[f"gpu_state_after_measure{i}"] = gpu_state()
            meta["status"] = "COLLECTED_CROSS_CONFIG_CORRECTNESS_PENDING"
        except BaseException as exc:
            meta.update(status="FAILED", error=repr(exc))
            raise
        finally:
            cleanup(process)
            meta["end_unix"] = time.time()
            meta["gpu_state_end"] = gpu_state()
            meta["shm_exists_after_cleanup"] = Path("/dev/shm/nanovllm").exists()
            save()


if __name__ == "__main__":
    main()
