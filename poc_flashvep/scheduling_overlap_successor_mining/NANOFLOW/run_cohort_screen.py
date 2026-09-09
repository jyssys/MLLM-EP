"""Randomized native plan/cohort screen, not an online NanoFlow full port."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "FASTPP"))
from run_configuration import assert_gpus_empty, cleanup, gpu_state


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4,5,6,7"
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--workloads", nargs="+", default=["b4_c128", "b16_c512", "b32_c128", "b64_c128"])
    parser.add_argument("--variants", nargs="+", choices=["graph", "plan1_graph", "plan2_graph", "plan4_graph", "plain", "plan1", "plan2", "plan4", "plan2_c112_n16", "plan2_c112_n16_graph"],
                        default=["graph", "plan1_graph", "plan2_graph", "plan4_graph"])
    parser.add_argument("--nsight", action="store_true",
                        help="CUDA/NVTX diagnostic only; exclude profiled request latency")
    parser.add_argument("--phase", choices=["decode", "prefill"], default="decode")
    parser.add_argument("--trace-dir", type=Path,
                        help="Frozen alternative real-content cohort directory")
    args = parser.parse_args()
    if args.phase == "prefill":
        assert all(v in {"plain", "plan1", "plan2", "plan4", "plan2_c112_n16"} for v in args.variants)
    out = args.results / "nanoflow_runs" / args.name
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(20260908)
    plan = []
    for block in range(args.restarts):
        conditions = [(w, v) for w in args.workloads for v in args.variants]
        rng.shuffle(conditions)
        for workload, variant in conditions:
            plan.append({"block": block, "workload": workload, "variant": variant})
    manifest = {"plan": plan, "records": [], "status": "RUNNING",
                "phase": args.phase,
                "trace_dir": str(args.trace_dir) if args.trace_dir else None,
                "nsight_profiled": args.nsight,
                "contract": "real-content fixed cohorts; no dynamic scheduler; per-cohort fresh engine",
                "quality": "cross-plan generated token checks required before performance acceptance"}
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    model = "/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"
    for item in plan:
        assert_gpus_empty()
        name = f"b{item['block']}_{item['workload']}_{item['variant']}"
        target = out / (name + ".json")
        trace_dir = "nanoflow_prefill_cohorts_v1" if args.phase == "prefill" else "nanoflow_cohorts_v1"
        trace_root = args.trace_dir or args.results / "request_traces" / trace_dir
        trace = trace_root / (item["workload"] + ".jsonl")
        assert trace.is_file(), "Validate frozen input trace before starting workers"
        command = ["bash", str(ROOT / "NANOFLOW/launch_native_smoke.sh"),
                   "--cache", str(args.results / "nanoflow_weight_cache"), "--model", model,
                   "--out", str(target), "--cohort-trace", str(trace), "--plan-phase", args.phase]
        if args.phase == "decode":
            command += ["--cuda-graph"]
        if item["variant"].startswith("plan"):
            command += ["--nano-parts", item["variant"][4]]
        if "_c112_n16" in item["variant"]:
            command += ["--compute-sms", "112", "--collective-sms", "16"]
        if args.nsight:
            # Child workers use spawn; no unsafe fork-before-exec tracing.
            # No system-wide GPU metrics or sampling of unrelated processes.
            command = ["/usr/local/bin/nsys", "profile", "--trace=cuda,nvtx",
                       "--sample=none", "--cpuctxsw=none", "--cuda-graph-trace=node",
                       "--gpu-metrics-devices=none", "--force-overwrite=false",
                       "--output", str(out / name)] + command
        record = {**item, "name": name, "start_unix": time.time(),
                  "command": command, "gpu_state_start": gpu_state()}
        print("START", name, flush=True)
        with (out / (name + ".log")).open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True,
                                       env=dict(os.environ, SUCCESSOR_NANO_NVTX="1" if args.nsight else "0"))
            record["pid"] = process.pid
            try:
                record["returncode"] = process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                record["status"] = "BOUNDED_TIMEOUT"
            except KeyboardInterrupt:
                record["status"] = "INTERRUPTED_FOR_DIAGNOSIS"
            finally:
                cleanup(process)
        record.update(end_unix=time.time(), gpu_state_end=gpu_state())
        if record.get("returncode") == 0 and target.exists():
            data = json.loads(target.read_text())
            record["status"] = ("NSIGHT_DIAGNOSTIC_NOT_PERFORMANCE" if args.nsight
                                else "COLLECTED_CORRECTNESS_PENDING")
            subprocess.run([
                sys.executable, str(ROOT / "COMMON/record_gpu_interval.py"),
                "--log", str(ROOT / "GPU_TIME_LOG.csv"), "--candidate", "NANOFLOW",
                "--run-id", args.name + ":" + name, "--question",
                "Native operation-plan regret on actual fixed-cohort request completion",
                "--start", str(data["start_unix"]), "--end", str(data["end_unix"]),
                "--kind", "mechanism_diagnostic" if args.nsight else "measurement",
                "--valid", record["status"],
                "--artifact", str(target)], check=True)
        else:
            record.setdefault("status", "PORT_OR_RUNTIME_FAILURE")
        manifest["records"].append(record)
        path.write_text(json.dumps(manifest, indent=2))
        print("DONE", name, record["status"], flush=True)
        if record.get("returncode") != 0:
            manifest["status"] = "STOPPED_ON_PORT_OR_RUNTIME_FAILURE_NOT_METHOD_FAILURE"
            path.write_text(json.dumps(manifest, indent=2))
            return
    manifest["status"] = ("NSIGHT_SCREEN_COMPLETE" if args.nsight
                          else "SCREEN_COMPLETE_CORRECTNESS_PENDING")
    path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
