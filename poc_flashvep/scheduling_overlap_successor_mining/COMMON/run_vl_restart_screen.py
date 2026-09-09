"""Paired clean/sparse-observer restarts; transfer diagnostics, not native ports."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"FASTPP"))
from run_configuration import cleanup, assert_gpus_empty


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4,5,6,7"
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--batch-per-dp", nargs="+", type=int, default=[1, 4, 8])
    p.add_argument("--max-pixels", type=int, default=512*512)
    args = p.parse_args()
    out = args.results/"vl_transfer"/args.name
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(20260909)
    plan = []
    for block in range(args.restarts):
        modes = ["clean", "layer_sampled"]
        rng.shuffle(modes)
        plan.extend({"block": block, "mode": mode} for mode in modes)
    manifest = {"plan": plan, "records": [], "status": "RUNNING",
                "scope": "vLLM real-image observer control, not native scheduler reproduction"}
    for item in plan:
        assert_gpus_empty()
        name = f"b{item['block']}_{item['mode']}"
        command = [sys.executable, str(ROOT/"COMMON/run_vl_operation_trace.py"),
                   "--model", args.model, "--trace", str(args.trace), "--out", str(out/name),
                   "--repetitions", "2", "--observer-mode", "layer_sampled",
                   "--observer-step-every", "8", "--observer-moe-every", "8",
                   "--seed", str(20260909+item["block"]),
                   "--max-pixels", str(args.max_pixels),
                   "--batch-per-dp", *[str(n) for n in args.batch_per_dp]]
        if item["mode"] != "clean":
            command.append("--instrument")
        record = {**item, "name": name, "command": command, "start_unix": time.time()}
        print("START", name, flush=True)
        with (out/(name+".log")).open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                record["returncode"] = process.wait(timeout=2400)
            finally:
                cleanup(process)
        record["end_unix"] = time.time()
        manifest["records"].append(record)
        (out/"manifest.json").write_text(json.dumps(manifest, indent=2))
        print("DONE", name, record["returncode"], flush=True)
        if record["returncode"] != 0:
            manifest["status"] = "FAILED_NOT_METHOD_FAILURE"
            break
    else:
        manifest["status"] = "PAIRED_OBSERVER_CONTROL_COLLECTED"
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
