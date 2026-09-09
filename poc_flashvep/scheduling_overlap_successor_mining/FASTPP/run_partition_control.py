"""Uninstrumented existing-static-partition attack, not a successor method.

Equal KV capacity and full workload warmup in both arms. The alternative comes
from a separate profile and is fixed before these request comparisons.
"""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4,5,6,7"
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--correctness-trace", required=True)
    p.add_argument("--restarts", type=int, default=3)
    args = p.parse_args()
    out = args.results/"fastpp_runs"/args.name
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(20260910)
    plan = []
    for block in range(args.restarts):
        variants = ["equal", "cost_static"]
        rng.shuffle(variants)
        workloads = ["heterogeneous_steady", "heterogeneous_bursty"]
        rng.shuffle(workloads)
        plan.extend({"block": block, "variant": v, "workloads": workloads} for v in variants)
    manifest = {"plan": plan, "completed": [], "status": "RUNNING",
                "scope": "existing static partition knob on native Qwen3 PP4, not EP",
                "partitions": {"equal": "12,12,12,12", "cost_static": "8,12,14,14"},
                "fixed_max_total_tokens": 262144, "workload_warmup_repeats": 1,
                "policy": "greedy", "source_profile": "qwen3_mechanism_alp_v3"}
    target = out/"plan.json"
    target.write_text(json.dumps(manifest, indent=2))
    try:
        for item in plan:
            name = f"b{item['block']}_{item['variant']}"
            command = [sys.executable, str(ROOT/"FASTPP/run_configuration.py"),
                       "--out", str(out/name), "--variant", "greedy", "--p2p-disable", "0",
                       "--model", args.model, "--correctness-trace", args.correctness_trace,
                       "--warmup", str(args.results/"request_traces/text_v1/warmup.jsonl"),
                       "--workload-warmup-repeats", "1", "--max-total-tokens", "262144",
                       "--partition", manifest["partitions"][item["variant"]]]
            for workload in item["workloads"]:
                command += ["--trace", str(args.results/"request_traces/text_v1"/(workload+".jsonl"))]
            print("START", name, flush=True)
            with (out/(name+".log")).open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
            manifest["completed"].append(name)
            target.write_text(json.dumps(manifest, indent=2))
            print("DONE", name, flush=True)
        manifest["status"] = "COLLECTED_CROSS_CONFIG_CORRECTNESS_PENDING"
    except BaseException as exc:
        manifest.update(status="FAILED_NOT_METHOD_FAILURE", error=repr(exc))
        raise
    finally:
        target.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
