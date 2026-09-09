"""Fresh-worker same-prefix logit checks; no performance conclusions."""
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
from run_configuration import assert_gpus_empty, cleanup


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4,5,6,7"
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--phase", choices=["decode", "prefill"], default="decode")
    parser.add_argument("--variants", nargs=2,
                        choices=["graph", "plan2_graph", "plain", "plan2", "plan2_c112_n16"])
    args = parser.parse_args()
    out = args.results / "nanoflow_runs" / args.name
    out.mkdir(parents=True, exist_ok=False)
    variants = args.variants or (["graph", "plan2_graph"] if args.phase == "decode" else ["plain", "plan2_c112_n16"])
    assert all(not v.endswith("graph") for v in variants) if args.phase == "prefill" else all(v.endswith("graph") for v in variants)
    manifest = {"timing_excluded": True, "reference": str(args.reference), "records": [],
                "phase": args.phase, "variants": variants}
    plan = [(block, variant) for block in range(2) for variant in variants]
    random.Random(20260908).shuffle(plan)
    manifest["plan"] = plan
    for block, variant in plan:
        assert_gpus_empty()
        name = f"b{block}_{variant}"
        command = ["bash", str(ROOT / "NANOFLOW/launch_native_smoke.sh"),
                   "--cache", str(args.results / "nanoflow_weight_cache"),
                   "--model", "/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9",
                   "--out", str(out / (name + ".json")), "--cohort-trace", str(args.trace),
                   "--forced-output", str(args.reference), "--capture-logits", str(out / (name + "_logits")),
                   "--plan-phase", args.phase]
        if args.phase == "decode":
            command += ["--cuda-graph"]
        if variant.startswith("plan2"):
            command += ["--nano-parts", "2"]
        if "c112_n16" in variant:
            command += ["--compute-sms", "112", "--collective-sms", "16"]
        record = {"name": name, "block": block, "variant": variant,
                  "command": command, "start_unix": time.time()}
        print("START", name, flush=True)
        with (out / (name + ".log")).open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            record["pid"] = process.pid
            try:
                record["returncode"] = process.wait(timeout=600)
            finally:
                cleanup(process)
        record["end_unix"] = time.time()
        manifest["records"].append(record)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print("DONE", name, record["returncode"], flush=True)
        if record["returncode"] != 0:
            break


if __name__ == "__main__":
    main()
