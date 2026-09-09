"""Sequential, bounded native correctness probes before any plan-regret claim."""
import argparse
import json
import os
from pathlib import Path
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
    parser.add_argument("--only", nargs="+",
                        choices=["eager", "graph", "plan1", "plan2", "plan2_graph"])
    args = parser.parse_args()
    out = args.results / "nanoflow_runs" / args.name
    out.mkdir(parents=True, exist_ok=False)
    variants = {"eager": [], "graph": ["--cuda-graph"],
                "plan1": ["--nano-parts", "1"], "plan2": ["--nano-parts", "2"],
                "plan2_graph": ["--nano-parts", "2", "--cuda-graph"]}
    plan = args.only or ["eager", "graph", "plan1", "plan2"]
    manifest = {"purpose": "native_correctness_and_activation_not_request_regret",
                "plan": plan, "records": [], "physical_gpus": [4, 5, 6, 7]}
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    model = Path("/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9")
    for name in plan:
        assert_gpus_empty()
        native = out / (name + ".json")
        command = ["bash", str(ROOT / "NANOFLOW/launch_native_smoke.sh"),
                   "--model", str(model), "--cache", str(args.results / "nanoflow_weight_cache"),
                   "--out", str(native), *variants[name]]
        record = {"name": name, "command": command, "start_unix": time.time()}
        print("START", name, flush=True)
        with (out / (name + ".log")).open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            record["pid"] = process.pid
            try:
                record["returncode"] = process.wait(timeout=900)
            except subprocess.TimeoutExpired:
                record["status"] = "BOUNDED_TIMEOUT"
            finally:
                cleanup(process)
                record["end_unix"] = time.time()
        if record.get("returncode") == 0 and native.exists():
            result = subprocess.run([
                sys.executable, str(ROOT / "NANOFLOW/check_hf_tokens.py"),
                "--native", str(native), "--reference", str(args.results / "raw/qwen15_hf_reference.jsonl"),
                "--out", str(out / (name + ".correctness.json"))], check=False)
            record["status"] = "EXACT_SMOKE_PASS" if result.returncode == 0 else "CORRECTNESS_FAIL"
        else:
            record.setdefault("status", "PORT_OR_RUNTIME_FAILURE")
        manifest["records"].append(record)
        path.write_text(json.dumps(manifest, indent=2))
        print("DONE", name, record["status"], flush=True)
        # Each requested probe has its own fresh workers and remains diagnostic.
        # Failure never becomes a latency datapoint or a method-failure claim.


if __name__ == "__main__":
    main()
