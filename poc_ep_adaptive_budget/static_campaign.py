"""Sequential, independently restarted static-threshold EP4 frontier.

Every launch records physical UUID/free memory/process ownership and NVLink.
It refuses to start while another compute process is using GPU 4--7.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_UUIDS = (
    "GPU-6076e2f2-5b63-3761-5586-56ceb7df8139",
    "GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730",
    "GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28",
    "GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b",
)


def command_output(*args):
    return subprocess.check_output(args, text=True)


def audit(label):
    result = ROOT / "results" / label
    result.mkdir(parents=True, exist_ok=False)
    inventory = command_output(
        "nvidia-smi", "-i", "4,5,6,7", "--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    rows = [row.strip().split(", ") for row in inventory.splitlines() if row.strip()]
    if len(rows) != 4 or tuple(row[1] for row in rows) != EXPECTED_UUIDS:
        raise RuntimeError(f"GPU UUID or ordering changed: {inventory}")
    if min(int(row[2]) for row in rows) < 70000:
        raise RuntimeError(f"insufficient free GPU memory: {inventory}")
    processes = command_output(
        "nvidia-smi", "-i", "4,5,6,7", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    )
    if processes.strip():
        pids = sorted(set(re.findall(r", (\d+),", processes)))
        owners = command_output("ps", "-o", "user,pid,ppid,args", "-p", ",".join(pids)) if pids else ""
        raise RuntimeError(f"GPUs busy; no process will be stopped:\n{processes}\n{owners}")
    (result / "physical_gpu_inventory.csv").write_text(inventory)
    (result / "processes_before.csv").write_text(processes)
    (result / "logical_mapping.txt").write_text("CUDA_VISIBLE_DEVICES=4,5,6,7; logical 0->4, 1->5, 2->6, 3->7\n")
    (result / "topology.txt").write_text(command_output("nvidia-smi", "topo", "-m"))
    (result / "nvlink.txt").write_text(command_output("nvidia-smi", "nvlink", "-s", "-i", "4,5,6,7"))
    print(f"AUDIT {label}: {inventory.strip().replace(chr(10), ' | ')}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("gsm8k", "humaneval"), required=True)
    parser.add_argument("--n", type=int, choices=(8, 32, 100, 164, 512, 1319), default=32)
    parser.add_argument("--submitted", type=int, default=32)
    parser.add_argument("--mini", type=int, default=32)
    parser.add_argument("--generation", type=int, default=128)
    parser.add_argument("--thresholds", nargs="+", type=float, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4,5,6,7":
        raise RuntimeError("only CUDA_VISIBLE_DEVICES=4,5,6,7 permitted")
    if args.n in (164, 512, 1319):
        if (args.n, args.task) not in ((164, "humaneval"), (512, "gsm8k"), (1319, "gsm8k")):
            raise ValueError("164 is full HumanEval; 512/1319 are GSM8K promotion")
        data = ROOT / "data" / f"{args.task}_{args.n}.json"
    else:
        data_subdir = {8: "bounded_eval", 32: "bounded_eval_32", 100: "bounded_eval_100"}[args.n]
        data = Path(__file__).resolve().parents[1] / "poc_llada2_flash_ep" / "data" / data_subdir / f"{args.task}_{args.n}.json"
    truth = data.with_name(data.stem + "_truth.json")
    for threshold in args.thresholds:
        label = f"{args.prefix}_{args.task}{args.n}_t{int(round(threshold * 1000)):03d}"
        audit(label)
        command = [
            sys.executable, str(ROOT / "run_ep4.py"), "--dataset", str(data), "--label", label,
            "--submitted", str(args.submitted), "--mini", str(args.mini),
            "--generation", str(args.generation), "--threshold", str(threshold),
        ]
        # run_ep4 already owns the per-run directory created by audit.
        environment = os.environ.copy()
        environment["EP_BUDGET_PRECREATED_RESULT"] = "1"
        subprocess.run(command, check=True, env=environment)
        answers = next((ROOT / "results" / label / "benchmark").glob("*.jsonl"))
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parents[1] / "poc_llada2_flash_ep" / "scripts" / "evaluate_bounded_quality.py"),
             "--task", args.task, "--predictions", str(answers), "--truth", str(truth),
             "--output", str(ROOT / "results" / label / "quality.json")],
            check=True,
        )
        log = (ROOT / "results" / label / "benchmark.log").read_text()
        points = re.findall(r"\[iter\s+\d+\]nfe=\s*(\d+).*?sample_time=([\d.]+)", log)
        score = json.loads((ROOT / "results" / label / "quality.json").read_text())
        print(f"DONE {label}: NFE={sum(int(p[0]) for p in points)} BCT={sum(float(p[1]) for p in points):.4f}s score={score['correct']}/{score['total']}", flush=True)


if __name__ == "__main__":
    main()
