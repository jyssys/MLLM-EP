"""Actually roll out phase-wise or one-step budget interventions on EP4."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from static_campaign import audit


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("gsm8k", "humaneval"), required=True)
    parser.add_argument("--n", type=int, choices=(8, 32, 100, 164, 512, 1319), required=True)
    parser.add_argument("--submitted", type=int, required=True)
    parser.add_argument("--mini", type=int, required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--policies", nargs="+", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--trace-decisions", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4,5,6,7":
        raise RuntimeError("only physical GPU 4,5,6,7")
    data_root = Path(__file__).resolve().parents[1] / "poc_llada2_flash_ep" / "data"
    if args.n in (164, 512, 1319):
        if (args.n, args.task) not in ((164, "humaneval"), (512, "gsm8k"), (1319, "gsm8k")):
            raise ValueError("164 is full HumanEval; 512/1319 are GSM8K promotion")
        data = ROOT / "data" / f"{args.task}_{args.n}.json"
    else:
        subdir = {8: "bounded_eval", 32: "bounded_eval_32", 100: "bounded_eval_100"}[args.n]
        data = data_root / subdir / f"{args.task}_{args.n}.json"
    truth = data.with_name(data.stem + "_truth.json")
    for index, policy in enumerate(args.policies):
        label = f"{args.prefix}_{args.task}{args.n}_p{index:02d}"
        audit(label)
        command = [
            sys.executable, str(ROOT / "run_ep4.py"), "--dataset", str(data), "--label", label,
            "--submitted", str(args.submitted), "--mini", str(args.mini),
            "--generation", str(args.generation), "--threshold", "0.9",
            "--policy", policy,
        ]
        if args.trace_decisions:
            command.append("--trace-decisions")
        environment = os.environ.copy()
        environment["EP_BUDGET_PRECREATED_RESULT"] = "1"
        subprocess.run(command, check=True, env=environment)
        answer = next((ROOT / "results" / label / "benchmark").glob("*.jsonl"))
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parents[1] / "poc_llada2_flash_ep" / "scripts" / "evaluate_bounded_quality.py"),
             "--task", args.task, "--predictions", str(answer), "--truth", str(truth),
             "--output", str(ROOT / "results" / label / "quality.json")],
            check=True,
        )
        log = (ROOT / "results" / label / "benchmark.log").read_text()
        points = re.findall(r"\[iter\s+\d+\]nfe=\s*(\d+).*?sample_time=([\d.]+)", log)
        quality = json.loads((ROOT / "results" / label / "quality.json").read_text())
        (ROOT / "results" / label / "policy.txt").write_text(policy + "\n")
        print(f"DONE {label} {policy}: NFE={sum(int(p[0]) for p in points)} BCT={sum(float(p[1]) for p in points):.3f}s score={quality['correct']}/{quality['total']}", flush=True)


if __name__ == "__main__":
    main()
