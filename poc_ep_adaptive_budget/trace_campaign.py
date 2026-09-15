"""One opt-in observer-heavy trajectory, never used for clean BCT claims."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from static_campaign import ROOT, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--policy", default="none")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4,5,6,7":
        raise RuntimeError("only physical GPUs 4,5,6,7")
    data = ROOT.parent / "poc_llada2_flash_ep" / "data" / "bounded_eval_32" / "gsm8k_32.json"
    truth = data.with_name("gsm8k_32_truth.json")
    audit(args.label)
    if args.policy != "none":
        (ROOT / "results" / args.label / "policy.txt").write_text(args.policy + "\n")
    env = os.environ.copy()
    env["EP_BUDGET_PRECREATED_RESULT"] = "1"
    subprocess.run([
        sys.executable, str(ROOT / "run_ep4.py"),
        "--dataset", str(data), "--label", args.label,
        "--submitted", "32", "--mini", "32", "--generation", "128",
        "--threshold", "0.9", "--policy", args.policy,
        "--trace-decisions", "--ep-trace-layers", "1,16,31",
        "--shape-trace-layers", "16",
    ], env=env, check=True)
    answer = next((ROOT / "results" / args.label / "benchmark").glob("*.jsonl"))
    evaluator = ROOT.parent / "poc_llada2_flash_ep" / "scripts" / "evaluate_bounded_quality.py"
    subprocess.run([
        sys.executable, str(evaluator), "--task", "gsm8k",
        "--predictions", str(answer), "--truth", str(truth),
        "--output", str(ROOT / "results" / args.label / "quality.json"),
    ], check=True)
    print(json.dumps(json.loads((ROOT / "results" / args.label / "quality.json").read_text()), indent=2)[:200])


if __name__ == "__main__":
    main()
