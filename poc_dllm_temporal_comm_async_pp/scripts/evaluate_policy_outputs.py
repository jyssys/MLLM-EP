#!/usr/bin/env python3
"""Score multi-policy boundary-staleness outputs and baseline agreement."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import tempfile
from pathlib import Path


def number(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{\s*([-+]?[$]?[\d,]+(?:\.\d+)?)\s*\}", text)
    candidates = boxed or re.findall(r"[-+]?[$]?[\d,]+(?:\.\d+)?", text)
    return candidates[-1].replace("$", "").replace(",", "") if candidates else None


def humaneval_pass(prompt: str, answer: str, test: str, entry_point: str) -> bool:
    continuation = answer
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", answer, flags=re.S | re.I)
    if fenced:
        continuation = fenced[0]
        source = continuation if f"def {entry_point}" in continuation else prompt + continuation
    else:
        source = prompt + continuation
    source += f"\n\n{test}\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py") as handle:
        handle.write(source)
        handle.flush()
        try:
            result = subprocess.run(
                ["python", "-I", handle.name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0


def score(task: str, truth: list[dict], rows: list[dict]) -> tuple[int, list[bool]]:
    passed = []
    by_id = {int(row["original_index"]): row for row in rows}
    for index, item in enumerate(truth):
        answer = by_id[index]["answer"]
        if task == "gsm8k":
            gold = item["answer"].split("####")[-1].strip().replace(",", "")
            passed.append(number(answer) == gold)
        else:
            passed.append(
                humaneval_pass(item["prompt"], answer, item["test"], item["entry_point"])
            )
    return sum(passed), passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("gsm8k", "humaneval"), required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    truth = json.loads(args.truth.read_text())
    policies = {}
    for path in sorted(args.policy_dir.glob("policy_*.json")):
        payload = json.loads(path.read_text())
        policies[payload["policy"]["name"]] = payload
    baseline = policies["baseline"]
    baseline_ids = {
        int(row["original_index"]): row["token_ids"] for row in baseline["rows"]
    }
    baseline_score, baseline_pass = score(args.task, truth, baseline["rows"])
    output_rows = []
    for name, payload in policies.items():
        quality_score, passed = score(args.task, truth, payload["rows"])
        exact = sum(
            row["token_ids"] == baseline_ids[int(row["original_index"])]
            for row in payload["rows"]
        )
        pass_agreement = sum(a == b for a, b in zip(passed, baseline_pass))
        output_rows.append(
            {
                "task": args.task,
                "policy": name,
                "score": quality_score,
                "total": len(truth),
                "relative_quality_pct": 100.0 * quality_score / max(baseline_score, 1),
                "baseline_score": baseline_score,
                "sequence_exact": exact,
                "sequence_exact_pct": 100.0 * exact / len(truth),
                "benchmark_pass_agreement": pass_agreement,
                "nfe": payload["nfe"],
                "nfe_delta": payload["nfe"] - baseline["nfe"],
                "model_seconds_observer": payload["model_seconds"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(output_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(sorted(output_rows, key=lambda row: row["policy"]))


if __name__ == "__main__":
    main()
