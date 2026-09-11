#!/usr/bin/env python3
"""Score bounded Stage-A outputs and aggregate speed/work metrics."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import re
import sys
from pathlib import Path


def gsm8k_answer(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].replace(",", "").strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else None


def humaneval_postprocess(text: str) -> str:
    blocks = re.findall(r"```\w*\n(.*?)```", text, re.DOTALL)
    if blocks:
        text = blocks[0]
    return text.lstrip()


def score_file(path: Path, dataset_by_id: dict, check_correctness) -> dict:
    payload = json.loads(path.read_text())
    task_counts: dict[str, int] = {}
    task_correct: dict[str, int] = {}
    for record in payload["records"]:
        task = record["task"]
        task_counts[task] = task_counts.get(task, 0) + 1
        if task == "gsm8k":
            correct = gsm8k_answer(record["output"]) == str(record["reference"])
        elif task == "humaneval" and check_correctness is not None:
            source = dataset_by_id[record["id"]]
            original_prompt = source["prompt"].split(
                "Your response should only contain the code for this function.\n", 1
            )[-1]
            problem = {
                "task_id": record["id"],
                "prompt": original_prompt,
                "canonical_solution": source["canonical_solution"],
                "test": source["test"],
                "entry_point": source["entry_point"],
            }
            completion = humaneval_postprocess(record["output"])
            result = check_correctness(problem, completion, timeout=3.0, completion_id=0)
            correct = bool(result["passed"])
            record["humaneval_result"] = result["result"]
        else:
            correct = None
        record["bounded_score"] = correct
        if correct is True:
            task_correct[task] = task_correct.get(task, 0) + 1
    return {
        "path": str(path),
        "mode": payload["mode"],
        "mean_latency_s": sum(x["elapsed_s"] for x in payload["records"]) / len(payload["records"]),
        "task_counts": task_counts,
        "task_correct": task_correct,
        "records": payload["records"],
        "trace_summary": payload.get("trace_summary"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--human-eval-dir", type=Path)
    args = parser.parse_args()
    dataset_by_id = {
        row["id"]: row
        for row in (json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip())
    }
    check_correctness = None
    if args.human_eval_dir:
        sys.path.insert(0, str(args.human_eval_dir.resolve()))
        from human_eval.execution import check_correctness as official_check_correctness
        check_correctness = official_check_correctness
    results = [score_file(path, dataset_by_id, check_correctness) for path in args.inputs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
