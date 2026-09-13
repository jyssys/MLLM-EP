#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "poc_llada2_flash_ep" / "scripts"))
from evaluate_bounded_quality import humaneval_pass, number  # noqa: E402


def evaluate(task: str, answers: list[str]) -> list[dict]:
    count = len(answers)
    subdir = {1: "smoke_eval", 8: "bounded_eval", 32: "bounded_eval_32"}[count]
    truth_path = ROOT / "poc_llada2_flash_ep" / "data" / subdir / f"{task}_{count}_truth.json"
    truths = json.loads(truth_path.read_text())
    details = []
    for index, (answer, truth) in enumerate(zip(answers, truths)):
        if task == "gsm8k":
            gold = truth["answer"].split("####")[-1].strip().replace(",", "")
            predicted = number(answer)
            passed = predicted == gold
            details.append(
                {"id": index, "gold": gold, "predicted": predicted, "pass": passed}
            )
        else:
            passed = humaneval_pass(
                truth["prompt"], answer, truth["test"], truth["entry_point"]
            )
            details.append(
                {"id": index, "entry_point": truth["entry_point"], "pass": passed}
            )
    return details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["gsm8k", "humaneval"])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.run_dir.glob("policy_*.json")):
        payload = json.loads(path.read_text())
        answers = [row["answer"] for row in sorted(payload["rows"], key=lambda x: x["original_index"])]
        quality_rows = evaluate(args.task, answers)
        score = sum(bool(row["pass"]) for row in quality_rows)
        rows.append(
            {
                "policy": payload["policy"].get("name", path.stem),
                "block_schedule": payload["policy"].get("block_schedule", []),
                "model_seconds": payload["model_seconds"],
                "elapsed_seconds": payload["elapsed_seconds"],
                "nfe": payload["nfe"],
                "generated_tokens": payload["generated_tokens"],
                "correct": score,
                "total": len(quality_rows),
                "accuracy": score / len(quality_rows),
                "details": quality_rows,
            }
        )
    output = args.run_dir / "schedule_quality.json"
    output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
