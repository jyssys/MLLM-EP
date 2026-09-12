#!/usr/bin/env python3
"""Evaluate bounded GSM8K exact-answer and HumanEval functional accuracy."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path


def read_concatenated_json(path: Path) -> list[dict]:
    text = path.read_text()
    decoder = json.JSONDecoder()
    position = 0
    rows = []
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        row, position = decoder.raw_decode(text, position)
        rows.append(row)
    return rows


def number(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{\s*([-+]?[$]?[\d,]+(?:\.\d+)?)\s*\}", text)
    candidates = boxed or re.findall(r"[-+]?[$]?[\d,]+(?:\.\d+)?", text)
    return candidates[-1].replace("$", "").replace(",", "") if candidates else None


def humaneval_pass(prompt: str, answer: str, test: str, entry_point: str) -> bool:
    continuation = answer
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", answer, flags=re.S | re.I)
    if fenced:
        continuation = fenced[0]
        # Some models repeat the signature inside a fenced complete solution.
        source = continuation if f"def {entry_point}" in continuation else prompt + continuation
    else:
        source = prompt + continuation
    source += f"\n\n{test}\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py") as f:
        f.write(source)
        f.flush()
        try:
            result = subprocess.run(
                ["python", "-I", f.name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("gsm8k", "humaneval"), required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = read_concatenated_json(args.predictions)
    truths = json.loads(args.truth.read_text())
    by_id = {int(row["id"]): row for row in predictions}
    details = []
    for index, truth in enumerate(truths):
        prediction = by_id[index]["answer"]
        if args.task == "gsm8k":
            gold = truth["answer"].split("####")[-1].strip().replace(",", "")
            predicted = number(prediction)
            passed = predicted == gold
            detail = {"id": index, "gold": gold, "predicted": predicted, "pass": passed}
        else:
            passed = humaneval_pass(
                truth["prompt"], prediction, truth["test"], truth["entry_point"]
            )
            detail = {"id": index, "entry_point": truth["entry_point"], "pass": passed}
        details.append(detail)
    result = {
        "task": args.task,
        "correct": sum(row["pass"] for row in details),
        "total": len(details),
        "accuracy": sum(row["pass"] for row in details) / len(details),
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("task", "correct", "total", "accuracy")}, indent=2))


if __name__ == "__main__":
    main()
