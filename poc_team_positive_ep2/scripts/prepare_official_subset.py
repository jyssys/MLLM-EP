#!/usr/bin/env python3
"""Prepare deterministic GSM8K/HumanEval subsets with official prompt templates."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import urllib.request
from pathlib import Path

from datasets import load_dataset


HUMANEVAL_URL = (
    "https://raw.githubusercontent.com/openai/human-eval/"
    "master/data/HumanEval.jsonl.gz"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gsm8k", type=int, default=4)
    parser.add_argument("--humaneval", type=int, default=4)
    args = parser.parse_args()

    records = []
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    for index, row in enumerate(gsm.select(range(args.gsm8k))):
        reference = row["answer"].split("#### ")[-1].replace(",", "").strip()
        records.append({
            "id": f"gsm8k_{index}",
            "task": "gsm8k",
            "prompt": row["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
            "reference": reference,
            "source": "openai/gsm8k:test",
        })

    compressed = urllib.request.urlopen(HUMANEVAL_URL).read()
    rows = [json.loads(line) for line in gzip.GzipFile(fileobj=io.BytesIO(compressed))]
    for index, row in enumerate(rows[: args.humaneval]):
        records.append({
            "id": row["task_id"],
            "task": "humaneval",
            "prompt": (
                "Read the following function signature and docstring, and fully implement the "
                "function described. Your response should only contain the code for this function.\n"
                + row["prompt"]
            ),
            "reference": row["task_id"],
            "canonical_solution": row["canonical_solution"],
            "test": row["test"],
            "entry_point": row["entry_point"],
            "source": "openai/human-eval:master",
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    print(f"wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
