#!/usr/bin/env python3
"""Fetch small immutable GSM8K/HumanEval slices in dInfer's JSON schema."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path


DATASETS = {
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "humaneval": ("openai/openai_humaneval", "openai_humaneval", "test"),
}


def fetch(name: str, count: int) -> list[dict]:
    dataset, config, split = DATASETS[name]
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": 0,
            "length": count,
        }
    )
    with urllib.request.urlopen(
        f"https://datasets-server.huggingface.co/rows?{query}", timeout=60
    ) as response:
        payload = json.load(response)
    rows = [item["row"] for item in payload["rows"]]
    if len(rows) != count:
        raise RuntimeError(f"requested {count} {name} rows, received {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for name in DATASETS:
        rows = fetch(name, args.count)
        details = []
        truths = []
        for index, row in enumerate(rows):
            prompt = row["question"] if name == "gsm8k" else row["prompt"]
            details.append({"prompt": prompt})
            truths.append({"id": index, **row})

        input_path = args.output / f"{name}_{args.count}.json"
        truth_path = args.output / f"{name}_{args.count}_truth.json"
        input_path.write_text(json.dumps({"details": details}, indent=2) + "\n")
        truth_path.write_text(json.dumps(truths, indent=2) + "\n")
        manifest[name] = {
            "count": args.count,
            "input": input_path.name,
            "truth": truth_path.name,
            "source": DATASETS[name],
        }

    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
