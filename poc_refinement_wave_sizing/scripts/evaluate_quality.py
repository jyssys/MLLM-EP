#!/usr/bin/env python3
"""Evaluate clean mini-size outputs using the bounded GSM8K task metric."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


def read_concatenated(path: Path) -> list[dict]:
    text = path.read_text()
    decoder = json.JSONDecoder()
    position, rows = 0, []
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position < len(text):
            row, position = decoder.raw_decode(text, position)
            rows.append(row)
    return rows


def number(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{\s*([-+]?[$]?[\d,]+(?:\.\d+)?)\s*\}", text)
    values = boxed or re.findall(r"[-+]?[$]?[\d,]+(?:\.\d+)?", text)
    return values[-1].replace("$", "").replace(",", "") if values else None


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    truth = json.loads((root.parent / "poc_llada2_flash_ep/data/bounded_eval_32/gsm8k_32_truth.json").read_text())
    rows = []
    for path in sorted((root / "results/clean/ep4/b32").glob("mini*/r*/*.jsonl")):
        predictions = {int(row["id"]): row for row in read_concatenated(path)}
        passed = []
        for index, item in enumerate(truth):
            gold = item["answer"].split("####")[-1].strip().replace(",", "")
            passed.append(number(predictions[index]["answer"]) == gold)
        mini = int(path.parents[1].name.removeprefix("mini"))
        repeat = path.parent.name.removeprefix("r")
        rows.append(
            {
                "topology": "ep4",
                "submitted_batch": 32,
                "mini_batch_size": mini,
                "repeat": repeat,
                "gsm8k_correct": sum(passed),
                "gsm8k_total": len(passed),
                "gsm8k_accuracy": sum(passed) / len(passed),
                "source": str(path),
            }
        )
    output = root / "QUALITY_RESULTS.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"runs": len(rows), "scores": sorted(set(row["gsm8k_correct"] for row in rows))}))


if __name__ == "__main__":
    main()
