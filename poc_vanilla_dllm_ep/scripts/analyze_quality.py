#!/usr/bin/env python3
"""Summarize bounded GSM8K/HumanEval quality by topology and restart."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path)
    parser.add_argument("--run-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.scores.read_text())
    run_rows = []
    by_ep = defaultdict(list)
    for entry in payload:
        match = re.search(r"ep(\d+)_restart(\d+)", entry["path"])
        ep, restart = (int(match.group(1)), int(match.group(2)))
        counts = entry["task_counts"]
        correct = entry["task_correct"]
        row = {
            "ep": ep,
            "restart": restart,
            "gsm8k_correct": int(correct.get("gsm8k", 0)),
            "gsm8k_count": int(counts.get("gsm8k", 0)),
            "humaneval_correct": int(correct.get("humaneval", 0)),
            "humaneval_count": int(counts.get("humaneval", 0)),
        }
        row["overall_accuracy"] = (
            row["gsm8k_correct"] + row["humaneval_correct"]
        ) / (row["gsm8k_count"] + row["humaneval_count"])
        run_rows.append(row)
        by_ep[ep].append(row)
    args.run_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.run_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)
    summary = {}
    for ep, rows in sorted(by_ep.items()):
        summary[str(ep)] = {
            "restarts": len(rows),
            "gsm8k_correct_median": statistics.median(x["gsm8k_correct"] for x in rows),
            "humaneval_correct_median": statistics.median(x["humaneval_correct"] for x in rows),
            "overall_accuracy_median": statistics.median(x["overall_accuracy"] for x in rows),
            "all_restart_scores": rows,
        }
    args.summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
