"""Read actual full-trajectory results and make a bounded static Pareto table."""

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def collect():
    rows = []
    for result in sorted((ROOT / "results").iterdir()):
        if (
            not result.is_dir() or not (result / "quality.json").exists()
            or (result / "policy.txt").exists() or (result / "decisions.jsonl").exists()
            or (result / "ep_trace").exists()
        ):
            continue
        command = (result / "command.txt").read_text() if (result / "command.txt").exists() else ""
        threshold = re.search(r"--threshold ([\d.]+)", command)
        if not threshold:
            continue
        log = (result / "benchmark.log").read_text()
        timings = re.findall(r"\[iter\s+\d+\]nfe=\s*(\d+), token number=\s*(\d+), sample_time=([\d.]+)", log)
        if not timings:
            continue
        quality = json.loads((result / "quality.json").read_text())
        nfe = sum(int(point[0]) for point in timings)
        generated = sum(int(point[1]) for point in timings)
        bct = sum(float(point[2]) for point in timings)
        rows.append({
            "label": result.name,
            "task": quality["task"],
            "n": quality["total"],
            "threshold": float(threshold.group(1)),
            "correct": quality["correct"],
            "accuracy": quality["accuracy"],
            "nfe": nfe,
            "generated_tokens": generated,
            "bct_s": bct,
            "tokens_per_forward": generated / nfe,
            "answer_file": str(next((result / "benchmark").glob("*.jsonl"))),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "STATIC_THRESHOLD_SWEEP.csv")
    args = parser.parse_args()
    rows = collect()
    if not rows:
        raise RuntimeError("no complete full-trajectory results")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for task in sorted(set(row["task"] for row in rows)):
        subset = [row for row in rows if row["task"] == task]
        print(task)
        for row in sorted(subset, key=lambda item: item["threshold"]):
            print(f"  tau={row['threshold']:.3f} score={row['correct']}/{row['n']} NFE={row['nfe']} BCT={row['bct_s']:.3f}s label={row['label']}")


if __name__ == "__main__":
    main()
