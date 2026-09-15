"""Summarize actual dynamic trajectories, labeling incomplete oracle search."""

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "DYNAMIC_ORACLE_RESULTS.csv")
    args = parser.parse_args()
    rows = []
    for result in sorted((ROOT / "results").iterdir()):
        if (
            not result.is_dir() or not (result / "policy.txt").exists()
            or not (result / "quality.json").exists() or (result / "decisions.jsonl").exists()
            or (result / "ep_trace").exists()
        ):
            continue
        log = (result / "benchmark.log").read_text()
        points = re.findall(r"\[iter\s+\d+\]nfe=\s*(\d+), token number=\s*(\d+), sample_time=([\d.]+)", log)
        if not points:
            continue
        quality = json.loads((result / "quality.json").read_text())
        rows.append({
            "label": result.name, "task": quality["task"], "n": quality["total"],
            "policy": (result / "policy.txt").read_text().strip(),
            "evidence_type": "actual_full_trajectory_candidate_not_perfect_future_oracle",
            "correct": quality["correct"], "accuracy": quality["accuracy"],
            "nfe": sum(int(point[0]) for point in points),
            "generated_tokens": sum(int(point[1]) for point in points),
            "bct_s": sum(float(point[2]) for point in points),
            "wave_bct_median_s": sorted(float(point[2]) for point in points)[len(points) // 2],
            "answer_file": str(next((result / "benchmark").glob("*.jsonl"))),
        })
    if not rows:
        raise RuntimeError("no complete dynamic rollouts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['label']}: {row['policy']} score={row['correct']}/{row['n']} NFE={row['nfe']} BCT={row['bct_s']:.3f}s")


if __name__ == "__main__":
    main()
