"""Summarize clean, separately initialized repeats without pooling their waves."""

import argparse
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(label):
    path = ROOT / "results" / label
    log = (path / "benchmark.log").read_text()
    points = re.findall(r"\[iter\s+\d+\]nfe=\s*(\d+).*?sample_time=([\d.]+)", log)
    if not points:
        raise RuntimeError(f"no completed waves in {label}")
    quality = json.loads((path / "quality.json").read_text())
    return {
        "label": label,
        "bct_s": sum(float(point[1]) for point in points),
        "nfe": sum(int(point[0]) for point in points),
        "correct": quality["correct"],
        "total": quality["total"],
        "answers": [(row["id"], row.get("predicted"), row["pass"]) for row in quality["details"]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    baseline = [load(label) for label in args.baseline]
    candidate = [load(label) for label in args.candidate]
    for cohort in (baseline, candidate):
        if any(row["answers"] != cohort[0]["answers"] for row in cohort[1:]):
            raise RuntimeError("within-policy answer/quality drift across restarts")
    base_median = statistics.median(row["bct_s"] for row in baseline)
    candidate_median = statistics.median(row["bct_s"] for row in candidate)
    report = {
        "baseline": [{key: value for key, value in row.items() if key != "answers"} for row in baseline],
        "candidate": [{key: value for key, value in row.items() if key != "answers"} for row in candidate],
        "baseline_median_bct_s": base_median,
        "candidate_median_bct_s": candidate_median,
        "median_gain_pct": (base_median - candidate_median) / base_median * 100,
        "baseline_range_s": [min(row["bct_s"] for row in baseline), max(row["bct_s"] for row in baseline)],
        "candidate_range_s": [min(row["bct_s"] for row in candidate), max(row["bct_s"] for row in candidate)],
        "caveat": "Restart median is descriptive at n=3; the paired sample quality interval is a separate statistical test.",
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
