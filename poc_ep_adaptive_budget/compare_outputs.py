"""Report answer-sequence drift separately from benchmark correctness."""

import argparse
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(label):
    file = next((ROOT / "results" / label / "benchmark").glob("*.jsonl"))
    payload = file.read_text()
    decoder = json.JSONDecoder()
    offset = 0
    rows = []
    while offset < len(payload):
        while offset < len(payload) and payload[offset].isspace():
            offset += 1
        if offset >= len(payload):
            break
        row, offset = decoder.raw_decode(payload, offset)
        rows.append(row)
    return {int(row["id"]): row for row in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    baseline = load(args.baseline)
    candidate = load(args.candidate)
    if baseline.keys() != candidate.keys():
        raise RuntimeError("outputs do not have identical sample IDs")
    exact = sum(baseline[index]["answer"] == candidate[index]["answer"] for index in baseline)
    length_deltas = [
        int(candidate[index]["generated_length"]) - int(baseline[index]["generated_length"])
        for index in baseline
    ]
    report = {
        "baseline": args.baseline, "candidate": args.candidate, "n": len(baseline),
        "answer_sequence_exact": exact, "answer_sequence_exact_pct": exact / len(baseline) * 100,
        "generated_length_delta_median": statistics.median(length_deltas),
        "generated_length_delta_nonzero": sum(value != 0 for value in length_deltas),
        "caveat": "Sequence identity is diagnostic; task metric is quality ground truth.",
    }
    path = ROOT / "results" / args.candidate / f"output_drift_vs_{args.baseline}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
