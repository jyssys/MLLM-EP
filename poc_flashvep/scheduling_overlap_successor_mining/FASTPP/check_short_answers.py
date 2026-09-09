"""Sharp regression: greedy short-answer requests must terminate at HF EOS."""
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()
rows = [json.loads(line) for line in open(args.path)]
expected = ["42", "Paris"]
for row, reference in zip(rows[:2], expected):
    actual = row["output_text"].strip()
    print(json.dumps({"request_id": row["request_id"], "expected": reference,
                      "actual": actual, "pass": actual == reference}))
assert len(rows) >= 2
assert all(r["output_text"].strip() == e for r, e in zip(rows[:2], expected)), \
    "HF-correct short answers were continued instead of stopping"
