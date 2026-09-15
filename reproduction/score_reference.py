"""Score captured official generations; do not confound parser and model errors."""

import argparse
import csv
import json
import math
import re
import statistics
from decimal import Decimal, InvalidOperation
from pathlib import Path


def wilson(correct, total, z=1.959963984540054):
    p = correct / total
    d = 1 + z * z / total
    middle = (p + z * z / (2 * total)) / d
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [middle - radius, middle + radius]


def visible_final(text):
    patterns = (
        r"####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)",
        r"\\boxed\{\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)\s*\}",
        r"(?:final answer|the answer is|answer:)\s*[^\n\d-]*([-+]?\$?\d[\d,]*(?:\.\d+)?)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.I)
        if matches:
            return matches[-1].replace("$", "").replace(",", "")
    return None


def numeric_equal(answer, gold):
    """Diagnostic: string-score failures that denote the same exact number."""
    if answer is None or gold is None:
        return False
    try:
        return Decimal(answer) == Decimal(gold)
    except InvalidOperation:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--parser-csv", type=Path)
    parser.add_argument("--manual-review", type=Path)
    parser.add_argument("--sample-id-max", type=int)
    parser.add_argument("--combined-output", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for path in args.inputs for line in path.read_text().splitlines() if line.strip()]
    if args.sample_id_max is not None:
        records = [row for row in records if row["sample_id"] <= args.sample_id_max]
    records.sort(key=lambda row: row["sample_id"])
    ids = [row["sample_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate sample IDs across captured workers")
    if not records:
        raise RuntimeError("no generated samples")
    if args.combined_output:
        args.combined_output.parent.mkdir(parents=True, exist_ok=True)
        args.combined_output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    task = records[0]["task"]
    if any(row["task"] != task for row in records):
        raise RuntimeError("mixed tasks")
    classified = []
    manual = {}
    if args.manual_review:
        with args.manual_review.open(newline="") as stream:
            for review in csv.DictReader(stream):
                sample_id = int(review["sample_id"])
                if sample_id in manual:
                    raise RuntimeError(f"duplicate manual review: {sample_id}")
                manual[sample_id] = review
    if task == "gsm8k":
        for row in records:
            text = row["postprocessed_generation"]
            human = visible_final(text)
            predicted = row["parsed_answer"]
            gold = row["ground_truth"]
            if not text.strip():
                category = "EMPTY_OUTPUT"
            elif row["termination_reason"] in ("GEN_LENGTH_CAP", "REMAINING_MASK") and human is None:
                category = "TRUNCATED"
            elif human is not None and human != predicted:
                category = "PARSER_WRONG"
            elif predicted != gold and numeric_equal(predicted, gold):
                category = "PARSER_WRONG"
            elif predicted is None:
                category = "FORMAT_MISMATCH"
            elif predicted != gold:
                category = "MODEL_WRONG"
            else:
                category = "CORRECT"
            classified.append({
                "sample_id": row["sample_id"], "raw_generation": row["raw_generation"],
                "human_visible_final_answer": human,
                "parser_extracted_answer": predicted, "ground_truth": gold,
                "numeric_equivalent_to_gold": numeric_equal(predicted, gold),
                "termination_reason": row["termination_reason"], "category": category,
                "needs_manual_review": human is None or category in ("TRUNCATED", "PARSER_WRONG", "FORMAT_MISMATCH"),
                "manual_checked": row["sample_id"] in manual,
                "manual_human_answer": manual[row["sample_id"]]["human_answer"] if row["sample_id"] in manual else None,
                "manual_category": manual[row["sample_id"]]["manual_category"] if row["sample_id"] in manual else None,
            })
    correct = sum(bool(row["correct"]) for row in records) if task == "gsm8k" else None
    nfe = [row["nfe"] for row in records]
    lengths = [row["generation_tokens"] for row in records]
    summary = {
        "task": task, "n": len(records), "sample_ids": [ids[0], ids[-1]],
        "correct": correct, "accuracy": correct / len(records) if correct is not None else None,
        "wilson_95_accuracy": wilson(correct, len(records)) if correct is not None else None,
        "mean_nfe": statistics.mean(nfe), "median_nfe": statistics.median(nfe),
        "mean_generation_tokens": statistics.mean(lengths),
        "median_generation_tokens": statistics.median(lengths),
        "termination_counts": {name: sum(row["termination_reason"] == name for row in records)
                               for name in ("EOS", "GEN_LENGTH_CAP", "REMAINING_MASK")},
        "parser_no_number": sum(row["parsed_answer"] is None for row in records),
        "numeric_equivalent_string_failures": sum(not row["correct"] and numeric_equal(row["parsed_answer"], row["ground_truth"]) for row in records) if task == "gsm8k" else None,
        "empty_output": sum(not row["postprocessed_generation"].strip() for row in records),
        "category_counts": {name: sum(row["category"] == name for row in classified)
                            for name in ("CORRECT", "MODEL_WRONG", "PARSER_WRONG", "TRUNCATED", "EMPTY_OUTPUT", "FORMAT_MISMATCH", "OTHER")},
        "classification_caveat": "Automatic triage only. Human-visible answers and all parser errors require manual audit before final quality verdict.",
        "manual_reviewed_n": sum(row["manual_checked"] for row in classified),
        "manual_reviewed_correct": sum(row["manual_human_answer"] == row["ground_truth"]
                                       for row in classified if row["manual_checked"]),
        "manual_parser_wrong": sum(row["manual_category"] == "PARSER_WRONG" for row in classified),
        "manual_truncated": sum(row["manual_category"] == "TRUNCATED" for row in classified),
        "manual_model_wrong": sum(row["manual_category"] == "MODEL_WRONG" for row in classified),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    if args.parser_csv and task == "gsm8k":
        args.parser_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.parser_csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(classified[0]))
            writer.writeheader()
            writer.writerows(classified)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
