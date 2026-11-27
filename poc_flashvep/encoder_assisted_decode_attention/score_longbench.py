"""Score measured text controls with the pinned publisher's LongBench evaluator."""
import argparse
from collections import defaultdict
import hashlib
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

from .longbench import TASKS


EVAL_REVISION = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
EVAL_FILES = {
    "eval.py": "1a3acfc25d9b053e9bb75c479f7e385d0cb9989f0f7115346b7d632655967721",
    "metrics.py": "e22e2a2662e0f7e683137fa3541f64edb6a801e9138d16d2f3459a6ab9941323",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publisher_evaluator(path):
    for name, expected in EVAL_FILES.items():
        if digest(path / "official" / name) != expected:
            raise ValueError(f"Unexpected publisher evaluator source in {name}")
    sys.path[:0] = [str(path / "official"), str(path / "deps")]
    return importlib.import_module("eval").scorer


def arm_scores(run_name, observations, reference, tokenizer, scorer):
    tasks = defaultdict(list)
    for row in observations:
        original = reference[row["id"]]
        if row["source"] != original["source"] or row["image_count"] != 0:
            raise ValueError("Measured request metadata differs from its frozen text input")
        ids = row["output_ids"]
        prediction = tokenizer.decode(ids, skip_special_tokens=True)
        tasks[row["source"]].append((row, prediction, original["reference_answers"]))
    per_task = {}
    per_request = {}
    for task in TASKS:
        measured = tasks[task]
        if len(measured) != 32:
            raise ValueError(f"Expected 32 arrivals in {task} for {run_name}")
        scores = [scorer(task, [prediction], [answers], None) for _, prediction, answers in measured]
        per_task[task] = {
            "metric": "ROUGE-L" if task == "gov_report" else "QA F1",
            "official_score_0_to_100": scorer(task, [item[1] for item in measured],
                                              [item[2] for item in measured], None),
            "requests": len(measured),
            "eos": sum(row["outcome"] == "eos" for row, _, _ in measured),
            "cap": sum(row["outcome"] == "cap" for row, _, _ in measured),
        }
        per_request.update({row["id"]: score for (row, _, _), score in zip(measured, scores, strict=True)})
    return {"by_task": per_task, "by_request_score_0_to_100": per_request}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--assisted", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scorer = publisher_evaluator(args.scorer.resolve())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    reference = {row["request_id"]: row for row in json.loads(args.manifest.read_text())["requests"]}
    assert len(reference) == 128
    input_paths = {"local": args.baseline, "prefill_assisted": args.assisted}
    arms = {name: json.loads(path.read_text()) for name, path in input_paths.items()}
    for name, result in arms.items():
        ids = [row["id"] for row in result["requests"]]
        if result["workload"] != "real-text" or len(ids) != 128 or set(ids) != set(reference):
            raise ValueError(f"Incorrect frozen LongBench roster for {name}")
    local, assisted = [result["requests"] for result in arms.values()]
    for base, candidate in zip(local, assisted, strict=True):
        if base["id"] != candidate["id"]:
            raise ValueError("Local and assisted LongBench arrival rosters differ")
    output = {
        "official_evaluator_revision": EVAL_REVISION,
        "official_source_sha256": EVAL_FILES,
        "dependency_versions": {name: version(name) for name in ("rouge", "jieba", "fuzzywuzzy", "six")},
        "manifest_sha256": digest(args.manifest),
        "tokenizer_sha256": {name: digest(args.model / name) for name in ("tokenizer.json", "tokenizer_config.json")},
        "cohort_sha256": {name: digest(path) for name, path in input_paths.items()},
        "served_generation_scope": "Natural EOS, shared experiment cap, full response text. "
                                   "Frozen 128-request subset, with publisher metrics.",
        "exact_output_tokens": sum(bool(base["output_ids"]) and base["output_ids"] == candidate["output_ids"]
                                   for base, candidate in zip(local, assisted, strict=True)),
        "arms": {name: arm_scores(name, result["requests"], reference, tokenizer, scorer)
                 for name, result in arms.items()},
    }
    with args.out.open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
