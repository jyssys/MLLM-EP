"""Reduce saved natural EOS text with pinned dataset-origin scorer functions."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from ..mllm_epd_prefill_heavy_token_split import quality_audit
from ..mllm_epd_prefill_heavy_token_split.quality_floor_diagnostic import normalized


SOURCE_PINS = {
    "MPDocVQA_metrics.py.json": (
        "rubenpt91/MP-DocVQA-Framework", "7939f521fe5a82a442f54831b1c0e96daab331d9", "metrics.py",
        "481fe17e8d443a6dcf852b8348fcf06b7973ac1e99336f230a749008e992bf70",
    ),
    "Mantis_mantis_benchmark_eval.py.json": (
        "TIGER-AI-Lab/Mantis", "f3a319202d882a1794d28440b77bba7c58869a9e", "mantis/benchmark/eval.py",
        "caaab22f215462371189a35b86eb6ab011ad6f4b6934f73fa09e0818fd2ef729",
    ),
    "MMLongBench-Doc_eval_eval_score.py.json": (
        "mayubo2333/MMLongBench-Doc", "d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384", "eval/eval_score.py",
        "cafa268358b08521cfc9a805c67d86732d943677e4408c5751f336123d09631f",
    ),
    "MMLongBench-Doc_eval_extract_answer.py.json": (
        "mayubo2333/MMLongBench-Doc", "d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384", "eval/extract_answer.py",
        "fa6a747bd6a8e96fb6884eb7201663268a5df700ad71d85db3bc5e04a157722f",
    ),
}
SCOPES = {
    "lmms-lab/DocVQA": "Whole-response ANLS via the pinned closely related author framework",
    "lmms-lab/MP-DocVQA": "Whole-response ANLS and exact accuracy via the benchmark authors' framework",
    "TIGER-Lab/Mantis-Eval": "Whole-response parsing and correctness via benchmark authors' get_prediction",
    "yubo2333/MMLongBench-Doc": "Downstream score on raw response as DIAGNOSTIC, official GPT extraction NOT_RUN",
}


def open_json(path):
    raw = path.read_bytes()
    return json.loads(raw), sha256(raw).hexdigest()


def load_scorer(campaign):
    provenance = {}
    for filename, (repository, revision, path, expected_digest) in SOURCE_PINS.items():
        reference = campaign / "references" / "quality_sources" / filename
        record, file_digest = open_json(reference)
        actual_digest = sha256(record["source_text"].encode()).hexdigest()
        if (actual_digest != expected_digest or record["repository"] != repository
                or record.get("source_SHA", record.get("revision")) != revision):
            raise ValueError(f"Unexpected scorer revision or source bytes in {reference}")
        provenance[filename] = {
            "url": f"https://github.com/{repository}/blob/{revision}/{path}",
            "source_sha256": actual_digest,
            "receipt_sha256": file_digest,
        }
    original = quality_audit.ROOT
    try:
        quality_audit.ROOT = campaign
        scorer = quality_audit.scorers()
    finally:
        quality_audit.ROOT = original
    return scorer, provenance


def read_predictions(path, manifest):
    raw = path.read_bytes()
    by_id = {}
    for line in raw.splitlines():
        prediction = json.loads(line)
        key = prediction["request_id"]
        if key not in manifest or key in by_id:
            raise ValueError(f"Unexpected or duplicated request {key} in {path}")
        expected = manifest[key]
        if any(prediction[k] != expected[k] for k in ("source", "prompt_SHA256")):
            raise ValueError(f"Prediction and original prompt differ for {key}")
        if prediction["input_tokens_manifest"] != expected["total_input_tokens"]:
            raise ValueError(f"Prediction and original prompt geometry differ for {key}")
        by_id[key] = prediction
    return by_id, sha256(raw).hexdigest()


def score_manifest(manifest, predictions, scorer):
    groups = defaultdict(list)
    for key, expected in manifest.items():
        source = expected["source"]
        targets = expected["reference_answers"]
        if source not in SCOPES or not targets:
            raise ValueError(f"Missing original scoring contract for {key}")
        item = predictions.get(key, {})
        status = item.get("status", "MISSING_RESULT")
        succeeded = status == "MEASURED_ONLINE"
        record = {
            "request_id": key, "source": source, "status": status,
            "prompt_sha256": expected["prompt_SHA256"], "finish_reason": item.get("finish_reason"),
            "question_type": expected["metadata"].get("question_type"),
        }
        if succeeded:
            value = quality_audit.score_request(expected, item["text"], scorer)
            record.update(score=value["score"], exact_accuracy=value.get("accuracy"),
                          output_sha256=sha256(item["text"].encode()).hexdigest())
            if source in ("lmms-lab/DocVQA", "lmms-lab/MP-DocVQA"):
                response = f" {normalized(item['text'])} "
                tokens = [normalized(target) for target in targets]
                record["reference_token_sequence_present_diagnostic"] = any(
                    token and f" {token} " in response for token in tokens)
        groups[source].append(record)
    summary = {}
    for source, records in groups.items():
        scores = [row["score"] for row in records if row.get("score") is not None]
        exact = [row["exact_accuracy"] for row in records if row.get("exact_accuracy") is not None]
        summary[source] = {
            "requests": len(records), "status": dict(Counter(row["status"] for row in records)),
            "finish_reasons": dict(Counter(row["finish_reason"] or "missing" for row in records)),
            "scored": len(scores), "mean_on_scored": mean(scores) if scores else None,
            "mean_exact_accuracy_on_scored": mean(exact) if exact else None,
            "scoreable_request_metric": SCOPES[source],
            "null_scores": len(records) - len(scores),
        }
        if source == "yubo2333/MMLongBench-Doc":
            summary[source]["official_answer_extraction"] = "NOT_RUN"
        if source in ("lmms-lab/DocVQA", "lmms-lab/MP-DocVQA"):
            summary[source]["reference_token_sequence_present_diagnostic"] = sum(
                row.get("reference_token_sequence_present_diagnostic", False) for row in records)
        if source == "TIGER-Lab/Mantis-Eval":
            types = Counter(row["question_type"] for row in records)
            summary[source]["question_types"] = dict(types)
    return summary, [row for group in groups.values() for row in group]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--scorer-campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    roster, manifest_digest = open_json(args.manifest)
    config, config_digest = open_json(args.run_config)
    manifest = {row["request_id"]: row for row in roster["requests"]}
    if (config["output_mode"] != "Natural EOS" or config["output_count"] is not None
            or config["count"] != len(manifest) or len(manifest) != len(roster["requests"])):
        raise ValueError("Use an unchanged natural EOS cohort with a unique manifest and matching run config")
    results, raw_digest = read_predictions(args.predictions, manifest)
    scorer, provenance = load_scorer(args.scorer_campaign)
    summary, requests = score_manifest(manifest, results, scorer)
    document = {
        "evidence": "DIAGNOSTIC", "overall_task_noninferiority": "NOT_RUN",
        "scope": "Saved natural-EOS Native EPD responses on a fixed subset. Separate task metrics. "
                 "Prediction text and original answers stay unchanged. No external API is used.",
        "manifest_sha256": manifest_digest, "raw_predictions_sha256": raw_digest,
        "run_config_sha256": config_digest, "scorers": provenance, "summary": summary, "requests": requests,
    }
    args.out.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
