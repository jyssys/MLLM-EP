"""Score completed H5 pilots offline without releasing answers or references."""

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
from statistics import mean

from ..mllm_epd_prefill_heavy_token_split import quality_audit


FROZEN_SHA = "910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1"
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "reports/mllm_epd_prefill_heavy_token_split_20261124"
CAPS = ("free", "b4")
METHODS = ("native_common", "native_strong", "static10", "idle10")


def read(path):
    return json.loads(path.read_text())


def require(condition, description):
    if not condition:
        raise ValueError(description)


def load_cohort(runs, cap, method, roster):
    start = runs / f"pilot_{cap}_{method}"
    tag = f"{method}_{cap}_5"
    receipt = read(start / "state" / f"{tag}_complete.json")
    metrics = read(start / "analysis" / f"{tag}_metrics.json")
    config = read(start / "configs" / f"{tag}.json")
    journal = start / "raw" / f"{tag}_completion_journal.jsonl"
    journal_bytes = journal.read_bytes()
    journal_hash = hashlib.sha256(journal_bytes).hexdigest()
    scope = receipt["client_evidence_policy"]
    require(receipt["passed"] and metrics["primary_execution_valid"]
            and metrics["completed_count"] == scope["full_rows"] == len(roster)
            and metrics["raw_SHA256"] == receipt["raw_SHA256"]
            and scope["mode"] == "deferred_full_trace"
            and scope["journal_SHA256"] == journal_hash,
            f"{start.name} invalid completion or journal receipt")
    require(config["output_mode"] == "Natural EOS" and config["QPS"] == 5
            and config["workload"]["subset"] == "all" and not receipt["serial"]
            and config["workload"]["paired_start_id"] == f"pilot_{cap}",
            f"{start.name} is not the clean Main100 pilot")
    rows = [json.loads(line) for line in journal_bytes.splitlines()]
    by_id = {row["request_id"]: row for row in rows}
    require(len(rows) == len(roster) and by_id.keys() == roster.keys(), f"{start.name} roster mismatch")
    for request_id, observed in by_id.items():
        original = roster[request_id]
        require(observed["status"] == "MEASURED_ONLINE"
                and observed["prompt_SHA256"] == original["prompt_SHA256"]
                and observed["input_tokens_manifest"] == original["total_input_tokens"]
                and observed["source"] == original["source"]
                and isinstance(observed["text"], str)
                and len(observed["token_ids"]) == observed["output_tokens"],
                f"{start.name} identity, input, or token ledger differs for {request_id}")
    require(sum(row["output_tokens"] for row in rows) == metrics["total_output_tokens"],
            f"{start.name} clean metric output total differs")
    return by_id, {
        "run": start.name, "cohort": tag, "requests": len(rows), "output_tokens": metrics["total_output_tokens"],
        "completion_journal_SHA256": journal_hash, "raw_trace_SHA256": receipt["raw_SHA256"],
        "arrival_SHA256": config["arrival_SHA256"], "derived_manifest_SHA256": config["workload"]["manifest_SHA256"],
    }


def evaluate(runs, source_root):
    frozen = source_root / "data/main100_manifest_template_v2.json"
    require(hashlib.sha256(frozen.read_bytes()).hexdigest() == FROZEN_SHA, "Frozen Main100 hash mismatch")
    manifest = read(frozen)["requests"]
    roster = {row["request_id"]: row for row in manifest}
    require(len(manifest) == len(roster) == 100, "Frozen Main100 roster is incomplete")
    quality_audit.ROOT = source_root
    scorers = quality_audit.scorers()
    request_rows, task_rows, evidence = [], [], []
    for cap in CAPS:
        baseline, baseline_receipt = load_cohort(runs, cap, "native_common", roster)
        reference_scores = {
            request_id: quality_audit.score_request(request, baseline[request_id]["text"], scorers)
            for request_id, request in roster.items()
        }
        for method in METHODS:
            if method == "native_common":
                observed, receipt = baseline, baseline_receipt
            else:
                observed, receipt = load_cohort(runs, cap, method, roster)
            require(all(receipt[field] == baseline_receipt[field] for field in
                        ("arrival_SHA256", "derived_manifest_SHA256")), f"{receipt['run']} paired contract mismatch")
            evidence.append(receipt)
            groups = defaultdict(list)
            for request_id, request in roster.items():
                item, ref = observed[request_id], baseline[request_id]
                # Per-cohort media UUIDs deliberately vary full-payload hashes;
                # the exact prompt hash and frozen derived manifest are checked above.
                require(item["index"] == ref["index"],
                        f"{receipt['run']} arrival order changed for {request_id}")
                value = (reference_scores[request_id] if method == "native_common" else
                         quality_audit.score_request(request, item["text"], scorers))
                score, baseline_score = value["score"], reference_scores[request_id]["score"]
                detail = {
                    "cap": cap, "method": method, "request_id": request_id, "task": request["source"],
                    "point_score": None if score is None else float(score),
                    "native_common_point_score": None if baseline_score is None else float(baseline_score),
                    "metric": value["metric"], "complete_task_scorer": value["complete_task_scorer"],
                    "official_full_pipeline": value.get("official_full_pipeline", "SCORED"),
                    "output_tokens": item["output_tokens"], "native_common_output_tokens": ref["output_tokens"],
                    "output_length_delta": item["output_tokens"] - ref["output_tokens"],
                    "output_length_equal": item["output_tokens"] == ref["output_tokens"],
                    "exact_output_token_ids_equal": item["token_ids"] == ref["token_ids"],
                }
                request_rows.append(detail)
                groups[request["source"]].append(detail)
            for task, details in sorted(groups.items()):
                point = [row["point_score"] for row in details if row["point_score"] is not None]
                paired = [row["point_score"] - row["native_common_point_score"] for row in details
                          if row["point_score"] is not None and row["native_common_point_score"] is not None]
                task_rows.append({
                    "cap": cap, "method": method, "task": task, "requests": len(details),
                    "scored": len(point), "point_mean": mean(point) if point else None,
                    "paired_scored": len(paired), "point_delta_vs_native_common_pp": 100 * mean(paired) if paired else None,
                    "metric": details[0]["metric"], "official_full_pipeline": details[0]["official_full_pipeline"],
                    "output_tokens": sum(row["output_tokens"] for row in details),
                    "output_length_equal_count": sum(row["output_length_equal"] for row in details),
                    "exact_output_token_ids_equal_count": sum(row["exact_output_token_ids_equal"] for row in details),
                })
    provenance = {
        name: {**data, "receipt": Path(data["receipt"]).name} for name, data in scorers[3].items()
    }
    return request_rows, task_rows, {
        "evidence": "TASK_SCORE_DIAGNOSTIC_N1", "manifest_SHA256": FROZEN_SHA,
        "cohorts": evidence, "scorer_provenance": provenance,
        "request_rows": len(request_rows), "per_task_rows": len(task_rows), "confidence_intervals": "NOT_MEASURED",
        "MMLongBench-Doc_official_extraction": "NOT_RUN",
        "scope": "Local pinned scorer definitions on unchanged Natural EOS completion text. "
                 "MMLongBench-Doc RAW is diagnostic because official extraction did not run. "
                 "Exact token identity and length are per request against same-cap NativeCommon. "
                 "Observed point differences from a single start cannot establish task quality noninferiority.",
    }


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    args = parser.parse_args()
    # The pinned MMLong code prints a list-length diagnostic on some rows.
    with redirect_stdout(io.StringIO()):
        requests, tasks, receipt = evaluate(args.runs, args.source_root)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "wide_pilot_quality_requests.csv", requests)
    write_csv(args.output / "wide_pilot_quality_by_task.csv", tasks)
    (args.output / "wide_pilot_quality.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
