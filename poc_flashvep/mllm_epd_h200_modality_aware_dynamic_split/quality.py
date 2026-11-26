"""Score verified Natural EOS cohorts with the pinned local quality definitions."""
import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from statistics import mean

from ..mllm_epd_h200_prefill_heavy_batch_matched import analysis as shared
from ..mllm_epd_h200_wide_decode_batch_token_split.quality import FROZEN_SHA, SOURCE_ROOT, require
from ..mllm_epd_prefill_heavy_token_split import quality_audit
from .online_evidence import DynamicCohort


def compare(runs, root, output, *, recovery_comparisons=False):
    manifest = root / "data/main100_manifest_template_v2.json"
    require(shared.digest(manifest) == FROZEN_SHA, "Frozen Main100 hash mismatch")
    frozen = {row["request_id"]: row for row in shared.read(manifest)["requests"]}
    require(len(frozen) == 100, "Incomplete Main100 manifest")
    quality_audit.ROOT = root
    scorers = quality_audit.scorers()
    cohorts, excluded = shared.cohort_records(runs, DynamicCohort)
    groups = defaultdict(list)
    for cohort in cohorts:
        if cohort.observation == "clean" and cohort.cycles == 1 and cohort.config["output_mode"] == "Natural EOS":
            groups[cohort.match].append(cohort)
    details, tasks = [], []
    native_prefixes = ("native-common", "native-strong", "native-matched") if recovery_comparisons else (
        "native-common", "native-strong",
    )
    modality_prefixes = ("hard", "soft", "c-new") if recovery_comparisons else ("hard", "soft")
    generic_prefixes = ("generic", "b-new") if recovery_comparisons else ("generic",)
    for arms in groups.values():
        if len(arms) < 2:
            continue
        scored = {}
        for cohort in arms:
            values = {}
            for item in cohort.client_rows:
                reference = frozen[item["request_id"]]
                require(item["prompt_SHA256"] == reference["prompt_SHA256"]
                        and item["input_tokens_manifest"] == reference["total_input_tokens"]
                        and len(item["token_ids"]) == item["output_tokens"], "Output ledger mismatch")
                values[item["request_id"]] = (item, quality_audit.score_request(reference, item["text"], scorers))
            scored[(cohort.run.name, cohort.tag)] = values
        for candidate in arms:
            references = [arm for arm in arms if arm.run != candidate.run and (
                arm.method.startswith(native_prefixes)
                or candidate.method.startswith(modality_prefixes) and arm.method.startswith(generic_prefixes))]
            for baseline in references:
                candidate_rows = scored[(candidate.run.name, candidate.tag)]
                baseline_rows = scored[(baseline.run.name, baseline.tag)]
                require(candidate_rows.keys() == baseline_rows.keys(), "Quality comparison rosters differ")
                start = len(details)
                for request_id, (a, a_score) in candidate_rows.items():
                    b, b_score = baseline_rows[request_id]
                    require(a_score["metric"] == b_score["metric"] and a["index"] == b["index"],
                            "Quality comparison changed scorer or arrival ordering")
                    details.append({**candidate.base, "baseline_start": baseline.run.name,
                                    "baseline_method": baseline.method, "request_id": request_id,
                                    "source": a["source"], "metric": a_score["metric"],
                                    "complete_task_scorer": a_score["complete_task_scorer"],
                                    "candidate_score": a_score["score"], "baseline_score": b_score["score"],
                                    "candidate_output_tokens": a["output_tokens"], "baseline_output_tokens": b["output_tokens"],
                                    "exact_token_identity": a["token_ids"] == b["token_ids"]})
                by_task = defaultdict(list)
                for row in details[start:]:
                    by_task[row["source"]].append(row)
                for task, rows in sorted(by_task.items()):
                    score_deltas = [float(row["candidate_score"]) - float(row["baseline_score"])
                                    for row in rows if row["candidate_score"] is not None and row["baseline_score"] is not None]
                    tasks.append({**candidate.base, "baseline_start": baseline.run.name, "baseline_method": baseline.method,
                                  "source": task, "requests": len(rows), "paired_scored": len(score_deltas),
                                  "same_output_tokens": sum(row["exact_token_identity"] for row in rows),
                                  "point_delta_percentage_points": 100 * mean(score_deltas) if score_deltas else None,
                                  "complete_task_scorer": rows[0]["complete_task_scorer"]})
    output.mkdir(parents=True, exist_ok=True)
    shared.csv_file(output / "dynamic_quality_requests.csv", details)
    shared.csv_file(output / "dynamic_quality_by_task.csv", tasks)
    receipt = {
        "evidence": "DIAGNOSTIC" if details else "NOT_MEASURED",
        "request_pairs": len(details), "task_rows": len(tasks), "excluded_cohorts": excluded,
        "paired_independent_start_quality_CI": "NOT_MEASURED",
        "official_MMLongBench_extraction": "NOT_RUN",
        "source_manifest_sha256": FROZEN_SHA,
        "scorer_provenance": {name: data | {"receipt": Path(data["receipt"]).name}
                              for name, data in scorers[3].items()},
        "scope": "Natural EOS outputs with verified arrival/manifest/runtime/lease matching and pinned local scorers. "
                 "Only point effects are reported. The MMLong RAW scorer is diagnostic. No answers/references exported.",
    }
    (output / "dynamic_quality_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--recovery-comparisons", action="store_true")
    args = parser.parse_args()
    with redirect_stdout(io.StringIO()):
        compare(args.runs, args.source_root, args.output, recovery_comparisons=args.recovery_comparisons)


if __name__ == "__main__":
    main()
