"""Export compact Native evidence with stable arrivals, timings and answer hashes."""

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
from shutil import copyfile

from .dataset import file_hash, read, require


STAGES = ("received", "E_submit", "E_complete", "P_submit", "P_ready", "D_submit",
          "D_first_SSE_bytes", "D_http_complete", "finished")
SCORE_KEYS = ("score", "metric", "complete_task_scorer", "official_full_pipeline")
LABELS = ("screen_1", "screen_2", "screen_3", "discovery200_native")


def sha_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def load_journal(run, tag, arrival, done, config, metrics):
    journal = run / "raw" / f"{tag}_completion_journal.jsonl"
    policy = done["client_evidence_policy"]
    require(file_hash(journal) == policy["journal_SHA256"] and metrics["raw_SHA256"] == done["raw_SHA256"],
            f"{tag} journal or raw provenance mismatch")
    require(done["passed"] and metrics["primary_execution_valid"] and config["output_mode"] == "Natural EOS",
            f"{tag} failed or changed output mode")
    journal_rows = [json.loads(line) for line in journal.read_text().splitlines()]
    planned = read(arrival)["trace"]
    require(len(journal_rows) == len(planned) == metrics["completed_count"] == policy["full_rows"],
            f"{tag} completion count mismatch")
    by_index = {row["index"]: row for row in journal_rows}
    for request in planned:
        row = by_index[request["index"]]
        require(row["request_id"] == request["request_id"] and row["status"] == "MEASURED_ONLINE" and
                len(row["token_times"]) == len(row["token_ids"]) == row["output_tokens"] and
                abs(row["scheduled_arrival"] - done["monotonic_start"] - request["offset_seconds"]) < 5e-6,
                f"{tag} scheduled identity or output ledger mismatch")
    require(sum(row["output_tokens"] for row in journal_rows) == metrics["total_output_tokens"],
            f"{tag} output aggregate mismatch")
    return journal_rows


def load_proxy(run):
    with (run / "raw/proxy_epd.jsonl").open() as stream:
        return {
            row["runtime_request_id"]: {key: row.get(key) for key in STAGES}
            for row in map(json.loads, stream) if row["status"] == "MEASURED_ONLINE"
        }


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_gzip(path, rows):
    # Fixed metadata ensures equivalent ledgers have identical compressed bytes.
    with path.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as archive:
            for row in rows:
                archive.write((json.dumps(row, separators=(",", ":")) + "\n").encode())


def export(run, run_manifest, grouped_manifest, output, native_analysis=None):
    output.mkdir(parents=True, exist_ok=True)
    arrivals = output / "arrival"
    arrivals.mkdir(exist_ok=True)
    configs = output / "config"
    configs.mkdir(exist_ok=True)
    load_config = run / "configs/frozen_native_load.json"
    copyfile(load_config, configs / load_config.name)
    env = read(run / "state/environment.json")
    require(env["manifest_SHA256"] == file_hash(run_manifest), "Execution manifest hash mismatch")
    original = {row["request_id"]: row for row in read(run_manifest)["requests"]}
    grouped = {row["request_id"]: row for row in read(grouped_manifest)["requests"]}
    require(original.keys() == grouped.keys()
            and all(row["prompt_SHA256"] == grouped[key]["prompt_SHA256"] for key, row in original.items()),
            "Confirmation manifest changes requests or prompts")
    proxy = load_proxy(run)
    natural = {}
    analysis_summary = None
    if native_analysis is not None:
        analysis_summary = read(native_analysis / "native_summary.json")
        require(analysis_summary["confirmation_manifest_SHA256"] == file_hash(grouped_manifest),
                "Native scorer analysis used another confirmation manifest")
        natural = {row["request_id"]: row for row in read(native_analysis / "native_requests.json")}
        require(natural.keys() == grouped.keys(), "Native scorer roster mismatch")

    cohort_receipts, requests, ledger = [], [], []
    for tag in LABELS:
        config_path = run / "configs" / f"{tag}.json"
        if not config_path.is_file():
            continue
        config = read(config_path)
        done = read(run / "state" / f"{tag}_complete.json")
        metrics = read(run / "analysis" / f"{tag}_metrics.json")
        arrival = run / "configs" / f"{tag}_arrival.json"
        require(file_hash(arrival) == config["arrival_SHA256"], f"{tag} arrival digest mismatch")
        copyfile(arrival, arrivals / arrival.name)
        copyfile(config_path, configs / config_path.name)
        journal_rows = load_journal(run, tag, arrival, done, config, metrics)
        if tag == "discovery200_native":
            require({row["request_id"] for row in journal_rows} == grouped.keys(), "Native full roster mismatch")
        nvml_path = run / "raw" / f"{tag}_nvml.json"
        nvml = read(nvml_path)
        ownership = (read(run / "state" / f"{tag}_ownership.json"))["state"]
        fold_counts = Counter(grouped[row["request_id"]]["study_split"] for row in journal_rows)
        cohort_receipts.append({
            "cohort": tag, "method": config["workload"]["method"], "output_mode": config["output_mode"],
            "seed": config["seed"], "order_seed": config["workload"]["order_seed"], "QPS": config["QPS"],
            "arrival_sha256": config["arrival_SHA256"], "config_sha256": file_hash(config_path),
            "journal_sha256": done["client_evidence_policy"]["journal_SHA256"],
            "monotonic_start": done["monotonic_start"], "metrics": metrics, "grouped_v2_counts": dict(fold_counts),
            "NVML_memory_samples": len(nvml["rows"]), "NVML_errors": nvml["errors"],
            "peak_sampled_memory_bytes": {
                role: max((sample["roles"][role]["memory_used_bytes"] for sample in nvml["rows"]), default=None)
                for role in ("encoder", "prefill", "decode")
            },
            "EC_completion_ownership": {key: ownership[key] for key in
                                        ("live_images", "live_storage_bytes", "peak_storage_bytes",
                                         "outstanding_exports", "saved_count", "released_count")},
        })
        for item in sorted(journal_rows, key=lambda row: row["index"]):
            request_id = item["request_id"]
            data = grouped[request_id]
            require(data["prompt_SHA256"] == item["prompt_SHA256"] and
                    data["total_input_tokens"] == item["input_tokens_manifest"] == item["usage"]["prompt_tokens"],
                    f"{tag} prompt or input geometry mismatch for {request_id}")
            timing = proxy[item["runtime_request_id"]]
            result = natural.get(request_id, {}) if tag == "discovery200_native" else {}
            if result:
                require(item["text"] == result["answer"] and item["output_tokens"] == result["output_tokens"],
                        f"Native scorer answer or token count changed for {request_id}")
            score = result.get("score", {})
            requests.append({
                "cohort": tag, "index": item["index"], "request_id": request_id, "source": data["source"],
                "benchmark": data["study_source"], "grouped_v2_fold": data["study_split"], "image_count": data["image_count"],
                "input_tokens": data["total_input_tokens"], "visual_tokens": data["visual_tokens"],
                "text_tokens": data["text_tokens"], "special_tokens": data["special_tokens"], "status": item["status"],
                "output_tokens": item["output_tokens"], "finish_reason": item["finish_reason"],
                "output_cap_hit": item["output_cap_hit"], "answer_sha256": sha_text(item["text"]),
                "output_token_ids_sha256": sha_text(json.dumps(item["token_ids"], separators=(",", ":"))),
                "TTFT_ms": item["TTFT_ms"], "ITL_count": len(item["ITL_ms"]),
                "ITL_max_ms": max(item["ITL_ms"], default=None), "TPOT_ms": item["TPOT_ms"], "E2E_ms": item["E2E_ms"],
                "exported_prompt_block_ids": result.get("exported_prompt_block_ids"),
                **{key: score.get(key) for key in SCORE_KEYS},
            })
            ledger.append({
                "cohort": tag, "index": item["index"], "request_id": request_id,
                "scheduled_arrival": item["scheduled_arrival"], "client_dispatch": item["dispatch"],
                "client_finish": item["finish"], "client_output_token_monotonic_times": item["token_times"],
                "coordinator_monotonic_times": timing,
            })
    require(cohort_receipts, "Completed Native evidence is unavailable")

    save_csv(output / "native_cohorts_requests.csv", requests)
    save_gzip(output / "native_timing_ledger.jsonl.gz", ledger)
    result = {
        "status": "MEASURED_ONLINE", "run": run.name, "load_config_sha256": file_hash(load_config),
        "load_policy": read(load_config), "execution_manifest_sha256": file_hash(run_manifest),
        "grouped_v2_manifest_sha256": file_hash(grouped_manifest), "upstream_asset_receipt_sha256": env["receipt_SHA256"],
        "base_head": env["base_head"], "runner_source_sha256": env["runner_SHA256"],
        "runtime_vllm_wheel_sha256": env["runtime"]["vllm_wheel_sha256"],
        "authorized_gpu_uuids": env["authorized_uuid_by_gpu"], "cohorts": cohort_receipts,
        "raw_and_serialization": "Original cohort configs and arrivals copied byte for byte. Verified journal supplies unchanged output token "
                                 "timestamps; coordinator records expose host boundaries. Only hashes of answers and token "
                                 "sequences are published, with per-task point scores where computed.",
        "timing_units": "Monotonic seconds in the compressed ledger; milliseconds in the request table.",
    }
    if analysis_summary:
        result["full_native_by_source"] = analysis_summary["by_source"]
        result["full_native_scorer_provenance"] = analysis_summary["scorer_provenance"]
    (output / "native_baseline_receipt.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--grouped-manifest", required=True, type=Path)
    parser.add_argument("--native-analysis", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export(args.run, args.run_manifest, args.grouped_manifest, args.output, args.native_analysis)


if __name__ == "__main__":
    main()
