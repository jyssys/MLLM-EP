"""Summarize serial and mixed late-bound state probes without copying tensors."""
import argparse
from collections import Counter
import json
from pathlib import Path

from ..mllm_epd_h200_prefill_heavy_batch_matched import analysis as shared
from .online_evidence import milliseconds, worker_records


def timing_evidence(proxy, tag, offer):
    row = proxy.get(offer["request_id"])
    if row is None:
        return None
    first = row.get("D_first_generated_token")
    reserved = row.get("D_admission", {}).get("reservation", {}).get("reserved")
    return {
        "cohort": tag, "D_coordinator_wait_ms": milliseconds(row.get("P_ready"), row.get("D_submit")),
        "D_reserved_to_first_token_host_ms": milliseconds(reserved, first),
        "P_cut_ready_to_first_token_host_ms": milliseconds(offer["boundary_ready"], first),
        "scope": "Instrumented correctness host landmarks. No benchmark latency qualification.",
    }


def case_evidence(proxy, tag, case):
    offer = case.get("offer", {})
    numeric = case.get("numerical_tolerance_checks", [])
    checks = case.get("checks", {})
    return {
        "request_id": case["request_id"], "case": tag, "ran_on_D": offer.get("accepted"),
        "reason": offer.get("reason", case.get("reason")), "prefix_tokens": offer.get("prefix_tokens"),
        "requested_prefix_tokens": case.get("requested_prefix_tokens", case.get("prefix_tokens")),
        "full_prompt_tokens": offer.get("full_prompt_tokens", case.get("full_prompt_tokens")),
        "passed_cache_and_coverage": checks.get("passed", case["passed"]),
        "exact_greedy_tokens": case["exact_greedy_tokens"],
        "exact_prefix_import": case.get("exact_prefix_import"),
        "exact_imported_prefix_layers": checks.get("exact_imported_prefix_layers"),
        "exact_completed_prompt_layers": checks.get("exact_d_final_prompt_layers", checks.get("exact_p_final_prompt_layers")),
        "original_fields": checks.get("original_fields"),
        "teacher_output_tokens": case.get("teacher_output_tokens"),
        "observed_output_tokens": case.get("observed_output_tokens"),
        "P_query_coverage": case.get("P_query_coverage"),
        "D_query_coverage": case.get("D_query_coverage"),
        "failed_layer_segments": sum(not row["tolerance_pass"] for row in numeric) if numeric else None,
        "max_relative_L2": max((row["relative_L2"] for row in numeric), default=None),
        "P_preemptions_through_cut": offer.get("prefill_recomputations"),
        "decision_host_ms": offer.get("decision_ms"),
        "natural_boundary": case.get("natural_boundary"),
        "native_path_completed": case.get("native_path_completed"),
        "natural_skips": case.get("natural_skips"),
        "held_before_retirement_ms": offer.get("held_before_retirement_ms"),
        "held_schedule_attempts": offer.get("held_schedule_attempts"),
        "time_scope": timing_evidence(proxy, tag, offer) if offer else None,
    }


def decode_histogram(run, tag):
    path = run / "raw/batch_prefix" / f"{tag}_D" / "manifest.json"
    if not path.is_file():
        return None
    data = shared.read(path)
    steps = Counter()
    for event in data["events"]:
        count = sum(row["phase"] == "decode" for row in event["requests"])
        if count:
            steps[count] += 1
    return {
        "decode_request_count_per_forward": dict(sorted(steps.items())),
        "decode_containing_steps": sum(steps.values()),
        "step_weighted_mean": (sum(size * count for size, count in steps.items()) / sum(steps.values()))
                              if steps else None,
        "single_decode_request_fraction": steps[1] / sum(steps.values()) if steps else None,
        "raw_manifest_sha256": shared.digest(path),
        "scope": "One count per actual prepared forward containing any decode row. "
                 "A mixed prefill/decode forward counts only the decode requests. Correctness instrumentation.",
    }


def actual_serial_prefill_chunks(run, tag):
    path = run / "raw/batch_prefix" / f"{tag}_P/manifest.json"
    if path.is_file():
        manifest = shared.read(path)
        rows = [row for event in manifest["events"] for row in event["requests"] if row["phase"] == "prefill"]
        chunks = [[row["start"], row["start"] + row["query_tokens"]] for row in rows]
        return {"chunks": chunks, "manifest_sha256": shared.digest(path)}
    path = run / "raw/llm_reference" / f"{tag}_P/manifest.json"
    if path.is_file():
        manifest = shared.read(path)
        chunks = [[row["start"], row["start"] + row["query_tokens"]] for row in manifest["events"]
                  if row["start"] < row["prompt_tokens"]]
        return {"chunks": chunks, "manifest_sha256": shared.digest(path)}
    return None


def paired_geometry_evidence(run):
    path = run / "analysis/geometry_control.json"
    if not path.is_file():
        return []
    controls = shared.read(path)
    result = []
    for case in controls["cases"]:
        rows = case.get("cases", [])
        tag = case["case"]
        result.append({
            "case": tag, "ratio": case["ratio"], "evidence": case["evidence"], "passed": case["passed"],
            "actual_P_D_concurrency": case.get("actual_P_D_concurrency"),
            "identical_E_count": case.get("verified_same_E"),
            "split_count": case.get("accepted"),
            "continue_count": case.get("continued"),
            "no_offer_count": case.get("no_offer_continuations"),
            "native_stock_cut_count": sum(row.get("cut_source") == "OBSERVED_NATIVE_CHUNK" for row in rows),
            "paired_KV_tolerance_pass_count": sum(row["late_vs_native_tolerance_pass"] for row in rows),
            "paired_exact_greedy_count": sum(row["late_vs_native_exact_greedy"] for row in rows),
            "paired_state_and_greedy_pass_count": sum(row["passed"] for row in rows),
            "lead_boundary_control": case.get("lead_boundary_control"),
            "lead_shared_decode": case.get("lead_shared_decode"),
            "decode_geometry_native": decode_histogram(run, tag + "_native"),
            "decode_geometry_late": decode_histogram(run, tag + "_late"),
            "raw_receipt_sha256": shared.digest(path),
        })
    return result


def run_receipt(run):
    serial, mixed = [], []
    proxy_path = run / "raw/proxy_epd.jsonl"
    proxy = ({row["runtime_request_id"]: row for row in shared.jsonl(proxy_path, appending=True)
              if row.get("status") == "MEASURED_ONLINE"} if proxy_path.is_file() else {})
    for path in sorted((run / "analysis").glob("*_late_correctness.json")):
        source = shared.read(path)
        tag = path.name.removesuffix("_late_correctness.json")
        ident = {"case": tag, "evidence": source["evidence"], "passed": source["passed"],
                 "raw_receipt_sha256": shared.digest(path)}
        if "cases" not in source:
            serial.append(ident | case_evidence(proxy, tag, source) | {
                "route": source["route"], "actual_prefill_chunks": actual_serial_prefill_chunks(run, tag)
            })
            continue
        items = [case_evidence(proxy, tag, row) for row in source["cases"]]
        mixed.append(ident | {
            "split_count": source["accepted"], "continued_count": source["continued"],
            "full_cache_and_coverage_pass_count": sum(row["passed_cache_and_coverage"] for row in items),
            "exact_greedy_count": sum(row["exact_greedy_tokens"] for row in items), "requests": items,
            "decode_geometry": decode_histogram(run, tag),
            "scope": source["scope"],
        })
    natural_result = run / "analysis/natural_correctness.json"
    if natural_result.is_file():
        observed = worker_records(run)
        recorded = {row["case"] for row in serial}
        audit = shared.read(natural_result)
        for case in audit["cases"]:
            tag = case["case"]
            if tag in recorded:
                continue
            offers = [row for row in observed[f"{tag}_0000"] if "accepted" in row]
            serial.append({
                "case": tag, "evidence": case["evidence"], "passed": case["passed"],
                "passed_cache_and_coverage": False,
                "ran_on_D": any(row["accepted"] for row in offers),
                "raw_receipt_sha256": shared.digest(natural_result), "error": case.get("error"),
                "actual_prefill_chunks": actual_serial_prefill_chunks(run, tag),
                "natural_boundary": "hold" if audit["hold"] else "nohold",
            })
    return {"start": run.name, "serial": serial, "mixed": mixed, "paired_geometry": paired_geometry_evidence(run),
            "execution_failure": shared.read(run / "state/failure.json")
            if (run / "state/failure.json").is_file() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = {"runs": [run_receipt(run) for run in args.runs],
             "scope": "Serial exactness and mixed correctness probes only. No timing or online quality claim."}
    args.output.write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
