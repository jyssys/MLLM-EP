"""Summarize real crossing descriptor use and staged correctness receipts."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def request_geometry(run, name, arm, index):
    identity = f"ratio_{name}_n8_{arm}"
    request_id = f"{identity}_{index:04d}"
    shapes = {}
    for role in ("P", "D"):
        manifest = read(run / "raw/batch_prefix" / f"{identity}_{role}" / "manifest.json")
        shapes[role] = [{"start": req["start"], "query_tokens": req["query_tokens"],
                         "actual_batch": event["actual_batch"], "query_rows": event["actual_query_tokens"]}
                        for event in manifest["events"] for req in event["requests"]
                        if req["base"] == request_id and req["phase"] == "prefill"]
    return shapes


def summarize(run):
    environment = read(run / "state/environment.json")
    receipt = read(run / "analysis/ratio_gates.json")
    close = read(run / "state/crossing_registry_closeout.json")
    if close["live_images"] or close["outstanding_exports"]:
        raise ValueError("The E source or descriptor registry did not retire")
    if close["crossing_issued"] != close["crossing_acknowledged"] + close["crossing_revoked"]:
        raise ValueError("Crossing descriptor counts did not balance")
    d = [row for path in (run / "raw").glob("ec_cuda_ipc_decode_gpu*.jsonl") for row in rows(path)]
    results = []
    for case in receipt["cases"]:
        records = {}
        for arm in ("N1", "N8"):
            gate = case[arm]
            transport = gate["D_transport"]
            image_keys = {clone["key"] for clone in transport["clones"]}
            observed = [event for event in d if image_keys.intersection(event["keys"])]
            records[arm] = {
                "strict_passed": gate["passed"], "exact_D_clone_ranges": transport["verified_range"],
                "D_images": len(image_keys), "D_smaller_clones": transport["strictly_smaller_clones"],
                "crossing_spares_consumed": sum(event["preissued_crossing_images"] for event in observed),
                "suffix_only_P_redirects": sum(event["redirected_inline_images"] for event in observed),
                "D_export_RPC_ms": sum(event["export_RPC_ms"] for event in observed),
                "D_clone_ACK": bool(observed) and all(event["clone_complete_before_ACK"] for event in observed)
                if image_keys else None,
            }
        mixed = case["N8"]
        failures = []
        for index, item in enumerate(mixed["cases"]):
            if not item["passed"]:
                keep = ("request_id", "actual_offloaded_tokens", "identical_serial_E", "state_passed",
                        "late_vs_native_exact_greedy", "native_vs_serial_exact_greedy", "numerical_checks")
                failures.append({key: item[key] for key in keep} | {"geometry": {
                    arm: request_geometry(run, case["name"], arm, index) for arm in ("native", "late")}})
        results.append({key: case[key] for key in ("name", "request_id", "cut", "nominal_ratio", "actual_ratio", "passed")}
                       | {"arms": records, "paired_exact_greedy_n8": sum(item["late_vs_native_exact_greedy"]
                                                                        for item in mixed["cases"]),
                          "paired_state_pass_n8": sum(item["state_passed"] for item in mixed["cases"]),
                          "lead_shared_decode_n8": mixed["lead_shared_decode"], "n8_failures": failures})
    warning = "Producer process has been terminated before all shared CUDA tensors released"
    return {
        "evidence": "DIAGNOSTIC", "numeric_regime": environment["numeric_regime"],
        "envelope": environment["crossing_preissue_envelope"], "argv": environment["argv"],
        "runtime_source_sha256": environment["source_SHA256"], "gpu_uuid": environment["authorized_uuid_by_gpu"],
        "D_decode_embedding_events": len(d), "D_export_RPC_events": sum(row["export_RPC_ms"] > 0 for row in d),
        "E_registry_closeout": close, "Torch_CUDA_IPC_lifetime_warning": any(
            warning in path.read_text(errors="replace") for path in (run / "logs").glob("*.log")),
        "natural_hold_continuation_passed": receipt["natural_hold_continuation"]["passed"],
        "continuation_greedy_exact_to_native_EPD": receipt.get("native_epd_resume_control", {}).get("Paired_Greedy_exact"),
        "cases": results,
        "scope": "Actual CUDA IPC routes and staged Natural EOS correctness. Correctness observers invalidate serving latency.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.run)
    with args.output.open("x") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
