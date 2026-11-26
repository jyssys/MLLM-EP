"""Join actual late-bound D suffixes with decoder CUDA IPC host intervals."""

import argparse
import json
from pathlib import Path

from ..mllm_epd_h200_prefill_heavy_batch_matched.analysis import csv_file, jsonl


def analyze(run):
    stages = jsonl(run / "raw/proxy_epd.jsonl", appending=True)
    records = {row["runtime_request_id"]: row for row in stages}
    ec = [row for path in (run / "raw").glob("ec_cuda_ipc_decode_gpu*.jsonl")
          for row in jsonl(path, appending=True)]
    results = []
    for path in sorted((run / "raw").glob("*_requests.jsonl")):
        clients = jsonl(path)
        if not clients:
            continue
        arrivals = sorted(row["scheduled_arrival"] for row in clients)
        left, right = arrivals[len(arrivals) // 5], arrivals[4 * len(arrivals) // 5]
        for client in clients:
            runtime_id = client.get("runtime_request_id")
            stage = records.get(runtime_id, {})
            partition = stage.get("token_partition") or {}
            suffix = partition.get("suffix_tokens", 0)
            if suffix <= 0 or stage.get("status") != "MEASURED_ONLINE":
                continue
            decision, = [d for d in stage["late_plan"]["decisions"] if d["accepted"]]
            submit, first = stage["D_submit"], stage["D_first_generated_token"]
            boundary = decision["P_boundary_state"]["boundary_ready"]
            loads = [row for row in ec if submit <= row["begin"] <= row["end"] <= first]
            descriptor_ms = sum(row["descriptor_prepare_host_ms"] for row in loads)
            export_ms = sum(row["export_RPC_ms"] for row in loads)
            e_requests_during_export = None
            export_begin = export_end = None
            if export_ms and loads:
                last = next(row for row in reversed(loads) if row["export_RPC_ms"] > 0)
                export_end = last["begin"] + last["descriptor_prepare_host_ms"] / 1000
                export_begin = export_end - last["export_RPC_ms"] / 1000
                e_requests_during_export = sum(
                    row["E_submit"] < export_end and row["E_complete"] > export_begin
                    for row in stages if "E_submit" in row and "E_complete" in row
                )
            result = {
                "start": run.name,
                "cohort": path.name.removesuffix("_requests.jsonl"),
                "request_id": client["request_id"],
                "runtime_request_id": runtime_id,
                "prefix_tokens": partition["prefix_tokens"],
                "suffix_tokens": suffix,
                "actual_suffix_fraction": suffix / (partition["prefix_tokens"] + suffix),
                "output_tokens": client["output_tokens"],
                "central_arrival_roi": left <= boundary < right,
                "post_arrival_drain": boundary > arrivals[-1],
                "D_coordinator_wait_ms": (submit - stage["P_ready"]) * 1000,
                "cut_to_D_submit_ms": (submit - boundary) * 1000,
                "D_submit_to_first_ms": (first - submit) * 1000,
                "D_cut_to_first_ms": (first - boundary) * 1000,
                "D_idle_running_at_reservation": decision.get("D_engine", {}).get("running"),
                "D_idle_waiting_at_reservation": decision.get("D_engine", {}).get("waiting"),
                "D_KV_free_blocks_at_reservation": decision.get("D_engine", {}).get("KV_free_blocks"),
                "D_EC_load_calls": len(loads),
                "D_EC_images": sum(row["images"] for row in loads),
                "D_EC_inline_images": sum(row["inline_images"] for row in loads),
                "D_EC_redirected_images": sum(row["redirected_inline_images"] for row in loads),
                "D_EC_preissued_crossing_images": sum(row.get("preissued_crossing_images", 0) for row in loads),
                "D_EC_clone_logical_bytes": sum(row["logical_bytes"] for row in loads),
                "D_EC_descriptor_ms": descriptor_ms if loads else None,
                "D_EC_export_RPC_ms": export_ms if loads else None,
                "D_EC_begin_from_cut_ms": (loads[0]["begin"] - boundary) * 1000 if loads else None,
                "D_EC_end_from_cut_ms": (loads[-1]["end"] - boundary) * 1000 if loads else None,
                "last_E_export_begin_from_cut_ms": (export_begin - boundary) * 1000 if export_begin else None,
                "last_E_export_end_from_cut_ms": (export_end - boundary) * 1000 if export_end else None,
                "D_EC_import_clone_sync_ms": sum(row["import_clone_sync_host_ms"] for row in loads) if loads else None,
                "overlapping_E_HTTP_intervals_during_last_export": e_requests_during_export,
                "TTFT_ms": client["TTFT_ms"],
                "max_client_ITL_ms": max(client["ITL_ms"], default=0.0),
                "TPOT_ms": client.get("TPOT_ms"),
            }
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [row for run in args.runs for row in analyze(run)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_file(args.output, rows)
    print(json.dumps({"scope": "Host intervals during the exclusive D suffix reservation", "rows": len(rows)}))


if __name__ == "__main__":
    main()
