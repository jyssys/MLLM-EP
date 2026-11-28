"""Join original E IPC publication bounds to the legal causal LM prefix."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

import numpy as np

from ..mllm_epd_prefill_heavy_modality_discovery.reduce_epd import (
    epoch_proxies, payload_receipt, read_image_witness, successful,
)
from ..mllm_epd_prefill_heavy_modality_discovery.timeline import read_jsonl


LEADS_MS = (5, 10, 25, 50)


def one(rows, identity):
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one record for {identity}, found {len(rows)}")
    return rows[0]


def check(value, description):
    if not value:
        raise ValueError(description)


def analyze(epoch, manifest, geometry):
    raw = epoch / "raw"
    clients = read_jsonl(raw / "image_ready_n24_requests.jsonl")
    proxies = epoch_proxies(epoch)
    events, manifests, producer_receipts = read_image_witness(epoch)
    frozen = {item["request_id"]: item for item in json.loads(manifest.read_bytes())["requests"]}
    cohort = one([item for item in json.loads(geometry.read_bytes())["cohorts"]
                  if item["cohort"] == "primary_very_long_n24"], "primary processor geometry")
    actual_geometry = {item["request_id"]: item for item in cohort["request_to_grid_and_prefix"]}
    check(len(clients) == len(frozen) == len(actual_geometry) == 24
          and {item["request_id"] for item in clients} == set(frozen) == set(actual_geometry),
          "The measured and anchored primary request sets differ")

    handoff_files = sorted(raw.glob("processor_handoff_prefill_*.jsonl*"))
    ec_files = sorted(raw.glob("ec_cuda_ipc_prefill_*.jsonl*"))
    handoffs = defaultdict(list)
    clones = defaultdict(list)
    for path in handoff_files:
        for row in read_jsonl(path):
            handoffs[row["key"]].append(row)
    for path in ec_files:
        for row in read_jsonl(path):
            check([item["key"] for item in row["clone_shapes"]] == row["keys"],
                  "P EC metadata and cloned key order differ")
            for part in row["clone_shapes"]:
                clones[part["key"]].append((row, part))

    per_request = []
    for client in clients:
        identity = client["request_id"]
        rid = client["runtime_request_id"]
        source = frozen[identity]
        verified = actual_geometry[identity]
        proxy = one(proxies[rid], rid)
        key = sha256(rid.encode()).hexdigest()
        ordered = one(manifests[key], key)
        consumer = one(handoffs[key], f"P handoff {key}")
        check(successful(client) and proxy["status"] == "MEASURED_ONLINE"
              and client["prompt_SHA256"] == source["prompt_SHA256"],
              f"Failed original request contract for {identity}")
        check(consumer["action"] == "CONSUME_SAME_REQUEST_PROCESSOR_RESULT"
              and consumer["manifest_SHA256"] == ordered["actual_manifest_sha256"],
              f"Prefill did not consume the producer manifest for {identity}")
        tokens_path = Path(source["token_metadata_file"])
        check(payload_receipt(tokens_path)["sha256"] == source["token_metadata_SHA256"],
              f"Frozen prompt tensor changed for {identity}")
        with np.load(tokens_path, allow_pickle=False) as tokens:
            original = tokens["input_ids"].reshape(-1).tolist()
        encoder = one(proxy["E_items"], f"encoder request {rid}")
        check(original == encoder["response"]["prompt_token_ids"]
              == proxy["P_response"]["prompt_token_ids"] == client["prompt_token_ids"],
              f"Exact E/P/D prompt token IDs differ for {identity}")
        check(len(original) == ordered["lm_input_tokens"] == consumer["prompt_tokens"]
              == verified["input_tokens"] == source["total_input_tokens"],
              f"Prompt row count differs for {identity}")
        markers = ordered["ordered_image_identifiers"]
        check(len(markers) == len(set(markers)) == len(verified["images"]) == client["image_count"]
              == consumer["images"] and all(marker.endswith(f"-image-{i}") for i, marker in enumerate(markers)),
              f"Image identity order differs for {identity}")

        bounds = []
        for i, (marker, image) in enumerate(zip(markers, verified["images"], strict=True)):
            start, end = image["visual_span"]
            check(image["image_index"] == i and image["SHA256"] == source["images"][i]["SHA256"]
                  and [start, end] == source["image_visual_spans"][i]
                  and image["after_original_vision_end_prefix_tokens"] == end + 1
                  and original[start - 1] == 151652 and original[end] == 151653
                  and all(token == 151655 for token in original[start:end]),
                  f"Exact original image cut differs for {identity} at {i}")
            matches = [event for event in events[marker]
                       if proxy["E_submit"] * 1e9 <= event["save_begin_ns"] <= proxy["E_complete"] * 1e9]
            event = one(matches, f"producer event {identity} image {i}")
            copies = [(row, part) for row, part in clones[marker]
                      if proxy["P_submit"] <= row["begin"] <= row["end"] <= proxy["P_ready"]]
            copy, clone = one(copies, f"P EC clone {identity} image {i}")
            check(copy["producer_pid"] == event["producer_pid"] and copy["clone_complete_before_ACK"]
                  and clone["first_source_row"] == 0 and clone["source_shape"] == clone["clone_shape"]
                  == [end - start, 3584], f"P cloned wrong original image data for {identity} at {i}")
            lo = event["completion_lower_bound_ns"]
            hi = event["completion_upper_bound_ns"]
            check(event["status"] == "CUDA_EVENT_COMPLETE" and end - start == event["embedding_rows"]
                  and isinstance(lo, int) and isinstance(hi, int) and event["save_begin_ns"] <= lo <= hi
                  and event["save_begin_ns"] <= event["save_returned_ns"],
                  f"Missing valid lower and upper event bounds for {identity} at {i}")
            bounds.append((lo, max(hi, event["save_returned_ns"]), end + 1))
        full_lo, full_hi = (max(pair[j] for pair in bounds) for j in (0, 1))
        prefix_hi = 0
        qualified = {str(ms): {"LM_rows": 0, "images": 0} for ms in LEADS_MS}
        for i, (_, latest, rows) in enumerate(bounds):
            prefix_hi = max(prefix_hi, latest)
            for ms in LEADS_MS:
                if full_lo - prefix_hi >= ms * 1_000_000:
                    qualified[str(ms)] = {"LM_rows": rows, "images": i + 1}
        per_request.append({"request_id": identity, "LM_input_rows": len(original),
                            "total_images": len(bounds), "full_transport_bound_width_ms": (full_hi - full_lo) / 1e6,
                            "first_image_lead_lower_ms": max(0, full_lo - bounds[0][1]) / 1e6,
                            "actual_P_submit_after_E_ack_ms": (proxy["P_submit"] - proxy["E_complete"]) * 1e3,
                            "max_complete_contiguous_prefix_by_lead_ms": qualified})

    inputs = [raw / "image_ready_n24_requests.jsonl", manifest, geometry, *handoff_files, *ec_files,
              *sorted(raw.glob("proxy_*.jsonl*")), *sorted((raw / "image_ready").glob("*.jsonl*"))]
    summary = {}
    for ms in LEADS_MS:
        cut = [row["max_complete_contiguous_prefix_by_lead_ms"][str(ms)] for row in per_request]
        rows = [item["LM_rows"] for item in cut if item["LM_rows"] > 0]
        summary[str(ms)] = {
            "requests_with_qualifying_prefix": len(rows), "total_original_LM_rows": sum(rows),
            "qualifying_LM_rows_min": min(rows, default=None),
            "qualifying_LM_rows_p50": median(rows) if rows else None,
            "qualifying_LM_rows_max": max(rows, default=None),
            "maximum_qualifying_images": max(item["images"] for item in cut),
        }
    return {
        "evidence": "DIAGNOSTIC_TRACE_ANALYSIS",
        "scope": "Producer GPU completion plus host marker bounds. Actual P clones confirm identity during original serving.",
        "requests_validated": len(per_request), "image_occurrences_validated": sum(r["total_images"] for r in per_request),
        "positive_first_image_lead_lower_requests": sum(r["first_image_lead_lower_ms"] > 0 for r in per_request),
        "actual_P_submit_after_E_ack_requests": sum(r["actual_P_submit_after_E_ack_ms"] >= 0 for r in per_request),
        "producer_terminal_receipts": len(producer_receipts),
        "threshold_summary": summary, "requests": sorted(per_request, key=lambda row: row["request_id"]),
        "input_SHA256": {p.name: payload_receipt(p)["sha256"] for p in inputs},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.epoch, args.manifest, args.geometry)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["threshold_summary"]))


if __name__ == "__main__":
    main()
