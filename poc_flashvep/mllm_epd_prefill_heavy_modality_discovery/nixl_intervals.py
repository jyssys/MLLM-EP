"""Match the original KV handoff descriptors to bounded native D-side READ lifetimes."""

from collections import defaultdict
from hashlib import sha256
import json

from .timeline import distribution, read_json, read_jsonl


def handoff_key(fields):
    engine, request = (fields.get(name) for name in ("remote_engine_id", "remote_request_id"))
    if isinstance(engine, str) and isinstance(request, str) and engine and request:
        return sha256(json.dumps([engine, request], separators=(",", ":")).encode()).hexdigest()
    return fields.get("remote_handoff_SHA256")


def read_nixl_witness(epoch):
    folder = epoch / "raw"
    paths = sorted(folder.glob("nixl_read_decode_*.jsonl*"))
    if not paths:
        return None
    by_handoff = defaultdict(list)
    for path in paths:
        for row in read_jsonl(path):
            identifiers = {handoff_key(item) for item in row["reads"]}
            if len(identifiers) == 1 and None not in identifiers:
                by_handoff[next(iter(identifiers))].append(row)
    receipt_keys = ("written_request_rows", "tracked_requests", "dropped_inflight_reads",
                    "dropped_buffered_requests")
    receipts = [{key: row.get(key) for key in receipt_keys}
                for row in (read_json(path) for path in sorted(folder.glob("nixl_read_decode_*.json")))]
    return by_handoff, receipts


def exact_read_spans(raw, submit, finish):
    if raw.get("native_failures") or raw.get("dropped_reads") or not raw.get("worker_kv_ready_observed_ns"):
        return None
    possible, definite = [], []
    for read in raw["reads"]:
        moments = read.get("possible_read_enclosure_ns")
        consumer_ready = raw["worker_kv_ready_observed_ns"]
        if (not moments or len(moments) != 2
                or not submit <= moments[0] <= moments[1] <= consumer_ready <= finish
                or read.get("post_begin_ns") != moments[0]
                or read.get("completion_upper_bound_ns") != moments[1]):
            return None
        possible.append(tuple(moment / 1e9 for moment in moments))
        lower = read.get("definite_outstanding_after_post_ns")
        if lower:
            if not moments[0] <= lower[0] <= lower[1] <= moments[1]:
                return None
            definite.append(tuple(moment / 1e9 for moment in lower))
    if not possible:
        return None
    return possible, definite, consumer_ready / 1e9


def join_nixl(witness, proxy, client_count):
    if witness is None:
        return None
    by_handoff, receipts = witness
    totals = defaultdict(int)
    bytes_reported, native_duration_us, ready_over_enclosure_us, post_to_ready_us = [], [], [], []
    for request in proxy:
        if request.get("status") != "MEASURED_ONLINE":
            continue
        totals["successful_proxy_requests"] += 1
        if request.get("deflected") is True and "P_ready" not in request and "H_ready" not in request:
            totals["destination_local_without_native_READ"] += 1
            request["native_nixl_READ_disposition"] = "D_LOCAL_NO_REMOTE_HANDOFF"
            continue
        params = (request.get("H_response") or request.get("P_response") or {}).get("kv_transfer_params") or {}
        key = handoff_key(params) if params.get("do_remote_prefill") else None
        if key is None or "D_submit" not in request or "D_http_complete" not in request:
            totals["unclassified_successful_requests"] += 1
            continue
        totals["descriptor_expected_requests"] += 1
        submit, finish = (request[field] * 1e9 for field in ("D_submit", "D_http_complete"))
        choices = [(row, proven) for row in by_handoff.get(key, ())
                   if (proven := exact_read_spans(row, submit, finish)) is not None]
        if len(choices) != 1:
            totals["missing_or_ambiguous_READ_rows"] += 1
            continue
        (raw, (possible, definite, ready)), = choices
        totals["matched_native_READ_requests"] += 1
        totals["matched_native_READ_operations"] += len(possible)
        for read in raw["reads"]:
            telemetry = read.get("native_telemetry")
            if telemetry:
                bytes_reported.append(telemetry["bytes"])
                native_duration_us.append(telemetry["transfer_duration_us"])
        ready_over_enclosure_us.append((ready - max(b for _, b in possible)) * 1e6)
        post_to_ready_us.append((ready - min(a for a, _ in possible)) * 1e6)
        records = []
        for index, read in enumerate(raw["reads"]):
            operation = {"read_index": index}
            fields = ("post_begin_ns", "post_return_ns", "completion_lower_bound_ns",
                      "completion_upper_bound_ns", "native_poll_count")
            operation.update({key: read.get(key) for key in fields})
            operation.update(read.get("native_telemetry") or {})
            records.append(operation)
        request.update(native_nixl_READ_disposition="REMOTE_READ_BOUNDED",
                       D_native_nixl_READ_possible=possible, D_native_nixl_READ_definite_outstanding=definite,
                       D_native_nixl_READ_operations=records,
                       observed_D_worker_KV_ready=ready, observed_D_worker_KV_ready_ns=raw["worker_kv_ready_observed_ns"])
    counts = dict(totals)
    counts["all_client_requests"] = client_count
    lossless = bool(receipts) and all(row.get("dropped_inflight_reads") == 0
                                    and row.get("dropped_buffered_requests") == 0 for row in receipts)
    complete = (lossless and client_count == len(proxy) == totals["successful_proxy_requests"]
                == totals["matched_native_READ_requests"] + totals["destination_local_without_native_READ"]
                and client_count > 0)
    return {"evidence": "DIAGNOSTIC", "counts": counts, "whole_cohort_READ_accounted": complete,
            "all_recorders_no_drops": lossless, "recorder_receipts": receipts,
            "reported_native_READ_bytes": sum(bytes_reported), "operations_with_telemetry": len(bytes_reported),
            "native_transfer_duration_us": distribution(native_duration_us),
            "observed_worker_ready_after_latest_DONE_us": distribution(ready_over_enclosure_us),
            "observed_worker_ready_after_first_post_us": distribution(post_to_ready_us),
            "scope": "Matched source descriptor and native D-side NIXL PULL async READ. The union of host post-to-observed-DONE "
                     "intervals is a conservative enclosure of possible outstanding READ. Its complement covers only "
                     "these accounted reads. Connector readiness includes the worker get_finished return. Neither "
                     "interval measures exclusive DMA wire utilization or arbitrary D transport."}
