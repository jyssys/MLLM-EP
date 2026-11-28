"""Reduce paired Native EPD cohorts and qualify same-window physical observations."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

from ..mllm_epd_h200_prefill_heavy_batch_matched.trace import elapsed, merged
from ..mllm_specific_disaggregation_discovery.native_transfer_telemetry import prom_delta
from .nixl_intervals import join_nixl, read_nixl_witness
from .timeline import ROLES, analyze_window, arrival_window, distribution, read_json, read_jsonl


OBSERVER_LIMIT = .05
OBSERVER_FIELDS = ("duration_seconds", "TTFT_p50_ms", "TTFT_p95_ms", "TPOT_p95_ms", "ITL_p99_ms")
DECODE_LIMITS = {"TPOT_ms": 33, "ITL_ms": 34}
TTFT_LIMIT_MS = 5000


def payload_receipt(path):
    with path.open("rb") as stream:
        return {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                "bytes": path.stat().st_size}


def source_path(folder, name):
    for path in (folder / name, folder / (name + ".gz")):
        if path.is_file():
            return path
    return None


def tags(epoch):
    ledger = epoch / "configs" / "cohorts.jsonl"
    if ledger.is_file():
        return read_jsonl(ledger)
    return [{"tag": path.stem.removesuffix("_requests")} for path in (epoch / "raw").glob("*_requests.jsonl")
            if path.stem != "warmup_requests" and not path.stem.endswith("_priming_requests")]


def successful(row):
    return row.get("status") == "MEASURED_ONLINE" and bool(row.get("token_ids"))


def decode_pass(row):
    return (successful(row) and row.get("TPOT_ms", math.inf) <= DECODE_LIMITS["TPOT_ms"]
            and max(row.get("ITL_ms", [math.inf]), default=0) <= DECODE_LIMITS["ITL_ms"])


def serving_metrics(client, planned):
    ok = [row for row in client if successful(row)]
    arrivals = [row["scheduled_arrival"] for row in client]
    duration = max(row.get("finish", row["scheduled_arrival"]) for row in client) - min(arrivals)
    counts_per_request = len({row["runtime_request_id"] for row in client})
    joint = [row for row in ok if decode_pass(row) and row["TTFT_ms"] <= TTFT_LIMIT_MS]
    output = sum(len(row["token_ids"]) for row in ok)
    fields = {name: distribution(row[name] for row in ok) for name in ("TTFT_ms", "TPOT_ms", "E2E_ms")}
    fields["ITL_ms"] = distribution(gap for row in ok for gap in row["ITL_ms"])
    return {"planned": planned, "recorded": len(client), "completed": len(ok),
            "failed_or_missing": max(len(client), planned) - len(ok),
            "complete": len(ok) == len(client) == planned == counts_per_request,
            "duration_seconds": duration, "throughput_req_s": len(ok) / duration if duration > 0 else None,
            "output_tokens_per_second": output / duration if duration > 0 else None,
            "absolute_decode_pass": sum(decode_pass(row) for row in client), "joint_pass": len(joint),
            "joint_goodput_req_s": len(joint) / duration if duration > 0 else None,
            "output_tokens": output,
            "metrics_ms": fields, "TTFT_p50_ms": fields["TTFT_ms"]["p50"],
            "TTFT_p95_ms": fields["TTFT_ms"]["p95"], "TPOT_p95_ms": fields["TPOT_ms"]["p95"],
            "ITL_p99_ms": fields["ITL_ms"]["p99"],
            "geometry": {key: distribution(row[key] for row in ok if row.get(key) is not None)
                         for key in ("image_count", "input_tokens_manifest", "visual_tokens", "visual_ratio", "output_tokens")}}


def compare_patterns(before, profile, after):
    def freeze(arm):
        ordered = sorted(arm, key=lambda row: row["scheduled_arrival"])
        origin = ordered[0]["scheduled_arrival"]
        schedule = [(row["request_id"], row.get("prompt_SHA256"), row.get("input_tokens_manifest"),
                     row["scheduled_arrival"] - origin) for row in ordered]
        outputs = [row.get("token_ids") for row in ordered]
        return schedule, outputs

    runs = [freeze(arm) for arm in (before, profile, after)]
    schedules = [arm[0] for arm in runs]
    match = len(schedules[0]) == len(schedules[1]) == len(schedules[2]) and all(
        a[:3] == b[:3] == c[:3] and max(a[-1], b[-1], c[-1]) - min(a[-1], b[-1], c[-1]) <= 2e-6
        for a, b, c in zip(*schedules))
    return {"request_order_contract_and_arrival_match": match,
            "clean_before_profile_exact_tokens": sum(a == b and bool(a) for a, b in zip(runs[0][1], runs[1][1])),
            "profile_clean_after_exact_tokens": sum(a == b and bool(a) for a, b in zip(runs[1][1], runs[2][1])),
            "all_exact_tokens": runs[0][1] == runs[1][1] == runs[2][1] and all(runs[1][1])}


def observer_gate(rows, metrics, window):
    """Use a 5% clean spread and deviation from the center of matching clean runs."""
    if len(rows) != 3:
        return {"passed": False, "status": "UNPAIRED", "metrics": {}}
    contract = compare_patterns(*rows)
    checks = {}
    for metric in OBSERVER_FIELDS:
        a, b, c = (point[metric] for point in metrics)
        center = (a + c) / 2 if a is not None and c is not None else 0
        spread = abs(a - c) / center if center else None
        shift = abs(b / center - 1) if center and b is not None else None
        checks[metric] = {"clean_before": a, "profile": b, "clean_after": c,
                          "clean_relative_spread": spread, "profile_relative_deviation": shift,
                          "passed": spread is not None and shift is not None and max(spread, shift) <= OBSERVER_LIMIT}
    captures = window["captures"] if window else {}
    collectors = (set(captures) == set(ROLES) and all(x["collector_ok"] and x["clock_ok"] for x in captures.values()))
    passed = (contract["request_order_contract_and_arrival_match"] and contract["all_exact_tokens"]
              and all(point["complete"] for point in metrics) and collectors
              and all(check["passed"] for check in checks.values()))
    return {"passed": passed, "status": "PASS" if passed else "FAIL", "threshold": OBSERVER_LIMIT,
            "collector_integrity_and_clock_coverage_pass": collectors, **contract, "metrics": checks}


def gpu_gpm(path, window):
    if path is None or window is None:
        return None
    root = read_json(path)
    left, right = window
    by_role = defaultdict(lambda: defaultdict(list))
    covered = defaultdict(list)
    for sample in root["samples"]:
        for role, value in sample["roles"].items():
            a, b = value["sample_window"]
            duration = min(right, b) - max(left, a)
            if duration <= 0:
                continue
            covered[role].append((a, b))
            for metric, result in value["values"].items():
                if isinstance(result, (int, float)) and math.isfinite(result):
                    by_role[role][metric].append((result, duration))
    return {"by_role": {role: {name: sum(v * w for v, w in series) / sum(w for _, w in series)
                                for name, series in metrics.items()} for role, metrics in by_role.items()},
            "coverage": {role: elapsed(merged(spans, left, right)) / (right - left)
                         for role, spans in covered.items()}, "units": root.get("units")}


def queue_summaries(path, window):
    if path is None or window is None:
        return None
    left, right = window
    data = [row for row in read_jsonl(path) if "values" in row]
    data.sort(key=lambda row: row["monotonic"])
    detail = defaultdict(lambda: defaultdict(list))
    for row in data:
        if left <= row["monotonic"] < right:
            for role, counters in row["values"].items():
                for key, value in counters.items():
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        detail[role][key].append(value)
    return {"sample_model": "Periodic user-space request snapshots within the ROI. Averages are over samples.",
            "samples": [{"seconds_from_roi": row["monotonic"] - left, "values": row["values"]}
                        for row in data if left <= row["monotonic"] < right],
            "by_role": {role: {name: {"sample_count": len(values), "mean": sum(values) / len(values),
                                     "max": max(values), "min": min(values)}
                               for name, values in metrics.items()} for role, metrics in detail.items()}}


def event_distributions(client, proxy, processor):
    matches = {row["runtime_request_id"]: row for row in client if successful(row)}
    rows = []
    for row in proxy:
        request = matches.get(row["runtime_request_id"])
        if request is None or row.get("status") != "MEASURED_ONLINE":
            continue
        measured = {"runtime_request_id": row["runtime_request_id"], "request_id": request["request_id"],
                    "image_count": request.get("image_count"), "input_tokens": request.get("input_tokens_manifest"),
                    "visual_tokens": request.get("visual_tokens"), "output_tokens": request.get("output_tokens"),
                    "route": row.get("name", row.get("mode")), "observed_E_item_count": len(row.get("E_items") or []),
                    "D_local": bool(row.get("deflected")), "E_assisted": bool(row.get("encoder_assisted")),
                    "completed": successful(request) and row.get("status") == "MEASURED_ONLINE"}
        for role, (begin, end) in {"encoder_http": ("E_submit", "E_complete"),
                                  "prefill_http": ("P_submit", "P_ready"),
                                  "encoder_assist_prefill_http": ("H_submit", "H_ready"),
                                  "decode_http": ("D_submit", "D_http_complete"),
                                  "embedding_ready_to_P_submit": ("E_complete", "P_submit"),
                                  "P_ready_to_D_submit": ("P_ready", "D_submit")}.items():
            if row.get(begin) is not None and row.get(end) is not None:
                measured[f"{role}_ms"] = 1000 * (row[end] - row[begin])
        measured.update({field: row[field] for field in ("E_submit", "E_complete", "P_submit", "P_ready",
                          "H_submit", "H_ready", "D_submit", "D_http_complete", "D_first_generated_token") if field in row})
        if request.get("token_times"):
            measured["first_client_token"] = request["token_times"][0]
        for key in ("native_nixl_READ_disposition", "D_native_nixl_READ_possible", "observed_D_worker_KV_ready",
                    "D_native_nixl_READ_operations", "observed_D_worker_KV_ready_ns"):
            if key in row:
                measured[key] = row[key]
        if measured["D_local"] and {"D_submit", "D_first_generated_token"} <= measured.keys():
            measured["D_local_prefill_and_first_token_HTTP_ms"] = 1000 * (
                measured["D_first_generated_token"] - measured["D_submit"])
        admission = row.get("D_admission") or {}
        if isinstance(admission, dict) and {"begin", "admitted"} <= admission.keys():
            measured["D_proxy_admission_ms"] = 1000 * (admission["admitted"] - admission["begin"])
            measured["D_proxy_isolated_on_admission"] = admission.get("suffix_isolated")
        transfer = (row.get("H_response") or row.get("P_response") or {}).get("kv_transfer_params") or {}
        if transfer.get("do_remote_prefill"):
            measured["kv_handoff_prompt_tokens_declared"] = transfer.get("remote_num_tokens")
            block_ids = transfer.get("remote_block_ids") or []
            measured["kv_handoff_descriptor_blocks"] = transfer.get(
                "remote_block_count", sum(len(group) if isinstance(group, list) else 1 for group in block_ids))
        key = hashlib.sha256(row["runtime_request_id"].encode()).hexdigest()
        spans = [x for x in processor.get(key, []) if x["action"] == "PRODUCE_ORIGINAL_PROCESSOR_RESULT"
                 and row.get("E_submit", math.inf) <= x["begin"] < x["end"] <= row.get("E_complete", -math.inf)]
        if len(spans) == 1:
            item = spans[0]
            measured.update(E_renderer_processor_handoff_begin=item["begin"], E_renderer_processor_handoff_end=item["end"],
                            E_renderer_processor_handoff_ms=1000 * (item["end"] - item["begin"]))
        measured["arrival"] = request["scheduled_arrival"]
        rows.append(measured)
    return rows


def route_accounting(request_events, connector):
    complete = [row for row in request_events if row["completed"]]
    local = [row for row in complete if row["D_local"]]
    assist = [row for row in complete if row["E_assisted"]]
    counters = (connector or {}).get("decode", {})
    operations = counters.get("vllm:nixl_bytes_transferred_count")
    handoffs = sum(row.get("kv_handoff_descriptor_blocks", 0) > 0 for row in complete)
    return {"completed_proxy_requests": len(complete), "D_local_completed": len(local), "E_assisted_completed": len(assist),
            "D_local_total_LM_input_tokens": sum(row["input_tokens"] for row in local)
            if all(row["input_tokens"] is not None for row in local) else None,
            "D_local_visual_tokens": sum(row["visual_tokens"] for row in local)
            if all(row["visual_tokens"] is not None for row in local) else None,
            "D_local_prefill_to_first_token_HTTP_ms": distribution(row["D_local_prefill_and_first_token_HTTP_ms"] for row in local
                                                                 if "D_local_prefill_and_first_token_HTTP_ms" in row),
            "requests_with_declared_KV_handoff": handoffs,
            "NIXL_reported_READ_operations": operations,
            "NIXL_reported_READ_bytes": counters.get("vllm:nixl_bytes_transferred_sum"),
            "declared_handoffs_match_NIXL_operation_count": operations == handoffs if operations is not None else None}


def epoch_proxies(epoch):
    all_rows = defaultdict(list)
    for path in sorted((epoch / "raw").glob("proxy_*.jsonl*")):
        if path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz"):
            for row in read_jsonl(path):
                all_rows[row["runtime_request_id"]].append(row)
    return all_rows


def epoch_processors(epoch):
    index = defaultdict(list)
    for path in sorted((epoch / "raw").glob("processor_handoff_encoder_*.jsonl*")):
        for row in read_jsonl(path):
            index[row["key"]].append(row)
    return index


def read_image_witness(epoch):
    root = epoch / "raw/image_ready"
    if not root.is_dir():
        return None
    events, manifests = defaultdict(list), defaultdict(list)
    for path in sorted(root.glob("events_*.jsonl*")):
        for item in read_jsonl(path):
            events[item["image_identifier"]].append(item)
    for path in sorted(root.glob("manifests_*.jsonl*")):
        for item in read_jsonl(path):
            manifests[item["request_key"]].append(item)
    receipt_keys = ("enqueued", "dropped", "complete", "unresolved", "manifest_count",
                    "manifest_limit_reached", "error")
    receipts = [{key: row.get(key) for key in receipt_keys}
                for row in (read_json(path) for path in sorted(root.glob("receipt_*.json")))]
    return events, manifests, receipts


def image_readiness(witness, items):
    """Bound native producer transport readiness against the whole-request HTTP acknowledgment."""
    if witness is None:
        return None
    events, manifests, receipts = witness
    coverage, requests = defaultdict(int), []
    for row in items:
        if not row.get("image_count") or not row.get("completed") or "E_complete" not in row:
            continue
        coverage["eligible_completed_requests"] += 1
        choices = manifests.get(hashlib.sha256(row["runtime_request_id"].encode()).hexdigest(), ())
        if len(choices) != 1:
            coverage["missing_or_ambiguous_processor_manifests"] += 1
            continue
        manifest, = choices
        identifiers = manifest["ordered_image_identifiers"]
        input_matches = row.get("input_tokens") is None or row["input_tokens"] == manifest["lm_input_tokens"]
        if len(identifiers) != row["image_count"] or len(set(identifiers)) != len(identifiers) or not input_matches:
            coverage["ambiguous_or_mismatched_manifest_geometry"] += 1
            continue
        low = row["E_submit"] * 1e9
        high = row["E_complete"] * 1e9
        ordered = [[x for x in events[item] if low <= x["save_begin_ns"] <= high] for item in identifiers]
        if any(len(matches) != 1 or matches[0]["status"] != "CUDA_EVENT_COMPLETE" for matches in ordered):
            coverage["missing_or_ambiguous_completed_native_events"] += 1
            continue
        markers = [matches[0] for matches in ordered]
        gpu = [(event["completion_lower_bound_ns"], event["completion_upper_bound_ns"]) for event in markers]
        if any(not isinstance(a, int) or not isinstance(b, int)
               or not event["save_begin_ns"] <= a <= b or not event["save_begin_ns"] <= event["save_returned_ns"]
               for event, (a, b) in zip(markers, gpu)):
            coverage["invalid_monotonic_bounds"] += 1
            continue
        spans = manifest.get("image_spans")
        if spans is not None and (len(spans) != len(markers)
                                  or any(int(end) - int(start) != event["embedding_rows"]
                                         for (start, end), event in zip(spans, markers))):
            coverage["ambiguous_or_mismatched_manifest_geometry"] += 1
            continue
        visual_total = row.get("visual_tokens")
        if visual_total is not None and sum(event["embedding_rows"] for event in markers) != visual_total:
            coverage["ambiguous_or_mismatched_manifest_geometry"] += 1
            continue
        coverage["validated_complete_requests"] += 1
        coverage["validated_native_events"] += len(markers)
        coverage["validated_span_geometry_requests"] += spans is not None
        coverage["validated_total_visual_token_requests"] += visual_total is not None
        # A consumer needs the transport marker and the backing GPU data. The
        # marker was published sometime before save returned, as established in source.
        usable = [(max(a, event["save_begin_ns"]), max(b, event["save_returned_ns"]))
                  for event, (a, b) in zip(markers, gpu)]
        p_start = row.get("P_submit")
        first = usable[0]
        last = tuple(map(max, zip(*usable)))
        record = {"runtime_request_id": row["runtime_request_id"], "images": len(markers),
                  "first_image_prefix_tokens": spans[0][1] if spans else None,
                  "first_image_before_E_ack_ms_lower": (high - first[1]) / 1e6,
                  "first_image_before_E_ack_ms_upper": (high - first[0]) / 1e6,
                  "full_request_before_E_ack_ms_lower": (high - last[1]) / 1e6,
                  "full_request_before_E_ack_ms_upper": (high - last[0]) / 1e6,
                  "first_image_before_all_ms_lower": max(0., (last[0] - first[1]) / 1e6),
                  "first_image_before_all_ms_upper": max(0., (last[1] - first[0]) / 1e6),
                  "publisher_first_to_last_return_ms": (
                      max(event["save_returned_ns"] for event in markers) - markers[0]["save_returned_ns"]) / 1e6,
                  "after_full_P_start_margin_ms_lower": (p_start * 1e9 - last[1]) / 1e6 if p_start else None,
                  "largest_image_completion_poll_bound_width_ms": max((b - a) / 1e6 for a, b in gpu)}
        requests.append(record)
    multidoc = [row for row in requests if row["images"] > 1]
    return {"evidence": "DIAGNOSTIC", "requests": requests, "coverage": dict(coverage),
            "producer_receipts": receipts,
            "multimodal_complete_requests": len(multidoc),
            "positive_guaranteed_first_image_gap_count": sum(row["first_image_before_all_ms_lower"] > 0 for row in multidoc),
            "possible_first_image_gap_ms": distribution(row["first_image_before_all_ms_upper"] for row in multidoc),
            "guaranteed_first_image_gap_ms": distribution(row["first_image_before_all_ms_lower"] for row in multidoc),
            "transport_ready_before_E_ack_ms_lower": distribution(row["full_request_before_E_ack_ms_lower"] for row in requests),
            "per_request_maximum_poll_bound_width_ms": distribution(
                row["largest_image_completion_poll_bound_width_ms"] for row in requests),
            "scope": "Original CUDA IPC producer event completion bounded by nonblocking checks. "
                     "Per-image transport readiness also requires the existing host marker. No individual vision-kernel "
                     "timing, early consumer clone, early LM execution, or uncontended scheduling opportunity is inferred."}


def epoch_rosters(epoch):
    plan = epoch / "configs/frozen_plan.json"
    rosters = {}
    manifests = {}
    if not plan.is_file():
        return rosters
    for point in read_json(plan)["cases"]:
        path = Path(point["manifest"])
        choices = (path, epoch.parent / "dataset" / path.name)
        ref = next((candidate for candidate in choices if candidate.is_file()), None)
        if ref is None:
            continue
        if ref not in manifests:
            manifests[ref] = {row["request_id"]: row for row in read_json(ref)["requests"]}
        rosters[point["tag"]] = manifests[ref]
    return rosters


def enrich_geometry(rows, manifest):
    for row in rows:
        original = re.sub(r"--cycle\d+$", "", row["request_id"])
        item = manifest.get(original)
        if (item and item.get("prompt_SHA256") == row.get("prompt_SHA256")
                and item.get("total_input_tokens") == row.get("input_tokens_manifest")):
            row["visual_tokens"] = item["visual_tokens"]
            row["visual_ratio"] = item["visual_tokens"] / item["total_input_tokens"]


def model_revision(epoch):
    path = epoch / "state/model_download_complete.json"
    return read_json(path).get("revision") if path.is_file() else None


def read_cohorts(epoch, *, keep_intervals=False, keep_clients=False):
    proxies = epoch_proxies(epoch)
    processors = epoch_processors(epoch)
    rosters = epoch_rosters(epoch)
    revision = model_revision(epoch)
    image_witness = read_image_witness(epoch)
    nixl_witness = read_nixl_witness(epoch)
    auxiliary = [name for name, witness in (("native image-ready polling", image_witness),
                                           ("native NIXL READ observation", nixl_witness)) if witness is not None]
    arms = {}
    for meta in tags(epoch):
        tag = meta["tag"]
        folder = epoch / "raw"
        path = source_path(folder, tag + "_requests.jsonl")
        if path is None:
            arms[tag] = {"tag": tag, "evidence": "NOT_RUN", "metadata": meta}
            continue
        client = read_jsonl(path)
        enrich_geometry(client, rosters.get(meta.get("group", tag), {}))
        cfg_path = epoch / "configs" / f"{tag}.json"
        cfg = read_json(cfg_path) if cfg_path.is_file() else {}
        schedule_path = epoch / "configs" / f"{tag}_arrival.json"
        schedule = read_json(schedule_path) if schedule_path.is_file() else {}
        if not client:
            arms[tag] = {"tag": tag, "evidence": "NOT_RUN", "metadata": meta}
            continue
        joined = [dict(x) for row in client for x in proxies.get(row["runtime_request_id"], [])]
        native_reads = join_nixl(nixl_witness, joined, len(client))
        profile_root = folder / "profiles" / tag
        profiles = {role: profile_root / role for role in ROLES if (profile_root / role).is_dir()}
        trace = analyze_window(profiles, client, joined, keep_intervals=keep_intervals,
                               whole_cohort_native_READ_accounted=bool(native_reads and native_reads["whole_cohort_READ_accounted"]))
        req_metrics = serving_metrics(client, meta.get("count", cfg.get("count", len(client))))
        window = arrival_window(client)
        measured = [row for row in client if successful(row) and window is not None and row.get("token_times")
                    and window[0] <= row["token_times"][0] <= row["token_times"][-1] < window[1]]
        if trace is not None:
            trace["decode_SLO_wholly_in_ROI"] = {"requests": len(measured),
                                                "passes": sum(decode_pass(row) for row in measured)}
        prom = folder / f"{tag}_prometheus.json"
        if not prom.is_file():
            prom = folder / f"{tag}_nixl_prometheus.json"
        snapshot = read_json(prom) if prom.is_file() else None
        nixl = prom_delta({k: snapshot[k].get("roles", snapshot[k]) for k in ("before", "after")}) \
            if snapshot and {"before", "after"} <= snapshot.keys() else None
        request_events = event_distributions(client, joined, processors)
        arms[tag] = {"tag": tag, "metadata": meta, "config": cfg, "client": client, "model_revision": revision,
                     "auxiliary_observers": auxiliary,
                     "evidence": "MEASURED_ONLINE", "serving": req_metrics, "trace": trace,
                     "request_event_rows": request_events,
                     "route_accounting": route_accounting(request_events, nixl) if request_events else None,
                     "image_readiness": image_readiness(image_witness, request_events),
                     "native_NIXL_READ": native_reads,
                     "offered_qps": meta.get("qps", cfg.get("QPS")),
                     "replay_source": schedule.get("source"),
                     "queue": queue_summaries(source_path(folder, f"{tag}_scheduler.jsonl"), window),
                     "gpm": gpu_gpm(source_path(folder, f"{tag}_gpm.json"), window), "NIXL": nixl,
                     "request_receipt": payload_receipt(path)}

    for tag, arm in arms.items():
        if "client" not in arm or not arm["trace"] or not arm["trace"]["captures"]:
            continue
        stem = tag.removesuffix("_profile")
        candidates = [arms.get(stem + "_clean_before"), arm, arms.get(stem + "_clean_after")]
        paired = [item for item in candidates if item is not None and "client" in item]
        gate = observer_gate([item["client"] for item in paired], [item["serving"] for item in paired], arm["trace"])
        if auxiliary:
            gate["CUPTI_pair_pass_with_auxiliary_observation"] = gate["passed"]
            gate["unqualified_auxiliary_observers"] = auxiliary
            gate.update(passed=False, status="AUXILIARY_OBSERVER_NOT_QUALIFIED")
        arm["observer_gate"] = gate
        arm["trace"]["evidence"] = "TRACE_ANALYSIS" if gate["passed"] else "DIAGNOSTIC"
    return [{k: v for k, v in arm.items() if k != "client" or keep_clients} for arm in arms.values()]


def reduce_capacity(root):
    output = defaultdict(list)
    for path in sorted(root.glob("*/analysis/*_capacity_*.json")):
        stem, kind = path.stem.rsplit("_capacity_", 1)
        record = read_json(path)
        if record.get("stage") not in ROLES or not record.get("requests_per_second"):
            continue
        epoch = path.parent.parent
        workload_path = epoch / "configs" / (path.stem + "_workload.json")
        workload = read_json(workload_path) if workload_path.is_file() else {}
        values = [row for f in sorted((epoch / "raw" / path.stem).glob("*_rows.json")) for row in read_json(f)]
        batch_keys = {"encoder": "maximum_actual_encoder_image_batch", "prefill": "maximum_actual_prefill_request_batch",
                      "decode": "maximum_actual_prepared_decode_requests"}
        batch = [x.get(batch_keys[record["stage"]]) for x in record.get("cohorts", ())]
        duration = record.get("seconds", record.get("duration_seconds"))
        compute = {"label": kind, "stage": record["stage"], "path": str(path),
                   "requests": record.get("requests"), "seconds": duration,
                   "observed_input_or_output_tokens": record.get("tokens"),
                   "requests_per_second": record["requests_per_second"],
                   "tokens_per_second": record["tokens_per_second"], "measurement_scope": record.get("scope"),
                   "output_contract": workload.get("output"), "p_budget": workload.get("p_budget"),
                   "model_revision": model_revision(epoch),
                   "input_tokens": distribution(row["input_tokens"] for row in values),
                   "finite_http_ms": distribution(1000 * (row["finish"] - row["submit"]) for row in values),
                   "maximum_actual_batch": max((value for value in batch if value is not None), default=None),
                   "evidence": "MEASURED_PHYSICAL"}
        if values and record["stage"] == "decode":
            passed = sum(decode_pass(row) for row in values)
            compute.update(absolute_decode_pass=passed, absolute_decode_goodput_req_s=passed / duration if duration else None,
                           decode_TPOT_p95_ms=distribution(row["TPOT_ms"] for row in values)["p95"],
                           decode_ITL_p99_ms=distribution(gap for row in values for gap in row["ITL_ms"])["p99"])
        output[stem].append(compute)
    return dict(output)


def operating_region(trace):
    busy = trace["fraction_of_window"]
    if not all(role in busy for role in ROLES):
        return "NOT_MEASURED"
    e, p, d = (busy[role] for role in ROLES)
    if e >= .8 and p >= .8 and d <= .30:
        return "R3"
    if e >= .8 and p >= .8 and d >= .5:
        return "R4"
    if p >= .8 and e < .8 and d <= .30:
        return "R2"
    if e >= .8 and p < .8 and d <= .30:
        return "R1"
    if d >= .50 and d >= max(e, p):
        return "R5"
    return "OUTSIDE_R1_R5"


def classify(case, capacity):
    trace = case.get("trace")
    if not trace:
        return
    busy = trace["fraction_of_window"]
    observed_p = trace["stage_rates"].get("prefill", {})
    cohort = case["metadata"].get("cohort_id")
    peers = [row for row in capacity.get(cohort, [])
             if row.get("model_revision") == case.get("model_revision") or not case.get("model_revision")]
    fastest = {role: max((row for row in peers if row["stage"] == role and (
                         role != "decode" or row["output_contract"] == case["metadata"].get("output"))),
                         key=lambda row: row["tokens_per_second"], default=None)
               for role in ROLES}
    proximity = {}
    for role, reference in fastest.items():
        if reference is not None and role in trace["stage_rates"]:
            observed = trace["stage_rates"][role]
            proximity[role] = {"tokens": observed["tokens_per_second"] / reference["tokens_per_second"]
                               if observed["tokens_per_second"] is not None and reference["tokens_per_second"] > 0 else None,
                               "requests": observed["requests_per_second"] / reference["requests_per_second"]}
    p = proximity.get("prefill", {}).get("tokens")
    e = proximity.get("encoder", {}).get("requests")
    decode = trace["decode_SLO_wholly_in_ROI"]
    scheduler = trace["scheduler"].get("prefill", {})
    ready = busy.get("prefill_known_compute_ready_waiting") if scheduler.get("compute_ready_field_observed") else None
    ready_signal = ready > 0 if ready is not None else None
    admission = scheduler.get("sampled_admission_operations", {})
    d_queue = (case.get("queue") or {}).get("by_role", {}).get("decode", {})
    d_capacity = d_queue.get("vllm:num_requests_waiting_by_reason:capacity", {}).get("max")
    kv_admission = (admission.get("kv_allocate_rejected") == 0 and d_capacity == 0
                    if admission.get("kv_allocate_attempts") and d_capacity is not None else None)
    native_route = case["metadata"].get("route", "native") in ("native", "epd", "native-epd")
    d_quiet = busy.get("D_kernel_idle_proxy_request_free")
    requirements = {
        "native_complete_online": case["serving"]["complete"] and native_route and trace["proxy_complete"],
        "same_window_observer_pass": trace.get("evidence") == "TRACE_ANALYSIS",
        "P_actual_compute_ready_backlog": ready_signal,
        "P_kernel_at_least_80pct": busy["prefill"] >= .80 if "prefill" in busy else None,
        "P_within_85pct_independent_token_rate": p >= .85 if p is not None else None,
        "D_kernel_at_most_30pct": busy["decode"] <= .30 if "decode" in busy else None,
        "D_absolute_SLO_within_common_window": (decode["passes"] == decode["requests"] if decode["requests"] else None)
        if decode else None,
        "D_proxy_quiet_intersects_P": busy["P_busy_D_kernel_idle_proxy_request_free"] > 0
        if d_quiet is not None and "P_busy_D_kernel_idle_proxy_request_free" in busy else None,
        "P_KV_or_transfer_credit_not_main_constraint_on_sampled_attempts": kv_admission,
        "E_supply_sufficient_in_ROI": ready_signal and observed_p.get("requests", 0) > 0
        if ready_signal is not None else None,
    }
    secondary = {"E_kernel_at_least_80pct": busy["encoder"] >= .8 if "encoder" in busy else None,
                 "E_within_85pct_independent_request_rate": e >= .85 if e is not None else None,
                 "E_P_joint_busy_D_idle_observed": busy["E_busy_P_busy_D_kernel_idle"] > 0
                 if "E_busy_P_busy_D_kernel_idle" in busy else None}
    primary_pass = all(value is True for value in requirements.values())
    secondary_pass = primary_pass and all(value is True for value in secondary.values())
    case.update({"capacity_references": fastest, "same_window_capacity_proximity": proximity,
                 "observed_region": operating_region(trace), "M1_criteria": requirements,
                 "M2_criteria": secondary, "M1_pass": primary_pass, "M2_pass": secondary_pass})


def file_csv(path, entries):
    if not entries:
        return
    fields = list(dict.fromkeys(key for row in entries for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(entries)


def case_key(case):
    return {"epoch": case["epoch"], "tag": case["tag"]}


def cohort_identity(rows):
    order = sorted(rows, key=lambda row: row["scheduled_arrival"])
    anchor = order[0]["scheduled_arrival"]
    fields = ("request_id", "prompt_SHA256", "input_tokens_manifest", "TTFT_ms", "TPOT_ms", "status")
    return [{**{field: row.get(field) for field in fields},
             "runtime_request_id": row["runtime_request_id"],
             "arrival_offset_s": row["scheduled_arrival"] - anchor,
             "output_SHA256": hashlib.sha256(json.dumps(row["token_ids"]).encode()).hexdigest() if successful(row) else None,
             "ITL_max_ms": max(row.get("ITL_ms", ()), default=None),
             "decode_pass": decode_pass(row)} for row in order]


def same_input_trace(a, b, identities):
    x, y = (identities[(row["epoch"], row["tag"])] for row in (a, b))
    controls = ("qps", "output", "p_budget", "d_budget", "cache_gib", "arrival")
    items = ("request_id", "prompt_SHA256", "input_tokens_manifest")
    return (len(x) == len(y) and a["model_revision"] is not None and a["model_revision"] == b["model_revision"]
            and all(a["metadata"].get(key) == b["metadata"].get(key) for key in controls)
            and all(abs(p["arrival_offset_s"] - q["arrival_offset_s"]) <= 2e-6 and all(p[key] == q[key] for key in items)
                    for p, q in zip(x, y)))


def gap_stats(stamps):
    stamps = sorted(stamps)
    gaps = [1000 * (b - a) for a, b in zip(stamps, stamps[1:])]
    mean = statistics.fmean(gaps) if gaps else 0
    duration = stamps[-1] - stamps[0] if len(stamps) > 1 else None
    return {"count": len(stamps), "duration_seconds": duration,
            "finite_event_span_rate_per_second": (len(stamps) - 1) / duration if duration else None,
            "gap_ms": distribution(gaps), "gap_CV": statistics.pstdev(gaps) / mean if mean else None}


def stage_clock_comparison(events, stage_arrivals):
    arrivals = [{"scheduled_arrival": moment} for moment in stage_arrivals]
    window = arrival_window(arrivals)
    if window is None:
        return None
    left, right = window
    duration = right - left
    decode_pending = [(row["P_ready"], row["D_http_complete"]) for row in events
                      if {"D_http_complete", "P_ready"} <= row.keys()]
    submitted = [row for row in events if left <= row.get("P_submit", -1) < right]
    completed = [row for row in events if left <= row.get("P_ready", -1) < right]
    return {"duration_seconds": duration,
            "P_submits": len(submitted), "P_completes": len(completed),
            "P_completion_input_tokens_per_second": sum(row["input_tokens"] for row in completed) / duration
            if all(row.get("input_tokens") is not None for row in completed) else None,
            "D_proxy_request_free_fraction": 1 - elapsed(merged(decode_pending, left, right)) / duration
            if len(decode_pending) == len(events) else None,
            "P_submit_to_first_client_token_ms": distribution(
                1000 * (row["first_client_token"] - row["P_submit"]) for row in submitted
                if row.get("first_client_token", -1) > row["P_submit"])}


def arrival_replay(cases, identities):
    results = []
    for replay in cases:
        spec = replay.get("replay_source") or {}
        if spec.get("kind") != "observed_e_ready":
            continue
        source = [row for row in cases if row["tag"] == spec["tag"]]
        if len(source) > 1 and spec.get("source"):
            epoch = Path(spec["source"]).parent.parent.name
            source = [row for row in source if row["epoch"] == epoch]
        record = {"replay": case_key(replay), "source_tag": spec["tag"],
                  "source_proxy_sha256": spec.get("source_SHA256"), "evidence": "INCONCLUSIVE"}
        results.append(record)
        if len(source) != 1:
            record["failure"] = "Selected E-ready source cohort is absent or ambiguous."
            continue
        original, = source
        record["source"] = case_key(original)
        old_identity, new_identity = (identities[(row["epoch"], row["tag"])] for row in (original, replay))
        observed = sorted((row["E_complete"], row["runtime_request_id"]) for row in original["request_event_rows"]
                          if "E_complete" in row)
        if not observed or not new_identity:
            record["failure"] = "Source E completions or replay client schedule missing."
            continue
        offsets = [time - observed[0][0] for time, _ in observed]
        planned = spec["offsets"]
        actual = [row["arrival_offset_s"] for row in new_identity]
        error = max((abs(a - b) for a, b in zip(offsets, actual)), default=None)
        match = len(offsets) == len(planned) == len(actual) == len(old_identity)
        old_lengths, new_lengths = ([row["input_tokens_manifest"] for row in cohort] for cohort in (old_identity, new_identity))
        shape_measured = len(old_lengths) == len(new_lengths) and all(isinstance(n, int) for n in old_lengths + new_lengths)
        shape_delta = max(abs(a - b) for a, b in zip(sorted(old_lengths), sorted(new_lengths))) if shape_measured else None
        record.update({"matched_input_length_distribution": shape_delta == 0 if shape_measured else None,
                       "max_sorted_input_token_count_difference": shape_delta,
                       "same_model_revision": original["model_revision"] is not None
                       and original["model_revision"] == replay["model_revision"],
                       "planned_source_request_id_order_match": [request for _, request in observed] == spec["request_ids"],
                       "planned_source_offsets_match": match and max(abs(a - b) for a, b in zip(offsets, planned)) <= 2e-6,
                       "client_source_offsets_max_error_us": error * 1e6 if error is not None and match else None,
                       "replay_complete": replay["serving"]["complete"],
                       "source_complete": original["serving"]["complete"]})
        record["stream_gaps"] = {
            "source_client_arrival": gap_stats(row["arrival_offset_s"] for row in old_identity),
            "source_E_complete": gap_stats(row[0] for row in observed),
            "source_P_submit": gap_stats(row["P_submit"] for row in original["request_event_rows"] if "P_submit" in row),
            "source_P_ready": gap_stats(row["P_ready"] for row in original["request_event_rows"] if "P_ready" in row),
            "replay_client_arrival": gap_stats(actual),
            "replay_P_submit": gap_stats(row["P_submit"] for row in replay["request_event_rows"] if "P_submit" in row),
            "replay_P_ready": gap_stats(row["P_ready"] for row in replay["request_event_rows"] if "P_ready" in row),
        }
        record["live_matched_stage_clocks"] = {
            "source_from_E_ready": stage_clock_comparison(original["request_event_rows"], [row[0] for row in observed]),
            "text_replay_from_client_schedule": stage_clock_comparison(
                replay["request_event_rows"], [row["arrival"] for row in replay["request_event_rows"]]),
        }
        peers = [row for row in cases if row["metadata"].get("route") == "text" and row["offered_qps"]
                 and row["metadata"].get("arrival") == "poisson" and row["serving"]["complete"]
                 and row["metadata"].get("cohort_id") == replay["metadata"].get("cohort_id")
                 and row["metadata"].get("output") == replay["metadata"].get("output")
                 and row["model_revision"] == replay["model_revision"]
                 and row["serving"]["planned"] == replay["serving"]["planned"]]
        if peers:
            peer = min(peers, key=lambda row: abs(row["offered_qps"] - replay["offered_qps"]))
            record["nearest_text_Poisson"] = {**case_key(peer), "offered_qps": peer["offered_qps"],
                "relative_offered_rate_difference": abs(peer["offered_qps"] / replay["offered_qps"] - 1),
                "serving_comparison": {name: peer["serving"][name] for name in OBSERVER_FIELDS}}
            record["stream_gaps"]["nearest_text_Poisson_arrival"] = gap_stats(
                row["arrival_offset_s"] for row in identities[(peer["epoch"], peer["tag"])])
        source_agrees = (record["planned_source_request_id_order_match"] and record["planned_source_offsets_match"]
                         and record["client_source_offsets_max_error_us"] <= 2)
        if source_agrees and record["replay_complete"] and record["source_complete"] and record["same_model_revision"]:
            record["evidence"] = "MEASURED_ONLINE"
    return results


def admission_experiment(cases, identities):
    probes = []
    for d in cases:
        if d["metadata"].get("route") != "probe_d":
            continue
        peers = [row for row in cases if row["epoch"] == d["epoch"] and same_input_trace(d, row, identities)]
        e = next((row for row in peers if row["metadata"].get("route") == "probe_e"), None)
        control = next((row for row in peers if row["metadata"].get("route") == "ondemand"), None)
        if e is None:
            continue
        routes, selected, outputs = {}, {}, {}
        for arm in (control, e, d):
            if arm is None:
                continue
            mode = arm["metadata"]["route"]
            identity = identities[(arm["epoch"], arm["tag"])]
            events = arm["request_event_rows"]
            slot = {entry["runtime_request_id"]: index for index, entry in enumerate(identity)}
            chosen = [event for event in events if event["completed"] and (event["D_local"] or event["E_assisted"])]
            selected[mode] = {slot[event["runtime_request_id"]] for event in chosen}
            outputs[mode] = [row["output_SHA256"] for row in identity]
            standby = [row for index, row in enumerate(identity) if index not in selected[mode]]
            metrics = ("duration_seconds", "TTFT_p50_ms", "TTFT_p95_ms", "TPOT_p95_ms", "ITL_p99_ms",
                       "absolute_decode_pass", "joint_pass", "joint_goodput_req_s", "planned", "completed")
            routes[mode] = {**case_key(arm), "completed_assisted": len(chosen),
                            "completed_assisted_prompt_tokens": sum(row["input_tokens"] or 0 for row in chosen),
                            "E_HTTP_ms": distribution(row["encoder_http_ms"] for row in events if "encoder_http_ms" in row),
                            "H_HTTP_ms": distribution(row["encoder_assist_prefill_http_ms"] for row in chosen
                                                      if "encoder_assist_prefill_http_ms" in row),
                            "D_local_prefill_and_first_token_HTTP_ms": distribution(
                                row["D_local_prefill_and_first_token_HTTP_ms"] for row in chosen
                                if "D_local_prefill_and_first_token_HTTP_ms" in row),
                            "D_proxy_admission_ms": distribution(row["D_proxy_admission_ms"] for row in chosen
                                                                if "D_proxy_admission_ms" in row),
                            "D_proxy_empty_on_local_admission": sum(
                                event.get("D_proxy_isolated_on_admission") is True for event in chosen) if mode == "probe_d" else None,
                            "other_requests": len(standby), "other_requests_decode_SLO_pass": sum(row["decode_pass"] for row in standby),
                            "other_requests_ITL_peak_p95_ms": distribution(
                                row["ITL_max_ms"] for row in standby if row["ITL_max_ms"] is not None)["p95"],
                            "declared_handoff_prompt_tokens": sum(row.get("kv_handoff_prompt_tokens_declared") or 0 for row in events),
                            "declared_handoffs": sum(row.get("kv_handoff_descriptor_blocks", 0) > 0 for row in events),
                            "native_NIXL_D_counter_delta": (arm.get("NIXL") or {}).get("decode"),
                            "serving": {key: arm["serving"][key] for key in metrics}}
        exact = sum(a is not None and a == b for a, b in zip(outputs["probe_e"], outputs["probe_d"]))
        matched_slots = {i for i, row in enumerate(identities[(d["epoch"], d["tag"])])
                         if int(row["runtime_request_id"].rsplit("_", 1)[-1]) % 4 == 0}
        actual_pairs = selected["probe_e"] & selected["probe_d"]
        control_agreement = {role: sum(a is not None and a == b for a, b in zip(outputs[role], outputs["ondemand"]))
                             for role in ("probe_e", "probe_d")} if "ondemand" in outputs else None
        probes.append({"routes": routes, "same_input_trace": True,
                       "matched_assisted_slots": sorted(actual_pairs),
                       "expected_assisted_slots": sorted(matched_slots),
                       "matched_assisted_work": actual_pairs == matched_slots and selected["probe_e"] == selected["probe_d"],
                       "E_vs_D_exact_generated_sequences": exact,
                       "same_runtime_control_exact_generated_sequences": control_agreement,
                       "cold_or_warm_run_order": "Sequential arms in one independently started epoch.",
                       "evidence": "MEASURED_ONLINE" if e["serving"]["complete"] and d["serving"]["complete"] else "DIAGNOSTIC"})
    return probes


def historical_ar(profile):
    """Same interval arithmetic on the pinned, independently qualified AR raw run."""
    commit = "5bdd6ef5e91840e37c776c2ad937866531e6a391"
    receipt_path = "poc_flashvep/pd_resource_complementary_parallelism/results/native/native_profiles.json"
    records = json.loads(subprocess.check_output(["git", "show", f"{commit}:{receipt_path}"]))
    case, = [row for row in records if row["path"].endswith("/16k_light_profile") and row["evidence"] == "TRACE_ANALYSIS"]
    raw = read_jsonl(profile / "requests.jsonl")
    meta = read_json(profile / "run.json")
    if any(row["status"] != "OK" or row["input"] != 16384 or len(row["tokens"]) != 16 for row in raw):
        raise ValueError("AR fixture differs from the qualified 16K/O16 contract")
    if len(raw) != meta["count"] or len(raw) != 24 or meta["qps"] != 2.6:
        raise ValueError("AR fixture count or offered rate changed")
    paths, receipts = {}, {}
    for role, code in (("prefill", "P0"), ("decode", "D0")):
        paths[role] = profile.parent / "activity" / code
        kernel, = sorted(paths[role].glob("worker-*.csv"))
        receipt = payload_receipt(kernel)
        if receipt["sha256"] != case["roles"][code]["raw_receipt"]["sha256"]:
            raise ValueError("AR kernel capture differs from the published receipt")
        receipts[role] = receipt
    client = [{"runtime_request_id": row["id"], "scheduled_arrival": row["arrival"], "output_tokens": len(row["tokens"]),
               "input_tokens_manifest": row["input"]} for row in raw]
    proxy = [{"runtime_request_id": row["id"], "status": "MEASURED_ONLINE", "P_submit": row["start"],
              "P_ready": row["p_finish"], "D_submit": row["d_submit"], "D_http_complete": row["finish"]} for row in raw]
    trace = analyze_window(paths, client, proxy)
    if not all(capture["clock_ok"] and capture["collector_ok"] for capture in trace["captures"].values()):
        raise ValueError("AR capture does not cover the MLLM reducer's interval")
    return {"snapshot": commit, "source_receipt": receipt_path, "qualified_case": case["path"],
            "scope": "Historical Qwen3-8B native 1P/1D N24 controlled 16K/O16 at 2.6 req/s. "
                     "Model, arrival transform, media and KV budget differ. It is a descriptive comparison.",
            "observer_gate_source": case["observer"], "raw_kernel_receipts": receipts, "trace": trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epoch", action="append", help="Restrict to named raw epoch folders")
    parser.add_argument("--keep-intervals", action="store_true", help="Retain verbatim intervals in addition to bins")
    parser.add_argument("--ar-profile", type=Path, help="Extracted published Qwen3-8B 16k_light_profile folder")
    args = parser.parse_args()
    epochs = [args.raw_root / name for name in args.epoch] if args.epoch else sorted(
        {path.parent.parent for path in args.raw_root.glob("*/configs/cohorts.jsonl")})
    cases = []
    identities = {}
    capacity = reduce_capacity(args.raw_root)
    for epoch in epochs:
        for case in read_cohorts(epoch, keep_intervals=args.keep_intervals, keep_clients=True):
            case["epoch"] = epoch.name
            if client := case.pop("client", None):
                identities[(case["epoch"], case["tag"])] = cohort_identity(client)
            classify(case, capacity)
            cases.append(case)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "SLO_ms": {"TTFT_ms": TTFT_LIMIT_MS, **DECODE_LIMITS},
              "capacity": capacity, "cohorts": cases}
    measured = [case for case in cases if "serving" in case]
    report["arrival_replay"] = arrival_replay(measured, identities)
    report["matched_E_D_interventions"] = admission_experiment(measured, identities)
    if args.ar_profile:
        report["historical_AR"] = historical_ar(args.ar_profile)
    (args.out / "operating_region.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    file_csv(args.out / "online_cohorts.csv", [
        {"epoch": row["epoch"], "tag": row["tag"], "workload": row["metadata"].get("cohort_id"),
         "qps": row.get("offered_qps"), "status": row["evidence"], "region": row.get("observed_region"),
         "trace_evidence": (row.get("trace") or {}).get("evidence"),
         "M1_pass": row.get("M1_pass"), "M2_pass": row.get("M2_pass"),
         **{key: row["metadata"].get(key) for key in ("route", "source", "output", "arrival", "observer",
                                                     "p_budget", "d_budget", "cache_gib")},
         "auxiliary_observers": ", ".join(row.get("auxiliary_observers", ())),
         **{f"median_{key}": value["p50"] for key, value in row.get("serving", {}).get("geometry", {}).items()},
         **{key: value for key, value in row.get("serving", {}).items() if not isinstance(value, dict)},
         **{f"{role}_kernel_busy_fraction": (row.get("trace") or {}).get("fraction_of_window", {}).get(role)
            for role in ROLES},
         **{f"{field}_fraction": (row.get("trace") or {}).get("fraction_of_window", {}).get(field) for field in (
             "D_native_nixl_READ_possible", "D_native_nixl_READ_definite_outstanding", "D_no_accounted_native_nixl_READ",
             "D_kernel_idle_outside_accounted_native_READ", "D_kernel_idle_proxy_free_outside_accounted_native_READ",
             "E_P_busy_D_kernel_idle_outside_accounted_native_READ")},
         **{name: (row.get("native_NIXL_READ") or {}).get("counts", {}).get(name) for name in (
             "matched_native_READ_requests", "matched_native_READ_operations", "destination_local_without_native_READ")}}
        for row in cases])
    file_csv(args.out / "request_timeline.csv", [{"epoch": case["epoch"], "tag": case["tag"], **row}
        for case in cases for row in case.get("request_event_rows", [])])
    file_csv(args.out / "native_nixl_READ_intervals.csv", [
        {"epoch": case["epoch"], "tag": case["tag"], "runtime_request_id": row["runtime_request_id"],
         "observed_D_worker_KV_ready_ns": row["observed_D_worker_KV_ready_ns"], **event}
        for case in cases for row in case.get("request_event_rows", []) for event in row.get("D_native_nixl_READ_operations", ())])
    print(f"Reduced {len(cases)} cohorts from {len(epochs)} epoch(s)")


if __name__ == "__main__":
    main()
