"""Compare a Native-matched Grouped Encode N24 open-loop pilot on CPU."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import re
from pathlib import Path
import shutil

from ..mllm_epd_modality_aware_idle_execution_discovery.online_evidence import (
    exact_pair,
    json_lines,
    load_online,
    payload_receipt,
)
from ..mllm_epd_prefill_heavy_modality_discovery.timeline import distribution

ROUTES = {
    "native": "Native",
    "naive_isolated": "Whole-request Isolation",
    "group_late_p": "Grouped Late P",
    "group_early_p": "Grouped Early P",
    "group_guarded_d": "Grouped Guarded D",
}


def records(path):
    return list(json_lines(path)) if path.is_file() else []


def read_optional(path):
    return json.loads(path.read_bytes()) if path.is_file() else None


def by_request(paths, key):
    hits = defaultdict(list)
    for path in sorted(paths):
        for record in json_lines(path):
            if (identity := record.get(key)) is not None:
                hits[identity].append(record)
    return hits


def gaps_ms(rows, start, end):
    return distribution(1000 * (row[end] - row[start]) for row in rows
                        if row.get(start) is not None and row.get(end) is not None)


def encoder_request_id(engine_request_id):
    match = re.fullmatch(r"chatcmpl-(.+):E:full-[a-f0-9]+", engine_request_id)
    return match[1] if match else None


def encoder_groups(epoch):
    by_request = defaultdict(list)
    for path in sorted(epoch.glob("raw/group_encoder.jsonl*")):
        for record in json_lines(path):
            for internal_id in record["requests"]:
                if (request := encoder_request_id(internal_id)) is not None:
                    by_request[request].append(record)
    return by_request


def gpu_counter_evidence(epoch, tag, clients):
    source = epoch / "raw" / f"{tag}_gpm.json"
    gpm = read_optional(source)
    if not gpm or not clients:
        return None
    first = min(row["scheduled_arrival"] for row in clients)
    last = max(row.get("finish", row["scheduled_arrival"]) for row in clients)
    by_role = {}
    for role in gpm["roles"]:
        counters = defaultdict(list)
        for sample in gpm["samples"]:
            values = sample.get("roles", {}).get(role, {})
            begin, end = values.get("sample_window", (0, 0))
            if first <= begin < end <= last:
                for key, value in values["values"].items():
                    if value is not None:
                        counters[key].append((value, end - begin))
        by_role[role] = {
            key: {
                "duration_weighted_mean": sum(value * duration for value, duration in entries)
                / sum(duration for _, duration in entries),
                "complete_windows": len(entries),
                "sampled_seconds": sum(duration for _, duration in entries),
            } for key, entries in counters.items()
        }
    return {"source_receipt": payload_receipt(source), "roles": by_role, "units": gpm["units"],
            "joined": gpm["joined"], "scope": "Whole GPM sample windows within first planned arrival and last finish. "
            "Device-level sampled counters. This has no separate Observer 5% qualification for attribution."}


def trace_evidence(epoch, tag, clients, proxies, retirements, groups):
    selected = {}
    for client in clients:
        rows = proxies.get(client["runtime_request_id"], ())
        if len(rows) == 1 and rows[0]["status"] == "MEASURED_ONLINE":
            selected[client["runtime_request_id"]] = rows[0]
    rows = list(selected.values())
    p_requests = sum(row.get("P_submit") is not None for row in rows)
    original_tokens = {c["runtime_request_id"]: c["input_tokens_manifest"] for c in clients}

    early_visual = defaultdict(list)
    early_visual_lower = defaultdict(list)
    retired_observed = Counter()
    for row in rows:
        full_cuda, start = row.get("all_images_cuda_ready_ns"), row.get("first_visual_offset")
        if not (isinstance(full_cuda, int) and isinstance(start, int) and row.get("cut")):
            continue
        role = "decode" if row.get("deflected") else "prefill"
        visual = [event["retire_ns"] for event in retirements.get(row["key"], ())
                  if event["role"] == role and event["retired_rows"] > start]
        if visual:
            retired_observed[role] += 1
            earliest = min(visual)
            early_visual[role].append((full_cuda - earliest) / 1e6)
            encoded = groups.get(row["runtime_request_id"], ())
            if encoded and max(value["cuda_ready_upper_ns"] for value in encoded) == full_cuda:
                lower = max(value["cuda_ready_lower_ns"] for value in encoded)
                early_visual_lower[role].append((lower - earliest) / 1e6)

    pressure = records(epoch / "raw" / f"{tag}_scheduler.jsonl")
    queue_metrics = {}
    for role in ("encoder", "prefill", "decode"):
        queue_metrics[role] = {}
        for metric in ("vllm:num_requests_running", "vllm:num_requests_waiting",
                       "vllm:num_requests_waiting_by_reason:capacity"):
            samples = [sample.get("values", {}).get(role, {}).get(metric) for sample in pressure]
            summary = distribution(value for value in samples if value is not None)
            if summary["min"] is not None:
                queue_metrics[role][metric] = summary
    return {
        "evidence": "TRACE_ANALYSIS",
        "proxy_rows_joined": len(selected),
        "route_counts": dict(Counter(row.get("arm") or ("D_local" if row.get("deflected") else "P_to_D") for row in rows)),
        "decision_counts": dict(Counter(row["reason"] for row in rows if row.get("reason"))),
        "D_local_completed": sum(row.get("deflected") is True for row in rows),
        "P_HTTP_submitted": p_requests,
        "P_HTTP_prompt_tokens_logical": sum(original_tokens[key] for key, row in selected.items() if row.get("P_submit") is not None),
        "E_HTTP_submitted": sum(row.get("E_submit") is not None for row in rows),
        "proxy_E_submit_to_ack_ms": gaps_ms(rows, "E_submit", "E_complete"),
        "proxy_E_ack_to_P_submit_ms": gaps_ms(rows, "E_complete", "P_submit"),
        "P_HTTP_submit_to_ready_ms": gaps_ms(rows, "P_submit", "P_ready"),
        "P_submit_before_E_ack": sum(row.get("P_submit") is not None and row.get("E_complete") is not None
                                     and row["P_submit"] < row["E_complete"] for row in rows),
        "visual_retirement": {
            role: {
                "observed_requests": retired_observed[role],
                "first_visual_retired_before_all_observed_ready": sum(value > 0 for value in early_visual[role]),
                "all_CUDA_ready_minus_first_visual_retire_ms": distribution(early_visual[role]),
                "producer_bounds_joined_requests": len(early_visual_lower[role]),
                "first_visual_retired_before_producer_lower_bound":
                    sum(value > 0 for value in early_visual_lower[role]),
                "all_CUDA_ready_conservative_lower_minus_first_visual_retire_ms": distribution(early_visual_lower[role]),
            } for role in ("prefill", "decode")
        },
        "scheduler_pressure_sample_count": len(pressure),
        "scheduler_pressure_samples": queue_metrics,
        "GPU_counter_windows": gpu_counter_evidence(epoch, tag, clients),
        "scope": "HTTP spans include queues and transfers. E readiness and retirement are observed event timestamps. "
                 "Logical prompt rows and periodic scheduler samples do not measure isolated GPU work.",
    }


def prime_evidence(epoch, setup):
    tag, count = setup["tag"], setup.get("prime_count")
    file = epoch / "state" / f"{tag}_prime_complete.json"
    receipt = read_optional(file) or {}
    return {
        "expected": count,
        "completed": receipt.get("completed"),
        "matched_24_request_prime": count == 24 and receipt.get("planned_count") == count
        and receipt.get("completed") == count and receipt.get("passed") is True,
        "receipt": payload_receipt(file) if file.is_file() else None,
    }


def raw_digest(receipt):
    path = Path(receipt["path"])
    if path.suffix != ".gz":
        return receipt["sha256"]
    with gzip.open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize_arm(epoch, arm, planned, proxies, retirements, groups, frozen, members):
    row, clients, arrival = arm
    row = dict(row)
    tag = row["tag"]
    row["prime"] = prime_evidence(epoch, planned)
    if row["evidence"] == "MEASURED_ONLINE":
        receipt = row["raw_receipt"]
        member = members.get(tag)
        uncompressed = raw_digest(receipt)
        row["client_member_uncompressed_sha256"] = uncompressed
        if member:
            row["client_scope"] = "SANITIZED_CLIENT_PROJECTION"
            row["raw_integrity"] = (
                member["original_raw_sha256"] == row["producer_declared_raw_sha256"]
                and member["sanitized_archive_sha256"] == receipt["sha256"]
                and member["sanitized_uncompressed_sha256"] == uncompressed
                and member["record_count"] == len(clients))
        else:
            row["client_scope"] = "ORIGINAL_CLIENT_TRACE"
            row["raw_integrity"] = uncompressed == row["producer_declared_raw_sha256"]
        row["pilot_contract"] = bool(
            row["planned"] == 24 and row["setup"]["qps"] == 1.465 and row["setup"]["output"] == 16
            and row["setup"]["arrival"] == "poisson"
            and arrival and arrival["seed"] == 2026112809 and not arrival.get("serial", False)
            and (frozen is None or all(arrival[key] == frozen[key] for key in ("seed", "QPS", "trace"))))
        row["matched_to_supplied_frozen_schedule"] = frozen is not None and row["pilot_contract"]
        row["trace"] = trace_evidence(epoch, tag, clients, proxies, retirements, groups)
        row["clients"] = sorted(clients, key=lambda item: item["index"])
    elif row["evidence"] not in {"DIAGNOSTIC", "NOT_RUN_TERMINATED"} and not (epoch/"state/complete.json").is_file():
        row["evidence"] = "PENDING"
    return row, clients, arrival


def paired(before, after):
    pair = exact_pair(before, after)
    a, b = before[0], after[0]
    pair["baseline_route"], pair["candidate_route"] = a["setup"]["route"], b["setup"]["route"]
    group = a["setup"].get("mode")
    pair["same_encoder_group_mode"] = bool(group and group == b["setup"].get("mode"))
    pair["matched_24_request_prime"] = all(item["prime"]["matched_24_request_prime"] for item in (a, b))
    pair["native_reference"] = pair["baseline_route"] == "native"
    pair["strict_gate_eligible"] &= bool(
        pair["same_server_start"] and pair["matched_24_request_prime"]
        and a["setup"].get("ec_policy") == b["setup"].get("ec_policy") == "inline"
        and all(item["raw_integrity"] and item["pilot_contract"] for item in (a, b))
        and (not (pair["baseline_route"].startswith("group_") and pair["candidate_route"].startswith("group_"))
             or pair["same_encoder_group_mode"]))
    pair["strict_decode_verdict"] = (
        "PASS" if all(pair["strict_decode_gates"].values()) else "FAIL"
    ) if pair["strict_gate_eligible"] else "INELIGIBLE"
    delta = pair["candidate_over_baseline_relative_change"]
    pair["TTFT_p95_at_least_15pct_faster"] = delta["TTFT_ms"] is not None and delta["TTFT_ms"] <= -.15
    pair["joint_goodput_at_least_15pct_higher"] = (
        delta["joint_goodput_req_s"] is not None and delta["joint_goodput_req_s"] >= .15)
    pair["numeric_native_gate_pass"] = (
        pair["strict_decode_verdict"] == "PASS" and pair["TTFT_p95_at_least_15pct_faster"]
    ) if pair["native_reference"] and pair["strict_gate_eligible"] else None
    pair["D_local_completed_witnesses"] = b["trace"]["D_local_completed"]
    return pair


def compare_recipients(arms):
    complete = {record["setup"]["route"]: (record, raw, arrival) for record, raw, arrival in arms
                if record["evidence"] == "MEASURED_ONLINE" and record["complete"]}
    controls = (
        ("native", ("naive_isolated", "group_late_p", "group_early_p", "group_guarded_d")),
        ("naive_isolated", ("group_early_p", "group_guarded_d")),
        ("group_late_p", ("group_early_p", "group_guarded_d")),
        ("group_early_p", ("group_guarded_d",)),
    )
    return [paired(complete[reference], complete[candidate])
            for reference, candidates in controls for candidate in candidates
            if reference in complete and candidate in complete]


def reduce_epoch(epoch, frozen, plan_path):
    plan_path = plan_path or epoch / "configs/frozen_plan.json"
    plan = read_optional(plan_path) or {}
    planned = {item["tag"]: item for item in plan.get("cases", ())}
    members = read_optional(epoch / "archive_client_members.json") or {}
    proxies = by_request(epoch.glob("raw/proxy_*.jsonl*"), "runtime_request_id")
    retirements = by_request(epoch.glob("raw/ready_retired_*.jsonl*"), "key")
    groups = encoder_groups(epoch)
    available = load_online(epoch.parent, "FRESH", names=[epoch.name])
    known = {row["tag"] for row, _, _ in available}
    pending = [{"setup": setup, "tag": tag, "evidence": "PENDING",
                "prime": prime_evidence(epoch, setup)} for tag, setup in planned.items() if tag not in known]
    arms = [summarize_arm(epoch, arm, planned.get(arm[0]["tag"], arm[0]["setup"]), proxies,
                          retirements, groups, frozen, members) for arm in available]
    native_pairs = compare_recipients(arms)
    return {"epoch": epoch.name, "pilot_only": True, "frozen_arrivals_supplied": frozen is not None,
            "plan_receipt": payload_receipt(plan_path) if plan_path.is_file() else None,
            "arms": [record for record, _, _ in arms] + pending, "pairs": native_pairs,
            "quality_scope": "Observed identity compares prompt and generated token sequences. Scored task quality is outside this pilot."}


def archive_pilot(epoch, destination, data, frozen_arrival=None):
    if destination.resolve().is_relative_to(epoch.resolve()):
        raise ValueError("Archive destination must be outside the live source epoch")
    terminal = ("state/complete.json", "state/failure.json")
    if not any((epoch / path).is_file() for path in terminal):
        raise ValueError("Archive after the online epoch finishes")
    cases = [row for row in data["arms"] if row["evidence"] == "MEASURED_ONLINE"]
    if not cases or not all(row["raw_integrity"] and row["client_scope"] == "ORIGINAL_CLIENT_TRACE"
                            for row in cases):
        raise ValueError("No SHA-verified original client trace to sanitize")
    frozen = ("configs/frozen_plan.json", "configs/cohorts.jsonl",
              "state/environment.json", "state/model_download_complete.json")
    receipts = {}
    for relative in (*frozen, *terminal):
        source = epoch / relative
        if source.is_file():
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            if relative == "state/failure.json":
                failure = read_optional(source)
                output.write_text(json.dumps({key: failure[key] for key in ("status", "type", "error_type", "time")
                                              if key in failure}) + "\n")
            else:
                shutil.copyfile(source, output)
            receipts[relative] = payload_receipt(source)
    if frozen_arrival is not None:
        key = "configs/frozen_primary_n24_arrival.json"
        shutil.copyfile(frozen_arrival, destination / key)
        receipts[key] = payload_receipt(frozen_arrival)

    def pack(path, rows):
        packed = destination / path
        with packed.open("wb") as file, gzip.GzipFile("", "wb", fileobj=file, mtime=0) as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode() + b"\n")

    client_ids = {client["runtime_request_id"] for row in cases for client in row["clients"]}
    compact_fields = {
        "runtime_request_id", "name", "mode", "status", "arm", "deflected", "encoder_assisted", "reason",
        "input_tokens", "image_count", "key", "cut", "prefix_images", "first_visual_offset",
        "prefix_cuda_ready_ns", "prefix_observed_ready_ns", "all_images_cuda_ready_ns",
        "all_images_observed_ready_ns", "E_submit", "E_complete", "P_submit", "P_ready", "P_KV_tokens",
        "p_http_outstanding", "d_http_outstanding", "D_submit", "D_headers", "D_first_generated_token",
        "D_http_complete", "E_embedding_release",
    }
    proxies = by_request(epoch.glob("raw/proxy_*.jsonl*"), "runtime_request_id")
    proxy_rows = [row for key in sorted(client_ids) for row in proxies.get(key, ())]
    effect_keys = {row["key"] for row in proxy_rows if row.get("key") is not None}
    retirements = by_request(epoch.glob("raw/ready_retired_*.jsonl*"), "key")
    (destination / "raw").mkdir(parents=True, exist_ok=True)
    pack("raw/proxy_compact.jsonl.gz",
         ({key: value for key, value in row.items() if key in compact_fields} for row in proxy_rows))
    pack("raw/ready_retired_compact.jsonl.gz",
         (item for key in sorted(effect_keys) for item in retirements.get(key, ())))
    encoded = epoch / "raw/group_encoder.jsonl"
    if encoded.is_file():
        pack("raw/group_encoder.jsonl.gz", (
            {key: row[key] for key in ("requests", "cuda_ready_lower_ns", "cuda_ready_upper_ns")}
            for row in json_lines(encoded)
            if any(encoder_request_id(request) in client_ids for request in row["requests"])))
    for pattern in ("raw/proxy_*.jsonl*", "raw/ready_retired_*.jsonl*", "raw/group_encoder.jsonl*"):
        for source in sorted(epoch.glob(pattern)):
            receipts[str(source.relative_to(epoch))] = payload_receipt(source)
    members = {}
    for row in cases:
        tag = row["tag"]
        relative = f"raw/{tag}_requests.jsonl.gz"
        pack(relative, row["clients"])
        sanitized = payload_receipt(destination / relative)
        members[tag] = {
            "original_raw_sha256": row["producer_declared_raw_sha256"],
            "sanitized_archive_sha256": sanitized["sha256"],
            "sanitized_uncompressed_sha256": raw_digest(sanitized),
            "record_count": len(row["clients"]),
            "projection": "Original client timestamps, scores, emitted token IDs and prompt/output digests.",
        }
        client = Path(row["raw_receipt"]["path"])
        receipts[str(client.relative_to(epoch))] = row["raw_receipt"]
        for relative in (f"configs/{tag}_arrival.json", f"state/{tag}_complete.json",
                         f"state/{tag}_prime_complete.json", f"raw/{tag}_scheduler.jsonl", f"raw/{tag}_gpm.json"):
            source = epoch / relative
            if source.is_file():
                shutil.copyfile(source, destination / relative)
                receipts[relative] = payload_receipt(source)
    (destination / "raw_source_receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
    (destination / "archive_client_members.json").write_text(json.dumps(members, indent=2) + "\n")


def number(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def pct(value):
    return "—" if value is None else f"{value:+.2%}"


def markdown(data):
    lines = [
        "# Grouped Encode open-loop N24 pilot",
        "",
        "Very-long Fixed16, QPS 1.465, seed 2026112809. Each measured arm uses the same H200 allocation if the matched checks pass. "
        "These pilot cohorts cover one server start. Quantiles describe completed requests and attainment uses the planned denominator.",
        "",
        "| Arm | Status | Complete | TTFT p95 (s) | TPOT p95 (ms) | ITL p99 (ms) | Decode | Joint req/s | D-local |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in data["arms"]:
        scope = ROUTES.get(record["setup"].get("route"), record["tag"])
        if record["evidence"] != "MEASURED_ONLINE":
            lines.append(f"| {scope} | {record['evidence']} | — | — | — | — | — | — | — |")
            continue
        metrics = record["metrics_ms"]
        ttft, tpot, itl = metrics["TTFT_ms"], metrics["TPOT_ms"], metrics["ITL_ms"]
        count = record["planned"]
        absolute = record["attainment_counts"]["absolute_decode"]
        d_count = record["trace"]["D_local_completed"] if record["trace"]["proxy_rows_joined"] else None
        fields = (scope, record["evidence"], f"{record['completed']}/{count}",
                  number(ttft["p95"] / 1000 if ttft["p95"] is not None else None), number(tpot.get("p95")),
                  number(itl.get("p99")), f"{absolute}/{count}", number(record["joint_goodput_req_s"]),
                  str(d_count) if d_count is not None else "—")
        lines.append("| " + " | ".join(fields) + " |")
    lines.extend((
        "",
        "| Matched reference → candidate | Reply prompts / outputs identical | Δ TTFT p95 | Δ TPOT p95 | Δ ITL p99 | Δ Joint goodput | Decode relative to reference | 15% TTFT gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ))
    for pair in data["pairs"]:
        root, candidate = (ROUTES[pair[name]] for name in ("baseline_route", "candidate_route"))
        pairs = pair["paired_requests_recorded"]
        identity = f"{pair['full_prompt_exact_requests']}/{pairs} / {pair['full_output_exact_requests']}/{pairs}"
        delta = pair["candidate_over_baseline_relative_change"]
        result = ("PASS" if pair["TTFT_p95_at_least_15pct_faster"] else "FAIL") if pair["strict_gate_eligible"] else "INELIGIBLE"
        fields = (f"{root} → {candidate}", identity,
                  *(pct(delta[key]) for key in ("TTFT_ms", "TPOT_ms", "ITL_ms", "joint_goodput_req_s")),
                  pair["strict_decode_verdict"], result)
        lines.append("| " + " | ".join(fields) + " |")
    lines.extend((
        "",
        "Native-relative Decode limits are +5% on TPOT p95 and ITL p99, at least 95% absolute Decode attainment, "
        "and a pass count at least equal to Native. Grouped comparisons against other references isolate the time and recipient policy.",
        "",
        "| Arm | Actual route counts | P submitted | P logical rows | E submit→ACK p50 (ms) | P submit→ready p50 (ms) | P submit before E ACK | Proven early Visual retire, P / D |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ))
    for record in data["arms"]:
        if not (trace := record.get("trace")) or not trace["proxy_rows_joined"]:
            continue
        routes = ", ".join(f"{route} {value}" for route, value in sorted(trace["route_counts"].items()))
        event = trace["visual_retirement"]
        timing = "/".join(f"{event[stage]['first_visual_retired_before_producer_lower_bound']} "
                          f"of {event[stage]['producer_bounds_joined_requests']}"
                          for stage in ("prefill", "decode"))
        fields = (ROUTES[record["setup"]["route"]], routes, str(trace["P_HTTP_submitted"]),
                  str(trace["P_HTTP_prompt_tokens_logical"]), number(trace["proxy_E_submit_to_ack_ms"].get("p50"), 1),
                  number(trace["P_HTTP_submit_to_ready_ms"].get("p50"), 1), str(trace["P_submit_before_E_ack"]), timing)
        lines.append("| " + " | ".join(fields) + " |")
    lines.extend(("", "A proven early retirement precedes even the producer's lower completion bound. "
                  "Observed upper-bound counts are retained separately in the receipt.",
                  "", data["quality_scope"], "", "[Raw-derived paired receipts, events and pressure samples](group_online_pilot.json)"))
    return "\n".join(lines) + "\n"


def plot_pilot(data, output):
    completed = [record for record in data["arms"] if record["evidence"] == "MEASURED_ONLINE"
                 and record["complete"]]
    if len(completed) < 2:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#376b85", "#a14f00", "#79568c", "#298775", "#ad3c6d")
    by_route = dict(zip(ROUTES, colors, strict=True))
    plt.rcParams["svg.hashsalt"] = "group-online-pilot"
    plt.rcParams["svg.fonttype"] = "none"
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, field, percent, absolute, label in (
        (axes[0], "TPOT_ms", "p95", 33, "TPOT p95 (ms)"),
        (axes[1], "ITL_ms", "p99", 34, "ITL p99 (ms)"),
    ):
        levels = []
        for row in completed:
            route = row["setup"]["route"]
            yy = row["metrics_ms"][field][percent]
            xx = row["metrics_ms"]["TTFT_ms"]["p95"] / 1000
            axis.scatter(xx, yy, color=by_route[route], s=55, label=ROUTES[route], zorder=3)
            levels.append(yy)
            if route == "native":
                bound = 1.05 * yy
                levels.append(bound)
                axis.axhline(bound, linestyle="--", color="#376b85", linewidth=1, label="Native +5%")
        margin = max(.05 * max(levels), .15 * (max(levels) - min(levels)))
        if min(levels) - margin <= absolute <= max(levels) + margin:
            axis.axhline(absolute, linestyle=":", color="#64748b", linewidth=1, label="Absolute limit")
        axis.set_ylim(max(0, min(levels) - margin), max(levels) + margin)
        axis.set_xlabel("TTFT p95 (s)")
        axis.set_ylabel(label)
        axis.grid(alpha=.25)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Grouped Encode open-loop N24 pilot · matched-epoch measurements")
    figure.savefig(output, metadata={"Date": None})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--frozen-arrival", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--archive-pilot", type=Path, help="Archive sanitized gzip client rows and compact independent audit receipts")
    options = parser.parse_args()
    frozen = read_optional(options.frozen_arrival) if options.frozen_arrival else None
    if options.frozen_arrival and frozen is None:
        parser.error(f"Missing frozen arrival file: {options.frozen_arrival}")
    data = reduce_epoch(options.epoch, frozen, options.plan)
    options.out.mkdir(parents=True, exist_ok=True)
    (options.out / "group_online_pilot.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    (options.out / "group_online_pilot.md").write_text(markdown(data))
    if options.figures:
        plot_pilot(data, options.out / "group_online_pilot.svg")
    if options.archive_pilot:
        archive_pilot(options.epoch, options.archive_pilot, data, options.frozen_arrival)


if __name__ == "__main__":
    main()
