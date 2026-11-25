"""Render the wide H200 experiment reports from validated run receipts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

from poc_flashvep.mllm_epd_h200_prefill_heavy_batch_matched import analysis as prior

from .analysis import CAPS, CLEAN_OBSERVATIONS, NATIVE_LOADS, SMALL_CAP_LOADS, WideCohort


PREFIX = "mllm_epd_h200_"
SANITY_LATENCIES = ("TTFT_p50_ms", "TTFT_p95_ms", "ITL_p99_ms", "TPOT_p95_ms")


def show(value, digits=2):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pct(value):
    return f"{100 * float(value):.2f}%" if value is not None and value != "" else "—"


def md_table(columns, rows):
    def line(cells):
        return "| " + " | ".join(show(cell).replace("|", "\\|").replace("\n", " ") for cell in cells) + " |"

    return "\n".join((line(columns), line(["---"] * len(columns)), *(line(row) for row in rows)))


def read_csv(path):
    if path is None or not path.is_file():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def put(docs, key, title, body):
    (docs / f"{PREFIX}{key}.md").write_text(f"# {title}\n\n{body.strip()}\n")


def link(key, label):
    return f"[{label}]({PREFIX}{key}.md)"


def sanity_evidence(runs):
    results, requests, pair_tokens = [], [], {}
    tag = "calibration32_sanity"
    for start, label in (("sanity_stock", "Stock vLLM 1 GPU"), ("sanity_native", "NativeStrong EPD 3 GPU")):
        root = runs / start
        failure = root / "state/failure.json"
        if failure.is_file():
            results.append({"arm": label, "evidence": "FAILED_EXECUTION",
                            "reason": prior.read(failure).get("message", "")})
            continue
        summary_path = root / "analysis" / f"{tag}_metrics.json"
        complete_path = root / "state" / f"{tag}_complete.json"
        if not summary_path.is_file() or not complete_path.is_file():
            results.append({"arm": label, "evidence": "NOT_RUN", "reason": "Completion receipt unavailable"})
            continue
        summary, complete = prior.read(summary_path), prior.read(complete_path)
        raw = root / "raw" / f"{tag}_requests.jsonl"
        rows = prior.jsonl(raw)
        cfg = prior.read(root / "configs" / f"{tag}.json")
        env = prior.read(root / "state/environment.json")
        if (not summary["primary_execution_valid"] or not complete["passed"]
                or prior.digest(raw) != summary["raw_SHA256"] or summary["raw_SHA256"] != complete["raw_SHA256"]
                or len(rows) != 32 or len({row["request_id"] for row in rows}) != 32 or not complete["serial"]
                or any(row["status"] != "MEASURED_ONLINE" for row in rows)):
            results.append({"arm": label, "evidence": "FAILED_EXECUTION",
                            "reason": "Completion, serial, frozen roster, or raw receipt validation failed"})
            continue
        proxy_path = root / "raw/proxy_epd.jsonl"
        proxy = {
            row["runtime_request_id"]: row for row in prior.jsonl(proxy_path, appending=True)
            if row["status"] == "MEASURED_ONLINE"
        } if proxy_path.is_file() else {}
        requests_start = len(requests)
        for row in rows:
            selected = proxy.get(row["runtime_request_id"])
            intervals = prior.stage_intervals(selected) if selected is not None else {}
            pair_tokens.setdefault(row["request_id"], []).append(row["token_ids"])
            requests.append({
                "arm": label, "index": row["index"], "request_id": row["request_id"],
                "input_tokens": row["input_tokens_manifest"], "prompt_sha256": row["prompt_SHA256"],
                "payload_sha256": row["payload_SHA256"], "actual_input_tokens": row["usage"]["prompt_tokens"],
                "engine_prompt_token_sha256": hashlib.sha256(json.dumps(row.get("prompt_token_ids")).encode()).hexdigest(),
                "output_tokens": row["output_tokens"], "finish_reason": row["finish_reason"],
                "output_token_sha256": hashlib.sha256(json.dumps(row["token_ids"]).encode()).hexdigest(),
                "TTFT_ms": row["TTFT_ms"], "TPOT_ms": row["TPOT_ms"], "ITL_ms": json.dumps(row["ITL_ms"]),
                **{f"{key}_ms": (right - left) * 1000 for key, (left, right) in intervals.items()},
            })
        results.append({
            "arm": label, "evidence": "MEASURED_ONLINE", "receipt_sha256": env["receipt_SHA256"],
            "wheel_sha256": env["runtime"]["vllm_wheel_sha256"],
            "gpu_map": tuple(sorted(env["authorized_uuid_by_gpu"].items())), "seed": cfg["seed"],
            "input_tokens": sum(row["usage"]["prompt_tokens"] for row in rows), "start": start,
            "validated_proxy_rows": sum(row["runtime_request_id"] in proxy for row in rows),
            "E_HTTP_p50_ms": prior.percentile(
                [row["E HTTP_ms"] for row in requests[requests_start:] if "E HTTP_ms" in row], .5),
            "P_HTTP_p50_ms": prior.percentile(
                [row["P HTTP_ms"] for row in requests[requests_start:] if "P HTTP_ms" in row], .5),
            "D_HTTP_p50_ms": prior.percentile(
                [row["D HTTP_ms"] for row in requests[requests_start:] if "D HTTP_ms" in row], .5),
            **{key: summary[key] for key in (*SANITY_LATENCIES, "completed_count", "total_output_tokens",
                                            "duration_seconds", "completed_requests_per_second",
                                            "raw_output_tokens_per_second", "ITL_mean_ms")},
        })
    paired = False
    if all(row["evidence"] == "MEASURED_ONLINE" for row in results):
        roster = [
            [(row["index"], row["request_id"], row["prompt_sha256"], row["input_tokens"],
              row["actual_input_tokens"], row["payload_sha256"], row["engine_prompt_token_sha256"])
             for row in requests if row["arm"] == result["arm"]]
            for result in results
        ]
        shared_state = ("receipt_sha256", "wheel_sha256", "gpu_map", "seed")
        paired = roster[0] == roster[1] and all(results[0][key] == results[1][key] for key in shared_state)
    if paired:
        for row in requests:
            a, b = pair_tokens[row["request_id"]]
            row["first_output_token_difference_1based"] = next(
                (i for i, (x, y) in enumerate(zip(a, b), 1) if x != y),
                min(len(a), len(b)) + 1 if len(a) != len(b) else None)
    return results, requests, paired


def wide_capacity(runs):
    arms = {}
    for path in sorted(runs.glob("*/analysis/capacity_decode_wide_*.json")):
        value = prior.read(path)
        if "cap" in value and "repeat" in value:
            arms[path.parent.parent.name, str(value["cap"]), value["repeat"]] = value
    for path in sorted(runs.glob("*/analysis/capacity_decode_wide.json")):
        value = prior.read(path)
        if value.get("evidence") == "ISOLATED_GPU_MEASUREMENT":
            for arm in value["arms"]:
                arms[path.parent.parent.name, str(arm["cap"]), arm["repeat"]] = arm
    result = []
    for (start, cap, repeat), arm in sorted(
            arms.items(), key=lambda item: (CAPS.index(item[0][1]), item[0][0], item[0][2])):
        keep = ("requests", "tokens", "seconds", "requests_per_second", "tokens_per_second", "service_p50_ms",
                "ITL_p99_ms", "TPOT_p95_ms", "KV_input_tokens_p50", "KV_input_tokens_p95",
                "maximum_D_HTTP_inflight", "actual_cap_reached", "mean_TBT_ms",
                "D_HTTP_cap_binding_with_pending_fraction")
        result.append({"start": start, "cap": cap, "repeat": repeat, "evidence": "ISOLATED_GPU_MEASUREMENT",
                       **{key: arm.get(key) for key in keep}, **arm.get("maximum_actual_D_geometry", {}),
                       "prepared_decode_call_histogram": arm.get("prepared_decode_call_histogram", {}),
                       "graph_mode_histogram": arm.get("graph_mode_histogram", {})})
    return result


def render_sanity(docs, summaries, requests, paired):
    path = docs / f"{PREFIX}runtime_performance_sanity_check.md"
    begin = "<!-- WIDE_SANITY_START -->"
    end = "<!-- WIDE_SANITY_END -->"
    text = path.read_text()
    before, remainder = text.split(begin, 1)
    _, after = remainder.split(end, 1)
    columns = ("Arm", "Evidence", "Completions", "Input / output tokens", "TTFT p50 / p95 ms", "ITL p99 ms",
               "TPOT p95 ms", "Requests/s", "Output tokens/s")
    table = md_table(columns, [
        (row["arm"], row["evidence"], row.get("completed_count"),
         f"{show(row.get('input_tokens'))} / {show(row.get('total_output_tokens'))}",
         " / ".join(show(row.get(key)) for key in SANITY_LATENCIES[:2]),
         row.get("ITL_p99_ms"), row.get("TPOT_p95_ms"),
         row.get("completed_requests_per_second"), row.get("raw_output_tokens_per_second"))
        for row in summaries
    ])
    lines = [table]
    stage_rows = [(row["arm"], row["validated_proxy_rows"], row["E_HTTP_p50_ms"], row["P_HTTP_p50_ms"],
                   row["D_HTTP_p50_ms"]) for row in summaries if row.get("validated_proxy_rows")]
    if stage_rows:
        lines.append(md_table(("Arm", "Matched proxy rows", "E HTTP p50 ms", "P HTTP p50 ms", "D HTTP p50 ms"),
                              stage_rows))
        lines.append("The separately addressed E/P/D workers expose these host phases. Stock has a single "
                     "HTTP serving worker. The phase medians include queues, transfers, and HTTP work.")
    if paired:
        left, right = summaries
        lines.append(
            f"Both arms completed the same ordered 32 requests with matching prompt and payload hashes, "
            f"engine input token IDs, runtime receipt, wheel hash, actual input lengths, and seed. The EPD minus stock TTFT difference was "
            f"{right['TTFT_p50_ms'] - left['TTFT_p50_ms']:+.2f} ms at p50 and "
            f"{right['TTFT_p95_ms'] - left['TTFT_p95_ms']:+.2f} ms at p95."
        )
        roster = {}
        for row in requests:
            roster.setdefault(row["request_id"], []).append((row["output_tokens"], row["output_token_sha256"]))
        lines.append(f"Natural EOS output counts agreed on {sum(a[0] == b[0] for a, b in roster.values())}/32 "
                     f"requests. Exact output token IDs agreed on {sum(a[1] == b[1] for a, b in roster.values())}/32. "
                     "The matched calibration cohort does not establish held-out task quality.")
        mismatches = {
            row["request_id"]: row["first_output_token_difference_1based"] for row in requests
            if row["first_output_token_difference_1based"] is not None
        }
        if mismatches:
            lines.append(md_table(("Request", "Stock / Native output tokens", "First divergent output position"), [
                (request_id, " / ".join(str(length) for length, _ in roster[request_id]), position)
                for request_id, position in mismatches.items()
            ]))
    elif len(requests) == 64:
        lines.append("The ordered input or environment checks differed. The two measured arms remain standalone.")
    for row in summaries:
        if row.get("reason"):
            lines.append(f"{row['arm']} · {row['reason']}.")
    path.write_text(before + begin + "\n\n" + "\n\n".join(lines) + "\n\n" + end + after)


def report_capacity(docs, arms, stages):
    columns = ("Cap", "Start / repetition", "Evidence", "Requests / tokens", "Requests/s", "Tokens/s",
               "Service p50 ms", "Mean TBT ms", "ITL p99 ms", "TPOT p95 ms", "HTTP max", "HTTP binding",
               "Scheduled / prepared max", "Padded rows max", "Target attained")
    table_rows = []
    for cap in CAPS:
        selected = [row for row in arms if row["cap"] == cap]
        if not selected:
            table_rows.append((cap, None, "NOT_RUN", *([None] * (len(columns) - 3))))
        for row in selected:
            table_rows.append((cap, f"{row['start']} r{row['repeat']}", row["evidence"],
                               f"{row['requests']} / {row['tokens']}", row["requests_per_second"],
                               row["tokens_per_second"], row["service_p50_ms"], row["mean_TBT_ms"],
                               row["ITL_p99_ms"], row["TPOT_p95_ms"], row["maximum_D_HTTP_inflight"],
                               pct(row["D_HTTP_cap_binding_with_pending_fraction"]), " / ".join(show(row.get(key)) for key in
                                ("scheduled_requests", "prepared_decode_requests")),
                               row["prepared_query_rows_after_padding"], row["actual_cap_reached"]))
    stage_section = md_table(("Role", "Start", "Requests/s", "Tokens/s", "Evidence"), [
        (role, row["start"], row["requests_per_second"], row["tokens_per_second"], row["evidence"])
        for role in ("encoder", "prefill") for row in (
            [row for row in stages if row["stage"] == role]
            or [{"start": None, "requests_per_second": None, "tokens_per_second": None, "evidence": "NOT_RUN"}]
        )
    ])
    put(docs, "wide_decode_batch_capacity", "H200 measured decode capacity by admission cap", f"""
{md_table(columns, table_rows)}

The timed D drain begins after actual E/P generation of KV in each cohort. It includes NIXL import, decoding and local SSE delivery. All arms use the frozen replay output targets. GPU capacity means isolated D throughput under this ordered cohort and scheduler. HTTP concurrency, prepared decode requests, query rows and CUDA graph padding are separate quantities. A cap that was never attained is an upper bound on admission for this cohort.

The aggregate and individual arm receipts are `analysis/capacity_decode_wide.json` and `analysis/capacity_decode_wide_*_r*.json` inside their run. They retain KV input lengths, prepared request-count and graph-mode distributions, original request IDs and KV block bounds.

## E and P controls

{stage_section}

The prior same-group H200 controls and their differences in observer and cohort context are available separately. [Previous independent-stage capacity report](mllm_epd_h200_independent_stage_capacity.md)
""")


def online_evidence(runs, tables_path):
    cohorts, failures = prior.cohort_records(runs, WideCohort)
    host = {name: [] for name in ("requests", "slo_breakdown", "host_timeline",
                                 "decode_idle_windows", "critical_path")}
    rows, pilot = [], []
    for cohort in cohorts:
        row = {
            **cohort.base,
            "completed_count": cohort.metrics["completed_count"],
            **{metric: cohort.metrics[metric] for metric in prior.METRICS},
            **cohort.metrics["gate_attainment"], **prior.request_table(cohort, host),
        }
        rows.append(row)
        start_group = cohort.config["workload"].get("paired_start_id")
        if start_group and str(start_group).startswith("pilot_") and cohort.observation == "clean":
            pilot.append((cohort, row))
    extra = {"pilot_pairs": pilot_pairs(pilot)}
    if tables_path is not None:
        for name in ("native_grid", "decode_cap", "same_cap_contrasts", "free_native_contrasts", "contrast_ci"):
            extra[name] = read_csv(tables_path / f"{name}.csv")
    return rows, failures, host, extra


def pilot_pairs(pilot):
    pairs = []
    for candidate, row in pilot:
        if not candidate.config["workload"]["method"].startswith(("static", "idle")):
            continue
        for baseline, native in pilot:
            family = baseline.config["workload"]["method"]
            if candidate.run == baseline.run:
                continue
            same_condition = candidate.match == baseline.match
            if same_condition and candidate.cap == baseline.cap and family in ("native-strong", "native-common", "static10"):
                pairs.append(("Same condition", native, row))
            # The two planned H5 groups share the frozen arrival and workload
            # but have separate per-cap group labels. This comparison is N1.
            if family == "native-strong" and baseline.cap == "free" and candidate.match[:-1] == baseline.match[:-1]:
                pairs.append(("Strong Native Free", native, row))
    return pairs


def planned_grid():
    return {(cap, qps) for cap in CAPS
            for qps in (SMALL_CAP_LOADS if cap in ("1", "2") else NATIVE_LOADS)}


def pilot_deltas(pairs):
    rows = [
        (scope, baseline["method"], candidate["method"], candidate["QPS"],
         baseline["TTFT_p95_ms"], candidate["TTFT_p95_ms"],
         100 * (candidate["TTFT_p95_ms"] / baseline["TTFT_p95_ms"] - 1),
         candidate["ITL_p99_ms"] / baseline["ITL_p99_ms"],
         candidate["TPOT_p95_ms"] / baseline["TPOT_p95_ms"],
         pct(baseline["D_absolute_SLO_attainment"]), pct(candidate["D_absolute_SLO_attainment"]),
         pct(baseline["JOINT_PASS"]), pct(candidate["JOINT_PASS"]),
         baseline["SLO_goodput"], candidate["SLO_goodput"])
        for scope, baseline, candidate in pairs
        if not baseline["method"].startswith("static") or candidate["method"].startswith("idle")
    ]
    columns = ("N1 contrast", "Baseline", "Candidate", "QPS", "Baseline TTFT p95 ms", "Candidate TTFT p95 ms",
               "TTFT p95 change %", "ITL p99 ratio", "TPOT p95 ratio", "Baseline Decode SLO", "Candidate Decode SLO",
               "Baseline Joint SLO", "Candidate Joint SLO", "Baseline SLO-goodput", "Candidate SLO-goodput")
    return md_table(columns, rows) if rows else "`NOT_MEASURED`."


def discovery_evidence(runs):
    profiles = []
    by_profile = {}
    paths = set(runs.glob("discovery_gate_*.json")) | set(runs.glob("discovery_gates_*.json"))
    for path in sorted(paths, key=lambda path: (path.name == "discovery_gates_wide.json", path.stat().st_mtime_ns)):
        value = prior.read(path)
        for row in value.get("profiles", []):
            by_profile[row["run"], row["tag"]] = row
    for gate in sorted(runs.glob("*/analysis/*_profile_gate.json")):
        start = gate.parent.parent
        tag = gate.name.removesuffix("_profile_gate.json")
        record = prior.read(gate)
        trace_path = start / "analysis" / f"{tag}_trace.json"
        status_path = start / "analysis" / f"{tag}_discovery_gate.json"
        status = prior.read(status_path) if status_path.is_file() else {}
        if not status:
            status = by_profile.get((start.name, tag), {})
        root_gate = runs / f"discovery_gate_{start.name}.json"
        if not status and root_gate.is_file():
            status = prior.read(root_gate)
        trace = prior.read(trace_path) if trace_path.is_file() and not status.get("kernel_busy_fraction") else {}
        busy = status.get("kernel_busy_fraction", trace.get("kernel_busy_fraction", {}))
        p_scheduler = status.get("p_scheduler", trace.get("scheduler_samples", {}).get("prefill", {}))
        d_scheduler = status.get("d_scheduler", trace.get("scheduler_samples", {}).get("decode", {}))
        config = prior.read(start / "configs" / f"{tag}_profile.json")
        failures = [f"{key} (clean {pct(value['clean_relative_spread'])}, profile "
                    f"{pct(value['profile_relative_deviation'])})"
                    for key, value in record["metrics"].items() if not value["passed"]]
        profiles.append({
            "start": start.name, "tag": tag, "QPS": config["QPS"],
            "cap": str(config["workload"].get("decode_cap", "1" if config["workload"]["batch_one"] else "free")), "evidence": record["evidence"],
            "collector_passed": record.get("collector_passed"), "observer_passed": record["passed"],
            "failed_metrics": ", ".join(failures), "status": status.get("status", "P_HEAVY_D_IDLE_NOT_MEASURED"),
            "checks": status.get("checks", {}), "capacity_reference": status.get("capacity_reference"),
            "capture": status.get("capture_setting"), "burner": status.get("other_GPU_burners_setting"),
            "cycles": status.get("main100_cycles", 1),
            "D_clean_before_SLO": status.get("clean_decode_SLO", {}).get("clean_before", {}).get("attainment"),
            "D_clean_after_SLO": status.get("clean_decode_SLO", {}).get("clean_after", {}).get("attainment"),
            "E_busy": busy.get("encoder"), "P_busy": busy.get("prefill"), "D_busy": busy.get("decode"),
            "P_backlog": p_scheduler.get("window_fraction_recent_sample_waiting"),
            "P_req_proximity": status.get("p_service_proximity", {}).get("requests_per_second"),
            "P_tok_proximity": status.get("p_service_proximity", {}).get("tokens_per_second"),
            "D_quiet_overlap_160": status.get("robust_window_160_ms", {}).get("p_busy_overlap_fraction"),
            "D_quiet_windows_160": status.get("robust_window_160_ms", {}).get("windows"),
            "D_absolute_SLO": status.get("d_decode_attainment"),
            "D_min_KV_blocks": d_scheduler.get("free_KV_blocks_min"),
        })
    return profiles


def perf_table(rows):
    return md_table(("Start", "Method", "QPS", "Output", "Scope", "Requests", "TTFT p50 / p95 ms",
                     "ITL p99 ms", "TPOT p95 ms", "Total tokens/s", "Output tokens/s", "Requests/s",
                     "Decode / Joint SLO", "SLO-goodput"), [
        (row["start"], row["method"], row["QPS"], row["output_mode"], row["observation"], row["completed_count"],
         f"{show(row['TTFT_p50_ms'])} / {show(row['TTFT_p95_ms'])}",
         row["ITL_p99_ms"], row["TPOT_p95_ms"], row["total_input_output_tokens_per_second"],
         row["raw_output_tokens_per_second"], row["completed_requests_per_second"],
         f"{pct(row['D_absolute_SLO_attainment'])} / {pct(row['JOINT_PASS'])}", row["SLO_goodput"])
        for row in rows
    ])


def report_motivation(docs, rows, failures, analysis):
    grid = []
    native = [row for row in rows if row["method"].startswith("native-common-")
              and row["observation"] == "clean" and row["output_mode"] == "Natural EOS"
              and row["Main100_cycles"] == 1 and row["start"].startswith("online_cap_")]
    for cap in CAPS:
        for qps in (SMALL_CAP_LOADS if cap in ("1", "2") else NATIVE_LOADS):
            observed = [row for row in native if row["decode_cap"] == cap and row["QPS"] == qps]
            if not observed:
                grid.append((cap, qps, "NOT_RUN", None, None, None, None, None, None))
            for row in observed:
                grid.append((cap, qps, row["evidence"], row["start"], row["TTFT_p95_ms"], row["ITL_p99_ms"],
                             row["TPOT_p95_ms"], pct(row["D_absolute_SLO_attainment"]), pct(row["JOINT_PASS"])))
    geometry = []
    for row in analysis.get("native_grid", []):
        if (row["design"] != "PRESPECIFIED" or
                all(row[field] == "NOT_MEASURED" for field in ("kernel_evidence", "batch_evidence"))):
            continue
        geometry.append((row["decode_cap"], row["QPS"], row["burner_condition"], row["batch_evidence"],
                         row["D_max_scheduled_requests"], row["D_max_prepared_decode_requests"],
                         row["D_prepared_decode_mode"], row["attains_cap"], row["kernel_evidence"],
                         pct(row["encoder_kernel_busy_fraction"]), pct(row["prefill_kernel_busy_fraction"]),
                         pct(row["decode_kernel_busy_fraction"])))
    profiles = [row for row in rows if row["start"].startswith("free_q5_burners_")]
    matched = {}
    for condition in ("on", "off"):
        cohorts = [row for row in profiles if row["start"] == f"free_q5_burners_{condition}"
                   and row["observation"] in ("clean_before", "clean_after")]
        if len(cohorts) == 2:
            matched[condition] = cohorts
    effect = ""
    if len(matched) == 2:
        metrics = ("TTFT_p95_ms", "ITL_p99_ms", "TPOT_p95_ms", "D_absolute_SLO_attainment")
        effect = md_table(("Metric", "On clean average", "Off clean average", "Off / On"),
                          [(key, on, off, off / on if on else None) for key in metrics
                           for on, off in [(mean(row[key] for row in matched["on"]),
                                            mean(row[key] for row in matched["off"]))]])
    failure_section = ""
    if failures:
        failure_section = "## Excluded executions\n\n" + md_table(
            ("Start", "Cohort", "Evidence", "Reason"),
            [(row["start"], row["cohort"], row["evidence"], row["reason"]) for row in failures])
    put(docs, "decode_batch_qps_motivation", "H200 decode cap and online load motivation", f"""
## NativeCommon Primary Main100 Natural EOS

{md_table(("D cap", "QPS", "Evidence", "Start", "TTFT p95 ms", "ITL p99 ms", "TPOT p95 ms",
           "Decode SLO", "Joint SLO"), grid)}

Completed {len({(row["decode_cap"], row["QPS"]) for row in native} & planned_grid())}/{len(planned_grid())} prespecified clean cells. Clean results use all Main100 requests from first scheduled arrival through drain. Actual batch distributions and queue and SLO breakdowns are recorded per start in the accompanying CSVs. The uninstrumented cells support latency and throughput claims. An observed engine geometry comes from the corresponding counter cohort.

## Prepared geometry and kernel busy on instrumented cells

{md_table(("Cap", "QPS", "Burners", "Batch evidence", "Scheduled max", "Prepared max", "Prepared mode", "Attained",
           "Kernel evidence", "E busy", "P busy", "D busy"), geometry) if geometry else "`NOT_MEASURED`."}

## Free QPS5 and host load

{perf_table(profiles) if profiles else "`NOT_RUN`."}

{effect}

The burners-on arm permits only the five off-group GPUs to run the existing burner. The burners-off arm stops all eight burners. Both reserve the same experiment E/P/D GPUs. Clean-before and clean-after are sequential cohorts within each one-worker epoch. Absolute host effects require comparing clean rows with identical request arrival and output contracts. Cross-arm timing differences may include thermal, CPU and independent-start effects. Early burners-on evidence has lease and process records. Full host snapshots were added for the burners-off and later arms, so temperature, clocks, power and CPU counter comparisons are incomplete for the initial pair. NVML utilization is auxiliary telemetry. The all-off full-cohort profile overlapped a CPU regression job for about 14 seconds. Its profile deviation is confounded, while the clean-before Decode SLO miss and clean spread were observed outside that overlap.

{failure_section}
""")
    return native


def report_discovery(docs, profiles):
    rows = [(row["start"], row["cap"], row["QPS"], row["cycles"], row["evidence"], pct(row["E_busy"]), pct(row["P_busy"]),
             pct(row["D_busy"]), pct(row["P_backlog"]),
             f"{pct(row['P_req_proximity'])} / {pct(row['P_tok_proximity'])}",
             row["D_quiet_windows_160"], pct(row["D_quiet_overlap_160"]),
             f"{pct(row['D_clean_before_SLO'])} / {pct(row['D_absolute_SLO'])} / {pct(row['D_clean_after_SLO'])}",
             row["status"]) for row in profiles]
    gates = [(row["start"], show(row["collector_passed"]), show(row["observer_passed"]),
              row["failed_metrics"], ", ".join(key for key, passed in row["checks"].items() if not passed))
             for row in profiles]
    references = sorted({row["capacity_reference"] for row in profiles if row["capacity_reference"]})
    basis = "P-capacity receipts used here · " + ", ".join(f"`{value}`" for value in references) if references else ""
    put(docs, "qualified_prefill_heavy_discovery", "H200 qualified prefill pressure and D quiet windows", f"""
Qualified observer captures: {sum(row["observer_passed"] for row in profiles)}/{len(profiles)}. Duplicate receipts for the same start and cohort count once.

{md_table(("Start", "Cap", "QPS", "Main100 cycles", "Evidence", "E busy", "P busy", "D busy", "P scheduler wait",
           "P req / token capacity proximity", "D quiet ≥160 ms", "P busy and backlog overlap",
           "Decode SLO before / profile / after", "Motivation"), rows) if rows else "`P_HEAVY_D_IDLE_NOT_MEASURED`."}

## Observer qualification

{md_table(("Start", "Collector", "Clean / profile / clean", "Failed metric and absolute deviations",
           "Failed combined prerequisites"), gates) if gates else "`NOT_RUN`."}

{basis}

Kernel-busy percentages use unioned CUPTI activity in the central p20–p80 arrival interval. P scheduler backlog is a recent sampled scheduler waiting state. P service proximity compares central completions with a finite isolated drain. Batch mix, KV drain, and cohort boundaries can yield a ratio above 100%. Quiet requires no D kernel activity and no coordinator-visible D work from P-ready through D completion. Overlap uses P kernel busy time as denominator and requires coincident sampled P backlog. It is possible to observe high E kernel busy and still meet all P and D criteria.

The combined gate requires P busy ≥80%, independent P capacity proximity ≥85% for requests and tokens, real scheduler backlog, low P starvation, at least two eligible quiet windows with ≥10% of P busy time overlapping sampled P backlog, at least 32,768 tokens of D KV headroom, and ≥95% absolute decode SLO. The clean/profile/clean tolerance is 5%. The combined gate uses the profiled cohort's Decode SLO.
Pilot eligibility also considers clean-before and clean-after attainment. Physical and operational windows measure observed availability. A future suffix fit or `DECODE_SAFE_HEADROOM_CONFIRMED` requires completed candidate evidence.
""")


def contrast_table(rows):
    return md_table(("Start", "Compared native arm", "Candidate", "QPS", "Output", "Metric",
                     "Native", "Candidate", "Candidate / Native", "Evidence"), [
        (row["start"], row["baseline"], row["method"], row["QPS"], row["output_mode"], row["metric"],
         show(float(row["baseline_value"]), 3), show(float(row["candidate_value"]), 3),
         show(float(row["ratio"]), 3) if row.get("ratio") else None, row["evidence"])
        for row in rows if row["metric"] in (*SANITY_LATENCIES, "SLO_goodput")
    ])


def comparison_effects(rows, metric, family):
    valid = [row for row in rows if row.get("metric") == metric and row.get("ratio")
             and row["evidence"] == "MEASURED_ONLINE" and row["method"].startswith(family)]
    return (" · ".join(f"{row['method']} vs {row['baseline']} at {row['QPS']} QPS "
                      f"({row['output_mode']}) {100 * (float(row['ratio']) - 1):+.2f}%"
                      for row in valid) if valid else None)


def candidate_correctness(runs, group):
    evidence = []
    for path in sorted(runs.glob("*/analysis/regression.json")):
        start = path.parent.parent
        env_path = start / "state/environment.json"
        args = prior.read(env_path).get("argv", []) if env_path.is_file() else []
        if "--mode" not in args:
            continue
        mode = args[args.index("--mode") + 1]
        if not mode.startswith(group):
            continue
        receipt = prior.read(path)
        evidence.append((start.name, mode, receipt["evidence"], str(path.relative_to(runs))))
    return evidence


def report_candidates(docs, runs, rows, analysis, profiles):
    selected = {}
    selection_path = runs / "pilot_selection.json"
    selection = prior.read(selection_path) if selection_path.is_file() else {}
    selection_note = ""
    if selection:
        selection_note = (f"`{selection['classification']}` selected {', '.join(selection['conditions'])}. "
                          f"{selection['limits']}\n\nFree rationale · {selection['Free']}\n\n"
                          f"B4 rationale · {selection['B4']}")
    for group, key, title in (
        ("static", "batch_preserving_static_token_split", "H200 batch-preserving static token split"),
        ("idle", "batch_preserving_idle_token_split", "H200 batch-preserving idle-aware token split"),
    ):
        selected[group] = [
            row for row in rows if row["method"].startswith(group) and row["observation"] in CLEAN_OBSERVATIONS
        ]
        candidate = selected[group]
        geometry = [row for row in analysis.get("decode_cap", []) if row.get("method", "").startswith(group)]
        split_rows = [(row["start"], row["method"], row["QPS"], row["split_requests"],
                       row["completed_count"], row["suffix_tokens"], pct(row["actual_offload_fraction"]),
                       pct(row["split_only_P_ready_D_HTTP_inactive_no_pending_fraction"]),
                       row["split_only_D_coordinator_wait_p50_ms"], row["split_only_D_coordinator_wait_p95_ms"])
                      for row in candidate]
        status = ("The following candidate cohorts completed. Any exploratory pilot remains bounded by its "
                  "authorization and does not qualify the underlying workload." if candidate
                  else "`NOT_RUN`. No validated online cohort for this policy is available.")
        if not profiles:
            status += " The current campaign has no combined discovery gate receipt."
        correctness = candidate_correctness(runs, group)
        pilot = [pair for pair in analysis["pilot_pairs"] if pair[2]["method"].startswith(group)]
        same = [row for row in analysis.get("same_cap_contrasts", []) if row.get("method", "").startswith(group)]
        free = [row for row in analysis.get("free_native_contrasts", []) if row.get("method", "").startswith(group)]
        geometry_rows = [(row["start"], row["cohort"], row.get("batch_evidence"),
                          row.get("D_prepared_decode_max"), row.get("D_prepared_decode_mean"),
                          pct(row.get("D_prepared_single_request_fraction")), row.get("suffix_selected"),
                          row.get("suffix_zero_active_D")) for row in geometry if row.get("D_prepared_decode_max")]
        put(docs, key, title, f"""
{status}

{selection_note}

{perf_table(candidate) if candidate else ""}

## Correctness

{md_table(("Start", "Method", "Evidence", "Receipt"), correctness) if correctness else "`NOT_MEASURED` for this campaign."}

These receipts cover the launched policy's exact reconstruction checks. Task-level quality requires separate evidence.

## Realized split and D state

{md_table(("Start", "Method", "QPS", "Split selected", "Completed", "Moved tokens", "Moved fraction",
           "Prefix-ready D inactive and no pending", "Selected wait p50 ms", "Selected wait p95 ms"),
          split_rows) if split_rows else "`NOT_MEASURED`."}

## Decode geometry and practical comparison

{md_table(("Start", "Cohort", "Batch evidence", "Prepared max", "Prepared mean", "Single-request fraction",
           "Suffix selected", "Suffix zero active D"), geometry_rows) if geometry_rows else "`NOT_MEASURED`."}

Candidate versus same-cap Native comparisons require shared
arrival trace, output mode, runtime, topology, and admission discipline. Practical benefit additionally requires the
Free Native reference. A change in ITL alone cannot establish a TTFT benefit. An actually smaller prepared-batch
distribution under split can confound the effect even under the same configured cap.

### Conditional pilot N1 contrasts

{pilot_deltas(pilot)}

### Same-cap paired cohorts

{contrast_table(same) if same else "`NOT_MEASURED`."}

### Free Native paired cohorts

{contrast_table(free) if free else "`NOT_MEASURED`."}
""")
    return selected


def report_critical(docs, host, rows):
    cohorts = {(row["start"], row["cohort"]) for row in rows if row["method"].startswith(("static", "idle"))}
    comparison_qps = {row["QPS"] for row in rows if (row["start"], row["cohort"]) in cohorts}
    comparisons = [row for row in host["critical_path"] if
                   (row["start"], row["cohort"]) in cohorts
                   or row["QPS"] in comparison_qps and row["method"].startswith("native") and row["observation"] == "clean"]
    ttft_paths = {}
    for row in comparisons:
        key = (row["start"], row["method"], row["QPS"], row["output_mode"], row["observation"])
        ttft_paths.setdefault(key, {})[row["stage"]] = row["mean_ms"]
    table = md_table(("Start", "Method", "QPS", "Output", "Observation", *prior.CRITICAL_PATH), [
        (*key, *(stages.get(label) for label in prior.CRITICAL_PATH)) for key, stages in ttft_paths.items()
    ])
    put(docs, "wide_batch_token_split_critical_path", "H200 wide-batch token split critical path", f"""
{table if ttft_paths else "`NOT_MEASURED`. No validated split cohort exists for a paired critical path."}

The seven columns partition scheduled arrival to first client output, for requests with monotonic timestamps
and complete proxy alignment. They are host clock boundaries in ms. The E and P HTTP spans include
submission, queues, transfer, execution, and response. The D submit to output boundary can include KV import,
scheduler, suffix, first worker emission, and client delivery. The mean along the common additive path can
be decomposed. Quantiles of individual stages do not add to TTFT p95.

For a split decision, check whether the drop in P HTTP/queue exceeds the extra D waiting and suffix work.
Compare the prepared decode distribution and D admission queue as well as ITL. A client-facing ITL gain
under a collapsed batch distribution is compatible with a loss in D capacity. Exact suffix kernel time and
a native-versus-split timeline require qualified GPU intervals aligned to the matching request boundary.
""")


def evidence_index(results):
    if results is None or results.name != "results":
        return ""
    relative = "../mllm_epd_h200_wide_decode_batch_token_split/results"
    figures = [
        f"- [{path.stem.replace('_', ' ')}]({relative}/{path.name})"
        for path in sorted(results.glob("*.pdf"))
    ]
    tables = [
        ("receipt.json", "Analysis receipt and table row counts"),
        ("execution_metadata.json", "Runtime fingerprint, manifest hashes, topology, launch arguments, and arrival seeds"),
        ("discovery_gates_wide.json", "Motivation audit and all eight quiet-window thresholds"),
        ("burner_restore.json", "Post-campaign GPU burner verification"),
        ("supplementary_figures_receipt.json", "Supplementary figures and timeline selection rule"),
        ("wide_sanity_requests.csv", "Calibration32 matched request inputs, outputs, and latencies"),
        ("wide_pilot_quality.json", "Pilot output equality, task-score scope, and pinned scorer provenance"),
        ("wide_pilot_quality_by_task.csv", "Pilot task-score point estimates by dataset and method"),
        ("wide_decode_capacity.csv", "Per-cap isolated decode capacity"),
        ("native_grid.csv", "Native QPS grid, scope and case status"),
        ("decode_cap.csv", "Actual wide decode geometry and D admission"),
        ("observer_qualification.csv", "Clean / profile / clean observer criteria"),
        ("quiet_overlap.csv", "D quiet overlap and denominators"),
        ("same_cap_contrasts.csv", "Same-cap candidate comparisons"),
        ("free_native_contrasts.csv", "Practical Free Native comparisons"),
        ("contrast_ci.csv", "Independent-start paired intervals"),
        ("critical_path.csv", "Additive host critical path"),
        ("slo_breakdown.csv", "Joint SLO breakdown"),
        ("missing.csv", "Measurements unavailable"),
        ("execution_failures.csv", "Failed executions"),
    ]
    references = [f"- [{label}]({relative}/{name})" for name, label in tables if (results / name).is_file()]
    sections = []
    if references:
        sections.append("## Evidence tables and inventory\n\n" + "\n".join(references))
    if figures:
        sections.append("## Measured figures\n\n" + "\n".join(figures))
    return "\n\n".join(sections)


def closeout(execution, motivation):
    if execution != "CONDITIONAL_ONLINE_PILOT" or motivation == "P_HEAVY_D_IDLE_CONFIRMED":
        return ""
    return ("\n\n## Campaign closeout\n\n"
            "The two QPS5 pilots were limited to Free and B4. All eight clean NativeStrong, "
            "NativeCommon, Static10 and Idle10 cohorts completed 100 requests each. Static10 increased "
            "TTFT p95 under both caps. Idle10 improved on Static10 and missed the 95% absolute Decode "
            "SLO under both caps. The campaign ended after the pilots. Additional ratios, online "
            "equal-output replay, independent N3 confirmation and Heldout50 were `NOT_RUN`. "
            "The independent D-only arms did use the frozen 5,269-token replay.\n\n"
            "A suffix-specific kernel cost, time-weighted actual decode geometry and active KV peak were "
            "`NOT_MEASURED`. Output counts varied under Natural EOS. The scoped Static10 correctness "
            "checks and an N1 task-score diagnostic cannot establish quality noninferiority.")


def quality_closeout(results):
    if results is None or not (results / "wide_pilot_quality_by_task.csv").is_file():
        return ""
    with (results / "wide_pilot_quality_by_task.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    comparisons = []
    for cap in ("free", "b4"):
        for method in ("static10", "idle10"):
            group = [row for row in rows if (row["cap"], row["method"]) == (cap, method)]
            exact = sum(int(row["exact_output_token_ids_equal_count"]) for row in group)
            count = sum(int(row["requests"]) for row in group)
            if count:
                comparisons.append(f"{method} {cap.upper()} {exact}/{count}")
    return ("\n\n## Pilot output and task quality\n\n"
            "The unchanged Natural EOS outputs were scored offline. Exact output token sequences "
            "matched same-cap NativeCommon on " + " · ".join(comparisons) + ". "
            "Official MMLongBench-Doc answer extraction and independent quality intervals remain unavailable. "
            "The per-task CSV retains scorer coverage and point estimates.")


def report_final(docs, sanity, paired, arms, native, profiles, candidates, analysis, results):
    completed = [row for row in sanity if row["evidence"] == "MEASURED_ONLINE"]
    if paired:
        stock, epd = sanity
        q1 = (f"Both paths completed the same 32 ordered Calibration requests. Stock and EPD TTFT p95 "
              f"were {stock['TTFT_p95_ms']:.2f} and {epd['TTFT_p95_ms']:.2f} ms. The results use different GPU counts.")
    else:
        q1 = f"{len(completed)}/2 arms completed with validated local evidence. The ordered paired comparison is pending."
    cap_rows = {cap: [row for row in arms if row["cap"] == cap] for cap in CAPS}
    q2 = " · ".join(
        f"{cap} " + (f"{mean(row['requests_per_second'] for row in rows):.2f} req/s, "
                    f"{mean(row['tokens_per_second'] for row in rows):.1f} output tok/s"
                    if rows else "`NOT_RUN`")
        for cap, rows in cap_rows.items()
    )
    q3_parts = []
    for cap in ("32", "64"):
        rows = cap_rows[cap]
        maxima = [row["prepared_decode_requests"] for row in rows if row["prepared_decode_requests"] is not None]
        text = (f"{sum(row['actual_cap_reached'] is True for row in rows)}/{len(rows)} arms reached the target, "
                f"maximum prepared decode requests {max(maxima)}" if maxima else "`NOT_MEASURED`")
        q3_parts.append(f"B{cap} {text}")
    q3 = " · ".join(q3_parts)
    def cap_label(cap):
        return "Free" if cap == "free" else "B" + cap

    def cap_result(row):
        return (f"{cap_label(row['cap'])} {row['requests_per_second']:.3f} req/s, "
                f"{row['tokens_per_second']:.1f} output tok/s, ITL p99 {row['ITL_p99_ms']:.2f} ms, "
                f"TPOT p95 {row['TPOT_p95_ms']:.2f} ms")

    eligible = [row for row in arms if row["ITL_p99_ms"] <= 34 and row["TPOT_p95_ms"] <= 33]
    q4 = "Isolated cap trade-off is unestablished."
    if arms and eligible:
        best = max(eligible, key=lambda row: row["requests_per_second"])
        peak = max(arms, key=lambda row: row["requests_per_second"])
        q4 = (f"The fastest observed isolated cap within ITL p99 ≤34 ms and TPOT p95 ≤33 ms was {cap_result(best)}. "
              f"The raw throughput peak was {cap_result(peak)}. "
              "Isolated latency quantiles do not measure per-request absolute Decode SLO or prove an online winner.")
    grid = {(row["decode_cap"], row["QPS"]) for row in native} & planned_grid()
    q4 += f" Clean online NativeCommon screening completed {len(grid)}/{len(planned_grid())} cells."

    confirmed = [row for row in profiles if row["status"] == "P_HEAVY_D_IDLE_CONFIRMED"]
    if confirmed:
        conditions = {f"{cap_label(row['cap'])} QPS{row['QPS']:g}" for row in confirmed}
        q5 = "The combined gate confirmed " + ", ".join(sorted(conditions)) + "."
    else:
        q5 = "No cap/QPS condition passed the combined gate."
    representative = [
        row for row in profiles if row["start"] in ("free_q5_burners_on", "b4_q5_short_burners_on")
    ]
    for row in representative:
        q5 += (f" {cap_label(row['cap'])} QPS{row['QPS']:g} diagnostic recorded P busy {pct(row['P_busy'])}, "
               f"request/token P proximity {pct(row['P_req_proximity'])}/{pct(row['P_tok_proximity'])}, "
               f"recent P wait {pct(row['P_backlog'])}, ≥160 ms D quiet overlapping busy and backlog "
               f"{pct(row['D_quiet_overlap_160'])} of P-busy time, and clean/profile/clean Decode SLO "
               f"{pct(row['D_clean_before_SLO'])}/{pct(row['D_absolute_SLO'])}/{pct(row['D_clean_after_SLO'])}.")
    ends = sorted((row for row in native if row["decode_cap"] == "free"), key=lambda row: row["QPS"])
    if len(ends) > 1:
        left, right = ends[0], ends[-1]
        q5 += (f" The separate clean Free screen at {left['QPS']:g}–{right['QPS']:g} QPS changed TTFT p95 "
               f"{left['TTFT_p95_ms']:.0f}→{right['TTFT_p95_ms']:.0f} ms and ITL p99 "
               f"{left['ITL_p99_ms']:.2f}→{right['ITL_p99_ms']:.2f} ms.")

    free_q5 = [row for row in profiles if row["cap"] == "free" and row["QPS"] == 5]
    q6 = "`P_HEAVY_D_IDLE_NOT_MEASURED`."
    if free_q5:
        n_free = sum(row["observer_passed"] for row in free_q5)
        n_all = sum(row["observer_passed"] for row in profiles)
        q6 = (f"{n_free}/{len(free_q5)} distinct Free QPS5 captures passed Clean/Profile/Clean "
              f"({n_all}/{len(profiles)} overall). "
              "The discovery report records every capture, every failed criterion, and clean Decode SLO. "
              "Repeated trace-only failures trigger the limited H5 pilot only with independent supporting evidence.")
    e_busy = [row for row in profiles if row["E_busy"] is not None and row["E_busy"] >= .8
              and all(row["checks"].get(field) for field in
                      ("p_kernel_busy", "p_scheduler_backlog", "d_quiet_window_coverage"))]
    qualified_e_busy = [row for row in e_busy if row in confirmed]
    q7 = ("The run qualifies." if qualified_e_busy else
          "The simultaneous E-busy, P-busy and D-quiet condition is supported at diagnostic level.")
    if e_busy:
        q7 += (f" {len(e_busy)} captures had E kernel busy "
               f"{pct(min(row['E_busy'] for row in e_busy))}–{pct(max(row['E_busy'] for row in e_busy))} "
               "with P pressure and operational D quiet overlap.")
    else:
        q7 = "No trace yet shows E kernel busy ≥80% with P pressure and eligible D quiet overlap."

    measured_static, measured_idle = candidates["static"], candidates["idle"]
    geometry = [row for row in analysis.get("decode_cap", []) if row.get("method", "").startswith(("static", "idle"))]
    contrasts = analysis.get("same_cap_contrasts", [])
    ci = analysis.get("contrast_ci", [])
    pairs = analysis["pilot_pairs"]
    static_ttft = comparison_effects(contrasts, "TTFT_p95_ms", "static")
    pilot_static = [(baseline, candidate) for scope, baseline, candidate in pairs
                    if scope == "Same condition" and baseline["method"].startswith("native-common")
                    and candidate["method"].startswith("static")]
    if pilot_static:
        q8 = " · ".join(
            f"N1 {candidate['method']} vs {baseline['method']} TTFT p95 "
            f"{candidate['TTFT_p95_ms']:.1f}/{baseline['TTFT_p95_ms']:.1f} ms "
            f"({100 * (candidate['TTFT_p95_ms'] / baseline['TTFT_p95_ms'] - 1):+.1f}%)"
            for baseline, candidate in pilot_static) + "."
    else:
        q8 = ("`NOT_RUN`." if not measured_static else static_ttft or
              "Static cohorts completed. No valid equal-arrival Native comparison is available.")
    measured_geometry = {row["start"]: row for row in analysis.get("decode_cap", [])
                         if row.get("D_prepared_decode_max")}
    geometry_deltas = []
    for scope, baseline, candidate in pairs:
        if scope != "Same condition" or not baseline["method"].startswith("native-common"):
            continue
        a, b = (measured_geometry.get(row["start"]) for row in (baseline, candidate))
        if a and b:
            geometry_deltas.append(
                f"{candidate['method']} vs {baseline['method']} prepared max "
                f"{b['D_prepared_decode_max']}/{a['D_prepared_decode_max']}, "
                f"single-request forward shares "
                f"{pct(b['D_prepared_single_request_fraction'])}/{pct(a['D_prepared_single_request_fraction'])} "
                f"({b['batch_evidence']}/{a['batch_evidence']})")
    if not geometry_deltas:
        geometry_deltas = [
            f"{row['start']} prepared max {row['D_prepared_decode_max']}, "
            f"single-request forwards {pct(row['D_prepared_single_request_fraction'])} ({row['batch_evidence']})"
            for row in geometry if row.get("D_prepared_decode_max")
        ]
    q9 = " · ".join(geometry_deltas) if geometry_deltas else "`NOT_MEASURED`."

    idle_vs_static = [(base, idle) for scope, base, idle in pairs
                      if scope == "Same condition" and base["method"].startswith("static")
                      and idle["method"].startswith("idle")]
    q10 = ("`NOT_RUN`." if not measured_idle else
           "No matched clean Idle/Static contrast is available." if not idle_vs_static else " · ".join(
               f"{cap_label(idle['decode_cap'])} QPS{idle['QPS']:g} Idle minus Static TTFT p95 "
               f"{idle['TTFT_p95_ms'] - static['TTFT_p95_ms']:+.1f} ms"
               for static, idle in idle_vs_static) + ".")
    for row in measured_idle:
        selected, count = row["split_requests"], row["completed_count"]
        fallback = count - selected if selected is not None else None
        q10 += (f" {row['method']} selected {show(selected)}/{count}, fell back on {show(fallback)}, "
                f"moved {show(row['suffix_tokens'])} tokens ({pct(row['actual_offload_fraction'])}), "
                f"prefix-ready D quiet {pct(row['split_only_P_ready_D_HTTP_inactive_no_pending_fraction'])}, "
                f"selected admission wait p95 {show(row['split_only_D_coordinator_wait_p95_ms'])} ms.")

    decode_ci = [row for row in ci if row.get("metric") in ("ITL_p99_ms", "TPOT_p95_ms")]
    q11 = ("Independent-start paired 95% CIs are `NOT_MEASURED`." if not decode_ci else
           " · ".join(f"{row['method']} vs {row['baseline']} {row['metric']} upper "
                      f"{float(row['CI_high']):.3f} ({row['pairs']} pairs)" for row in decode_ci))
    completed_pairs = [item for item in pairs if item[0] == "Same condition" and
                       item[1]["method"].startswith("native-common")]
    for _, baseline, candidate in completed_pairs:
        q11 += (f" N1 {candidate['method']} versus {baseline['method']} ITL p99 "
                f"{candidate['ITL_p99_ms']:.2f}/{baseline['ITL_p99_ms']:.2f} ms, TPOT p95 "
                f"{candidate['TPOT_p95_ms']:.2f}/{baseline['TPOT_p95_ms']:.2f} ms, absolute Decode SLO "
                f"{pct(candidate['D_absolute_SLO_attainment'])}/{pct(baseline['D_absolute_SLO_attainment'])}.")
    if not completed_pairs:
        for row in (*measured_static, *measured_idle):
            q11 += (f" N1 {row['method']} ITL p99 {row['ITL_p99_ms']:.2f} ms, "
                    f"TPOT p95 {row['TPOT_p95_ms']:.2f} ms, absolute Decode SLO "
                    f"{pct(row['D_absolute_SLO_attainment'])}.")

    q12 = ("Practical benefit remains unestablished. It requires lower TTFT against Strong Native Free "
           "with the absolute Decode SLO, paired ITL/TPOT bounds, Joint SLO/goodput, and quality guardrails met.")
    for scope, baseline, candidate in pairs:
        if scope != "Strong Native Free":
            continue
        ttft_change = 100 * (candidate["TTFT_p95_ms"] / baseline["TTFT_p95_ms"] - 1)
        q12 += (f" N1 {candidate['method']} versus {baseline['method']} TTFT p95 {ttft_change:+.2f}%, "
                f"ITL p99 ratio {candidate['ITL_p99_ms'] / baseline['ITL_p99_ms']:.3f}, "
                f"TPOT p95 ratio {candidate['TPOT_p95_ms'] / baseline['TPOT_p95_ms']:.3f}, "
                f"candidate Decode SLO {pct(candidate['D_absolute_SLO_attainment'])}.")
    q13 = "`HOLD`. The current evidence does not establish a quality-preserving wide-batch serving benefit."
    motivation = ("P_HEAVY_D_IDLE_CONFIRMED" if confirmed else
                  "P_HEAVY_D_IDLE_NOT_CONFIRMED" if any(row["status"] == "P_HEAVY_D_IDLE_NOT_CONFIRMED"
                                                       for row in profiles) else "P_HEAVY_D_IDLE_NOT_MEASURED")
    execution = "CONDITIONAL_ONLINE_PILOT" if any(row["start"].startswith("pilot_")
                                                 for row in (*measured_static, *measured_idle)) else "NOT_RUN"
    questions = (
        ("Q1. Did the local runtime pass the absolute performance sanity check?", q1),
        ("Q2. What isolated D capacity was reached at each cap?", q2),
        ("Q3. Were prepared B32 and B64 formed?", q3),
        ("Q4. Which cap had the best observed trade-off?", q4),
        ("Q5. Which H200 cap/QPS conditions satisfied the combined motivation gate?", q5),
        ("Q6. Did Free QPS5 pass qualification?", q6),
        ("Q7. Was an E-busy, P-heavy and D-idle regime qualified?", q7),
        ("Q8. Did Static Split reduce TTFT?", q8),
        ("Q9. Did token splitting collapse the actual D batch?", q9),
        ("Q10. Did Idle-aware Split improve on Static?", q10),
        ("Q11. Was ITL and TPOT protected with paired intervals?", q11),
        ("Q12. Was there practical benefit against Native Free?", q12),
        ("Q13. Is there enough evidence for modality-aware follow-up?", q13),
    )
    pointers = (
        ("runtime_performance_sanity_check", "Calibration32 runtime sanity"),
        ("wide_decode_batch_capacity", "Isolated capacity and actual decode geometry"),
        ("decode_batch_qps_motivation", "Native online grid and Burner comparison"),
        ("qualified_prefill_heavy_discovery", "P/D combined gates"),
        ("batch_preserving_static_token_split", "Static candidate results"),
        ("batch_preserving_idle_token_split", "Idle-aware candidate results"),
        ("wide_batch_token_split_critical_path", "Common timestamp analysis"),
        ("wide_decode_batch_token_split_poc_spec", "Frozen campaign specification"),
    )
    put(docs, "wide_decode_batch_final_summary", "H200 wide decode batch token split final summary",
        f"Serving status **`HOLD`** · Motivation **`{motivation}`** · Candidate evidence **`{execution}`**\n\n"
        + "\n\n".join(f"**{question}** {answer}" for question, answer in questions)
        + closeout(execution, motivation)
        + quality_closeout(results)
        + "\n\n## Evidence reports\n\n" + "\n".join(f"- {link(key, text)}" for key, text in pointers)
        + "\n\n" + evidence_index(results))


def calibration_artifacts(output, summaries, requests, figures):
    output.mkdir(parents=True, exist_ok=True)
    prior.csv_file(output / "wide_sanity.csv", summaries, ("arm", "evidence"))
    prior.csv_file(output / "wide_sanity_requests.csv", requests, ("arm", "request_id"))
    measured = [row for row in summaries if row["evidence"] == "MEASURED_ONLINE"]
    if not figures or not measured:
        return
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), layout="constrained")
    for axis, key, label in zip(
            axes, ("TTFT_ms", "ITL_ms", "TPOT_ms"), ("TTFT (ms)", "ITL (ms)", "TPOT (ms/token)"), strict=True):
        for row in measured:
            values = [req[key] for req in requests if req["arm"] == row["arm"]]
            if key == "ITL_ms":
                values = [gap for encoded in values for gap in json.loads(encoded)]
            values.sort()
            axis.step(values, [(i + 1) / len(values) for i in range(len(values))], where="post", label=row["arm"])
        axis.set_xlabel(label)
        axis.set_ylabel("Fraction of output token gaps" if key == "ITL_ms" else "Fraction of requests")
        axis.legend(fontsize=8)
        axis.grid(alpha=.25)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"wide_sanity_absolute_latency.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, help="Path generated by the wide analysis module")
    parser.add_argument("--extra-output", type=Path, help="Write Calibration32 source tables and figures here")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    sanity, requests, paired = sanity_evidence(args.runs)
    arms = wide_capacity(args.runs)
    rows, failures, host, analysis = online_evidence(args.runs, args.analysis)
    profiles = discovery_evidence(args.runs)
    render_sanity(args.docs, sanity, requests, paired)
    report_capacity(args.docs, arms, prior.capacity_table(args.runs))
    native = report_motivation(args.docs, rows, failures, analysis)
    report_discovery(args.docs, profiles)
    candidates = report_candidates(args.docs, args.runs, rows, analysis, profiles)
    report_critical(args.docs, host, rows)
    if args.extra_output is not None:
        calibration_artifacts(args.extra_output, sanity, requests, not args.no_figures)
    report_final(args.docs, sanity, paired, arms, native, profiles, candidates, analysis, args.analysis)


if __name__ == "__main__":
    main()
