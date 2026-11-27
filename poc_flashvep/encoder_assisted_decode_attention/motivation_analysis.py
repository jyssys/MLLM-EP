"""Analyze matched Native EPD measurements on a shared monotonic timeline."""
import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
import csv
import json
from pathlib import Path

from ..mllm_epd_h200_prefill_heavy_batch_matched.trace import complement, elapsed, intersect, merged

ROLES = ("encoder", "prefill", "decode")
ATTENTION_NAMES = ("FlashAttnFwdSm90", "FlashAttnFwdCombine")


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def quantiles(values):
    values = sorted(values)
    if not values:
        return None

    def at(fraction):
        index = (len(values) - 1) * fraction
        base = int(index)
        return values[base] + (index - base) * (values[min(base + 1, len(values) - 1)] - values[base])

    return {key: at(fraction) for key, fraction in (("min", 0), ("p50", .5), ("p95", .95),
                                                   ("p99", .99), ("max", 1))}


def classify(name):
    if any(pattern.lower() in name.lower() for pattern in ATTENTION_NAMES):
        return "attention"
    if "prepare_varlen_num_blocks_kernel" in name:
        return "attention_metadata"
    if "reshape_and_cache_flash_kernel" in name:
        return "KV_write"
    if any(part in name.lower() for part in ("gemm", "nvjet", "cublaslt19splitkreduce")):
        return "GEMM_unattributed"
    if any(part in name.lower() for part in ("rms_norm", "silu")):
        return "normalization_or_activation"
    if "argmax" in name.lower():
        return "sampling_argmax"
    return "other_unattributed"


def kernel_data(path, left, right, *, breakdown=False):
    roi = read(path / "actual_ROI.json")
    before, after = roi["left"], roi["right"]
    a, b = [(x["host_lower_ns"] + x["host_upper_ns"]) / 2 for x in (before, after)]
    scale = (b - a) / (after["cupti_ns"] - before["cupti_ns"])
    brackets = [x["host_upper_ns"] - x["host_lower_ns"] for x in (before, after)]
    if not a / 1e9 <= left < right <= b / 1e9 or not .99 <= scale <= 1.01 or max(brackets) > 1e6:
        raise ValueError(f"Clock alignment or ROI coverage failed for {path}")

    def seconds(ns):
        return (a + scale * (int(ns) - before["cupti_ns"])) / 1e9

    names = {}
    if breakdown:
        with (path / "kernels.csv.names.csv").open() as stream:
            names = {row["name_id"]: row["kernel_name"] for row in csv.DictReader(stream)}
    categories = {key: classify(name) for key, name in names.items()}
    spans, by_category, attention_starts = [], defaultdict(list), []
    seen_names = set()
    with (path / "kernels.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["kind"] != "kernel":
                continue
            start, end = seconds(row["start_ns"]), seconds(row["end_ns"])
            if not left < end or not start < right:
                continue
            spans.append((start, end))
            if breakdown:
                category = categories.get(row["name_id"], "other_unattributed")
                by_category[category].append((start, end))
                if category == "attention" and left <= start < right:
                    attention_starts.append(start)
                    seen_names.add(names[row["name_id"]])
    return merged(spans, left, right), {
        category: merged(intervals, left, right) for category, intervals in by_category.items()
    }, attention_starts, {
        "clock_scale": scale, "max_anchor_bracket_ns": max(brackets), "collector_passed": roi["passed"],
        "kernels_in_roi": len(spans), "attention_kernel_names": sorted(seen_names),
    }


def extent_by_http(rows, begin, end, left, right):
    return merged(((row[begin], row[end]) for row in rows if begin in row and end in row), left, right)


def start_opportunities(starts, windows, thresholds_us):
    beginnings = [first for first, _ in windows]
    remaining = []
    for start in starts:
        i = bisect_right(beginnings, start) - 1
        if i >= 0 and start < windows[i][1]:
            remaining.append(windows[i][1] - start)
    return {
        str(guard): {"eligible": count, "all_attention_starts": len(starts),
                     "eligible_fraction": count / len(starts) if starts else None}
        for guard, count in [(us, sum(slack >= us / 1e6 for slack in remaining)) for us in thresholds_us]
    }


def sampled_decode_batch(path, clients, left, right):
    contexts = {row["runtime_request_id"]: row["input_tokens_manifest"] for row in clients}
    batches, prompt_rows = [], []
    for call in read(path)["rows"]:
        if not left <= call["begin_ns"] / 1e9 < right:
            continue
        scheduled = call["scheduled_requests"]
        if not scheduled or any(rows != 1 for rows in scheduled.values()):
            continue
        batches.append(len(scheduled))
        owners = [name.split(":D", 1)[0].removeprefix("chatcmpl-") for name in scheduled]
        if all(owner in contexts for owner in owners):
            prompt_rows.append(sum(contexts[owner] for owner in owners))
    return {"decode_only_worker_calls": len(batches), "calls_joined_to_context": len(prompt_rows),
            "actual_scheduled_decode_batch": quantiles(batches), "batch_sum_prompt_rows": quantiles(prompt_rows),
            "scope": "Conditional worker-call samples where every scheduled row is exactly one token. "
                     "The sum excludes decode growth and does not estimate unsampled calls."}


def analyze_gpm(run, tag, left, right, attention):
    data = read(run / "raw" / f"{tag}_gpm.json")
    result = {"scope": data["scope"], "errors": data["errors"], "units": data["units"], "roles": {}}
    starts = [a for a, _ in attention]
    ends = [b for _, b in attention]
    cumulative_attention = [0.0]
    for a, b in attention:
        cumulative_attention.append(cumulative_attention[-1] + b - a)

    def attention_in_window(a, b):
        first, last = bisect_right(ends, a), bisect_left(starts, b)
        if first >= last:
            return 0.0
        return max(0, cumulative_attention[last] - cumulative_attention[first]
                   - max(0, a - starts[first]) - max(0, ends[last - 1] - b))

    for role in ROLES:
        across = defaultdict(list)
        windows, memory_used = [], []
        attention_overlapping = defaultdict(list)
        spans_by_metric = defaultdict(list)
        for row in data["samples"]:
            item = row["roles"][role]
            a, b = item["sample_window"]
            if a >= right or b <= left:
                continue
            time_window = [(max(left, a), min(right, b))]
            d_attention_seconds = attention_in_window(*time_window[0])
            windows.extend(time_window)
            memory_used.append(item["gpu_memory_used_bytes"])
            for key, value in item["values"].items():
                if value is not None:
                    dt = time_window[0][1] - time_window[0][0]
                    across[key].append((value, dt))
                    if d_attention_seconds:
                        attention_overlapping[key].append((value, d_attention_seconds))
                    if key == "DRAM_BW_UTIL":
                        for level in (20, 40, 60):
                            if value <= level:
                                spans_by_metric[level].extend(time_window)

        def summarize(group):
            return {
                name: {"weighted_mean": sum(value * weight for value, weight in values)
                       / sum(weight for _, weight in values), "window_sample_percentiles": quantiles(
                           [value for value, _ in values]
                       )}
                for name, values in group.items()
            }

        result["roles"][role] = {
            "sample_count": len(windows),
            "covered_seconds": elapsed(merged(windows, left, right)),
            "peak_sampled_device_used_bytes": max(memory_used, default=None),
            "device_metrics": summarize(across),
            "metrics_in_windows_overlapping_D_attention": summarize(attention_overlapping),
            "fraction_of_D_attention_in_low_bandwidth_counter_windows": {
                str(level): elapsed(intersect(merged(spans_by_metric[level], left, right), attention)) / elapsed(attention)
                if attention else None for level in (20, 40, 60)
            },
        }
    return result


def capacity_probes(run, baseline, role="encoder"):
    probes = []
    for path in sorted((run / "state").glob(f"owned_{role}_arena_*gib.json")):
        label = path.stem.removeprefix("owned_")
        ready = run / "state" / f"{label}_ready.json"
        measured = run / "analysis" / f"{label}_metrics.json"
        failure = run / "state" / f"{label}_failure.json"
        record = {"label": label, "ready": ready.exists(), "finished": measured.exists()}
        if failure.exists():
            record["failure"] = read(failure)
        if ready.exists():
            record["allocator_receipt"] = read(ready)
        if measured.exists():
            values = read(measured)
            utilization = read(run / "raw" / f"{label}_gpm.json")
            worker = run / "state" / f"{label}_{role}_memory.json"
            record.update({
                "online_execution": values,
                f"peak_{role}_device_used_bytes": max((row["roles"][role]["gpu_memory_used_bytes"]
                                                      for row in utilization["samples"]), default=None),
                "ratio_to_native_no_arena": {name: values[name] / value for name, value in baseline.items()},
            })
            if worker.exists():
                record[f"{role}_worker_after_load"] = read(worker)
        probes.append(record)
    return probes


def analyze_trace(run, tag, csv_path=None):
    from ..mllm_specific_disaggregation_discovery.native_transfer_telemetry import prom_delta

    client = list(jsonl(run / "raw" / f"{tag}_requests.jsonl"))
    identifiers = {row["runtime_request_id"] for row in client}
    proxy = [row for row in jsonl(run / "raw" / "proxy_epd.jsonl")
             if row["runtime_request_id"] in identifiers]
    if len(proxy) != len(client) or len(identifiers) != len(client):
        raise ValueError("Incomplete arrival to stage-timing match")
    arrivals = sorted(row["scheduled_arrival"] for row in client)
    left, right = (arrivals[int(len(arrivals) * fraction)] for fraction in (.2, .8))
    profile_dir = run / "raw" / "profiles" / tag
    busy, captures = {}, {}
    for role in ROLES:
        busy[role], categories, starts, captures[role] = kernel_data(
            profile_dir / role, left, right, breakdown=role == "decode",
        )
    attention = categories.get("attention", [])
    if not attention or not starts:
        raise ValueError("FA3 attention kernels were not recognized")
    idle = complement(busy["encoder"], left, right)
    http = extent_by_http(proxy, "E_submit", "E_complete", left, right)
    quiet = intersect(idle, complement(http, left, right))
    p_http = extent_by_http(proxy, "P_submit", "P_ready", left, right)
    p_quiet = intersect(complement(busy["prefill"], left, right), complement(p_http, left, right))
    gate = read(run / "analysis" / f'{tag.removesuffix("_profile")}_profile_gate.json')
    cohort_after = read(run / "state" / "worker_after_full_load.json")
    initial = {role: read(next((run / "state").glob(f"worker_*-{role}.json"), None)
                          or next((run / "state").glob(f"worker_*_{role}.json"))) for role in ROLES}
    opportunity = {
        label: {
            "window_seconds": elapsed(windows),
            "overlap_D_attention_seconds": elapsed(intersect(windows, attention)),
            "overlap_fraction_of_D_attention": elapsed(intersect(windows, attention)) / elapsed(attention),
            "window_span_ms": quantiles([(b - a) * 1000 for a, b in windows]),
            "retrospective_attention_kernel_start_fit": start_opportunities(
                starts, windows, (0, 25, 50, 100, 250, 500, 1000)
            ),
        }
        for label, windows in (("E_physical_kernel_idle", idle), ("E_no_kernel_or_HTTP", quiet),
                               ("P_no_kernel_or_HTTP", p_quiet))
    }
    traces = {**busy, "D_attention": attention, "E_no_kernel_or_HTTP": quiet, "P_no_kernel_or_HTTP": p_quiet}
    transfers = prom_delta({direction: read(run / "raw" / f"{tag}_nixl_{direction}.json")
                            for direction in ("before", "after")})
    reference = {name: (row["clean_before"] + row["clean_after"]) / 2 for name, row in gate["metrics"].items()}
    names = ("relative_monotonic_ms", *traces)
    if csv_path is not None:
        i = 0
        left_edges = {key: 0 for key in traces}
        with csv_path.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=names, lineterminator="\n")
            writer.writeheader()
            while left + i / 10 < right:
                a = left + i / 10
                b = min(right, a + .1)
                row = {"relative_monotonic_ms": i * 100}
                for label, spans in traces.items():
                    j = left_edges[label]
                    while j < len(spans) and spans[j][1] <= a:
                        j += 1
                    left_edges[label] = j
                    covered = 0.0
                    while j < len(spans) and spans[j][0] < b:
                        covered += min(b, spans[j][1]) - max(a, spans[j][0])
                        j += 1
                    row[label] = covered / (b - a)
                writer.writerow(row)
                i += 1

    return {
        "evidence": "TRACE_ANALYSIS" if gate["passed"] else "DIAGNOSTIC",
        "trace_tag": tag,
        "test_terminal_state": read(run / "state/failure.json") if (run / "state/failure.json").exists()
        else read(run / "state/complete.json"),
        "matched_requests": len(client),
        "central_arrival_window": [left, right],
        "capture": captures,
        "observer_gate": gate,
        "worker_allocator_initial": initial,
        "worker_allocator_after_load": cohort_after,
        "kernel_busy_fraction": {role: elapsed(spans) / (right - left) for role, spans in busy.items()},
        "D_kernel_category_fraction_of_D_busy": {category: elapsed(spans) / elapsed(busy["decode"])
                                                  for category, spans in categories.items()},
        "D_attention_time_fraction_of_D_busy": elapsed(attention) / elapsed(busy["decode"]),
        "D_attention_time_fraction_of_D_HTTP": elapsed(attention) / elapsed(
            extent_by_http(proxy, "D_submit", "finished", left, right)
        ),
        "D_attention_kernel_starts": len(starts),
        "D_sampled_decode_only_schedule": sampled_decode_batch(
            profile_dir / "decode/worker_calls.json", client, left, right,
        ),
        "optimistic_amdahl_bound_fraction": .20 * elapsed(attention) / elapsed(busy["decode"]),
        "E_and_P_opportunity": opportunity,
        "gpm": analyze_gpm(run, tag, left, right, attention),
        "NIXL_connector_transport_counter_delta": transfers,
        "encoder_persistent_arena_probes": capacity_probes(run, reference),
        "scope": "CPC-qualified physical timeline, when the gate passes. D kernel buckets use name signatures. "
                 "GEMM names do not identify QKV/O/FFN paths. Kernel bucket unions can overlap. "
                 "Future slack fits are retrospective oracle bounds, not scheduler admissions. "
                 "The 20%-attention Amdahl bound omits scheduling and uses the favorable GPU-busy denominator.",
    }


def summarize_native_scan(run):
    proxies = {}
    for row in jsonl(run / "raw/proxy_epd.jsonl"):
        proxies[row["runtime_request_id"]] = {key: row.get(key) for key in (
            "received", "E_submit", "E_complete", "P_submit", "P_ready", "D_submit", "D_http_complete", "finished",
        )}
    groups = []
    for metrics_path in sorted((run / "analysis").glob("native_strong_free_*_metrics.json")):
        values = read(metrics_path)
        tag = values["run"]
        config = read(run / "configs" / f"{tag}.json")
        clients = list(jsonl(run / "raw" / f"{tag}_requests.jsonl"))
        scheduled = sorted(row["scheduled_arrival"] for row in clients)
        left, right = (scheduled[int(len(scheduled) * fraction)] for fraction in (.2, .8))
        stage = [dict((k, v) for k, v in proxies[row["runtime_request_id"]].items() if v is not None)
                 for row in clients]
        encoder = extent_by_http(stage, "E_submit", "E_complete", left, right)
        prefill = extent_by_http(stage, "P_submit", "P_ready", left, right)
        decode = extent_by_http(stage, "D_submit", "finished", left, right)
        e_quiet = complement(encoder, left, right)
        stage_ms = {
            name: quantiles([(p[end] - p[begin]) * 1000 for p in stage if begin in p and end in p])
            for name, begin, end in (("encoder_http", "E_submit", "E_complete"),
                                     ("prefill_http", "P_submit", "P_ready"),
                                     ("decode_http", "D_submit", "D_http_complete"),
                                     ("E_to_P_host_wait", "E_complete", "P_submit"),
                                     ("P_to_D_host_wait", "P_ready", "D_submit"))
        }
        groups.append({
            "offered_qps": config["QPS"], "metrics": values,
            "quality_scope": "Natural-EOS output lengths; this scan does not score task accuracy.",
            "E2E_ms": quantiles([row["E2E_ms"] for row in clients if row["status"] == "MEASURED_ONLINE"]),
            "TPOT_ms": quantiles([row["TPOT_ms"] for row in clients if row["status"] == "MEASURED_ONLINE"]),
            "ITL_ms": quantiles([value for row in clients if row["status"] == "MEASURED_ONLINE"
                                for value in row["ITL_ms"]]),
            "host_stage_elapsed_ms": stage_ms,
            "host_ROI": [left, right],
            "stage_http_union_fraction": {key: elapsed(value) / (right - left) for key, value in
                                          (("encoder", encoder), ("prefill", prefill), ("decode", decode))},
            "decode_http_overlap_with_no_encoder_http_fraction": elapsed(intersect(e_quiet, decode))
            / elapsed(decode) if decode else None,
        })
    return {"runs": sorted(groups, key=lambda row: row["offered_qps"]),
            "scope": "Actual NativeStrong Free EPD finite cohorts. Stage HTTP occupancy includes pending work, "
                     "and these host clocks are distinct from physical kernel activity."}


def summarize_capacity_study(run):
    from ..mllm_epd_h200_prefill_heavy_batch_matched.trace import COMPARISON_METRICS

    path, = (run / "analysis").glob("native_strong_free_*_metrics.json")
    baseline = read(path)
    observer = read(run / "state/native_observer.json")
    role = observer.get("capacity_role", "encoder")
    probes = capacity_probes(run, {field: baseline[field] for field in COMPARISON_METRICS}, role)
    arrivals = read(run / "configs" / f"{baseline['run']}_arrival.json")
    for probe in probes:
        if not probe["finished"]:
            continue
        trace_path = run / "configs" / f"{probe['label']}_arrival.json"
        if read(trace_path) != arrivals:
            raise ValueError("Co-residency arms used different Native arrival traces")
        probe["matched_arrival_trace"] = True
    sampler = read(run / "raw" / f"{baseline['run']}_gpm.json")
    encoder_elapsed = defaultdict(list)
    for row in jsonl(run / "raw/proxy_epd.jsonl"):
        if "E_submit" in row and "E_complete" in row:
            label = row["runtime_request_id"].rsplit("_", 1)[0]
            encoder_elapsed[label].append((row["E_complete"] - row["E_submit"]) * 1000)
    relevant = {baseline["run"], *(row["label"] for row in probes if row["finished"])}
    return {"native_reference": baseline, f"{role}_persistent_arena_probes": probes,
            "post_warm_worker_memory": read(run / "state/worker_after_full_load.json"),
            f"native_peak_{role}_device_used_bytes": max((row["roles"][role]["gpu_memory_used_bytes"]
                                                        for row in sampler["samples"]), default=None),
            "encoder_http_service_plus_queue_ms": {label: quantiles(values) for label, values in sorted(
                encoder_elapsed.items()) if label in relevant},
            "offered_qps": arrivals["QPS"],
            "scope": f"Touched device memory held during stock {role} work. Capacity here is an observed "
                     "co-residency probe in one worker start."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tag")
    parser.add_argument("--native-scan", action="store_true")
    parser.add_argument("--capacity-study", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeline-csv", type=Path)
    args = parser.parse_args()
    if sum((args.native_scan, args.capacity_study, bool(args.tag))) != 1:
        parser.error("Choose --tag, --native-scan or --capacity-study")
    if args.native_scan:
        result = summarize_native_scan(args.run_dir)
    elif args.capacity_study:
        result = summarize_capacity_study(args.run_dir)
    else:
        result = analyze_trace(args.run_dir, args.tag, args.timeline_csv)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
