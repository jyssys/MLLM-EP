"""Build a common monotonic E/P/D timeline from native runtime receipts."""

from bisect import bisect_left
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import statistics

from ..mllm_epd_h200_prefill_heavy_batch_matched.trace import complement, elapsed, intersect, merged


ROLES = ("encoder", "prefill", "decode")
ROI_QUANTILES = (.20, .80)
IDLE_EDGES_MS = (0, 1, 5, 10, 25, 50, 100, 250, 500, math.inf)


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt") as stream:
        for line in stream:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if line.endswith("\n") or path.suffix == ".gz":
                        raise
                    # A live append can expose an unfinished last record.
                    break
    return rows


def distribution(values):
    points = sorted(values)

    def at(fraction):
        if not points:
            return None
        index = (len(points) - 1) * fraction
        low = int(index)
        return points[low] + (points[min(low + 1, len(points) - 1)] - points[low]) * (index - low)

    return {label: at(fraction) for label, fraction in
            (("min", 0), ("p50", .5), ("p95", .95), ("p99", .99), ("max", 1))}


def arrival_window(client):
    starts = sorted(row["scheduled_arrival"] for row in client)
    if len(starts) < 5:
        return None
    left, right = (starts[min(int(len(starts) * fraction), len(starts) - 1)] for fraction in ROI_QUANTILES)
    return (left, right) if left < right else None


def clock_pair(first, last, bound_keys):
    lower, upper = bound_keys
    host_first, host_last = ((int(row[lower]) + int(row[upper])) / 2 for row in (first, last))
    tick_first, tick_last = (int(row["cupti_ns"]) for row in (first, last))
    if not (host_first < host_last and tick_first < tick_last):
        raise ValueError("Non-increasing CUPTI and host clock anchors")
    slope = (host_last - host_first) / (tick_last - tick_first)
    brackets = [int(row[upper]) - int(row[lower]) for row in (first, last)]
    return ((int(first[upper]) * 1e-9, int(last[lower]) * 1e-9),
            (host_first, tick_first, slope), brackets)


def kernel_capture(folder, left, right):
    roi_path = folder / "actual_ROI.json"
    if roi_path.is_file():
        roi = read_json(roi_path)
        clock_first, clock_last = (roi[key] for key in ("left", "right"))
        keys = ("host_lower_ns", "host_upper_ns")
        paths = [path for path in (folder / "kernels.csv", folder / "kernels.csv.gz") if path.is_file()]
        copied = bool(read_json(folder / "provenance.json").get("include_copies"))
        counters = roi["counters"]
        integrity = (roi["passed"] and roi["returncode"] == 0 and counters["dropped"] == 0
                     and counters["invalid"] == 0)
        expected = counters["all_kernels"]
    else:
        paths = sorted((*folder.glob("worker-*.csv"), *folder.glob("worker-*.csv.gz")))
        if len(paths) != 1:
            raise ValueError(f"Expected one physical kernel collector per role in {folder}")
        meta_path = paths[0].with_name(paths[0].name.split(".csv")[0] + ".json")
        roi = read_json(meta_path)
        clock_first, clock_last = (roi[key] for key in ("clock_start", "clock_end"))
        keys = ("mono_before_ns", "mono_after_ns")
        copied = False
        integrity = roi["returncode"] == 0 and roi["dropped"] == roi["invalid"] == 0
        expected = roi["exported"]

    if len(paths) != 1:
        raise ValueError(f"Ambiguous CUPTI trace in {folder}")
    coverage, (host_epoch, tick_epoch, scale), brackets = clock_pair(clock_first, clock_last, keys)
    clock_ok = (coverage[0] <= left < right <= coverage[1] and .99 <= scale <= 1.01
                and min(brackets) >= 0 and max(brackets) <= 1_000_000)
    path = paths[0]
    opener = gzip.open if path.suffix == ".gz" else open
    raw = {"kernel": [], "copy": []}
    count = invalid = 0
    with opener(path, "rt", newline="") as stream:
        reader = csv.reader(stream)
        columns = next(reader)
        start_i, end_i = (columns.index(key) for key in ("start_ns", "end_ns"))
        kind_i = columns.index("kind") if "kind" in columns else None
        for row in reader:
            kind = "kernel" if kind_i is None else row[kind_i]
            if kind == "kernel":
                count += 1
            elif kind in {"copy", "memcpy", "memcpy2", "gpu_memcpy"}:
                kind = "copy"
            else:
                continue
            start = (host_epoch + scale * (int(row[start_i]) - tick_epoch)) * 1e-9
            end = (host_epoch + scale * (int(row[end_i]) - tick_epoch)) * 1e-9
            if start >= end:
                invalid += 1
            elif start < right and end > left:
                raw[kind].append((start, end))
    kernel = merged(raw["kernel"], left, right)
    copies = merged(raw["copy"], left, right) if copied else None
    return {"kernel": kernel, "copy": copies, "clock_ok": clock_ok,
            "collector_ok": integrity and invalid == 0 and count == expected,
            "kernel_records": count, "expected_kernel_records": expected,
            "invalid_interval_records": invalid, "clock_scale": scale,
            "max_anchor_bracket_ns": max(brackets), "coverage": coverage}


def host_enclosure(proxy, start, end, left, right):
    return merged(((row[start], row[end]) for row in proxy
                   if isinstance(row.get(start), (int, float)) and isinstance(row.get(end), (int, float))), left, right)


def sample_spans(scheduler, left, right):
    """Bound sample-and-hold by the sampling interval to preserve gaps."""
    rows = sorted(scheduler["rows"], key=lambda row: row["begin_ns"])
    max_hold = scheduler["sampling_interval_ms"] * .0015
    for seed, following in zip(rows, rows[1:]):
        a = seed["begin_ns"] * 1e-9
        b = min(following["begin_ns"] * 1e-9, a + max_hold)
        if a < right and b > left and a < b:
            yield seed["before"], max(a, left), min(b, right)


def scheduler_states(folder, left, right):
    path = folder / "scheduler.json"
    if not path.is_file():
        return None
    raw = read_json(path)
    predicates = {
        "waiting": lambda row: row.get("waiting"),
        "plain_waiting": lambda row: row.get("waiting_status", {}).get("WAITING"),
        "known_compute_ready_waiting": lambda row: row.get("prefill_compute_ready_waiting"),
        "recent_idle_snapshot": lambda row: row.get("waiting") == row.get("running") == 0,
        "recent_decode_ready": lambda row: row.get("decode_ready_waiting") or row.get("decode_active"),
    }
    stats = defaultdict(list)
    capacity = []
    for state, a, b in sample_spans(raw, left, right):
        stats["observed"].append((a, b))
        for key, predicate in predicates.items():
            if predicate(state):
                stats[key].append((a, b))
        if state.get("total_KV_blocks") and state.get("free_KV_blocks") is not None:
            capacity.append(1 - state["free_KV_blocks"] / state["total_KV_blocks"])
    rows = [row for row in raw["rows"] if left <= row["begin_ns"] * 1e-9 < right]
    if not rows:
        return None
    geometry = {"requests": distribution(row["actual_scheduled_requests"] for row in rows),
                "query_rows": distribution(row["scheduled_tokens"] for row in rows),
                "encoder_items": distribution(row["scheduled_encoder_items"] for row in rows)}
    admission = {key: sum(row[key] for row in rows) if rows and all(key in row for row in rows) else None
                 for key in ("kv_allocate_attempts", "kv_allocate_rejected", "ec_availability_attempts",
                             "ec_availability_deferred")}
    return {"intervals": {key: merged(stats[key], left, right) for key in ("observed", *predicates)},
            "compute_ready_field_observed": any("prefill_compute_ready_waiting" in row["before"] for row in rows),
            "observations": len(rows), "sample_interval_ms": raw["sampling_interval_ms"],
            "max_KV_occupancy_fraction": max(capacity, default=None), "scheduler_geometry": geometry,
            "sampled_admission_operations": admission}


def idle_histogram(windows, intersections=None, handoffs=()):
    total = elapsed(windows)
    groups = [[] for _ in IDLE_EDGES_MS[1:]]
    for left, right in windows:
        length = (right - left) * 1000
        index = next(i for i, end in enumerate(IDLE_EDGES_MS[1:]) if length < end)
        groups[index].append((left, right))
    result = []
    for lo, hi, items in zip(IDLE_EDGES_MS, IDLE_EDGES_MS[1:], groups):
        span = elapsed(items)
        overlaps = {key: elapsed(intersect(group, items)) for key, group in (intersections or {}).items()}
        gaps = []
        for left, right in items:
            point = bisect_left(handoffs, left)
            if point < len(handoffs):
                gaps.append(1000 * (handoffs[point] - left))
        result.append({"lower_ms": lo, "upper_ms": hi if math.isfinite(hi) else None,
                       "count": len(items), "seconds": span, "fraction_idle_time": span / total if total else None,
                       "intersection_seconds": overlaps, "next_P_ready_proxy_ms": distribution(gaps)})
    return result


def bins(intervals, left, right, n=100):
    spans = []
    current = 0
    for i in range(n):
        a = left + i * (right - left) / n
        b = left + (i + 1) * (right - left) / n
        while current < len(intervals) and intervals[current][1] <= a:
            current += 1
        j = current
        covered = 0
        while j < len(intervals) and intervals[j][0] < b:
            x, y = intervals[j]
            covered += max(0, min(b, y) - max(a, x))
            j += 1
        spans.append(covered / (b - a))
    return spans


def all_intersections(first, *others):
    for other in others:
        first = intersect(first, other)
    return first


def arrival_transform(client, proxy, left, right):
    events = {"scheduled_arrival": [row["scheduled_arrival"] for row in client]}
    for tag in ("E_complete", "P_submit", "P_ready", "H_submit", "H_ready", "D_submit"):
        events[tag] = [row[tag] for row in proxy if isinstance(row.get(tag), (int, float))]
    result = {}
    for tag, timestamps in events.items():
        stamps = sorted(t for t in timestamps if left <= t < right)
        if not stamps:
            continue
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        deviation = statistics.pstdev(gaps) if gaps else None
        mean = statistics.fmean(gaps) if gaps else None
        result[tag] = {"events": len(stamps), "events_per_second": len(stamps) / (right - left),
                       "interevent_gap_ms": distribution(gap * 1000 for gap in gaps),
                       "gap_CV": deviation / mean if mean else None}
    return result


def analyze_window(profile_paths, client, proxy, *, quantum_ms=(5, 10, 25, 50, 100, 250, 500, 750, 1000, 1150, 1250),
                   keep_intervals=False,
                   whole_cohort_native_READ_accounted=False):
    """Return one window's measured unions and a distinctly labeled retrospective fit bound."""
    window = arrival_window(client)
    if window is None:
        return None
    left, right = window
    duration = right - left
    traces, captures, schedulers = {}, {}, {}
    errors = {}
    for role in ROLES:
        path = profile_paths.get(role)
        if path is None:
            continue
        try:
            result = kernel_capture(Path(path), left, right)
        except (ValueError, KeyError, OSError) as error:
            errors[role] = str(error)
        else:
            traces[role] = result.pop("kernel")
            copies = result.pop("copy")
            if copies is not None:
                traces[f"{role}_CUPTI_copy"] = copies
            required = role != "encoder" or any(row.get("image_count", 1) for row in client)
            if required and result["kernel_records"] == 0:
                result["collector_ok"] = False
            captures[role] = result
        scheduled = scheduler_states(Path(path), left, right)
        if scheduled is not None:
            for key, spans in scheduled.pop("intervals").items():
                traces[f"{role}_{key}"] = spans
            schedulers[role] = scheduled

    enclosure = {
        "E_HTTP": ("E_submit", "E_complete"), "P_HTTP": ("P_submit", "P_ready"),
        "H_HTTP": ("H_submit", "H_ready"),
        "D_HTTP": ("D_submit", "D_http_complete"),
        "embedding_ready_host_queue": ("E_complete", "P_submit"),
        "prefill_ready_host_queue": ("P_ready", "D_submit"),
        "KV_connector": ("D_KV_transfer_begin", "D_KV_transfer_end"),
        "D_local_prefill_and_first_token_HTTP": ("D_submit", "D_first_generated_token"),
    }
    for key, (start, end) in enclosure.items():
        rows = [row for row in proxy if row.get("deflected")] if key == "D_local_prefill_and_first_token_HTTP" else proxy
        if any(start in row and end in row for row in rows):
            traces[key] = host_enclosure(rows, start, end, left, right)

    number = len(client)
    joined = (len(proxy) == number == len({row["runtime_request_id"] for row in proxy})
              and all(row.get("status") == "MEASURED_ONLINE" for row in proxy))
    if joined and all("D_submit" in row and ("D_http_complete" in row or "finished" in row) for row in proxy):
        def eligible(row):
            """Native prefills become D work when ready. D-local prefills also queue at D."""
            return row.get("P_ready", row.get("H_ready", (row.get("D_admission") or {}).get("begin", row["D_submit"])))

        pending = merged(((eligible(row), row.get("D_http_complete", row.get("finished"))) for row in proxy), left, right)
        traces["D_proxy_request_free"] = complement(pending, left, right)
    for signal in ("D_native_nixl_READ_possible", "D_native_nixl_READ_definite_outstanding"):
        if any(signal in row for row in proxy):
            traces[signal] = merged((span for row in proxy for span in row.get(signal, ())), left, right)
    if whole_cohort_native_READ_accounted:
        traces["D_no_accounted_native_nixl_READ"] = complement(
            traces.get("D_native_nixl_READ_possible", []), left, right)
        if "D_proxy_request_free" in traces:
            traces["D_proxy_request_free_and_no_accounted_native_READ"] = intersect(
                traces["D_proxy_request_free"], traces["D_no_accounted_native_nixl_READ"])
    if "decode" in traces:
        traces["D_kernel_idle"] = complement(traces["decode"], left, right)
        for name, sources in (
            ("P_busy_D_kernel_idle", ("prefill", "D_kernel_idle")),
            ("E_busy_D_kernel_idle", ("encoder", "D_kernel_idle")),
            ("E_busy_P_busy", ("encoder", "prefill")),
            ("E_busy_P_busy_D_kernel_idle", ("encoder", "prefill", "D_kernel_idle")),
            ("D_kernel_idle_proxy_request_free", ("D_kernel_idle", "D_proxy_request_free")),
            ("D_kernel_idle_recent_idle_snapshot", ("D_kernel_idle", "decode_recent_idle_snapshot")),
            ("D_kernel_idle_inside_D_HTTP", ("D_kernel_idle", "D_HTTP")),
            ("D_kernel_idle_during_possible_native_READ", ("D_kernel_idle", "D_native_nixl_READ_possible")),
            ("D_kernel_idle_CUPTI_copy_free", ("D_kernel_idle", "decode_no_CUPTI_copy")),
            ("D_kernel_idle_outside_accounted_native_READ", ("D_kernel_idle", "D_no_accounted_native_nixl_READ")),
            ("D_kernel_idle_proxy_free_outside_accounted_native_READ",
             ("D_kernel_idle", "D_proxy_request_free", "D_no_accounted_native_nixl_READ")),
            ("E_P_busy_D_kernel_idle_outside_accounted_native_READ",
             ("encoder", "prefill", "D_kernel_idle", "D_no_accounted_native_nixl_READ")),
            ("D_kernel_idle_proxy_free_with_E_HTTP", ("D_kernel_idle", "D_proxy_request_free", "E_HTTP")),
            ("D_kernel_idle_proxy_free_with_P_HTTP", ("D_kernel_idle", "D_proxy_request_free", "P_HTTP")),
            ("P_busy_D_kernel_idle_proxy_request_free", ("prefill", "D_kernel_idle", "D_proxy_request_free")),
            ("P_plain_waiting_D_kernel_idle_proxy_request_free",
             ("prefill_plain_waiting", "D_kernel_idle", "D_proxy_request_free")),
            ("P_compute_ready_D_recent_idle", ("prefill_known_compute_ready_waiting", "decode_recent_idle_snapshot")),
            ("E_busy_P_compute_ready_D_recent_idle",
             ("encoder", "prefill_known_compute_ready_waiting", "decode_recent_idle_snapshot")),
        ):
            if name.endswith("CUPTI_copy_free") and "decode_CUPTI_copy" in traces:
                traces["decode_no_CUPTI_copy"] = complement(traces["decode_CUPTI_copy"], left, right)
            if all(source in traces for source in sources):
                traces[name] = all_intersections(*(traces[source] for source in sources))

    stage_rates = {}
    if joined:
        by_id = {row["runtime_request_id"]: row for row in client}
        for role, event, key in (("encoder", "E_complete", "visual_tokens"),
                                 ("prefill", "P_ready", "input_tokens_manifest"),
                                 ("decode", "D_http_complete", "output_tokens")):
            selected = [by_id[row["runtime_request_id"]] for row in proxy if left <= row.get(event, -1) < right]
            total = sum(item[key] for item in selected) if selected and all(key in item for item in selected) else None
            stage_rates[role] = {"requests": len(selected), "requests_per_second": len(selected) / duration,
                                 "tokens_per_second": total / duration if total is not None else None}

    opportunity = traces.get("D_kernel_idle_proxy_request_free")
    ready = traces.get("prefill_known_compute_ready_waiting") if schedulers.get("prefill", {}).get(
        "compute_ready_field_observed") else None
    fits = None
    opportunity_durations = None
    if opportunity is not None:
        fits = []
        for quantum in quantum_ms:
            duration_required = quantum / 1000
            eligible_windows = [(a, b) for a, b in opportunity if b - a >= duration_required]
            # P only needs ready work at the candidate start. P's short kernel
            # launch gaps should not shorten a physically quiet D interval.
            eligible_launches = [(a, b - duration_required) for a, b in eligible_windows]
            fit_ready = sum(bool(intersect([interval], ready)) for interval in eligible_launches) if ready is not None else None
            fits.append({"required_ms": quantum, "fit_windows": len(eligible_windows),
                         "fit_windows_with_recent_P_ready_possible_launch": fit_ready,
                         "eligible_interval_seconds": elapsed(eligible_windows)})
        opportunity_durations = {
            "D_kernel_idle_proxy_request_free_ms": distribution((b - a) * 1000 for a, b in opportunity),
            "D_kernel_idle_proxy_request_free_with_recent_P_ready_ms": distribution(
                (b - a) * 1000 for a, b in intersect(opportunity, ready)) if ready is not None else None,
        }
    handoffs = sorted(row["P_ready"] for row in proxy if left <= row.get("P_ready", -1) < right)
    cross = {key: traces[key] for key in ("encoder", "prefill", "prefill_known_compute_ready_waiting", "D_HTTP")
             if key in traces and (key != "prefill_known_compute_ready_waiting"
                 or schedulers.get("prefill", {}).get("compute_ready_field_observed"))}
    output = {"window_monotonic_s": [left, right], "duration_seconds": duration,
              "window_rule": "Central p20–p80 of scheduled arrivals, before cohort drain.",
              "joined_request_count": len(proxy), "proxy_complete": joined,
              "fraction_of_window": {name: elapsed(values) / duration for name, values in traces.items()},
              "bins": {name: bins(values, left, right) for name, values in traces.items()},
              "captures": captures, "capture_errors": errors, "scheduler": schedulers,
              "kernel_idle_histogram": idle_histogram(traces["D_kernel_idle"], cross, handoffs)
              if "D_kernel_idle" in traces else None,
              "proxy_quiet_idle_histogram": idle_histogram(traces["D_kernel_idle_proxy_request_free"])
              if "D_kernel_idle_proxy_request_free" in traces else None,
              "stage_rates": stage_rates, "retrospective_quantum_fit": fits,
              "observed_opportunity_durations": opportunity_durations,
              "arrival_transform": arrival_transform(client, proxy, left, right),
              "transfer_free_idle": None, "deadline_safe_idle": None, "physical_useful_idle": None}
    if keep_intervals:
        output["intervals"] = traces
    return output
