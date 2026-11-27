"""Reduce native open-loop serving logs and paired Kineto traces without a GPU."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import mean
import re


PERCENTILES = (0.50, 0.95, 0.99)
SAMPLE_PATTERN = re.compile(r"^(vllm:[\w:]+)(\{.*?\})?\s+([-+\d.eE]+)$")
ITERATION_ANNOTATION = re.compile(r"^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
DETAILED_ITERATION_ANNOTATION = re.compile(
    r"^execute_\d+_context_(\d+)\(sq(\d+)sk(\d+)sqsq\d+sqsk\d+\)"
    r"_generation_(\d+)\(sq(\d+)sk(\d+)sqsq\d+sqsk\d+\)$"
)
PROM_COUNTERS = {"vllm:num_preemptions_total", "vllm:prompt_tokens_by_source_total",
                 "vllm:generation_tokens_total", "vllm:iteration_tokens_total_count",
                 "vllm:iteration_tokens_total_sum", "vllm:iteration_tokens_total_bucket",
                 "vllm:request_queue_time_seconds_count",
                 "vllm:request_queue_time_seconds_sum", "vllm:request_prefill_time_seconds_count",
                 "vllm:request_prefill_time_seconds_sum", "vllm:request_decode_time_seconds_count",
                 "vllm:request_decode_time_seconds_sum", "vllm:nixl_bytes_transferred_count",
                 "vllm:nixl_bytes_transferred_sum", "vllm:nixl_xfer_time_seconds_count",
                 "vllm:nixl_xfer_time_seconds_sum", "vllm:nixl_post_time_seconds_count",
                 "vllm:nixl_post_time_seconds_sum", "vllm:nixl_num_failed_transfers_total",
                 "vllm:nixl_num_failed_notifications_total", "vllm:nixl_num_kv_expired_reqs_total",
                 "vllm:nixl_num_descriptors_sum", "vllm:nixl_num_descriptors_count"}
PLOT_COLORS = ("#327688", "#BB6332", "#746097", "#557B48", "#41619A", "#DBA643", "#788087")
GPM_METRICS = (
    "graphics_or_compute_pct", "sm_util_pct", "sm_occupancy_pct", "tensor_util_pct",
    "hmma_util_pct", "dram_bw_pct", "nvlink_rx_mib_s", "nvlink_tx_mib_s", "vram_B",
)
NIXL_COMPAT_HASH = re.compile(r"NIXL compatibility check passed \(hash: ([0-9a-f]+)\)")


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def receipt(path):
    with path.open("rb") as stream:
        return {"bytes": path.stat().st_size, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def describe(name, values):
    return {f"{name}_p{int(q * 100)}": quantile(values, q) for q in PERCENTILES}


def sample_summary(samples, begin, end):
    roles = defaultdict(list)
    for row in samples:
        roles[row["rank"]].append(row)
    result = {}
    for role, rows in roles.items():
        rows.sort(key=lambda row: row["s"])
        fields = set().union(*(row.keys() for row in rows)) - {"s", "rank"}
        aggregates = {}
        for field in sorted(fields):
            observed = [row[field] for row in rows if begin <= row["s"] <= end and field in row]
            segments = [
                (min(latest["s"], end) - max(earliest["s"], begin), earliest[field])
                for earliest, latest in zip(rows, rows[1:])
                if latest["s"] > begin and earliest["s"] < end and field in earliest
            ]
            coverage = sum(duration for duration, _ in segments)
            aggregates[field.removeprefix("vllm:")] = {
                "samples": len(observed),
                "max_sample": max(observed, default=None),
                "covered_s": coverage,
                "weighted_mean": (sum(duration * value for duration, value in segments) / coverage
                                  if coverage else None),
                "positive_fraction": (sum(duration for duration, value in segments if value > 0) / coverage
                                      if coverage else None),
            }
        result[role] = aggregates
    return result


def prom_deltas(directory):
    def parse(path):
        values = {}
        for line in path.read_text().splitlines():
            match = SAMPLE_PATTERN.fullmatch(line)
            if match and match[1] in PROM_COUNTERS:
                if match[1] == "vllm:iteration_tokens_total_bucket" and 'le="1.0"' not in (match[2] or ""):
                    continue
                values[match[1], match[2]] = float(match[3])
        return values

    output = {}
    for initial in sorted(directory.glob("before_*.prom")):
        role = initial.stem.removeprefix("before_")
        final = directory / f"after_{role}.prom"
        if final.exists():
            before, after = parse(initial), parse(final)
            output[role] = [
                {"metric": name, "labels": labels, "delta": value - before.get((name, labels), 0.0)}
                for (name, labels), value in sorted(after.items(), key=lambda x: str(x[0]))
            ]
    return output


def gpu_summary(directory, begin, end):
    physical = {}
    for path in sorted(directory.parent.glob("*_command.json")):
        role = path.stem.removesuffix("_command")
        devices = read_json(path)["physical"].split(",")
        for index, uuid in enumerate(devices):
            physical[uuid] = role if len(devices) == 1 else f"{role}/TP{index}"
    samples, errors = defaultdict(list), []
    for row in read_jsonl(directory / "gpu.jsonl"):
        if "error" in row:
            errors.append(row)
            continue
        overlap = min(row["end_s"], end) - max(row["begin_s"], begin)
        if overlap > 0.0:
            samples[row["uuid"]].append((overlap, row))
    devices = {}
    for uuid, intervals in samples.items():
        metrics = {}
        for metric in GPM_METRICS:
            values = [(seconds, row[metric]) for seconds, row in intervals if metric in row]
            coverage = sum(seconds for seconds, _ in values)
            unsupported = Counter(row[f"{metric}_status"] for _, row in intervals if f"{metric}_status" in row)
            metrics[metric] = {"seconds": coverage, "count": len(values), "status_counts": unsupported}
            if coverage:
                weighted_sum = sum(seconds * value for seconds, value in values)
                metrics[metric].update({
                    "weighted_mean": weighted_sum / coverage, "p95_sample": quantile([value for _, value in values], .95),
                    "max_sample": max(value for _, value in values),
                })
        devices[physical.get(uuid, uuid)] = {"uuid": uuid, "metrics": metrics}
    return {"devices": devices, "errors": errors}


def sample_metric(case, role_prefix, name, aggregate):
    values = [
        fields.get(name, {}).get(aggregate)
        for role, fields in case["queue_samples"].items() if role.startswith(role_prefix)
    ]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return max(values) if aggregate == "max_sample" else sum(values)


def counter_metric(case, role_prefix, metric, labels=None):
    values = [entry["delta"] for role, entries in case["counter_delta"].items()
              if role.startswith(role_prefix) for entry in entries
              if entry["metric"] == "vllm:" + metric
              and (labels is None or labels in (entry["labels"] or ""))]
    return sum(values) if values else None


def transport_layout(engine):
    receipt_path = engine / "kv_layout.json"
    layouts, compatibility, runners = {}, {}, {}
    if receipt_path.is_file():
        for role, response in read_json(receipt_path).items():
            try:
                if response["status"] != 200:
                    raise ValueError(f"HTTP {response['status']}")
                layouts[role] = json.loads(response["body"])["results"]
            except (KeyError, ValueError) as error:
                layouts[role] = {"status": str(error).replace("|", "/")}
    for log in sorted(engine.glob("[PD]*.log")):
        hashes = set()
        with log.open(errors="replace") as stream:
            for line in stream:
                if match := NIXL_COMPAT_HASH.search(line):
                    hashes.add(match[1])
                if "Using V2 Model Runner" in line:
                    runners[log.stem] = "V2"
                elif "Using V1 Model Runner" in line:
                    runners[log.stem] = "V1"
        if hashes:
            compatibility[log.stem] = sorted(hashes)
    return {"worker_layouts": layouts, "nixl_compatibility_hashes": compatibility, "logged_model_runners": runners}


def summarize(root, path):
    meta = read_json(path)
    case_path = path.parent.relative_to(root)
    rows = read_jsonl(path.with_name("requests.jsonl"))
    valid = [row for row in rows if row["status"] == "OK"]
    arrivals = [row["arrival"] for row in rows]
    drain_s = meta["finish"] - meta["start"]
    first, last = min(arrivals), max(arrivals)
    arrival_span = last - first
    trim = min(2.0, arrival_span * 0.10)
    begin, end = first + trim, last - trim
    itl = [value for row in valid for value in row["itl_ms"]]
    first_to_submit = [1000 * (row["first_token"] - row["d_submit"]) for row in valid]
    p_response = [1000 * (row["p_finish"] - row["start"]) for row in valid if "p_finish" in row]
    slo = meta["slo"]
    satisfied = [
        row for row in valid if row["arrival_ttft_ms"] <= slo["ttft_ms"]
        and row["tpot_ms"] <= slo["tpot_ms"] and max(row["itl_ms"], default=0.0) <= slo["max_itl_ms"]
    ]
    outputs = [len(row["tokens"]) for row in valid]
    total_prompt_tokens = sum(row["input"] for row in valid)
    physical = {uuid for command in path.parent.parent.glob("*_command.json")
                for uuid in read_json(command)["physical"].split(",")}
    correct_contract = len(rows) == meta["count"] and len({row["id"] for row in rows}) == len(rows)
    result = dict(
        path=str(case_path), evidence="MEASURED_ONLINE", **meta,
        gpu_uuids=sorted(physical),
        gpu_budget=len(physical) if physical else meta["prefill_workers"] * meta["p_tp"] + meta["decode_workers"],
        logged=len(rows), ok=len(valid), failed=len(rows) - len(valid), complete_contract=correct_contract,
        errors=Counter(row.get("error", "unspecified") for row in rows if row["status"] != "OK"),
        inputs=sorted({row["input"] for row in rows}),
        output_contract=sorted({row["expected_output"] for row in rows}),
        actual_output=describe("tokens", outputs), drain_s=drain_s, offered_span_s=arrival_span,
        realized_arrival_qps=((len(rows) - 1) / arrival_span if arrival_span else None),
        request_throughput=len(valid) / drain_s, prompt_tokens_per_s=total_prompt_tokens / drain_s,
        output_tokens_per_s=sum(outputs) / drain_s,
        slo_ok=len(satisfied), slo_client_submit_ok=sum(row["slo_pass"] for row in valid),
        goodput=len(satisfied) / drain_s if meta["mode"] != "prefill" else None,
        decode_slo_ok=sum(
            len(row["tokens"]) >= 2 and row["tpot_ms"] <= slo["tpot_ms"]
            and max(row["itl_ms"], default=0.0) <= slo["max_itl_ms"] for row in valid
        ),
        max_itl_ms=max(itl, default=None), grouped_sse_events=sum(
            size > 1 for row in valid for size in row.get("sse_groups", [])
        ),
        n_itl=len(itl),
        ready_to_submit_p95_ms=quantile(
            [1000 * (row["d_submit"] - row["p_finish"]) for row in valid if "p_finish" in row], .95),
        decode_submit_to_first_p95_ms=quantile(first_to_submit, .95),
        prefill_rpc_p95_ms=quantile(p_response, .95),
        central_roi_mono=[begin, end],
        prompt_rows=total_prompt_tokens, recorded_remote_block_count=sum(
            sum(row.get("remote_block_groups", [])) for row in valid
        ),
        queue_samples=sample_summary(read_jsonl(path.with_name("queues.jsonl")), begin, end),
        counter_delta=prom_deltas(path.parent),
        gpm=gpu_summary(path.parent, begin, end),
        kv_layout=transport_layout(path.parent.parent),
        raw_receipts={source.name: receipt(source) for source in sorted(path.parent.iterdir())
                      if source.is_file() and (source.suffix == ".prom" or source.name
                                               in ("run.json", "requests.jsonl", "queues.jsonl", "gpu.jsonl", "shapes.json"))},
    )
    for label, values in {
        "ttft_ms": [row["arrival_ttft_ms"] for row in valid],
        "submit_ttft_ms": [row["ttft_ms"] for row in valid],
        "tpot_ms": [row["tpot_ms"] for row in valid if row["itl_ms"]],
        "itl_ms": itl,
        "e2e_ms": [1000 * (row["token_times"][-1] - row["arrival"]) for row in valid],
        "request_dispatch_ms": [1000 * (row["start"] - row["arrival"]) for row in rows],
    }.items():
        result.update(describe(label, values))
    result["goodput_per_gpu"] = result["goodput"] / result["gpu_budget"] if result["goodput"] is not None else None
    result["shape_observed"] = (
        bool(meta.get("shape_observed")) or path.parent.name.endswith("_shape") or path.with_name("shapes.json").is_file()
    )
    archived_sampling = {
        "native-pd": "inline", "dense-controls": "background", "native-static-3gpu": "background",
        "native-steady": "background", "native-p-only": "background",
    }
    result["gpu_sampling_regime"] = meta.get("gpu_sampling_regime") or archived_sampling.get(
        next(iter(case_path.parts), ""), "unspecified")
    result["token_sequences"] = {row["id"]: row["tokens"] for row in valid}
    if meta["mode"] in ("pd", "prefill"):
        completion_field = "p_finish" if meta["mode"] == "pd" else "first_token"
        completed_prompt = sum(row["input"] for row in valid
                               if completion_field in row and begin <= row[completion_field] < end)
        result["p_completion_roi_tokens"] = completed_prompt
        result["p_completion_roi_tokens_per_s"] = completed_prompt / (end - begin) if end > begin else None
    remote = [row for row in valid if "p_finish" in row]
    if remote:
        boundaries = [("arrival", "start"), ("start", "p_finish"), ("p_finish", "d_submit"), ("d_submit", "first_token")]
        result["mean_ttft_host_segments_ms"] = {
            f"{left}_to_{right}": 1000 * mean(row[right] - row[left] for row in remote)
            for left, right in boundaries
        }
        result["mean_ttft_handoff_requests_ms"] = mean(row["arrival_ttft_ms"] for row in remote)
    result["native_transfer_bytes"] = counter_metric(result, "D", "nixl_bytes_transferred_sum")
    result["native_transfer_count"] = counter_metric(result, "D", "nixl_bytes_transferred_count")
    n_desc = counter_metric(result, "D", "nixl_num_descriptors_sum")
    n_posts = counter_metric(result, "D", "nixl_num_descriptors_count")
    result["descriptors_per_transfer"] = n_desc / n_posts if n_posts and n_desc is not None else None
    for timer in ("xfer", "post"):
        seconds = counter_metric(result, "D", f"nixl_{timer}_time_seconds_sum")
        count = counter_metric(result, "D", f"nixl_{timer}_time_seconds_count")
        result[f"mean_nixl_{timer}_ms"] = seconds * 1000 / count if count and seconds is not None else None
    result["transferred_prompt_tokens"] = counter_metric(
        result, "D", "prompt_tokens_by_source_total", 'source="external_kv_transfer"')
    result["p_preemptions"] = counter_metric(result, "P", "num_preemptions_total")
    result["d_preemptions"] = counter_metric(result, "D", "num_preemptions_total")
    result["d_accounting_iterations"] = counter_metric(result, "D", "iteration_tokens_total_count")
    result["d_accounting_tokens"] = counter_metric(result, "D", "iteration_tokens_total_sum")
    result["d_accounting_iterations_le_one"] = counter_metric(
        result, "D", "iteration_tokens_total_bucket", 'le="1.0"')
    return result


def union(intervals, begin, end):
    merged = []
    for left, right in sorted((max(left, begin), min(right, end)) for left, right in intervals):
        if right <= left:
            continue
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return merged


def duration(intervals):
    return sum(right - left for left, right in intervals)


def intersection(lhs, rhs):
    pairs = []
    i = j = 0
    while i < len(lhs) and j < len(rhs):
        left = max(lhs[i][0], rhs[j][0])
        right = min(lhs[i][1], rhs[j][1])
        if right > left:
            pairs.append((left, right))
        if lhs[i][1] <= rhs[j][1]:
            i += 1
        else:
            j += 1
    return pairs


def kernel_family(name):
    name = name.lower()
    for family, indicators in (
        ("Collective", ("nccl", "rccl", "all_reduce", "reduce_scatter", "all_gather")),
        ("KV cache write", ("reshape_and_cache", "concat_and_cache", "append_paged_kv")),
        ("Attention", ("flash_fwd", "flashattn", "fmha", "flash_fwd_kernel", "pagedattention",
                       "batchdecodewithpagedkv", "batchprefillwithpagedkv", "varlen_fwd", "flash_attn",
                       "flash::", "flashinfer::attention")),
        ("Norm", ("rmsnorm", "rms_norm", "layer_norm")),
        ("RoPE", ("rotary_embedding", "apply_rope")),
        ("FFN activation", ("silu_and_mul", "gelu_and_mul", "mul_silu")),
        ("Embedding", ("embedding_cuda", "embedding_forward")),
        ("Sampling", ("topk", "top_k", "topp", "top_p", "sampling", "argmax", "multinomial")),
        ("Linear / GEMM", ("gemm", "gemv", "xmma", "cublas", "cutlass", "matmul", "nvjet_")),
    ):
        if any(token in name for token in indicators):
            return family
    return "Other / unresolved"


def trace_records(path):
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        import ijson
    except ImportError:
        with opener(path, "rt") as stream:
            data = json.load(stream)
        return data.get("baseTimeNanoseconds"), iter(data.get("traceEvents", []))
    with opener(path, "rb") as stream:
        base = next(ijson.items(stream, "baseTimeNanoseconds"), None)

    def iterate():
        with opener(path, "rb") as stream:
            yield from ijson.items(stream, "traceEvents.item", use_float=True)

    return base, iterate()


def kernel_activity_events(path, clock):
    first, last = (clock[f"clock_{boundary}"] for boundary in ("start", "end"))
    start_ns = (first["mono_before_ns"] + first["mono_after_ns"]) / 2
    end_ns = (last["mono_before_ns"] + last["mono_after_ns"]) / 2
    cupti_span = last["cupti_ns"] - first["cupti_ns"]
    if cupti_span <= 0 or end_ns <= start_ns:
        raise ValueError("CUPTI calibration clocks did not advance")
    mono_per_cupti = (end_ns - start_ns) / cupti_span

    def aligned(tick):
        return (start_ns + (int(tick) - first["cupti_ns"]) * mono_per_cupti) * 1e-9

    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            yield "kernel", row["name"], aligned(row["start_ns"]), aligned(row["end_ns"])


def kineto_events(events, offset):
    for event in events:
        if event.get("ph") == "X" and "dur" in event and "ts" in event:
            start = offset + float(event["ts"]) * 1e-6
            yield event.get("cat", ""), event.get("name", ""), start, start + float(event["dur"]) * 1e-6


def read_trace(path, case, role):
    begin, end = case["central_roi_mono"]
    kernel_only = path.suffix == ".csv"
    metadata = {"collector": "cupti_kernel_only" if kernel_only else "kineto"}
    if kernel_only:
        clock = read_json(path.with_suffix(".json"))
        markers = [(clock["clock_start"]["mono_after_ns"] * 1e-9, clock["clock_end"]["mono_before_ns"] * 1e-9)]
        events = kernel_activity_events(path, clock)
        metadata["calibration_and_integrity"] = clock
    else:
        base, trace = trace_records(path)
        if base is None:
            return {"error": "Kineto baseTimeNanoseconds missing"}
        offset = float(base) * 1e-9 - case["clock_epoch_minus_mono"]
        markers = []
        events = kineto_events(trace, offset)
        metadata["clock_offset_s"] = offset

    kernels, copy_intervals, memset_intervals = [], [], []
    families = defaultdict(list)
    names = Counter()
    captures = Counter()
    shapes = Counter()
    detailed_shapes = Counter()
    total_kernels = 0
    for category, name, start, stop in events:
        if category == "kernel":
            total_kernels += 1
        if category == "Trace" or name == "PyTorch Profiler (0)":
            markers.append((start, stop))
        if stop <= begin or start >= end:
            continue
        captures[category] += 1
        if category == "user_annotation" and begin <= start < end:
            geometry = ITERATION_ANNOTATION.match(name)
            if geometry:
                shapes[tuple(map(int, geometry.groups()))] += 1
            else:
                detailed = DETAILED_ITERATION_ANNOTATION.match(name)
                if detailed:
                    ctx, cq, ck, gen, gq, gk = map(int, detailed.groups())
                    detailed_shapes[(ctx, cq, ck, gen, gq, gk)] += 1
                    shapes[(ctx, cq, gen, gq)] += 1
        if category == "kernel":
            kernels.append((start, stop))
            families[kernel_family(name)].append((start, stop))
            names[name] += (min(stop, end) - max(start, begin)) * 1000
        elif category == "gpu_memcpy":
            copy_intervals.append((start, stop))
        elif category == "gpu_memset":
            memset_intervals.append((start, stop))
    active = union(kernels, begin, end)
    if not active:
        return {"error": "No aligned GPU kernel intervals in ROI", "category_counts": captures,
                "profiler_markers_mono": markers, **metadata}
    copy = union(copy_intervals, begin, end)
    bins = []
    for index in range(100):
        lo = begin + (end - begin) * index / 100
        hi = begin + (end - begin) * (index + 1) / 100
        bins.append(duration(intersection(active, [(lo, hi)])) / (hi - lo))
    return {
        "role": role,
        "path": str(path),
        "raw_receipt": receipt(path),
        **metadata,
        "record_integrity_pass": (not kernel_only or (
            clock["returncode"] == 0 and clock["dropped"] == 0 and clock["invalid"] == 0
            and total_kernels == clock["exported"]
        )),
        "copy_capture_available": not kernel_only,
        "profiler_markers_mono": markers,
        "marker_covers_roi": any(start <= begin and stop >= end for start, stop in markers),
        "category_counts": captures,
        "iteration_geometry": [
            {"context_requests": shape[0], "context_rows": shape[1],
             "generation_requests": shape[2], "generation_rows": shape[3], "iterations": count}
            for shape, count in sorted(shapes.items())
        ],
        "detailed_iteration_geometry": [
            {"context_requests": shape[0], "context_rows": shape[1], "context_kv_rows": shape[2],
             "generation_requests": shape[3], "generation_rows": shape[4], "generation_kv_rows": shape[5], "iterations": count}
            for shape, count in sorted(detailed_shapes.items())
        ],
        "kernel_busy_fraction": duration(active) / (end - begin),
        "memcpy_busy_fraction": duration(copy) / (end - begin) if not kernel_only else None,
        "kernel_busy_bins": bins,
        "logical_kernel_count": len(kernels),
        "top_kernel_names_by_event_sum_ms": names.most_common(30),
        "family_union_ms": {name: 1000 * duration(union(intervals, begin, end))
                            for name, intervals in sorted(families.items())},
        "family_event_sum_ms": {name: 1000 * sum(min(stop, end) - max(start, begin)
                                                  for start, stop in intervals)
                                for name, intervals in sorted(families.items())},
        "intervals": active,
        "captured_gpu_activity": union(kernels + copy_intervals + memset_intervals, begin, end),
    }


def profile_gate(case, previous, following):
    if previous is None or following is None:
        return {"status": "UNPAIRED", "max_abs_change_pct": None}
    profile, clean_before, clean_after = case, previous, following
    metrics = ("ttft_ms_p95", "tpot_ms_p95", "itl_ms_p99", "request_throughput")
    deltas = {}
    for metric in metrics:
        observed, before, after = (arm[metric] for arm in (profile, clean_before, clean_after))
        if all(value is not None and value > 0 for value in (observed, before, after)):
            deltas[metric] = [100 * (observed / before - 1), 100 * (observed / after - 1),
                              100 * (after / before - 1)]
    maximum = max((abs(value) for values in deltas.values() for value in values), default=None)
    contract_fields = ("trace_sha256", "qps", "mode", "model", "apc", "natural", "batch_tokens", "chunked",
                       "p_tp", "cache_gib_per_device", "prefill_workers", "decode_workers", "nixl_cross_layer_blocks_requested", "deflect_at")
    contract = all(profile.get(key) == clean_before.get(key) == clean_after.get(key) for key in contract_fields)
    parity = profile["token_sequences"] == clean_before["token_sequences"] == clean_after["token_sequences"]
    complete = all(arm["complete_contract"] and not arm["failed"] for arm in (profile, clean_before, clean_after))
    ok = complete and contract and parity and len(deltas) == len(metrics) and maximum <= 5.0
    return {"status": "PASS" if ok else "FAIL", "max_abs_change_pct": maximum,
            "contract_match": contract, "exact_tokens": parity, "all_requests_ok": complete,
            "relative_pct_profile_before_profile_after_after_before": deltas}


def matched_observer_controls(case, by_path):
    directory = Path(case["path"])
    return [by_path.get(str(directory.with_name(case["case"] + suffix)))
            for suffix in ("_clean_before", "_clean_after")]


def shape_roles(path, begin, end):
    roles = {}
    for role, response in read_json(path).items():
        for index, worker in enumerate(response["results"]):
            role_label = role if len(response["results"]) == 1 else f"{role}/rank{index}"
            by_geometry = Counter()
            for sample in worker["samples"]:
                if begin <= sample["host_s"] < end:
                    geometry = tuple(zip(sample["rows"], sample["kv_rows"], sample["context"], strict=True))
                    by_geometry[geometry] += 1
            roles[role_label] = {
                "observer_stride": worker["stride"],
                "roi_samples": sum(by_geometry.values()),
                "schedule_geometry": [
                    {"query_rows": [item[0] for item in geometry],
                     "kv_rows": [item[1] for item in geometry],
                     "context": [item[2] for item in geometry], "samples": count}
                    for geometry, count in sorted(by_geometry.items())
                ],
            }
    return roles


def scheduler_shapes(root, cases):
    by_path = {case["path"]: case for case in cases}
    groups = []
    for case in cases:
        if not case["shape_observed"]:
            continue
        path = root / case["path"] / "shapes.json"
        gate = profile_gate(case, *matched_observer_controls(case, by_path))
        group = {"path": case["path"], "observer": gate, "evidence": "NOT_RUN", "roles": {}}
        groups.append(group)
        if not path.is_file():
            continue
        group["raw_receipt"] = receipt(path)
        try:
            group["roles"] = shape_roles(path, *case["central_roi_mono"])
        except (ValueError, KeyError, TypeError, OSError) as error:
            group["error"] = str(error)
        coverage = {"P0", "D0"} <= set(group["roles"]) and all(role["roi_samples"] for role in group["roles"].values())
        group["evidence"] = "TRACE_ANALYSIS" if gate["status"] == "PASS" and coverage else "DIAGNOSTIC"
    return groups


def profiles(root, cases):
    by_path = {case["path"]: case for case in cases}
    output = []
    for case in cases:
        if not case["profiled"]:
            continue
        directory = Path(case["path"])
        gate = profile_gate(case, *matched_observer_controls(case, by_path))
        roles = {}
        expected_roles = tuple(case.get("profile_roles") or ("P0", "D0"))
        engine_dir = root / directory.parent
        collector = case.get("profiler") or "kineto"
        if collector == "cupti_kernel_only":
            trace_paths = (path for path in (engine_dir / "activity").glob("*/worker-*.csv")
                           if path.with_suffix(".json").is_file())
        else:
            trace_paths = (path for path in (engine_dir / "profiles").glob("*/*.pt.trace.json*")
                           if path.suffix in (".json", ".gz"))
        for path in sorted(trace_paths):
            role = path.parent.name
            if role not in expected_roles:
                continue
            role_label = role if role not in roles else f"{role}/{path.name}"
            try:
                roles[role_label] = read_trace(path, case, role_label)
            except (ValueError, KeyError, OSError) as error:
                roles[role_label] = {"error": str(error)}
            if "path" in roles[role_label]:
                roles[role_label]["path"] = str(path.relative_to(root))
        paired_roles = ("P0", "D0")
        coverage = (set(roles) == set(expected_roles)
                    and all(role.get("marker_covers_roi") for role in roles.values()))
        integrity = (set(roles) == set(expected_roles)
                     and all(role.get("record_integrity_pass") for role in roles.values()))
        evidence = "TRACE_ANALYSIS" if gate["status"] == "PASS" and coverage and integrity else "DIAGNOSTIC"
        p_busy_d_kernel_inactive = None
        p_busy_d_no_capture = None
        overlap = None
        if all("intervals" in roles.get(role, {}) for role in paired_roles):
            width = case["central_roi_mono"][1] - case["central_roi_mono"][0]
            p_intervals, d_intervals = [roles[role]["intervals"] for role in paired_roles]
            overlap_duration = duration(intersection(p_intervals, d_intervals))
            overlap = overlap_duration / width
            p_busy_d_kernel_inactive = (duration(p_intervals) - overlap_duration) / width
            if roles["D0"]["copy_capture_available"]:
                all_d = roles["D0"]["captured_gpu_activity"]
                p_busy_d_no_capture = (duration(p_intervals) - duration(intersection(p_intervals, all_d))) / width
        for role in roles.values():
            role.pop("intervals", None)
            role.pop("captured_gpu_activity", None)
        output.append({"path": case["path"], "case": case["case"], "qps": case["qps"],
                       "inputs": case["inputs"], "outputs": case["output_contract"],
                       "central_roi_mono": case["central_roi_mono"],
                       "collector": collector, "profile_roles": expected_roles, "record_integrity_pass": integrity,
                       "evidence": evidence, "observer": gate, "marker_coverage_all_roles": coverage,
                       "roles": roles, "both_kernel_active_fraction": overlap,
                       "p_active_d_kernel_inactive_fraction": p_busy_d_kernel_inactive,
                       "p_active_d_without_recorded_gpu_activity_fraction": p_busy_d_no_capture})
    return output


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def label(case):
    if len(case["inputs"]) != 1 or len(case["output_contract"]) != 1:
        shape = case["case"].replace("_", " ")
    else:
        shape = f"{case['inputs'][0] // 1024}K/{case['output_contract'][0]}"
    if case["natural"]:
        shape += " EOS"
    if case["mode"] == "pd":
        ranks = f"{case['prefill_workers']}P"
        if case["p_tp"] > 1:
            ranks += f"(TP{case['p_tp']})"
        ranks += f"/{case['decode_workers']}D"
    elif case["mode"] == "dp":
        ranks = f"DP{case['decode_workers']}"
    else:
        ranks = "P reference"
    model = Path(case["model"]).name
    model_label = f" {model}" if model != "qwen3-8b" else ""
    variant = (
        " deflect" if case.get("deflect_at")
        else " cross-layer requested" if case.get("nixl_cross_layer_blocks_requested")
        else ""
    )
    return f"{ranks}{model_label}{variant} {shape} · {case['qps']:g} req/s"


def reference(case, cases, match_batch):
    match = [ref for ref in cases if ref["mode"] == "prefill" and not ref["failed"]
             and ref["inputs"] == case["inputs"] and ref["p_tp"] == case["p_tp"]
             and (not match_batch or ref["batch_tokens"] == case["batch_tokens"])
             and ref["model"] == case["model"] and ref["cache_gib_per_device"] == case["cache_gib_per_device"]]
    return max(match, key=lambda ref: ref["prompt_tokens_per_s"], default=None)


def add_motivation(cases, profile_data):
    by_path = {profile["path"]: profile for profile in profile_data}
    for case in cases:
        case["p_waiting_mean"] = sample_metric(case, "P", "num_requests_waiting", "weighted_mean")
        case["p_waiting_positive_fraction"] = sample_metric(case, "P", "num_requests_waiting", "positive_fraction")
        case["d_waiting_mean"] = sample_metric(case, "D", "num_requests_waiting", "weighted_mean")
        case["d_running_mean"] = sample_metric(case, "D", "num_requests_running", "weighted_mean")
        for role in ("P", "D"):
            key = role.lower()
            case[f"{key}_kv_peak_sample"] = sample_metric(case, role, "kv_cache_usage_perc", "max_sample")
            for reason in ("capacity", "deferred"):
                name = f"num_requests_waiting_by_reason:{reason}"
                case[f"{key}_{reason}_waiting_mean"] = sample_metric(case, role, name, "weighted_mean")
                case[f"{key}_{reason}_waiting_positive_fraction"] = sample_metric(case, role, name, "positive_fraction")
        if case["mode"] != "pd":
            continue
        for match_batch, suffix in ((True, ""), (False, "_best")):
            ref = reference(case, cases, match_batch)
            case[f"p_reference_path{suffix}"] = ref["path"] if ref else None
            case[f"p_reference_ratio_finite{suffix}"] = (
                case["prompt_tokens_per_s"] / ref["prompt_tokens_per_s"]
                if ref and not case.get("deflected_count") and case["prefill_workers"] == 1 else None
            )
            if not match_batch:
                roi_rate = case["p_completion_roi_tokens_per_s"]
                case["p_roi_ratio_to_best_finite_reference"] = (
                    roi_rate / ref["prompt_tokens_per_s"]
                    if ref and roi_rate is not None and not case.get("deflected_count") and case["prefill_workers"] == 1
                    else None
                )
        profile = by_path.get(case["path"])
        if profile is None:
            case["prefill_heavy_gate"] = "NOT_RUN_PROFILE"
        elif profile["evidence"] != "TRACE_ANALYSIS":
            case["prefill_heavy_gate"] = "DIAGNOSTIC_PROFILE"
        elif "P0" not in profile["roles"]:
            case["prefill_heavy_gate"] = "NOT_RUN_P_PROFILE"
        else:
            p_busy = profile["roles"]["P0"]["kernel_busy_fraction"]
            no_queue = case["p_waiting_mean"] is None or case["p_waiting_mean"] <= 0.0
            misses_decode_tail = (
                case["tpot_ms_p95"] is None
                or case["tpot_ms_p95"] > case["slo"]["tpot_ms"]
                or case["itl_ms_p99"] is None
                or case["itl_ms_p99"] > case["slo"]["max_itl_ms"]
            )
            p_rate_ratio = case["p_roi_ratio_to_best_finite_reference"]
            case["observed_prefill_gate_components"] = {
                "p_kernel_busy_at_least_80_percent": p_busy >= .80,
                "p_wait_queue_observed": not no_queue,
                "p_roi_rate_at_least_85_percent_of_fastest_finite_reference":
                    p_rate_ratio >= .85 if p_rate_ratio is not None else None,
                "decode_tail_slo": not misses_decode_tail,
                "d_capacity_wait_observed": (
                    case["d_capacity_waiting_mean"] > 0 if case["d_capacity_waiting_mean"] is not None else None
                ),
                "d_kv_admission_feedback_excluded": None,
                "d_kernel_intersection_measured": profile["both_kernel_active_fraction"] is not None,
                "deadline_safe_d_opportunity_verified": None,
            }
            if p_busy < .80 or no_queue or misses_decode_tail:
                case["prefill_heavy_gate"] = "FAIL_OBSERVED_COMPONENT"
            else:
                case["prefill_heavy_gate"] = "INCONCLUSIVE_SERVICE_AND_CREDIT_DECOMPOSITION"


def fmt(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def pct(value):
    return fmt(value * 100) + "%" if value is not None else "—"


def compare_native(cases):
    groups = defaultdict(list)
    for case in cases:
        if (case["mode"] != "prefill" and not case["profiled"] and not case["shape_observed"]
                and not case["path"].endswith("_clean_after")):
            groups[(case["model"], case["trace_sha256"], case["qps"], case["apc"], case["natural"],
                    case["gpu_budget"], tuple(case["gpu_uuids"]), case["gpu_sampling_regime"])].append(case)
    result = []
    for group in groups.values():
        for first, second in combinations(group, 2):
            first_outputs, second_outputs = first["token_sequences"], second["token_sequences"]
            matched = first_outputs.keys() & second_outputs.keys()
            result.append({
                "first": first["path"], "second": second["path"], "count": first["count"],
                "both_ok": len(matched), "exact": sum(first_outputs[key] == second_outputs[key] for key in matched),
                "first_ttft_p95_ms": first["ttft_ms_p95"], "second_ttft_p95_ms": second["ttft_ms_p95"],
                "first_goodput_req_s": first["goodput"], "second_goodput_req_s": second["goodput"],
                "first_decode_p95_ms": first["tpot_ms_p95"], "second_decode_p95_ms": second["tpot_ms_p95"],
                "first_itl_p99_ms": first["itl_ms_p99"], "second_itl_p99_ms": second["itl_ms_p99"],
            })
    return result


def save_reports(report_dir, cases, profile_data, shapes, incomplete, contrasts, artifact_link, figures):
    report_dir.mkdir(parents=True, exist_ok=True)
    clean_online = [case for case in cases if not case["profiled"] and not case["shape_observed"] and case["mode"] != "prefill"]
    p_reference = [case for case in cases if case["mode"] == "prefill"]
    motivation = [
        "# H200 PD prefill-heavy resource motivation",
        "",
        "## Measurement contract",
        "",
        "The client uses frozen token IDs, a seeded open-loop Poisson clock and greedy BF16 native vLLM. "
        "PD uses a physical NIXL lease returned by P and consumed by D. DP sends the same token contract to a native colocated worker. "
        "Primary cohorts disable automatic prefix caching. Fixed-length serving returns at least 16 tokens. "
        "The independent prefill reference returns one token as an internal phase capacity probe.",
        "",
        "## Dataset lineage",
        "",
        "The controlled trace source is EADA's `qwen_128.json`, SHA256 "
        "`c3472d660ffa61988e04205a1dcbb02b0ff7737ece729a1fdd001cb5ba85b904`. "
        "The public LongBench v1 archive at revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672` has SHA256 "
        "`cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64`. "
        "The earlier cohort builder applies the publisher's task templates and the pinned Qwen2.5-7B-Instruct tokenizer at "
        "`a09a35458c702b33eeacc393d103063234e8bc28`. It filters to 2,048–28,672 templated tokens, sorts within each task by "
        "the SHA256 of the EADA salt and official request ID, and retains 32 each from Qasper, HotpotQA, NarrativeQA and GovReport. "
        "The stored template maps equal the publisher's maps, with two-space UTF-8 JSON formatting and a final newline. "
        "The cohort was reproduced byte-for-byte on CPU against the pinned assets. "
        "[Pinned public dataset](https://huggingface.co/datasets/zai-org/LongBench/tree/5e628be450b7e67fb7ae6e201bd6d8f7056f7672) "
        "[Pinned publisher template](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/config/dataset2prompt.json) "
        "[Prior cohort builder](https://github.com/jyssys/MLLM-EP/blob/07892cbe7193366ec422f1df82006c6e69f91cec/poc_flashvep/encoder_assisted_decode_attention/longbench.py) "
        "[Source ID and hash receipt](https://github.com/jyssys/MLLM-EP/blob/07892cbe7193366ec422f1df82006c6e69f91cec/poc_flashvep/encoder_assisted_decode_attention/results/longbench_manifest_receipt.json)",
        "",
        "The current trace builder shuffles the source with seed 20261129, re-tokenizes its user bodies with the model-specific non-thinking "
        "chat template, and truncates or extends the token bodies to exact input lengths. Extensions use subsequent corpus rows. "
        "Arrival offsets use seed 20261128 and are scaled by offered load. The 16K/16 cohort extends 17 of 24, 49 of 64 and 103 of 128 inputs. "
        "The 128-request variant has 123 distinct token streams. APC remains disabled in the primary arms. "
        "These controlled traces measure serving behavior for LongBench-derived text. Task quality requires untouched task inputs. "
        "[Current trace builder](../pd_resource_complementary_parallelism/workload.py)",
        "",
        "## Metric definitions",
        "",
        "TTFT starts at scheduled arrival. TPOT is the mean inter-token interval per request. ITL is pooled across client-observed token timestamps. "
        "SSE multi-token events share one arrival timestamp. The reducer preserves their counts and resulting zero intervals. "
        "The frozen joint SLO is TTFT ≤5,000 ms, TPOT ≤33 ms and every ITL ≤34 ms. Failures count in the offered cohort and never contribute to goodput. "
        "Successful-request percentiles and finite arrival-to-drain throughput use different denominators. "
        "These finite cohorts cannot certify sustainable throughput.",
        "",
        "The queue and GPU-profile ROI uses the common arrival window with each end trimmed by 10% of its span, capped at two seconds. "
        "Sampled waiting measures the scheduler's published queue. It does not resolve P-ready work, D-credit stalls, or transfer wait. "
        "The P-reference ratio uses completed input tokens per finite wall second. It differs from a same-ROI independent P service-rate comparison.",
        "",
        "The archived initial `native-pd` sweep polled GPM from its client loop. The later `dense-controls` comparisons use a separate "
        "background sampler. Use the latter matched pairs for tail-latency comparisons.",
        "",
        "In the archived campaign, the steady shell wrapper reported exit 2 while parsing its trailing clause after recording "
        "all six complete cohorts. The per-cohort receipts and their successful request counts remain the serving evidence.",
        "",
        "## Independent P reference",
        "",
        "| P witness | Input | Batch token budget | Offered req/s | OK / offered | Wall s | Completed req/s | Input tokens/s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in p_reference:
        motivation.append(f"| {case['path']} | {case['inputs']} | {case['batch_tokens']} | {case['qps']:g} | {case['ok']} / {case['count']} | "
                          f"{fmt(case['drain_s'])} | {fmt(case['request_throughput'], 3)} | {fmt(case['prompt_tokens_per_s'], 0)} |")
    motivation.extend([
        "",
        "## Clean native serving",
        "",
        "| Run | GPUs | Offered req/s | APC | OK / offered | TTFT p95 ms | TPOT p95 ms | ITL p99 ms | Joint SLO | Finite req/s | Goodput req/s / per GPU | P waiting mean | P-ref finite ratio matched / best |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for case in clean_online:
        motivation.append(f"| {case['path']} | {case['gpu_budget']} | {case['qps']:g} | {'on' if case['apc'] else 'off'} | {case['ok']} / {case['count']} | "
                          f"{fmt(case['ttft_ms_p95'])} | {fmt(case['tpot_ms_p95'])} | {fmt(case['itl_ms_p99'])} | "
                          f"{case['slo_ok']} / {case['count']} | {fmt(case['request_throughput'], 3)} | "
                          f"{fmt(case['goodput'], 3)} / {fmt(case['goodput_per_gpu'], 3)} | {fmt(case['p_waiting_mean'])} | "
                          f"{pct(case.get('p_reference_ratio_finite'))} / {pct(case.get('p_reference_ratio_finite_best'))} |")
    if not clean_online:
        motivation.append("| NOT_RUN | — | — | — | — | — | — | — | — | — | — | — | — |")
    if incomplete:
        motivation.extend([
            "",
            "Interrupted cohorts have raw-stage receipts without a valid terminal run record. "
            f"The incomplete cohort manifest preserves them without assigning latency or success rates. "
            f"[Incomplete raw manifest]({artifact_link}/native_incomplete.json)",
        ])
    motivation.extend([
        "",
        "## Matched native output controls",
        "",
        "Pairs share the model, physical GPU set, trace digest, offered load, APC, sampling regime and generation contract. "
        "Output parity compares every emitted token "
        "of requests successful in both arms. A match supports output reproducibility for that cohort. "
        "The table does not infer bitwise logit equality or task accuracy.",
        "",
        "| First run | Second run | Exact / both OK / total | TTFT p95 first / second ms | Goodput first / second req/s |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for pair in contrasts:
        motivation.append(f"| {pair['first']} | {pair['second']} | {pair['exact']} / {pair['both_ok']} / {pair['count']} | "
                          f"{fmt(pair['first_ttft_p95_ms'])} / {fmt(pair['second_ttft_p95_ms'])} | "
                          f"{fmt(pair['first_goodput_req_s'], 3)} / {fmt(pair['second_goodput_req_s'], 3)} |")
    if not contrasts:
        motivation.append("| NOT_RUN | — | — | — | — |")
    motivation.extend([
        "",
        "## Stage and handoff observations",
        "",
        "The client observes P-response completion and the first D token. Ready-to-submit is an application boundary. "
        "D submit-to-first includes transfer, scheduling and model work. NIXL byte and time deltas are measured by the connector. "
        "The client issues the D RPC after the P response. P can still retain block leases from previously completed prefills. "
        "Transport-time sums can include overlapping transfers. The D waiting gauge includes every published reason, "
        "and the preemption counter does not count all admission waits.",
        "",
        "| Run | P RPC p95 ms | D submit-to-first p95 ms | D wait mean | Transfer count | Connector KV GiB | Mean transfer / post ms | Descriptors / transfer | P / D preemptions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for case in clean_online:
        if case["mode"] == "pd":
            byte_count = case["native_transfer_bytes"]
            motivation.append(f"| {case['path']} | {fmt(case['prefill_rpc_p95_ms'])} | "
                              f"{fmt(case['decode_submit_to_first_p95_ms'])} | {fmt(case['d_waiting_mean'])} | "
                              f"{fmt(case['native_transfer_count'], 0)} | {fmt(byte_count / 1024**3 if byte_count is not None else None)} | "
                              f"{fmt(case['mean_nixl_xfer_ms'])} / {fmt(case['mean_nixl_post_ms'])} | "
                              f"{fmt(case['descriptors_per_transfer'], 0)} | "
                              f"{fmt(case['p_preemptions'], 0)} / {fmt(case['d_preemptions'], 0)} |")
    if not any(case["mode"] == "pd" for case in clean_online):
        motivation.append("| NOT_RUN | — | — | — | — | — | — | — | — |")
    layouts = {}
    for case in clean_online:
        if case["mode"] == "pd" and case["kv_layout"]["worker_layouts"]:
            layouts[str(Path(case["path"]).parent)] = case["kv_layout"]
    if layouts:
        motivation.extend([
            "",
            "## Physical connector layout audit",
            "",
            "The experimental cross-layer flag records a request. The table reports the actual allocated shared tensor from the local "
            "worker RPC. The NIXL compatibility hashes in the JSON use the transfer topology, including its cross-layer mode.",
            "",
            "| Native start | Role | Logged runner | Connector / layout | Connector prefers cross-layer | Allocated cross-layer tensor | Shape | Stride |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for start, layout in layouts.items():
            for role, measurements in layout["worker_layouts"].items():
                runner = layout["logged_model_runners"].get(role, "—")
                if isinstance(measurements, dict):
                    motivation.append(f"| {start} | {role} | {runner} | {measurements['status']} | — | — | — | — |")
                    continue
                for worker in measurements:
                    motivation.append(f"| {start} | {role} | {runner} | {worker['connector']} / {worker['layout']} | "
                                      f"{worker['connector_prefers_cross_layer']} | {worker['allocated_cross_layer']} | "
                                      f"{worker['shape'] or '—'} | {worker['stride'] or '—'} |")
    with_reasons = [case for case in cases if case["mode"] == "pd" and not case["shape_observed"]
                    and any(case[key] is not None for key in ("p_capacity_waiting_mean", "d_deferred_waiting_mean"))]
    if with_reasons:
        motivation.extend([
            "",
            "## Scheduler capacity and deferred waits",
            "",
            "`capacity` counts requests in the ordinary waiting queue. `deferred` counts skipped requests, including transient "
            "constraints such as remote KV receipt. A capacity wait alone leaves its scheduler cause open. "
            "The mean gauges are weighted over the common ROI and KV values are sampled peaks of allocated cache-pool utilization. "
            "The preemption counters span the full finite cohort.",
            "",
            "| Run | P capacity / deferred wait | D capacity / deferred wait | P / D KV peak (sampled) | P / D preemptions |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for case in with_reasons:
            motivation.append(f"| {case['path']} | {fmt(case['p_capacity_waiting_mean'], 3)} / {fmt(case['p_deferred_waiting_mean'], 3)} | "
                              f"{fmt(case['d_capacity_waiting_mean'], 3)} / {fmt(case['d_deferred_waiting_mean'], 3)} | "
                              f"{pct(case['p_kv_peak_sample'])} / {pct(case['d_kv_peak_sample'])} | "
                              f"{fmt(case['p_preemptions'], 0)} / {fmt(case['d_preemptions'], 0)} |")
    handoff = [case for case in clean_online if case.get("mean_ttft_host_segments_ms")]
    if handoff:
        motivation.extend([
            "",
            "Mean host-segment durations below use the same successful handoff rows and sum to the mean TTFT of those requests.",
            "",
            "| Run | Arrival → client start ms | Client start → P ready ms | P ready → D submit ms | D submit → first token ms | Mean TTFT ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for case in handoff:
            segments = " | ".join(fmt(value) for value in case["mean_ttft_host_segments_ms"].values())
            motivation.append(f"| {case['path']} | {segments} | {fmt(case['mean_ttft_handoff_requests_ms'])} |")
    motivation.extend([
        "",
        "## Same-window motivation gate",
        "",
        "| Profile | Evidence | P kernel busy | D kernel busy | P active and D kernel inactive | P waited | P ROI rate / fastest finite P witness | Decode SLO | Overall gate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    case_map = {case["path"]: case for case in cases}
    for profile in profile_data:
        case = case_map[profile["path"]]
        p_role, d_role = [profile["roles"].get(role, {}) for role in ("P0", "D0")]
        motivation.append(f"| {case['path']} | {profile['evidence']} | {pct(p_role.get('kernel_busy_fraction'))} | "
                          f"{pct(d_role.get('kernel_busy_fraction'))} | {pct(profile['p_active_d_kernel_inactive_fraction'])} | "
                          f"{pct(case['p_waiting_positive_fraction'])} | {pct(case.get('p_roi_ratio_to_best_finite_reference'))} | "
                          f"{case['decode_slo_ok']} / {case['count']} | "
                          f"{case.get('prefill_heavy_gate', 'NOT_APPLICABLE')} |")
    if not profile_data:
        motivation.append("| NOT_RUN | NOT_RUN | — | — | — | — | — | — | — |")
    for profile in profile_data:
        both, no_capture = (profile[key] for key in (
            "both_kernel_active_fraction", "p_active_d_without_recorded_gpu_activity_fraction",
        ))
        if both is not None:
            coverage_detail = (f"P was active with no recorded D kernel, copy or memset for {pct(no_capture)}."
                               if no_capture is not None else "The kernel-only collector excludes memory-copy events.")
            motivation.extend([
                "",
                f"{profile['case']} [{profile['evidence']}] records simultaneous P/D kernel activity for {pct(both)} of the ROI. "
                f"{coverage_detail} Captured intervals do not resolve interruption-safe time.",
            ])
        elif tuple(profile["profile_roles"]) == ("P0",):
            motivation.extend([
                "",
                f"{profile['case']} captured P kernels only. D kernel occupancy and P/D temporal intersection are unavailable for this arm.",
            ])
        case = case_map[profile["path"]]
        if profile["evidence"] == "TRACE_ANALYSIS" and case["p_roi_ratio_to_best_finite_reference"] is not None:
            motivation.extend([
                "",
                f"{profile['case']} recorded {fmt(case['p_completion_roi_tokens_per_s'], 1)} P input tokens/s across the central window, "
                f"{pct(case['p_roi_ratio_to_best_finite_reference'])} of the fastest same-input finite P witness. "
                f"The ROI mean sampled D capacity/deferred gauges were "
                f"{fmt(case['d_capacity_waiting_mean'], 3)}/{fmt(case['d_deferred_waiting_mean'], 3)}. "
                f"P/D peak sampled cache occupancy was {pct(case['p_kv_peak_sample'])}/{pct(case['d_kv_peak_sample'])}.",
            ])
    motivation.extend([
        "",
        "A qualifying P busy interval alone establishes kernel occupancy. A stricter prefill-heavy claim also requires independently "
        "matched P service rate, P-ready backlog and exclusion of D KV admission and transfer credit stalls. "
        "The ROI rate counts prefill completions inside the central window, while the independent witness uses its finite drain. "
        "D kernel inactivity leaves DMA, hardware counter headroom and future deadline safety unestablished.",
        "",
        "Regenerate both reports and figures from the native raw directory with "
        "`python poc_flashvep/pd_resource_complementary_parallelism/analysis_native.py --raw-root <runs> --out "
        "poc_flashvep/pd_resource_complementary_parallelism/results/native --reports poc_flashvep/reports --figures`. "
        "The plots and streaming trace reductions were generated with Matplotlib 3.11.2 and ijson 3.6.0 in a separate CPU environment.",
        "",
        f"Reducer outputs include complete request counts, all configured percentiles, grouped SSE events, P and D host-boundary times "
        f"and Prometheus deltas. [Native reduction data]({artifact_link}/native_summary.json)",
    ])
    if figures:
        motivation.extend(["", " ".join(
            f"[{name.replace('_', ' ')}]({artifact_link}/figures/{name}.svg)"
            for name in sorted(figures) if name.startswith(("native_qps", "native_pd_operating", "native_pd_kernel_busy"))
        )])
    report_dir.joinpath("pd_prefill_heavy_resource_motivation.md").write_text("\n".join(motivation) + "\n")

    profiling = [
        "# Native H200 operator resource profiling",
        "",
        "CUPTI GPU activities exported by PyTorch Kineto are aligned to the client monotonic clock as "
        "`baseTimeNanoseconds × 10⁻⁹ + kernel.ts × 10⁻⁶ − clock_epoch_minus_mono`. "
        "Chrome event durations use microseconds. Kernel-only CUPTI data instead use the affine map from worker start and end "
        "CUPTI readings to their monotonic-clock bracket midpoints. The inner bounds of the brackets must cover the ROI "
        "for each selected role, with matching export counts and zero dropped or invalid records. "
        "Kineto coverage is checked against the selected profiler markers. Both roles are required to measure temporal intersection. "
        "A profile earns `TRACE_ANALYSIS` only when matched clean-before/profile/clean-after cohorts complete with equal token sequences "
        "and the absolute change is at most 5% on every pair for TTFT p95, TPOT p95, ITL p99, and finite request throughput. "
        "Other traces are `DIAGNOSTIC`. The three runs share workers and are an observer control, not independent server starts.",
        "",
        "| Profile | Collector | Traced roles | Trace status | Observer gate | Largest paired change % | Full selected-role coverage | Record check |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for profile in profile_data:
        profiling.append(f"| {profile['path']} | {profile['collector']} | {', '.join(profile['profile_roles'])} | "
                         f"{profile['evidence']} | {profile['observer']['status']} | "
                         f"{fmt(profile['observer']['max_abs_change_pct'])} | {profile['marker_coverage_all_roles']} | "
                         f"{profile['record_integrity_pass']} |")
    if not profile_data:
        profiling.append("| NOT_RUN | — | — | NOT_RUN | NOT_RUN | — | — | — |")
    for profile in profile_data:
        observer = profile["observer"]
        tp = observer.get("relative_pct_profile_before_profile_after_after_before", {}).get("tpot_ms_p95")
        if tp:
            profiling.extend([
                "",
                f"{profile['case']} TPOT p95 shifted {tp[0]:+.2f}% versus clean-before and "
                f"{tp[1]:+.2f}% versus clean-after. Clean-after versus clean-before shifted {tp[2]:+.2f}%. "
                f"Matched emitted token sequences were {'identical' if observer['exact_tokens'] else 'different'}.",
            ])
    profiling.extend([
        "",
        "## Operator families on the common clock",
        "",
        "Kernel name heuristics separate Attention, GEMM, explicit normalization, activation and cache operations. "
        "The linear family includes cuBLAS and NVJet names. "
        "A generic GEMM kernel name does not identify the QKV, output projection or FFN call site. "
        "The same warning applies to fused operators and GPU graphs. Each value below is a within-family interval union. "
        "Different family intervals may coincide. Neither their sum nor the sum of kernel durations is a serving critical path.",
        "",
        "| Profile | Role | GPU family | Union ms | Event-sum ms | Share of role's active kernel intervals | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ])
    for profile in profile_data:
        window_ms = 1_000 * (profile["central_roi_mono"][1] - profile["central_roi_mono"][0])
        for name, role in profile["roles"].items():
            for family, latency in role.get("family_union_ms", {}).items():
                share = latency / (window_ms * role["kernel_busy_fraction"])
                profiling.append(f"| {profile['case']} | {name} | {family} | {fmt(latency, 3)} | "
                                 f"{fmt(role['family_event_sum_ms'][family], 3)} | {pct(share)} | {profile['evidence']} |")
            if "error" in role:
                profiling.append(f"| {profile['case']} | {name} | missing role trace | — | — | — | {role['error']} |")
    if not any(role.get("family_union_ms") for profile in profile_data for role in profile["roles"].values()):
        profiling.append("| NOT_RUN | — | — | — | — | — | — |")
    profiling.extend([
        "",
        "## Actual engine iteration geometry",
        "",
        "A CPU `execute_context_…_generation_…` annotation counts scheduled context requests and token rows, plus generation requests and rows. "
        "Counts include annotation starts within the common ROI. The reducer uses CPU annotations once per iteration and excludes duplicate GPU annotations. "
        "Rows with no scheduled tokens include waiting or communication progress. These counts supply online shapes for operator controls. "
        "A trace with a failed observer gate establishes diagnostic shapes only.",
        "",
        "| Profile | Role | Context requests / rows | Generation requests / rows | Iterations | Evidence |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ])
    for profile in profile_data:
        for name, role in profile["roles"].items():
            for row in role.get("iteration_geometry", []):
                profiling.append(f"| {profile['case']} | {name} | {row['context_requests']} / {row['context_rows']} | "
                                 f"{row['generation_requests']} / {row['generation_rows']} | {row['iterations']} | "
                                 f"{profile['evidence']} |")
    if not any(role.get("iteration_geometry") for profile in profile_data for role in profile["roles"].values()):
        profiling.append("| NOT_RUN | — | — | — | — | — |")
    profiling.extend([
        "",
        "Qwen3-8B TP1 has 32 query heads, eight KV heads, head dimension 128, hidden width 4,096 and FFN width 12,288. "
        "For an annotated group of M token rows, the logical GEMM geometry before padding is QKV 6,144 × 4,096, "
        "attention output 4,096 × 4,096, fused gate/up 24,576 × 4,096, and down projection 4,096 × 12,288 in N × K order. "
        "These dimensions come from the checkpoint and runtime model definition. The role-specific M values above come from the actual trace. "
        "[Pinned checkpoint geometry](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json) "
        "[vLLM Qwen3 implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py)",
    ])
    profiling.extend([
        "",
        "## Strided scheduler geometry observer",
        "",
        "The worker samples every fourth dispatch and records each request's scheduled query rows, "
        "computed prefix positions plus query rows as its logical KV length, and context/generation flags. "
        "The raw arrays preserve sampled ragged shapes. Weighting by samples leaves unsampled iterations outside the population claim. "
        "This CPU observer uses the same clean-before/observer/clean-after 5% and token-parity contract.",
        "",
        "| Cohort | Role | ROI samples | Batch size range | Scheduled-query total p50 / p95 | Context-only / generation-only / mixed | Logical KV rows per request range | Observer max change % | Evidence |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for shape in shapes:
        for name, role in shape["roles"].items():
            queries = [sum(row["query_rows"]) for row in role["schedule_geometry"] for _ in range(row["samples"])]
            batch = [len(row["query_rows"]) for row in role["schedule_geometry"]]
            positions = [position for row in role["schedule_geometry"] for position in row["kv_rows"]]
            context_count = sum(row["samples"] for row in role["schedule_geometry"] if all(row["context"]))
            generation_count = sum(row["samples"] for row in role["schedule_geometry"] if not any(row["context"]))
            mixed = role["roi_samples"] - context_count - generation_count
            profiling.append(f"| {shape['path']} | {name} | {role['roi_samples']} | "
                             f"{fmt(min(batch, default=None), 0)}–{fmt(max(batch, default=None), 0)} | "
                             f"{fmt(quantile(queries, .5), 0)} / {fmt(quantile(queries, .95), 0)} | "
                             f"{context_count} / {generation_count} / {mixed} | "
                             f"{fmt(min(positions, default=None), 0)}–{fmt(max(positions, default=None), 0)} | "
                             f"{fmt(shape['observer']['max_abs_change_pct'])} | {shape['evidence']} |")
    if not any(shape["roles"] for shape in shapes):
        profiling.append("| NOT_RUN | — | — | — | — | — | — | — | — |")
    profiling.extend(["", f"[Per-worker logical shape distributions]({artifact_link}/native_scheduler_shapes.json)"])
    accounting = [case for case in clean_online if case["mode"] == "pd" and case["count"] >= 64
                  and not case.get("deflect_at") and not case.get("nixl_cross_layer_blocks_requested")
                  and case["d_accounting_iterations"] and not case["path"].endswith("_clean_after")]
    if accounting:
        profiling.extend([
            "",
            "## D engine token accounting in clean cohorts",
            "",
            "The Prometheus `iteration_tokens_total` histogram accumulates computed prompt and generation tokens in engine output accounting. "
            "The ≤1 bucket can contain zero-token updates. This evidence helps locate small decode geometry and leaves actual CUDA launch "
            "batch size to the worker observers and trace annotations.",
            "",
            "| Cohort | D iterations in histogram | Accounted tokens | Iterations reporting ≤1 token |",
            "| --- | ---: | ---: | ---: |",
        ])
        for case in accounting:
            profiling.append(f"| {case['path']} | {fmt(case['d_accounting_iterations'], 0)} | "
                             f"{fmt(case['d_accounting_tokens'], 0)} | "
                             f"{fmt(case['d_accounting_iterations_le_one'], 0)} |")
    profiling.extend([
        "",
        "## Same-window NVML GPM device samples",
        "",
        "These hardware samples are interval averages over roughly 250 ms. Means are weighted by central-ROI exposure. "
        "They do not resolve individual operator or decode deadlines. Device RX and TX are aggregate rates across NVLinks, "
        "which can account for the same transfer at both endpoints. [NVML GPM metric definitions]"
        "(https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlGpmEnums.html)",
        "",
        "| Cohort | Role | SM active % | SM occupancy % | Tensor active % | DRAM bandwidth % | NVLink RX / TX MiB/s | Peak sampled VRAM GiB | SM sample coverage |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for case in clean_online:
        width = case["central_roi_mono"][1] - case["central_roi_mono"][0]
        for name, device in case["gpm"]["devices"].items():
            metrics = device["metrics"]
            sm, occ, tensor, dram, rx, tx = (
                metrics[field].get("weighted_mean") for field in (
                    "sm_util_pct", "sm_occupancy_pct", "tensor_util_pct", "dram_bw_pct",
                    "nvlink_rx_mib_s", "nvlink_tx_mib_s",
                )
            )
            vram = metrics["vram_B"].get("max_sample")
            profiling.append(f"| {case['path']} | {name} | {fmt(sm)} | {fmt(occ)} | {fmt(tensor)} | {fmt(dram)} | "
                             f"{fmt(rx)} / {fmt(tx)} | {fmt(vram / 1024**3 if vram is not None else None)} | "
                             f"{pct(metrics['sm_util_pct']['seconds'] / width if width else None)} |")
    if not any(case["gpm"]["devices"] for case in clean_online):
        profiling.append("| NOT_RUN | — | — | — | — | — | — | — | — |")
    profiling.extend([
        "",
        "## Counter and geometry scope",
        "",
        "This capture identifies GPU kernel intervals and PyTorch-observed memory-copy intervals. "
        "NIXL copy-engine activity may execute outside the traced worker, so observed memcpy cannot measure all communication. "
        "NVML GPM supplies the device-level activity and bandwidth ratios. Achieved FLOPs, actual HBM read/write bytes, L2 and per-operator "
        "hardware counters remain unmeasured. The vLLM theoretical FLOP and byte counters are excluded. "
        "The iteration annotations identify row counts, while operation-specific GEMM N/K follow the model weight geometry. "
        "The trace does not map each GEMM invocation to its layer/operand or recover every active attention KV-length distribution. "
        "The Prometheus iteration token counter is an engine output-accounting aggregate. "
        "Computed prompt tokens can be credited together on output, so these sums cannot recover chunk-level dispatch shapes. "
        "Stream or kernel overlap is temporal evidence. These traces leave physical SM partitioning unverified.",
        "",
        f"Per-role offsets, interval union, same-ROI occupancy bins, observer shifts, event counts, and leading concrete kernel names are preserved. "
        f"[CUPTI/Kineto reduction data]({artifact_link}/native_profiles.json)",
    ])
    if figures:
        profiling.extend(["", " ".join(
            f"[{name.replace('_', ' ')}]({artifact_link}/figures/{name}.svg)"
            for name in sorted(figures) if name.startswith(("native_tensor", "native_operator", "native_pd_temporal"))
        )])
    report_dir.joinpath("pd_operator_resource_profiling.md").write_text("\n".join(profiling) + "\n")


def render(out, cases, profile_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .25, "svg.hashsalt": "pd-resource-phase0"})
    figures = out / "figures"
    figures.mkdir(exist_ok=True)

    def save(fig, name):
        path = figures / f"{name}.svg"
        fig.savefig(path, bbox_inches="tight", metadata={"Date": None})
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        plt.close(fig)

    clean = [
        case for case in cases if case["mode"] != "prefill" and not case["profiled"]
        and not case["shape_observed"]
        and not case["path"].endswith("_clean_after")
    ]
    if clean:
        groups = defaultdict(list)
        for case in clean:
            groups[(case["model"], case["mode"], case["prefill_workers"], case["decode_workers"], case["p_tp"],
                    tuple(case["inputs"]), tuple(case["output_contract"]), case["apc"], case["chunked"], case["natural"],
                    case.get("deflect_at"), case.get("nixl_cross_layer_blocks_requested"))].append(case)
        for metrics, filename in (
            ([("ttft_ms_p95", "TTFT p95 (ms)")], "native_qps_vs_ttft"),
            ([("tpot_ms_p95", "TPOT p95 (ms)"), ("itl_ms_p99", "ITL p99 (ms)")], "native_qps_vs_decode_tails"),
            ([("goodput", "Joint-SLO goodput (req/s)")], "native_qps_vs_goodput"),
        ):
            fig, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 4.2), squeeze=False)
            for ax, (metric, ylabel) in zip(axes[0], metrics):
                for index, group in enumerate(groups.values()):
                    ordered = sorted((case for case in group if case[metric] is not None), key=lambda case: case["qps"])
                    if ordered:
                        ax.plot([case["qps"] for case in ordered], [case[metric] for case in ordered],
                                color=PLOT_COLORS[index % len(PLOT_COLORS)], marker="o", alpha=.9,
                                linestyle=("-", "--", "-.")[index // len(PLOT_COLORS) % 3],
                                label=label(ordered[0]).split(" · ")[0] + (" APC" if ordered[0]["apc"] else ""))
                ax.set(xlabel="Offered load (req/s)", ylabel=ylabel)
                ax.legend(fontsize=7, frameon=False)
            fig.suptitle("Native serving · finite open-loop cohorts")
            save(fig, filename)

    gpm_rows = [(case, role, data["metrics"]) for case in clean if case["mode"] == "pd"
                for role, data in case["gpm"]["devices"].items()]
    if gpm_rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, max(4.5, .29 * len(gpm_rows))))
        ticks = [f"{label(case)} {role}" for case, role, _ in gpm_rows]
        for ax, metric, title in zip(axes, ("tensor_util_pct", "dram_bw_pct"),
                                    ("Tensor Core activity", "DRAM bandwidth utilization")):
            y, rates, colors = [], [], []
            for index, (_, role, metrics) in enumerate(gpm_rows):
                value = metrics[metric].get("weighted_mean")
                if value is not None:
                    y.append(index)
                    rates.append(value)
                    colors.append(PLOT_COLORS[0 if role.startswith("P") else 1])
            ax.barh(y, rates, color=colors)
            ax.set(xlabel="Interval-weighted device utilization (%)", xlim=(0, 100), title=title,
                   yticks=range(len(ticks)), yticklabels=ticks)
            if ax is axes[1]:
                ax.tick_params(labelleft=False)
            ax.invert_yaxis()
        fig.suptitle("Native PD · NVML GPM central-arrival samples")
        save(fig, "native_tensor_and_dram_gpm")

    operating = [case for case in clean if case["mode"] == "pd"
                 and case["p_reference_ratio_finite_best"] is not None and case["p_waiting_mean"] is not None]
    if operating:
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        points = ax.scatter([case["p_reference_ratio_finite_best"] for case in operating],
                            [case["decode_slo_ok"] / case["count"] for case in operating],
                            c=[case["p_waiting_mean"] for case in operating], s=40, cmap="viridis")
        for case in operating:
            ax.annotate(case["case"], (case["p_reference_ratio_finite_best"], case["decode_slo_ok"] / case["count"]),
                        xytext=(4, -9), textcoords="offset points", fontsize=7)
        fig.colorbar(points, ax=ax, label="Mean sampled P waiting in ROI")
        ax.set(xlabel="Finite input throughput / fastest independent P reference", ylabel="Decode SLO attainment",
               ylim=(-.04, 1.08), title="Observed PD operating points · finite cohort proxy")
        ax.axvline(.85, color="#697783", linestyle="--", linewidth=.8)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        save(fig, "native_pd_operating_region")

    measured = [profile for profile in profile_data if any(
        "kernel_busy_fraction" in role for role in profile["roles"].values())]
    if not measured:
        return
    fig, ax = plt.subplots(figsize=(max(6.0, len(measured) * 2.1), 4))
    ticks = [f"{profile['case']}\n{profile['evidence']}" for profile in measured]
    for offset, role in ((-.18, "P0"), (.18, "D0")):
        for i, profile in enumerate(measured):
            value = profile["roles"].get(role, {}).get("kernel_busy_fraction")
            if value is not None:
                ax.bar(i + offset, value, width=.32, color=PLOT_COLORS[0 if role == "P0" else 1],
                       label=role if i == 0 else None, hatch="//" if profile["evidence"] == "DIAGNOSTIC" else None)
    ax.set(xticks=range(len(ticks)), xticklabels=ticks, ylabel="Kernel interval union / ROI", ylim=(0, 1.05))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False)
    save(fig, "native_pd_kernel_busy")

    for i, profile in enumerate(measured):
        fig, ax = plt.subplots(figsize=(7.5, 3.5))
        for j, (name, role) in enumerate(profile["roles"].items()):
            if "kernel_busy_bins" in role:
                bins = role["kernel_busy_bins"]
                width = profile["central_roi_mono"][1] - profile["central_roi_mono"][0]
                ax.step([width * k / len(bins) for k in range(len(bins) + 1)], bins + [bins[-1]],
                        where="post", label=name, color=PLOT_COLORS[j % len(PLOT_COLORS)], alpha=.85)
        ax.set(xlabel="Time from central window start (s)", ylabel="Kernel interval union per bin",
               ylim=(0, 1.05), title=f"{profile['case']} · {profile['evidence']}")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(frameon=False)
        save(fig, f"native_pd_temporal_kernel_timeline_{i}")

    breakdown = [(profile, name, role) for profile in measured for name, role in profile["roles"].items()
                 if role.get("family_union_ms")]
    fig, axes = plt.subplots(len(breakdown), 1, figsize=(8, 2.3 * len(breakdown)), squeeze=False)
    for ax, (profile, name, role) in zip(axes[:, 0], breakdown):
        families = sorted(role["family_union_ms"], key=role["family_union_ms"].get)
        ax.barh(families, [role["family_union_ms"][family] for family in families], color=PLOT_COLORS[0])
        ax.set(xlabel="Per-family wall union (ms, overlapping classes)", title=f"{profile['case']} {name} · {profile['evidence']}")
    fig.tight_layout()
    save(fig, "native_operator_time_breakdown")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--group", action="append", help="Limit inputs to this directory under raw-root")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reports", type=Path)
    parser.add_argument("--artifact-link", default="../pd_resource_complementary_parallelism/results/native")
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    cases, incomplete = [], []
    sources = [args.raw_root / group for group in args.group] if args.group else [args.raw_root]
    directories = {path.parent for source in sources for pattern in ("**/requests.jsonl", "**/before_*.prom")
                   for path in source.glob(pattern)}
    for directory in sorted(directories):
        path = directory / "requests.jsonl"
        run_path = directory / "run.json"
        try:
            cases.append(summarize(args.raw_root, run_path))
        except (KeyError, ValueError, OSError) as error:
            incomplete.append({"path": str(directory.relative_to(args.raw_root)),
                               "evidence": "NOT_RUN", "error": str(error),
                               "requests": receipt(path) if path.is_file() else None})
    for source in sources:
        for status_path in sorted(source.glob("**/exit_codes.txt")):
            for line in status_path.read_text().splitlines():
                fields = line.split()
                if len(fields) == 2 and fields[1].isdigit() and int(fields[1]) != 0:
                    label, status = fields
                    failed_start = status_path.parent / label
                    archived_attempt = failed_start.with_name(label + "_failed")
                    if archived_attempt.is_dir():
                        failed_start = archived_attempt
                    incomplete.append({"path": str(failed_start.relative_to(args.raw_root)),
                                       "attempt_label": label, "evidence": "NOT_RUN", "process_exit": int(status),
                                       "status_receipt": receipt(status_path)})
    profile_data = profiles(args.raw_root, cases)
    shapes = scheduler_shapes(args.raw_root, cases)
    add_motivation(cases, profile_data)
    contrasts = compare_native(cases)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = [{key: value for key, value in case.items() if key != "token_sequences"}
               for case in cases]
    (args.out / "native_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "native_profiles.json").write_text(json.dumps(profile_data, indent=2) + "\n")
    (args.out / "native_scheduler_shapes.json").write_text(json.dumps(shapes, indent=2) + "\n")
    (args.out / "native_incomplete.json").write_text(json.dumps(incomplete, indent=2) + "\n")
    write_csv(args.out / "native_cohorts.csv", [
        {key: value for key, value in case.items() if not isinstance(value, (dict, list))}
        | {"inputs": "|".join(map(str, case["inputs"])), "output": "|".join(map(str, case["output_contract"]))}
        for case in summary
    ])
    write_csv(args.out / "native_operator_families.csv", [
        {"path": profile["path"], "evidence": profile["evidence"], "role": name, "family": family,
         "union_ms": latency, "event_sum_ms": role["family_event_sum_ms"][family]}
        for profile in profile_data for name, role in profile["roles"].items()
        for family, latency in role.get("family_union_ms", {}).items()
    ])
    write_csv(args.out / "native_iteration_geometry.csv", [
        {"path": profile["path"], "evidence": profile["evidence"], "role": name, **row}
        for profile in profile_data for name, role in profile["roles"].items()
        for row in role.get("iteration_geometry", [])
    ])
    write_csv(args.out / "native_gpm.csv", [
        {"path": case["path"], "role": role, "uuid": device["uuid"], "metric": metric,
         **{key: value for key, value in stats.items() if key != "status_counts"}}
        for case in cases for role, device in case["gpm"]["devices"].items()
        for metric, stats in device["metrics"].items()
    ])
    write_csv(args.out / "native_matched_controls.csv", contrasts)
    if args.figures:
        render(args.out, cases, profile_data)
    if args.reports:
        figures = [path.stem for path in (args.out / "figures").glob("*.svg")] if args.figures else []
        save_reports(args.reports, cases, profile_data, shapes, incomplete, contrasts, args.artifact_link, figures)
    print(f"Reduced {len(cases)} native cohorts and {len(profile_data)} profile arms.")


if __name__ == "__main__":
    main()
