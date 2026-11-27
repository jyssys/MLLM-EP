"""Measure prior EPD admission windows using archived client and host traces."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def receipt(path):
    with path.open("rb") as stream:
        return {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def jsonl(path):
    with path.open() as stream:
        yield from map(json.loads, stream)


def percentiles(data):
    values = sorted(data)
    result = {}
    for q in (.5, .95, 1.):
        pos = (len(values) - 1) * q
        left, right = math.floor(pos), math.ceil(pos)
        result[f"p{int(q * 100)}"] = values[left] + (pos - left) * (values[right] - values[left])
    return result


def simultaneous(windows):
    events = []
    for ready, finish, nbytes in windows:
        if ready < finish:
            events.extend(((ready, nbytes), (finish, -nbytes)))
    nrequests = total_bytes = peak_bytes = peak_requests = 0
    for _, delta in sorted(events):
        nrequests += 1 if delta > 0 else -1
        total_bytes += delta
        peak_requests = max(peak_requests, nrequests)
        peak_bytes = max(peak_bytes, total_bytes)
    return {"peak_requests": peak_requests, "peak_token_geometry_bytes": peak_bytes}


def analyze(run):
    proxy = run / "raw" / "proxy_epd.jsonl"
    records = {}
    for row in jsonl(proxy):
        key = row["runtime_request_id"]
        if key in records:
            raise ValueError(f"Duplicate host request ID {key}")
        records[key] = row
    arms = {}
    for path in sorted((run / "raw").glob("native_strong*_requests.jsonl")):
        rows = list(jsonl(path))
        starts, first_sse, pending, cap_binding = [], [], [], []
        for client in rows:
            host = records[client["runtime_request_id"]]
            start = host["P_ready"]
            if not start <= host["D_submit"] <= host["D_first_SSE_bytes"]:
                raise ValueError("Historical host stages are out of order")
            geometry_bytes = math.ceil(client["input_tokens_manifest"] / 128) * 128 * 57344
            starts.append((start, host["D_submit"], geometry_bytes))
            first_sse.append((start, host["D_first_SSE_bytes"], geometry_bytes))
            pending.append(host["D_at_P_ready"]["pending_admissions"])
            cap_binding.append(host["D_at_P_ready"]["cap_binding"])
        arms[path.name.removesuffix("_requests.jsonl")] = {
            "completed": sum(r["status"] == "MEASURED_ONLINE" for r in rows), "offered": len(rows),
            "D_pending_at_P_ready": sum(x > 0 for x in pending), "D_saturated_at_P_ready": sum(cap_binding),
            "P_ready_D_submit_ms": percentiles([(hi - lo) * 1e3 for lo, hi, _ in starts]),
            "P_ready_D_first_SSE_bytes_ms": percentiles([(hi - lo) * 1e3 for lo, hi, _ in first_sse]),
            "P_ready_D_submit_geometry": simultaneous(starts),
            "P_ready_D_first_SSE_geometry": simultaneous(first_sse), "client_raw": receipt(path),
        }
    snapshots = {}
    for path in sorted((run / "logs").glob("native-strong_*.log")):
        samples = [(int(n), int(w), float(kv)) for n, w, kv in re.findall(
            r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%", path.read_text())]
        if samples:
            snapshots[path.stem] = {"samples": len(samples), "max_sampled_running": max(s[0] for s in samples),
                                    "max_sampled_waiting": max(s[1] for s in samples),
                                    "max_sampled_kv_usage_pct": max(s[2] for s in samples), "raw": receipt(path)}
    return {"evidence": "DIAGNOSTIC", "host_trace_raw": receipt(proxy), "arms": arms,
            "sampled_worker_logs_all_loads": snapshots, "preemptions": "NOT_MEASURED_WITH_PROMETHEUS_COUNTER",
            "scope": "Historical NativeStrong EADA control, P ready and D submit are host boundaries. "
                     "First SSE includes transfer and service. Neither boundary is a P-worker KV-release ACK. "
                     "Concurrent bytes use 128-token rounding and Qwen2.5-VL 56 KiB per token. "
                     "Worker log storage and queue fractions are sparse samples pooled across all loads."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run)
    with args.out.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
