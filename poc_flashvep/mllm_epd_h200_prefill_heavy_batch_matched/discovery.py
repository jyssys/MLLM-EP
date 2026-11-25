"""Evaluate the preregistered Main100 P-heavy/D-idle gate."""
import argparse
import json
from pathlib import Path

from .trace import complement, elapsed, intersect, merged

P_BUSY = .80
P_PROXIMITY = .85
DECODE_SLO = .95
IDLE_WINDOW = .160
IDLE_SENSITIVE = .225
P_BUSY_IDLE_COVERAGE = .10


def load(path):
    return json.loads(path.read_text())


def evaluate(trace_path, capacity_path):
    trace = load(trace_path)
    tag = trace_path.name.removesuffix("_trace.json")
    profile_path = trace_path.with_name(tag + "_profile_metrics.json")
    metrics = load(profile_path)
    run = trace_path.parent.parent
    p_capacity = load(capacity_path) if capacity_path.is_file() else None
    l, r = trace["window_monotonic"]
    p_busy = trace["kernel_intervals"]["prefill"]
    d_busy = trace["kernel_intervals"]["decode"]
    trace_ids = {json.loads(line)["runtime_request_id"]
                 for line in (run / "raw" / (tag + "_profile_requests.jsonl")).read_text().splitlines()}
    raw = [json.loads(line) for line in (run / "raw/proxy_epd.jsonl").read_text().splitlines()]
    raw = [row for row in raw if row["runtime_request_id"] in trace_ids and row.get("status") == "MEASURED_ONLINE"]
    d_work = [(row["P_ready"], row.get("D_http_complete", row["finished"])) for row in raw]
    d_quiet = complement(merged(d_busy + d_work, l, r), l, r)
    p_busy_seconds = elapsed(p_busy)

    def coverage(threshold):
        eligible = [(a, b) for a, b in d_quiet if b - a >= threshold]
        return dict(windows=len(eligible), p_busy_overlap_fraction=elapsed(intersect(p_busy, eligible))
                    / p_busy_seconds if p_busy_seconds else 0)

    window = coverage(IDLE_WINDOW)
    d_good = 0
    for line in (run / "raw" / (tag + "_profile_requests.jsonl")).read_text().splitlines():
        row = json.loads(line)
        d_good += row["status"] == "MEASURED_ONLINE" and row["TPOT_ms"] <= 33 and all(
            value <= 34 for value in row["ITL_ms"])
    online = trace["online_completed_stage_rates"]["prefill"]
    proximity = {field: online[field] / p_capacity[field] if p_capacity else None
                 for field in ("requests_per_second", "tokens_per_second")}
    p_schedule = trace["scheduler_samples"]["prefill"]
    d_schedule = trace["scheduler_samples"]["decode"]
    d_capacity = d_schedule["free_KV_blocks_min"]
    checks = {
        "trace_qualified": trace["evidence"] == "TRACE_ANALYSIS",
        "p_kernel_busy": trace["kernel_busy_fraction"]["prefill"] >= P_BUSY,
        "p_service_proximity": all(value is not None and value >= P_PROXIMITY for value in proximity.values()),
        "p_scheduler_backlog": p_schedule["observations"] >= 10
            and p_schedule["window_fraction_recent_sample_waiting"] >= .10,
        "p_not_starved": trace["P_kernel_idle_no_P_HTTP_fraction"] <= .10,
        "d_quiet_window_coverage": window["windows"] >= 2
            and window["p_busy_overlap_fraction"] >= P_BUSY_IDLE_COVERAGE,
        "d_kv_capacity": d_capacity is not None and d_capacity * 128 >= 32768,
        "absolute_decode_slo": d_good / metrics["planned_count"] >= DECODE_SLO,
    }
    if all(checks.values()):
        status = "P_HEAVY_D_IDLE_CONFIRMED"
    elif not checks["trace_qualified"] or p_capacity is None:
        status = "P_HEAVY_D_IDLE_NOT_MEASURED"
    else:
        status = "P_HEAVY_D_IDLE_NOT_CONFIRMED"
    return dict(status=status, checks=checks, p_busy=trace["kernel_busy_fraction"]["prefill"],
                p_service_proximity=proximity, d_decode_attainment=d_good / metrics["planned_count"],
                robust_window_160_ms=window, robust_window_225_ms=coverage(IDLE_SENSITIVE),
                p_scheduler=p_schedule, d_scheduler=d_schedule,
                scope="Observed kernel and conservative HTTP quiet conjunction. No assertion of future Suffix safety.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--capacity", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.trace, args.capacity), indent=2))


if __name__ == "__main__":
    main()
