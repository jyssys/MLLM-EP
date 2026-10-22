"""Explicit request and population SLO denominators for measured traces."""

import numpy as np


def percentile(values, p):
    return float(np.percentile(values, p)) if values else None


def compliance(request, ttft, tbt, require_natural_stop=False):
    times = request.get("token_times", [])
    intervals = request.get("ITLs_s", [])
    valid = request.get("status") == "COMPLETE" and bool(times)
    ttft_pass = (
        bool(times)
        and request.get("TTFT_s", float("inf")) <= ttft
        and request.get("status") != "ERROR"
    )
    tbt_pass = valid and max(intervals, default=0) <= tbt
    stopped = request.get("finish_reason") in ("stop", "frontend_stop")
    nonempty = len(request.get("output_ids", [])) > 1
    complete_response = valid and (not require_natural_stop or (stopped and nonempty))
    return dict(
        TTFT_pass=ttft_pass,
        TBT_observed_all_token_pass=tbt_pass,
        observed_joint_pass=bool(ttft_pass and tbt_pass),
        natural_response_complete=complete_response,
        joint_pass=bool(complete_response and ttft_pass and tbt_pass),
    )


def summarize(data, ttft, tbt, require_natural_stop=False):
    requests = data["requests"]
    total = len(requests)
    assert total > 0
    outcomes = [compliance(r, ttft, tbt, require_natural_stop) for r in requests]
    start, window = data["start"], data["arrival_window_s"]
    end_window = start + window
    all_intervals = [v for r in requests for v in r.get("ITLs_s", [])]
    ttfts = [r["TTFT_s"] for r in requests if "TTFT_s" in r]
    e2es = [r["E2E_s"] for r in requests]
    pass_count = sum(v["joint_pass"] for v in outcomes)
    complete_in_window = sum(
        r.get("status") == "COMPLETE" and r["completion"] <= end_window
        for r in requests
    )
    unfinished_at_window = sum(r["completion"] > end_window for r in requests)
    errors = sum(r["status"] == "ERROR" for r in requests)
    timeouts = sum(r["status"] == "TIMEOUT" for r in requests)
    queued = []
    for sample in data["queue_series"]:
        tick = start + sample["time"]
        if window * 0.25 <= sample["time"] <= window * 0.95:
            count = sum(
                r["scheduled_arrival"] <= tick
                and (
                    r.get("first_GPU_submission") is None
                    or r["first_GPU_submission"] > tick
                )
                and r["completion"] > tick
                for r in requests
            )
            queued.append((sample["time"], count))
    slope = (
        float(np.polyfit([v[0] for v in queued], [v[1] for v in queued], 1)[0])
        if len(queued) >= 3
        else None
    )
    stable = (
        slope is not None
        and slope <= 0.05
        and unfinished_at_window / total <= 0.10
        and errors == 0
        and timeouts == 0
    )
    duration = data["end"] - min(r["scheduled_arrival"] for r in requests)
    result = dict(
        offered=total,
        realized_offered_rate_per_s=total / window,
        completed=sum(r["status"] == "COMPLETE" for r in requests),
        errors=errors,
        timeouts=timeouts,
        completed_in_arrival_window=complete_in_window,
        drained_after_window=sum(
            r["status"] == "COMPLETE" and r["completion"] > end_window for r in requests
        ),
        unfinished_at_arrival_window_end=unfinished_at_window,
        arrival_window_s=window,
        drained_cohort_s=duration,
        TTFT_limit_s=ttft,
        TBT_limit_s=tbt,
        TTFT_pass=sum(v["TTFT_pass"] for v in outcomes),
        TBT_all_token_pass=sum(v["TBT_observed_all_token_pass"] for v in outcomes),
        observed_joint_pass=sum(v["observed_joint_pass"] for v in outcomes),
        joint_pass=pass_count,
        natural_complete=sum(v["natural_response_complete"] for v in outcomes),
        joint_attainment=pass_count / total,
        TTFT_attainment=sum(v["TTFT_pass"] for v in outcomes) / total,
        TBT_attainment=sum(v["TBT_observed_all_token_pass"] for v in outcomes) / total,
        goodput_offered_window_with_declared_drain=pass_count / window,
        goodput_completed_within_arrival_window=sum(
            v["joint_pass"] and r["completion"] <= end_window
            for r, v in zip(requests, outcomes)
        )
        / window,
        drained_requests_s=sum(r["status"] == "COMPLETE" for r in requests) / duration,
        drained_output_tokens_s=sum(len(r.get("output_ids", [])) for r in requests)
        / duration,
        pooled_ITL_p50_s=percentile(all_intervals, 50),
        pooled_ITL_p95_s=percentile(all_intervals, 95),
        pooled_ITL_p99_s=percentile(all_intervals, 99),
        pooled_p99_TBT_pass=percentile(all_intervals, 99) <= tbt
        if all_intervals
        else None,
        per_request_max_ITL_p95_s=percentile(
            [max(r.get("ITLs_s", []), default=0) for r in requests], 95
        ),
        TTFT_p50_s=percentile(ttfts, 50),
        TTFT_p95_s=percentile(ttfts, 95),
        TTFT_p99_s=percentile(ttfts, 99),
        E2E_p50_s=percentile(e2es, 50),
        E2E_p95_s=percentile(e2es, 95),
        queue_slope_requests_s=slope,
        stable_under_preregistered_finite_window_rule=stable,
        output_quality_scope="NATURAL_STOP_REQUIRED"
        if require_natural_stop
        else "BOUNDED_RESPONSE_COMPLETION",
        sample_adequacy="EXPLORATORY"
        if total < 1000
        else "CHECK_REPLICATION_AND_STABILITY",
    )
    for threshold in (50, 100, 150, 200, 250):
        result[f"ITLs_over_{threshold}ms"] = sum(
            v > threshold / 1000 for v in all_intervals
        )
        result[f"ITL_fraction_over_{threshold}ms"] = (
            sum(v > threshold / 1000 for v in all_intervals) / len(all_intervals)
            if all_intervals
            else None
        )
    return result
