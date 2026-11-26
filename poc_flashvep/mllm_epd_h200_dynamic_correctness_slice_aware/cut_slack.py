"""Cut-aligned coordinator evidence and counterfactual idle on the same monotonic clock."""
import argparse
from bisect import bisect_right
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import median


RATIOS = (0, .01, .025, .05, .10, .20, .30)
TOKEN_TARGETS = (64, 128, 256, 512, 1024, 2048, 4096)
THRESHOLDS_MS = (32, 64, 100, 128, 160, 225, 300, 500)
CANDIDATE_BANDS = (
    (0.001, .0175), (.0175, .0375), (.0375, .075), (.075, .15), (.15, .25), (.25, .35),
)


def eligible_geometry(length, remaining):
    return (0 < remaining < length and (
        64 <= remaining <= min(4096, length * .35) or
        any(low <= remaining / length <= high for low, high in CANDIDATE_BANDS)
    ))


def read_jsonl(path):
    with path.open() as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def joined_cohort(run, cohort):
    client = list(read_jsonl(run / "raw" / f"{cohort}_completion_journal.jsonl"))
    by_id = {row["runtime_request_id"]: row for row in client}
    if len(by_id) != len(client):
        raise ValueError("Duplicate client request ID")
    rows = [row for row in read_jsonl(run / "raw" / "proxy_epd.jsonl")
            if row["runtime_request_id"] in by_id]
    if len(rows) != len(by_id) or len({row["runtime_request_id"] for row in rows}) != len(rows):
        raise ValueError("Incomplete or ambiguous coordinator trace")
    successful = [row for row in rows if row["status"] == "MEASURED_ONLINE"]
    if len(successful) != len(rows):
        raise ValueError("A failed request can censor D occupancy. Inspect failed cohort separately.")
    return by_id, successful


def merge_intervals(intervals):
    merged = []
    for left, right in sorted(intervals):
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return merged


def _gap(intervals, starts, at, right):
    slot = bisect_right(starts, at)
    if slot > 0 and at < intervals[slot - 1][1]:
        return 0., False
    if slot == len(intervals) or starts[slot] >= right:
        return max(0., 1000 * (right - at)), True
    return 1000 * (starts[slot] - at), False


def _request_waiters(rows, request_id, at):
    # All three timestamps denote transitions that have already happened at `at`;
    # the condition reconstructs the in-memory P request map, without later durations.
    return sum(row["runtime_request_id"] != request_id and row["P_submit"] <= at < row["P_ready"]
               for row in rows)


def shadow_cuts(rows):
    for row in rows:
        for event in row.get("late_plan", {}).get("decisions", ()):
            p = event["P_boundary_state"]
            state = event["D_before"]
            engine = event.get("D_engine")
            recovered = event.get("policy") in {"b-new", "c-new-hard", "c-new-soft"}
            split = row.get("token_partition", {}).get("prefix_tokens") == event["prefix_tokens"]
            yield {
                "request_id": row["runtime_request_id"], "prefix": event["prefix_tokens"],
                "input_tokens": event["full_prompt_tokens"], "time": p["boundary_ready"],
                "callback_time": event["callback_begin"], "later_tokens_inflight": 0,
                "hold_attempts": p.get("held_schedule_attempts"), "first_held_at": p.get("first_held_at"),
                "held_before_retirement_ms": p.get("held_before_retirement_ms"),
                "P_retirement_to_callback_end_ms": 1000 * (event["callback_end"] - p["boundary_ready"]),
                "competitors": p.get("competitors"), "prefill_running": p["prefill_running"],
                "prefill_waiting": p["prefill_waiting"], "batch_tokens": p.get("scheduled_chunk_tokens"),
                "progress": p.get("progress"),
                "batch_size": None, "issue_to_result_ms": None, "D_active_callback": state["active_http_requests"],
                "D_pending_callback": state["admission_waiting"], "D_reservation_callback": state["prefill_owner"],
                "D_engine_at_callback": engine, "D_KV_free_callback": engine["KV_free_blocks"] if engine else None,
                "original_reason": event["reason"],
                "old_budget_ms": (None if recovered else event.get("budget_ms")),
                "old_cost_ms": (None if recovered else event.get("cost_prediction_ms")),
                "new_budget_ms": (event.get("budget_ms") if recovered else None),
                "new_cost_ms": (event.get("cost_prediction_ms") if recovered else None),
                "new_expected_gain_ms": event.get("expected_gain_ms"),
                "transport_ready_at_cut": event.get("transport_ready_at_cut"),
                "transport_required_suffix_images": event.get("transport_required_suffix_images"),
                "transport_missing_inline_images": event.get("transport_missing_inline_images"),
                "actual_boundary_to_D_first_token_ms": (
                    1000 * (row["D_first_generated_token"] - p["boundary_ready"]) if split else None),
                "callback": True,
                "stock_observed": False, "reserved": bool(event.get("reservation")),
                "D_reserved_at": (event.get("reservation") or {}).get("reserved"),
                "hypothetical_reserve": bool(event.get("shadow_would_reserve")),
                "actual_offload": split,
            }


def stock_cuts(paths, by_id):
    dropped = 0
    result = []
    for path in paths:
        for event in read_jsonl(path):
            if "dropped_steps" in event:
                dropped += event["dropped_steps"]
                continue
            for row in event["rows"]:
                request_id, length, prefix = row["request_id"], row["prompt_tokens"], row["computed_through"]
                if request_id not in by_id or not row["valid_retirement"] or not 0 < prefix < length:
                    continue
                result.append({
                    "request_id": request_id, "prefix": prefix, "input_tokens": length,
                    "time": event["worker_result_at"], "later_tokens_inflight": row["later_tokens_inflight"],
                    "prefill_running": event["running"], "prefill_waiting": event["waiting"],
                    "batch_tokens": event["batch_tokens"], "batch_size": event["batch_size"],
                    "issue_to_result_ms": event["issue_to_result_ms"], "competitors": row.get("competitors"),
                    "progress": {"competitors": row.get("competitors"), "running": event["running"],
                                 "waiting": event["waiting"], "ready_competitors": row.get("ready_competitors"),
                                 "batch_size": event["batch_size"], "batch_tokens": event["batch_tokens"]},
                    "callback": False, "stock_observed": True,
                })
    if dropped:
        raise ValueError(f"Stock trace incomplete ({dropped} dropped steps)")
    return result


def decode_queue(run):
    paths = list((run / "raw").glob("stock_decode_queue_*.jsonl"))
    if not paths:
        return []
    if len(paths) != 1:
        raise ValueError("Multiple D queue captures in a single-vLLM trace")
    rows = list(read_jsonl(paths[0]))
    if not rows or any("dropped_steps" in row for row in rows) or rows[-1]["source"] not in ("checkpoint", "end"):
        raise ValueError("D queue observer did not flush a complete trace")
    return rows


def annotate(rows, client, seeds, d_states, physical, decode_ec):
    scheduled = sorted(row["scheduled_arrival"] for row in client.values())
    left, right = scheduled[len(scheduled) // 5], scheduled[4 * len(scheduled) // 5]
    by_id = {row["runtime_request_id"]: row for row in rows}
    intervals = [(row["P_ready"], row["D_http_complete"], row["runtime_request_id"]) for row in rows]
    if any(b <= a for a, b, _ in intervals):
        raise ValueError("Coordinator stage boundaries are not monotonic")
    merged = merge_intervals((a, b) for a, b, _ in intervals)
    merged_starts = [pair[0] for pair in merged]
    last_decode = max(end for _, end, _ in intervals)
    ready_times = sorted((row["P_ready"], row["runtime_request_id"]) for row in rows)
    engine_times = [row["time"] for row in d_states]
    engine_busy = merge_intervals(
        (a["time"], b["time"]) for a, b in zip(d_states, d_states[1:])
        if a["running"] or a["waiting"] or a["skipped_waiting"])
    engine_busy_starts = [pair[0] for pair in engine_busy]
    if physical is not None:
        d_kernels, coverage = physical
        kernel_and_http = merge_intervals([*d_kernels, *merged, *engine_busy])
        kernel_and_http_starts = [pair[0] for pair in kernel_and_http]
    for seed in seeds:
        request_id = seed["request_id"]
        at = seed["time"]
        prefix, length = seed["prefix"], seed["input_tokens"]
        source = by_id[request_id]
        suffix = length - prefix
        # Keep the source's native completion in the factual series. Excluding
        # it estimates time to the next *other* P-ready request if the source
        # were diverted. This counterfactual still holds the rest of P fixed.
        horizon = right if left <= at < right else last_decode
        other = merge_intervals((a, b) for a, b, rid in intervals if rid != request_id)
        factual_gap, factual_censor = _gap(merged, merged_starts, at, horizon)
        other_gap, other_censor = _gap(other, [pair[0] for pair in other], at, horizon)
        next_ready = next((stamp for stamp, rid in ready_times if at <= stamp < horizon and rid != request_id), None)
        judge = client[request_id]
        failures = judge["ITL_ms"] or ()
        paces = judge["TPOT_ms"]
        d_state_index = bisect_right(engine_times, at) - 1
        d_state = d_states[d_state_index] if 0 <= d_state_index < len(d_states) - 1 else None
        if d_state is not None:
            engine_gap, engine_gap_censor = _gap(engine_busy, engine_busy_starts, at, min(horizon, engine_times[-1]))
        else:
            engine_gap, engine_gap_censor = None, None
        if physical is not None and coverage[0] <= at < coverage[1]:
            quiet, quiet_censor = _gap(kernel_and_http, kernel_and_http_starts, at, min(horizon, coverage[1]))
        else:
            quiet, quiet_censor = None, None
        reservation_start = seed.get("D_reserved_at") or source["D_submit"]
        affected_native = [row for row in rows if row["runtime_request_id"] != request_id
                           and not row.get("token_partition")
                           and reservation_start <= row["P_ready"] < source["D_first_generated_token"]
                           ] if seed.get("actual_offload") else None
        active_ids = [row["runtime_request_id"] for row in rows if row["D_submit"] <= at < row["D_http_complete"]]
        token_observed = [client[rid].get("token_times") for rid in active_ids]
        if all(stamps is not None for stamps in token_observed):
            next_tokens = [next((stamp for stamp in stamps if stamp > at), None) for stamps in token_observed]
            next_tokens = [stamp for stamp in next_tokens if stamp is not None]
        else:
            next_tokens = None
        ec_events = []
        if seed.get("actual_offload"):
            image_keys = set(source["late_plan"]["image_hashes"])
            ec_events = [event for event in decode_ec if source["D_submit"] <= event["begin"]
                         and event["end"] <= source["D_first_generated_token"]
                         and set(event["keys"]) <= image_keys]
        yield seed | {
            "dataset_request_id": judge["request_id"],
            "central_ROI": left <= at < right, "drain": at > scheduled[-1],
            "suffix": suffix, "ratio": suffix / length,
            "candidate_geometry": eligible_geometry(length, suffix),
            "usable_boundary_without_hold": seed["later_tokens_inflight"] == 0,
            "P_http_competitors": _request_waiters(rows, request_id, at),
            "D_active_http_at_cut": len(active_ids),
            "active_D_streams_with_later_client_output": len(next_tokens) if next_tokens is not None else None,
            "next_active_D_client_token_ms": 1000 * (min(next_tokens) - at) if next_tokens else None,
            "D_pending_from_P_ready_at_cut": sum(row["P_ready"] <= at < row["D_submit"] for row in rows),
            "D_engine_scheduler_at_cut": d_state,
            "D_engine_KV_free_blocks_at_cut": d_state["KV_free_blocks"] if d_state is not None else None,
            "D_engine_queue_quiet_gap_ms": engine_gap,
            "D_engine_queue_quiet_gap_right_censored": engine_gap_censor,
            "D_cupti_kernel_and_operational_gap_ms": quiet,
            "D_cupti_kernel_and_operational_gap_right_censored": quiet_censor,
            "observed_D_HTTP_gap_ms": factual_gap, "observed_D_HTTP_gap_right_censored": factual_censor,
            "oracle_other_D_HTTP_gap_ms": other_gap, "oracle_other_D_HTTP_gap_right_censored": other_censor,
            "P_next_other_ready_after_cut_ms": 1000 * (next_ready - at) if next_ready is not None else None,
            "P_first_handoff_after_cut_ms": 1000 * (source["P_ready"] - at),
            "P_own_native_ready_after_cut_ms": (1000 * (source["P_ready"] - at)
                                                if not seed.get("actual_offload") else None),
            "source_decode_SLO_pass": (all(gap <= 34. for gap in failures) and (paces is None or paces <= 33.)),
            "D_submit_to_first_token_ms": (1000 * (source["D_first_generated_token"] - source["D_submit"])
                                           if seed.get("actual_offload") else None),
            "D_EC_export_RPC_ms": sum(event["export_RPC_ms"] for event in ec_events) if ec_events else None,
            "D_EC_clone_enclosure_ms": (sum(event["import_clone_sync_host_ms"] for event in ec_events)
                                         if ec_events else None),
            "D_EC_inline_images": sum(event["inline_images"] for event in ec_events) if ec_events else None,
            "D_EC_total_images": sum(event["images"] for event in ec_events) if ec_events else None,
            "D_EC_spare_images": (sum(event.get("preissued_crossing_images", 0) for event in ec_events)
                                  if ec_events else None),
            "Native_ready_during_actual_suffix_reservation": [
                {"request_id": row["runtime_request_id"], "P_ready": row["P_ready"],
                 "D_coordinator_wait_ms": 1000 * (row["D_submit"] - row["P_ready"]),
                 "during_D_HTTP": row["P_ready"] >= source["D_submit"],
                 "D_admission_causes": row.get("D_admission", {}).get("blocked_on_enqueue")}
                for row in affected_native] if affected_native is not None else None,
            "future_evidence": "OFFLINE_ORACLE",
        }


def funnel(rows, request_count):
    result = {"requests": request_count, "cuts": len(rows),
              "cuts_with_positive_future_HTTP_slack": sum(row["observed_D_HTTP_gap_ms"] > 0 for row in rows)}
    stages = {
        "internal_chunk": rows,
        "candidate_geometry": [row for row in rows if row["candidate_geometry"]],
        "candidate_and_no_later_inflight": [row for row in rows if row["candidate_geometry"]
                                            and row["usable_boundary_without_hold"]],
        "callback": [row for row in rows if row["callback"]],
        "callback_and_D_idle": [row for row in rows if row["callback"] and row["D_active_callback"] == 0
                                and row["D_pending_callback"] == 0 and row["D_reservation_callback"] is None],
        "transport_readiness_checked": [row for row in rows if row.get("transport_ready_at_cut") is not None],
        "known_inline_transport_ready": [row for row in rows if row.get("transport_ready_at_cut") is True],
        "sampled_D_KV": [row for row in rows if row.get("D_KV_free_callback") is not None],
        "sampled_KV_at_least_32768_tokens": [row for row in rows if (row.get("D_KV_free_callback") or 0) >= 256],
        "traced_D_KV_at_least_32768_tokens": [
            row for row in rows if (row.get("D_engine_KV_free_blocks_at_cut") or 0) >= 256],
        "old_supported_cost": [row for row in rows if row.get("old_cost_ms") is not None],
        "old_predicted_fit": [row for row in rows if row.get("old_cost_ms") is not None
                              and row["old_cost_ms"] + 10 <= row["old_budget_ms"]],
        "new_absolute_fit": [row for row in rows if row.get("new_cost_ms") is not None
                             and row["new_cost_ms"] + 10 <= row["new_budget_ms"]],
        "new_absolute_fit_and_positive_gain": [row for row in rows
             if row.get("new_cost_ms") is not None and row["new_cost_ms"] + 10 <= row["new_budget_ms"]
             and row["new_expected_gain_ms"] > 0],
        "shadow_would_reserve": [row for row in rows if row.get("hypothetical_reserve")],
        "reservation": [row for row in rows if row.get("reserved")],
        "actual_offload": [row for row in rows if row.get("actual_offload")],
        "actual_offload_and_decode_SLO": [row for row in rows if row.get("actual_offload")
                                          and row["source_decode_SLO_pass"]],
    }
    result["stages"] = {key: {"cuts": len(value), "requests": len({row["request_id"] for row in value})}
                        for key, value in stages.items()}
    eligible = [row for row in rows if row["candidate_geometry"] and row["D_active_http_at_cut"] == 0
                and row["D_pending_from_P_ready_at_cut"] == 0]
    result["geometry_and_operationally_idle"] = len(eligible)
    result["geometry_without_hold_and_operationally_idle"] = sum(
        row["usable_boundary_without_hold"] for row in eligible)
    result["geometry_engine_and_HTTP_idle"] = sum(
        row["D_engine_queue_quiet_gap_ms"] is not None and row["D_engine_queue_quiet_gap_ms"] > 0
        for row in eligible)
    result["engine_state_available_at_cut"] = sum(row["D_engine_scheduler_at_cut"] is not None for row in rows)
    result["cut_aligned_thresholds"] = {
        threshold: {key: sum((row[key] or 0.) >= threshold for row in eligible) for key in (
            "observed_D_HTTP_gap_ms", "oracle_other_D_HTTP_gap_ms", "P_own_native_ready_after_cut_ms")}
        for threshold in THRESHOLDS_MS
    }
    result["D_cupti_coverage_cuts"] = sum(row["D_cupti_kernel_and_operational_gap_ms"] is not None
                                         for row in eligible)
    for threshold, threshold_counts in result["cut_aligned_thresholds"].items():
        threshold_counts["D_cupti_kernel_and_operational_gap_ms"] = sum(
            (row["D_cupti_kernel_and_operational_gap_ms"] or 0.) >= threshold for row in eligible)
    result["absolute_suffix_targets_in_actual_chunks"] = dict(
        Counter(row["suffix"] for row in rows if row["suffix"] in TOKEN_TARGETS))
    result["ratio_bands_in_actual_chunks"] = {
        str(ratio): sum(low <= row["ratio"] <= high for row in rows)
        for ratio, (low, high) in zip(RATIOS[1:], CANDIDATE_BANDS, strict=True)
    }
    result["right_censored_other_windows"] = sum(row["oracle_other_D_HTTP_gap_right_censored"] for row in eligible)
    result["old_reasons"] = dict(Counter(row["original_reason"] for row in rows if "original_reason" in row))
    result["active_budget_exceeded"] = sum(1 for row in rows
        if row.get("actual_offload") and row.get("new_budget_ms") is not None
        and row["actual_boundary_to_D_first_token_ms"] > row["new_budget_ms"])
    actual = [row for row in rows if row.get("actual_offload")]
    result["actual_cost_upper_exceeded"] = sum(
        row["actual_boundary_to_D_first_token_ms"] > row["new_cost_ms"]
        for row in actual if row.get("new_cost_ms") is not None)
    native_waits = [item["D_coordinator_wait_ms"] for row in actual
                    for item in row["Native_ready_during_actual_suffix_reservation"]]
    result["Native_ready_during_suffix"] = len(native_waits)
    result["maximum_Native_wait_during_suffix_ms"] = max(native_waits, default=None)
    result["actual_offload_tokens"] = sum(row["suffix"] for row in actual)
    result["actual_offload_requests_with_spare"] = sum((row.get("D_EC_spare_images") or 0) > 0 for row in actual)
    result["actual_offload_requests_without_RPC"] = sum(row.get("D_EC_export_RPC_ms") == 0. for row in actual)
    holds = [row for row in rows if row.get("hold_attempts")]
    result["callback_with_actual_retirement_hold"] = len(holds)
    result["max_held_schedule_attempts"] = max((row["hold_attempts"] for row in holds), default=None)
    result["P_median_retirement_to_callback_end_ms"] = (
        median(row["P_retirement_to_callback_end_ms"] for row in holds) if holds else None)
    positive_delivery = [row for row in rows if row["next_active_D_client_token_ms"] is not None]
    result["cut_with_later_delivery_from_active_D"] = len(positive_delivery)
    result["median_to_next_active_D_client_token_ms"] = (
        median(row["next_active_D_client_token_ms"] for row in positive_delivery) if positive_delivery else None)
    for field in ("observed_D_HTTP_gap_ms", "oracle_other_D_HTTP_gap_ms", "P_own_native_ready_after_cut_ms",
                  "old_budget_ms", "old_cost_ms"):
        valid = [row[field] for row in eligible if row.get(field) is not None]
        result[f"median_{field}"] = median(valid) if valid else None
    return result


def analyze_cohort(run, cohort, source, profile=None):
    client, rows = joined_cohort(run, cohort)
    if source == "stock":
        paths = list((run / "raw").glob("stock_prefill_chunks_*.jsonl"))
        if not paths:
            raise FileNotFoundError(f"No stock P scheduler trace in {run}")
        cuts = stock_cuts(paths, client)
    elif source in {"shadow", "active"}:
        cuts = list(shadow_cuts(rows))
    else:
        raise ValueError(source)
    states = decode_queue(run)
    physical = None
    if profile is not None:
        from ..mllm_epd_h200_prefill_heavy_batch_matched.trace import kernel_intervals

        intervals, coverage, clock = kernel_intervals(profile / "decode")
        if not clock["passed"]:
            raise ValueError("The D collector reports an invalid raw kernel trace")
        physical = intervals, coverage
    ec_events = [event for path in (run / "raw").glob("ec_cuda_ipc_decode*.jsonl")
                 for event in read_jsonl(path)] if source == "active" else []
    annotated = list(annotate(rows, client, cuts, states, physical, ec_events))
    report = {
        "source": source, "run": run.name, "cohort": cohort,
        "observed_evidence": "DIAGNOSTIC",
        "future_evidence": "OFFLINE_ORACLE",
        "CUPTI_scope": ("DIAGNOSTIC unless independent Clean/Profile/Clean also passes"
                        if profile is not None else "NOT_MEASURED"),
        "clock": "System-wide time.monotonic for workers, arrival client and HTTP coordinator.",
        "operational_scope": "A request occupies D from its P-ready timestamp until its D HTTP EOF. "
                             "The optional D transition log supplies scheduler/KV state on the same clock. "
                             "Neither trace establishes exclusive GPU-idle time.",
        "counterfactual_scope": "Exclude the source's own native D interval when measuring time until "
                                "next other P-ready. All other native completions are frozen.",
        "censoring": "p20–p80 scheduled-arrival ROI for central cuts. Windows extending beyond p80 are "
                     "lower bounds excluded from error evaluation. Noncentral cuts are bounded by last D EOF "
                     "and explicitly identify post-arrival drain.",
        "central": funnel([row for row in annotated if row["central_ROI"]], len(rows)),
        "full": funnel(annotated, len(rows)),
    }
    return annotated, report


def write_evidence(folder, rows, report):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (report["cohort"] + "_cut_slack.json")).write_text(json.dumps(report, indent=2) + "\n")
    if rows:
        fields = sorted(set().union(*(row.keys() for row in rows)))
        with (folder / (report["cohort"] + "_cut_slack.csv")).open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                                 for key, value in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--source", choices=("stock", "shadow", "active"), required=True)
    parser.add_argument("--profile", type=Path, help="Same-cohort CUPTI folder whose decode/ holds the raw trace")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, report = analyze_cohort(args.run, args.cohort, args.source, args.profile)
    write_evidence(args.output, rows, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
