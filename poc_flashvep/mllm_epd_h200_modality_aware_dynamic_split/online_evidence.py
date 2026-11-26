"""Extract verified clean/observed serving metrics and late-bound decisions."""
import argparse
from collections import Counter, defaultdict
from functools import cached_property
import hashlib
from itertools import combinations
import json
from pathlib import Path
from statistics import mean

from ..mllm_epd_h200_prefill_heavy_batch_matched import analysis as shared
from ..mllm_epd_h200_wide_decode_batch_token_split.analysis import CLEAN_OBSERVATIONS, WideCohort, geometry_summary


class DynamicCohort(WideCohort):
    @cached_property
    def variant(self):
        env = self.environment
        args = env.get("argv", [])
        return {
            "transport": ("inline-redirect" if env.get("EC_inline_redirect") else "on-demand"
                          if env.get("EC_inline_P") is False else "established"),
            "P_boundary_policy": ("natural-hold" if "--natural-hold" in args else "natural"
                                  if "--natural-boundaries" in args else "forced"
                                  if env.get("late") else "stock"),
            "execution_mode": "shadow" if "--shadow" in args else "active" if env.get("late") else "baseline",
        }

    @cached_property
    def serving_source_sha256(self):
        env = self.environment
        dynamic = env["runner_SHA256"]
        common = env["source_SHA256"]
        runtime = {name: dynamic.get(name) for name in
                   ("online.py", "coordinator.py", "admission.py", "policy.py", "modality.py", "late_worker.py")}
        runtime.update({name: common.get(name) for name in
                        ("proxy.py", "token_nixl.py", "cuda_ipc_ec.py", "server_entry.py", "idle_admission.py")})
        return hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()

    @property
    def family(self):
        base = self.config["workload"]["method"]
        if self.variant["P_boundary_policy"] != "stock":
            base += f"-{self.variant['P_boundary_policy']}"
        if self.variant["transport"] != "established":
            base += f"-{self.variant['transport']}"
        if self.variant["execution_mode"] == "shadow":
            base += "-shadow"
        return base

    @property
    def method(self):
        cap = self.cap if self.cap == "free" else "B" + self.cap
        return f"{self.family}-{cap}"

    @property
    def base(self):
        return {**super().base, **self.variant, "method_family": self.family,
                "serving_source_sha256": self.serving_source_sha256}


def milliseconds(begin, end):
    if begin is not None and end is not None and begin <= end:
        return (end - begin) * 1000
    return None


def request_record(raw, proxy, worker, overlapping_reservation):
    plan = proxy.get("late_plan")
    decisions = plan.get("decisions", []) if plan is not None else []
    executed = proxy.get("token_partition") or {}
    prefix = executed.get("prefix_tokens")
    full = raw.get("input_tokens_manifest")
    suffix = executed.get("suffix_tokens")
    suffix = int(suffix) if suffix is not None else (0 if proxy else None)
    partition_valid = suffix is not None and full and 0 <= suffix <= full
    counter = Counter(row.get("reason", "UNKNOWN_DECISION_REASON") for row in decisions)
    fit_checks = [(row["budget_ms"], row["cost_prediction_ms"], row["safety_ms"]) for row in decisions
                  if row.get("budget_ms") is not None and row.get("cost_prediction_ms") is not None]
    before = proxy.get("D_at_P_ready", {})
    critical = shared.critical_path(raw, proxy) if all(k in proxy for k in (
        "received", "E_submit", "E_complete", "P_submit", "P_ready", "D_submit")) else None
    accepted = next((row for row in decisions if row.get("accepted")), {})
    eligible = [row for row in decisions if row.get("shadow_would_reserve") is True]
    first_eligible = eligible[0] if eligible else {}
    shadow_prefix = first_eligible.get("prefix_tokens")
    shadow_suffix = full - shadow_prefix if shadow_prefix is not None else None
    first_arrival = raw.get("scheduled_arrival")
    boundary_ready = accepted.get("P_boundary_state", {}).get("boundary_ready")
    first_token_s = proxy.get("D_first_generated_token")
    fitted_path_ms = milliseconds(boundary_ready, first_token_s)
    reservation = proxy.get("D_admission", {}).get("reservation", {})
    reserved_time = reservation.get("reserved")
    d_engine = accepted.get("D_engine", {})
    budget_ms = accepted.get("budget_ms")
    if not suffix:
        budget_ms = fitted_path_ms = None
    raw_itl = raw["ITL_ms"]
    decode_pass = all(gap <= shared.SLO["ITL_ms"] for gap in raw_itl) and raw.get(
        "TPOT_ms", 0.0) <= shared.SLO["TPOT_ms"]
    return {
        "request_id": raw["request_id"], "runtime_request_id": raw["runtime_request_id"],
        "source": raw.get("source"), "stratum": raw.get("stratum"), "image_count": raw.get("image_count"),
        "input_tokens": full, "output_tokens": raw.get("output_tokens"), "TTFT_ms": raw["TTFT_ms"],
        "ITL_max_ms": max(raw_itl, default=0.0), "TPOT_ms": raw.get("TPOT_ms"),
        "decode_pass": decode_pass, "joint_slo": decode_pass and raw["TTFT_ms"] <= shared.SLO["TTFT_ms"],
        "prefix_tokens": prefix, "suffix_tokens": suffix,
        "actual_offload_ratio": suffix / full if partition_valid else None,
        "suffix_attention_pairs_per_layer": ((full - suffix) * suffix + suffix * (suffix + 1) // 2
                                             if partition_valid else None),
        "image_spans": json.dumps(plan.get("image_spans", [])) if plan else None,
        "late_policy": plan.get("policy") if plan else None,
        "P_natural_mode": plan.get("natural") if plan else None,
        "P_natural_hold_mode": plan.get("natural_hold") if plan else None,
        "shadow_decision_mode": plan.get("shadow") if plan else None,
        "shadow_eligible_count": len(eligible),
        "shadow_first_eligible_prefix_tokens": shadow_prefix,
        "shadow_first_eligible_suffix_tokens": shadow_suffix,
        "shadow_first_eligible_ratio": shadow_suffix / full if shadow_suffix is not None else None,
        "shadow_first_eligible_cost_ms": first_eligible.get("cost_prediction_ms"),
        "shadow_first_eligible_budget_ms": first_eligible.get("budget_ms"),
        "P_continued": proxy.get("late_P_continued"), "decision_count": len(decisions),
        "P_completed_boundary_observations": sum("accepted" in event for event in worker),
        "P_rejected_worker_boundaries": sum(event.get("accepted") is False for event in worker),
        "P_natural_skipped_inflight_boundaries": sum(
            event.get("reason") == "NATURAL_BOUNDARY_PASSED_BY_INFLIGHT_WORK" for event in worker
        ),
        "P_max_held_before_retirement_ms": max((event["held_before_retirement_ms"] for event in worker
                                                if "held_before_retirement_ms" in event), default=None),
        "P_scheduler_preemptions_at_boundary_max": max((event["prefill_recomputations"] for event in worker
                                                         if "prefill_recomputations" in event), default=None),
        "last_decision_reason": decisions[-1]["reason"] if decisions else None,
        "decision_reason_counts": json.dumps(counter, sort_keys=True),
        "decision_fit_budget_cost_margin_ms": json.dumps(fit_checks),
        "decision_RPC_ms": sum(milliseconds(row.get("callback_begin"), row.get("callback_end")) or 0
                               for row in decisions) if decisions else None,
        "D_quiet_at_P_ready_HTTP": before.get("active_http_requests") == 0
                                  and before.get("pending_admissions") == 0 if before else None,
        "D_coordinator_wait_ms": milliseconds(proxy.get("P_ready"), proxy.get("D_submit")),
        "D_ready_to_first_token_host_ms": milliseconds(proxy.get("D_submit"), first_token_s),
        "D_reserved_to_submit_host_ms": milliseconds(reserved_time, proxy.get("D_submit")),
        "D_reserved_to_first_token_host_ms": milliseconds(reserved_time, first_token_s),
        "D_engine_running_at_accept": d_engine.get("running"),
        "D_engine_waiting_at_accept": d_engine.get("waiting"),
        "D_engine_KV_free_blocks_at_accept": d_engine.get("KV_free_blocks"),
        "Native_P_ready_during_other_suffix_reservation": overlapping_reservation is not None,
        "Native_D_wait_overlapping_other_suffix_reservation_ms": overlapping_reservation,
        "accepted_cut_cost_ms": accepted.get("cost_prediction_ms") if suffix else None,
        "accepted_cut_budget_ms": budget_ms, "observed_cut_ready_to_first_token_ms": fitted_path_ms,
        "observed_budget_exceeded": fitted_path_ms > budget_ms if fitted_path_ms is not None
                                    and budget_ms is not None else None,
        "P_start_after_arrival_ms": milliseconds(first_arrival, proxy.get("P_submit")),
        "cut_ready_after_arrival_ms": milliseconds(first_arrival, boundary_ready),
        "D_submit_after_arrival_ms": milliseconds(first_arrival, proxy.get("D_submit")),
        "D_first_token_after_arrival_ms": milliseconds(first_arrival, first_token_s),
        "critical_path_ms": json.dumps(critical, sort_keys=True) if critical else None,
    }


def worker_records(run):
    result = defaultdict(list)
    paths = sorted(run.glob("raw/late_prefill_cut_*.jsonl"))
    paths.extend(sorted(run.glob("raw/late_prefill_natural_skips_*.jsonl")))
    for path in paths:
        for row in shared.jsonl(path, appending=True):
            result[row["request_id"]].append(row)
    return result


def ec_enclosure_records(run):
    result = {}
    for path in sorted(run.glob("raw/ec_cuda_ipc_*_gpu*.jsonl")):
        tag = path.stem.split("_gpu", 1)[0].removeprefix("ec_cuda_ipc_")
        result[tag] = shared.jsonl(path, appending=True)
    return result


def ec_enclosure_summary(cohort, by_role):
    completion = shared.read(cohort.run / "state" / f"{cohort.tag}_complete.json")
    begin = completion["monotonic_start"]
    end = begin + (completion["wall_end"] - completion["wall_start"])
    result = []
    for role, calls in sorted(by_role.items()):
        observed = [row for row in calls if begin <= row["begin"] <= row["end"] <= end]
        if not observed:
            continue
        result.append({
            **cohort.base, "role": role, "clone_events": len(observed),
            "cloned_images": sum(row["images"] for row in observed),
            "inline_images": sum(row.get("inline_images", 0) for row in observed),
            "redirected_inline_images": (sum(row.get("redirected_inline_images", 0) for row in observed)
                                         if "redirected_inline_images" in observed[0] else None),
            "logical_cloned_BF16_bytes": sum(row["logical_bytes"] for row in observed),
            "descriptor_prepare_mean_host_ms": mean(row["descriptor_prepare_host_ms"] for row in observed),
            "export_rpc_mean_host_ms": mean(row["export_RPC_ms"] for row in observed),
            "clone_sync_mean_host_ms": mean(row["import_clone_sync_host_ms"] for row in observed),
            "scope": "Worker EC host enclosures wholly inside the cohort. Logical clone payload and host time. "
                     "Physical link utilization, causal request TTFT and GPU-exclusive durations are separate.",
        })
    return result


def reservation_intervals(proxy_rows):
    intervals = []
    for request_id, row in proxy_rows.items():
        reserve = row.get("D_admission", {}).get("reservation", {}).get("reserved")
        first = row.get("D_first_generated_token")
        if milliseconds(reserve, first) is not None:
            intervals.append((request_id, reserve, first))
    return intervals


def native_overlap(proxy, request_id, intervals):
    p_ready = proxy.get("P_ready")
    d_submit = proxy.get("D_submit")
    if proxy.get("token_partition") or milliseconds(p_ready, d_submit) is None:
        return None
    return next((milliseconds(p_ready, min(d_submit, end)) for owner, start, end in intervals
                 if owner != request_id and start <= p_ready < end), None)


def decode_token_weighted(histogram):
    weighted = [(row["decode_only_requests"] + row["mixed_requests"], row["decode_query_rows"] * row["calls"])
                for row in histogram if row["series"] == "prepared_query_histogram" and row["counter_valid"]]
    total = sum(rows for _, rows in weighted)
    return sum(batch * rows for batch, rows in weighted) / total if total else None


def summarize(cohort, requests, tables):
    counts = Counter(row["last_decision_reason"] for row in requests if row["last_decision_reason"])
    decisions = Counter()
    for row in requests:
        decisions.update(json.loads(row["decision_reason_counts"]))
    fit_checks = [check for row in requests for check in json.loads(row["decision_fit_budget_cost_margin_ms"])]
    token_total = sum(row["input_tokens"] for row in requests)
    suffix_total = sum(row["suffix_tokens"] or 0 for row in requests)
    observed = [row for row in requests if row["suffix_tokens"]]
    waits = [row["D_coordinator_wait_ms"] for row in requests if row["D_coordinator_wait_ms"] is not None]
    observed_waits = [row["D_coordinator_wait_ms"] for row in observed if row["D_coordinator_wait_ms"] is not None]
    fits = [row["observed_budget_exceeded"] for row in observed if row["observed_budget_exceeded"] is not None]
    reservation_spans = [row["D_reserved_to_first_token_host_ms"] for row in observed
                         if row["D_reserved_to_first_token_host_ms"] is not None]
    callback = [row["decision_RPC_ms"] for row in requests if row["decision_RPC_ms"] is not None]
    blocked = [row["Native_D_wait_overlapping_other_suffix_reservation_ms"] for row in requests
               if row["Native_P_ready_during_other_suffix_reservation"]]
    complete_proxy = len(cohort.proxy_rows) == len(requests)
    shadow = cohort.variant["execution_mode"] == "shadow"
    if shadow and suffix_total:
        raise ValueError(f"Shadow cohort {cohort.tag} executed a suffix")
    start = len(tables["actual_d_batch"])
    batch = shared.batch_table(cohort, tables)
    histogram = tables["actual_d_batch"][start:]
    geometry = geometry_summary(cohort, histogram)
    m = cohort.metrics
    cfg = cohort.config
    return {
        **cohort.base, **batch, **geometry,
        "independent_start_id": cfg["workload"].get("paired_start_id"),
        "manifest_sha256": cfg["workload"]["manifest_SHA256"],
        "arrival_sha256": cfg["arrival_SHA256"],
        "completed": len(requests), "proxy_coverage": len(cohort.proxy_rows),
        "absolute_decode_slo": sum(row["decode_pass"] for row in requests) / len(requests),
        "joint_slo": sum(row["joint_slo"] for row in requests) / len(requests),
        "TTFT_mean_ms": m["TTFT_mean_ms"], "TTFT_p50_ms": m["TTFT_p50_ms"], "TTFT_p95_ms": m["TTFT_p95_ms"],
        "TTFT_p99_ms": m["TTFT_p99_ms"], "ITL_mean_ms": m["ITL_mean_ms"], "ITL_p95_ms": m["ITL_p95_ms"],
        "ITL_p99_ms": m["ITL_p99_ms"], "ITL_max_ms": m["ITL_max_ms"], "TPOT_p95_ms": m["TPOT_p95_ms"],
        "SLO_goodput": m["SLO_goodput"], "total_output_tokens": m["total_output_tokens"],
        "completed_requests_per_second": m["completed_requests_per_second"],
        "output_tokens_per_second": m["raw_output_tokens_per_second"],
        "total_input_output_tokens_per_second": m["total_input_output_tokens_per_second"],
        "SLO_failure_categories": json.dumps(m["categories"], sort_keys=True),
        "suffix_executions": len(observed) if complete_proxy else None,
        "suffix_token_total": suffix_total if complete_proxy else None,
        "workload_offload_ratio": suffix_total / token_total if complete_proxy else None,
        "shadow_requests_with_callbacks": sum(row["decision_count"] > 0 for row in requests) if shadow else None,
        "shadow_eligible_requests": sum(row["shadow_eligible_count"] > 0 for row in requests) if shadow else None,
        "shadow_eligible_decisions": sum(row["shadow_eligible_count"] for row in requests) if shadow else None,
        "shadow_first_eligible_suffix_tokens_total": sum(row["shadow_first_eligible_suffix_tokens"] or 0
                                                         for row in requests) if shadow else None,
        "policy_budgeted_decisions": len(fit_checks),
        "policy_budget_p50_ms": shared.percentile([b for b, _, _ in fit_checks], .5),
        "policy_cost_prediction_p50_ms": shared.percentile([c for _, c, _ in fit_checks], .5),
        "policy_predicted_over_budget_count": sum(cost + margin > budget for budget, cost, margin in fit_checks),
        "late_continued": sum(row["P_continued"] is True for row in requests),
        "P_natural_completed_boundary_observations": sum(
            row["P_completed_boundary_observations"] for row in requests if row["P_natural_mode"]
        ),
        "P_natural_skipped_inflight_boundaries": sum(row["P_natural_skipped_inflight_boundaries"] for row in requests),
        "P_max_held_before_retirement_ms": max((row["P_max_held_before_retirement_ms"] for row in requests
                                               if row["P_max_held_before_retirement_ms"] is not None), default=None),
        "D_wait_mean_ms": mean(waits) if waits else None, "D_wait_p95_ms": shared.percentile(waits, .95),
        "D_split_wait_p95_ms": shared.percentile(observed_waits, .95),
        "D_split_quiet_at_P_ready_HTTP": sum(row["D_quiet_at_P_ready_HTTP"] is True for row in observed),
        "D_reserved_to_first_token_host_p95_ms": shared.percentile(reservation_spans, .95),
        "P_callbacks_host_p95_ms_per_request": shared.percentile(callback, .95),
        "D_prepared_decode_query_token_weighted_mean": decode_token_weighted(histogram),
        "D_prepared_decode_time_weighted_mean": None,
        "Native_P_ready_during_other_suffix_reservation_count": len(blocked),
        "Native_D_wait_overlapping_other_suffix_reservation_p95_ms": shared.percentile(blocked, .95),
        "available_budget_exceed_checks": len(fits), "observed_exceed_count": sum(fits),
        "last_decision_reasons": json.dumps(counts, sort_keys=True),
        "all_decision_reasons": json.dumps(decisions, sort_keys=True),
    }


def paired_requests(cohorts, requests, mode="active"):
    records = defaultdict(dict)
    for row in requests:
        records[(row["start"], row["cohort"])][row["request_id"]] = row
    matched = defaultdict(list)
    for cohort in cohorts:
        if (cohort.observation == "clean" and cohort.cycles == 1
                and cohort.variant["execution_mode"] == mode):
            matched[cohort.match].append(cohort)
    fields = ("TTFT_ms", "ITL_max_ms", "TPOT_ms", "output_tokens", "suffix_tokens",
              "actual_offload_ratio", "suffix_attention_pairs_per_layer",
              "D_coordinator_wait_ms", "D_reserved_to_first_token_host_ms",
              "accepted_cut_budget_ms", "accepted_cut_cost_ms", "observed_cut_ready_to_first_token_ms")
    if mode == "shadow":
        fields = ("TTFT_ms", "ITL_max_ms", "TPOT_ms", "output_tokens", "decision_count", "shadow_eligible_count",
                  "shadow_first_eligible_suffix_tokens", "shadow_first_eligible_ratio",
                  "shadow_first_eligible_cost_ms", "shadow_first_eligible_budget_ms",
                  "decision_RPC_ms", "P_natural_skipped_inflight_boundaries")
    result = []
    for arms in matched.values():
        for generic, aware in combinations(sorted(arms, key=lambda cohort: not cohort.method.startswith("generic")), 2):
            if (not generic.method.startswith("generic") or not aware.method.startswith(("hard", "soft", "fake_boundary"))
                    or generic.cap != aware.cap or generic.run == aware.run or generic.variant != aware.variant
                    or generic.serving_source_sha256 != aware.serving_source_sha256):
                continue
            left = records[(generic.run.name, generic.tag)]
            right = records[(aware.run.name, aware.tag)]
            if left.keys() != right.keys():
                raise ValueError("Matched online rosters differ")
            for request_id, a in left.items():
                b = right[request_id]
                result.append({**aware.base, "generic_start": generic.run.name, "generic_method": generic.method,
                               "request_id": request_id, "input_tokens": a["input_tokens"], "image_count": a["image_count"],
                               "generic_decode_pass": a["decode_pass"], "aware_decode_pass": b["decode_pass"],
                               **{f"generic_{key}": a[key] for key in fields},
                               **{f"aware_{key}": b[key] for key in fields}})
    return result


def native_aa(cohorts):
    by_start = defaultdict(dict)
    for cohort in cohorts:
        if cohort.family == "native-common" and cohort.observation in ("clean_before", "clean_after"):
            by_start[(cohort.run.name, cohort.config["QPS"])][cohort.observation] = cohort
    summary, details = [], []
    for pair in by_start.values():
        if len(pair) != 2:
            continue
        before, after = pair["clean_before"], pair["clean_after"]
        fields = ("arrival_SHA256", "output_mode", "seed")
        if (any(before.config[key] != after.config[key] for key in fields)
                or before.config["workload"]["manifest_SHA256"] != after.config["workload"]["manifest_SHA256"]):
            raise ValueError("Native A/A requires identical workload and scheduled arrival pattern")
        b = {row["request_id"]: row["token_ids"] for row in before.client_rows}
        a = {row["request_id"]: row["token_ids"] for row in after.client_rows}
        if b.keys() != a.keys():
            raise ValueError("Native A/A request rosters differ")
        identity = {
            "start": before.run.name, "method": before.method, "QPS": before.config["QPS"],
            "seed": before.config["seed"], "arrival_sha256": before.config["arrival_SHA256"],
        }
        for request_id, reference in b.items():
            current = a[request_id]
            identical = reference == current
            difference = None if identical else next((i for i, (x, y) in enumerate(zip(reference, current)) if x != y),
                                                      min(len(reference), len(current)))
            details.append({
                **identity, "request_id": request_id, "identical_token_ids": identical,
                "before_output_tokens": len(reference), "after_output_tokens": len(current),
                "first_different_token_index_zero_based": difference,
            })
        summary.append({
            **identity, "request_count": len(b), "exact_token_sequence_count": sum(b[key] == a[key] for key in b),
            "same_output_length_count": sum(len(b[key]) == len(a[key]) for key in b),
            "before_total_output_tokens": sum(map(len, b.values())),
            "after_total_output_tokens": sum(map(len, a.values())),
            "before_raw_sha256": before.metrics["raw_SHA256"],
            "after_raw_sha256": after.metrics["raw_SHA256"],
            "scope": "Native A/A clean-before/after within one server start. "
                     "Output identity is a diagnostic; no independent timing or correctness threshold changes.",
        })
    return summary, details


def aggregate(runs, output, cohort_factory=DynamicCohort):
    cohorts, excluded = shared.cohort_records(runs, cohort_factory)
    tables = {"requests": [], "cohorts": [], "actual_d_batch": [], "observer": [],
              "ec_enclosures": [], "excluded": excluded}
    worker_by_run = {}
    ec_by_run = {}
    for cohort in cohorts:
        if cohort.run not in worker_by_run:
            worker_by_run[cohort.run] = worker_records(cohort.run)
            ec_by_run[cohort.run] = ec_enclosure_records(cohort.run)
        worker = worker_by_run[cohort.run]
        intervals = reservation_intervals(cohort.proxy_rows)
        requests = []
        for raw in cohort.client_rows:
            request_id = raw["runtime_request_id"]
            proxy = cohort.proxy_rows.get(request_id, {})
            requests.append(request_record(raw, proxy, worker.get(request_id, []),
                                           native_overlap(proxy, request_id, intervals)))
        tables["requests"].extend(cohort.base | row for row in requests)
        tables["cohorts"].append(summarize(cohort, requests, tables))
        tables["ec_enclosures"].extend(ec_enclosure_summary(cohort, ec_by_run[cohort.run]))
    tables["observer"] = shared.observer_qualification(runs)
    contrasts = shared.paired_contrasts(cohorts)
    tables["matched_contrasts"] = contrasts
    tables["matched_BC_requests"] = paired_requests(cohorts, tables["requests"])
    tables["shadow_matched_BC_requests"] = paired_requests(cohorts, tables["requests"], mode="shadow")
    tables["paired_95_ci"] = shared.paired_intervals(contrasts)
    tables["native_aa_summary"], tables["native_aa_requests"] = native_aa(cohorts)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        shared.csv_file(output / f"dynamic_{name}.csv", rows)
    clean = [row for row in tables["cohorts"] if row["observation"] in CLEAN_OBSERVATIONS]
    bundle = {
        "counts": {name: len(rows) for name, rows in tables.items()},
        "clean_cohorts": clean,
        "scope": "Verified against the frozen arrival trace, full client log and receipt digests. Counter and "
                 "CPC arms stay separate. Matched contrasts are reported without inferring independent N1 as paired N3.",
        "budget_observation": "Observed cut-ready-to-first-token includes decision, handoff, and worker/HTTP service. "
                              "It is compared with the recorded policy budget and does not reconstruct future idle truth.",
        "D_state": "P-ready D quiet in this table uses the observed HTTP coordinator queue. The reservation "
                   "receipt and Scheduler probes give engine-state evidence in the underlying trace.",
        "D_actual_batch": "Prepared step and decode query-row weighted batch from observed worker histograms. "
                          "Forward time weighting is NOT_MEASURED.",
        "D_reservation": "Native overlap is the observed host interval from Native P-ready (occurring during "
                         "another request's reservation) through its submit or reservation owner first token. "
                         "The overlap bounds time; it does not identify incremental delay causally.",
        "shadow": "Shadow runs execute only full P prefill and Native D decode. Eligible decisions observe a policy "
                  "and D state snapshot without taking its reservation. Actual transfer and clean timing stay distinct.",
    }
    (output / "online_evidence.json").write_text(json.dumps(bundle, indent=2) + "\n")
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = aggregate(args.runs, args.output)
    print(json.dumps(bundle["counts"]))


if __name__ == "__main__":
    main()
