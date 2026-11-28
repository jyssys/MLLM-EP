"""Reduce native EPD quiet windows and replay admission decisions using past-only state."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from ..mllm_epd_h200_prefill_heavy_batch_matched.trace import complement, elapsed, intersect, merged
from ..mllm_epd_prefill_heavy_modality_discovery import timeline
from ..mllm_epd_prefill_heavy_modality_discovery.nixl_intervals import join_nixl, read_nixl_witness
from ..mllm_epd_prefill_heavy_modality_discovery.reduce_epd import (
    decode_pass, observer_gate, payload_receipt, queue_summaries, serving_metrics, source_path,
)
from .graph_observer import read_envelopes


TRAIN = (
    "long:calibration-sweep/long_q096", "long:calibration-sweep/long_q120",
    "long:calibration-sweep/long_q148", "v16k:controlled-sweep/v25_q082",
    "v16k:controlled-sweep/v25_q140", "long-low:calibration-sweep/long_q072",
)
EVALUATE = (
    "long:observer-long64/long_primed_n64_clean_before",
    "long:observer-long64/long_primed_n64_clean_after",
    "long:natural-confirmation/long_persistence128",
    "long:deflection-confirm/confirm_native_n64",
    "v16k:late-native/v25_primed_n24_clean_before",
    "v16k:controlled-sweep/v95_q112",
)
PROFILES = (
    "long:observer-long64/long_primed_n64_profile",
    "long:nixl-ready/nixl_readiness_n24_profile",
    "v16k:late-native/v25_primed_n24_profile",
)
QUANTA_MS = (25, 50, 100, 250, 500, 750, 1000, 1150, 1250)


@dataclass(frozen=True)
class NativeRequest:
    scheduled: float
    encoder_ready: float
    prefill_submit: float
    prefill_ready: float
    decode_complete: float
    input_tokens: int


def parse_spec(spec):
    family, sep, case = spec.partition(":")
    if not sep or "/" not in case:
        raise ValueError(f"Expected FAMILY:EPOCH/TAG, received {spec}")
    epoch, tag = case.split("/", 1)
    if Path(epoch).name != epoch or Path(tag).name != tag or not tag:
        raise ValueError(f"Expected one epoch and one tag in {spec}")
    return family, epoch, tag


def load_case(root, spec, proxies):
    family, epoch, tag = parse_spec(spec)
    folder = root / epoch
    cfg = next(row for row in timeline.read_jsonl(folder / "configs/cohorts.jsonl") if row["tag"] == tag)
    if cfg["route"] != "native":
        raise ValueError(f"{spec}: frozen no-deflection native route required")
    if folder not in proxies:
        path = source_path(folder / "raw", "proxy_epd.jsonl")
        if path is None:
            raise FileNotFoundError(folder / "raw/proxy_epd.jsonl")
        index = defaultdict(list)
        for row in timeline.read_jsonl(path):
            index[row["runtime_request_id"]].append(row)
        proxies[folder] = path, index
    proxy_path, proxy_index = proxies[folder]
    path = source_path(folder / "raw", f"{tag}_requests.jsonl")
    if path is None:
        raise FileNotFoundError(folder / "raw" / f"{tag}_requests.jsonl")
    clients = timeline.read_jsonl(path)
    if len(clients) != cfg["count"] or len({row["runtime_request_id"] for row in clients}) != len(clients):
        raise ValueError(f"{spec}: planned and unique raw cohort sizes differ")
    requests, joined = [], []
    for row in clients:
        identity = row["runtime_request_id"]
        choices = proxy_index[identity]
        if len(choices) != 1 or row["status"] != "MEASURED_ONLINE":
            raise ValueError(f"{spec}: missing or failed client/proxy match")
        proxy = choices[0]
        chain = ("E_complete", "P_submit", "P_ready", "D_submit", "D_http_complete")
        if (proxy["status"] != "MEASURED_ONLINE" or any(key not in proxy for key in chain)
                or row["scheduled_arrival"] > proxy["E_complete"]
                or any(proxy[b] < proxy[a] for a, b in zip(chain, chain[1:]))):
            raise ValueError(f"{spec}: no complete monotonic native E/P/D chain")
        requests.append(NativeRequest(row["scheduled_arrival"], proxy["E_complete"], proxy["P_submit"],
                                      proxy["P_ready"], proxy["D_http_complete"], row["input_tokens_manifest"]))
        joined.append(dict(proxy))
    window = timeline.arrival_window(clients)
    if window is None:
        raise ValueError(f"{spec}: too few scheduled arrivals")
    origin = {"spec": spec, "epoch": epoch, "tag": tag, "family": family, "QPS": cfg["qps"],
              "output": cfg["output"], "count": cfg["count"], "observer": cfg["observer"],
              "client": payload_receipt(path), "proxy": payload_receipt(proxy_path)}
    return origin, requests, window, clients, joined


def decision_rows(requests, window):
    """The E-ready admission surface is observed before the target consumes P resources."""
    left, right = window
    ordered = sorted(requests, key=lambda req: req.encoder_ready)
    ready = sorted(req.prefill_ready for req in requests)
    rows = []
    for target in ordered:
        now = target.encoder_ready
        if not left <= now < right:
            continue
        index = bisect_right(ready, now)
        # The native copy of the target is kept in the untouched trace. Its
        # own P-ready remains eligible in outcome labels, just like other work.
        delay = ready[index] - now if index < len(ready) else None
        inflight = [req for req in requests if req is not target and req.prefill_submit <= now < req.prefill_ready]
        rows.append({"seconds_from_ROI_start": now - left,
                     "input_tokens": target.input_tokens,
                     "P_outstanding": len(inflight),
                     "D_proxy_quiet": not any(req.prefill_ready <= now < req.decode_complete for req in requests),
                     "recent_P_ready_age_ms": (now - ready[index - 1]) * 1000 if index else None,
                     "oldest_P_submit_age_ms": max((now - req.prefill_submit for req in inflight), default=0) * 1000,
                     "next_native_P_ready_observed_ms": delay * 1000 if delay is not None else None})
    return rows


def control_predictions(rows, cadence_ms):
    for row in rows:
        age = row["recent_P_ready_age_ms"]
        row["cadence_predicted_ms"] = max(0., cadence_ms - age) if cadence_ms is not None and age is not None else None


def admit(row, policy, threshold_ms, minimum_pending):
    if not row["D_proxy_quiet"] or row["P_outstanding"] < minimum_pending:
        return False
    if policy == "idle_only":
        return True
    if policy == "causal_handoff_cadence":
        prediction = row["cadence_predicted_ms"]
    elif policy == "offline_next_ready":
        prediction = row["next_native_P_ready_observed_ms"]
    else:
        raise ValueError(f"Unknown admission policy {policy}")
    return prediction is not None and prediction >= threshold_ms


def score_policy(rows, policy, quantum_ms, reserve_ms, minimum_pending):
    """Retain the baseline handoffs and forbid overlap of hypothetical assist work."""
    selections, outcomes = [], Counter()
    busy_until = -math.inf
    threshold = quantum_ms + reserve_ms
    for row in rows:
        now = row["seconds_from_ROI_start"]
        if now < busy_until or not admit(row, policy, threshold, minimum_pending):
            continue
        busy_until = now + quantum_ms / 1000
        selections.append(row)
        horizon = row["next_native_P_ready_observed_ms"]
        if horizon is None:
            outcomes["unobserved"] += 1
        elif horizon < quantum_ms:
            outcomes["crosses_native_handoff"] += 1
        elif horizon < threshold:
            outcomes["consumes_reserve"] += 1
        else:
            outcomes["within_reserve"] += 1
    return {"evidence": "OFFLINE_ORACLE" if policy == "offline_next_ready" else "DIAGNOSTIC",
            "policy": policy, "quantum_ms": quantum_ms, "reserve_ms": reserve_ms,
            "admissions": len(selections), "crosses_native_handoff": outcomes["crosses_native_handoff"],
            "consumes_reserve": outcomes["consumes_reserve"], "within_reserve": outcomes["within_reserve"],
            "unobserved": outcomes["unobserved"], "slots_times_quantum_ms": len(selections) * quantum_ms}


def intervals_ms(spans):
    lengths = [(right - left) * 1000 for left, right in spans]
    return {"count": len(lengths), "total_ms": sum(lengths), "quantiles_ms": timeline.distribution(lengths),
            "interval_ms": lengths}


def case_analysis(requests, window, rows, cadence_ms, reserves, minimum_pending):
    left, right = window
    pending = merged(((row.prefill_ready, row.decode_complete) for row in requests), left, right)
    quiet = complement(pending, left, right)
    control_predictions(rows, cadence_ms)
    eligible = [row for row in rows if row["D_proxy_quiet"] and row["P_outstanding"] >= minimum_pending]
    return {
        "evidence": "DIAGNOSTIC",
        "scope": "Actual host timestamps establish proxy outstanding state. Physical GPU, DMA, and future deadlines need separate witnesses.",
        "ROI_monotonic_seconds": list(window),
        "ROI_seconds": right - left,
        "D_proxy_quiet_fraction": elapsed(quiet) / (right - left),
        "D_proxy_quiet_windows": intervals_ms(quiet),
        "E_ready_decisions": len(rows),
        "E_ready_D_proxy_quiet": sum(row["D_proxy_quiet"] for row in rows),
        "E_ready_D_proxy_quiet_P_pending": len(eligible),
        "E_ready_to_actual_next_native_P_ready_ms": timeline.distribution(
            row["next_native_P_ready_observed_ms"] for row in eligible
            if row["next_native_P_ready_observed_ms"] is not None),
        "offline_capacity": [{"quantum_ms": q, "quiet_windows_wide_enough": sum((b - a) * 1000 >= q for a, b in quiet)}
                             for q in QUANTA_MS],
        "policy_replays": [score_policy(rows, rule, quantum, reserve, minimum_pending)
                           for reserve in reserves for quantum in QUANTA_MS
                           for rule in ("idle_only", "causal_handoff_cadence", "offline_next_ready")],
    }


def matched_observer_check(epoch, origin, clients, observed):
    name = origin["tag"].removesuffix("_profile")
    raw = epoch / "raw"
    before, after = (source_path(raw, f"{name}_{arm}_requests.jsonl") for arm in ("clean_before", "clean_after"))
    if before is None or after is None:
        return {"passed": False, "status": "UNPAIRED"}
    paired = (timeline.read_jsonl(before), clients, timeline.read_jsonl(after))
    if any(len(rows) != origin["count"] for rows in paired):
        return {"passed": False, "status": "INCOMPLETE"}
    return observer_gate(paired, [serving_metrics(rows, origin["count"]) for rows in paired], {"captures": observed})


def current_exact_profile(epoch, origin, clients, proxy, paths, nixl):
    nominal = timeline.arrival_window(clients)
    coverage = {}
    for role, path in paths.items():
        meta = timeline.read_json(path / "actual_ROI.json")
        coverage[role] = timeline.clock_pair(meta["left"], meta["right"], ("host_lower_ns", "host_upper_ns"))[0]
    left = max(nominal[0], *(bound[0] for bound in coverage.values()))
    right = min(nominal[1], *(bound[1] for bound in coverage.values()))
    if right <= left:
        raise ValueError(f"{origin['spec']}: role anchors and central scheduled interval have no common span")
    duration = right - left
    capture = {role: timeline.kernel_capture(path, left, right) for role, path in paths.items()}
    if any(not row["collector_ok"] or not row["clock_ok"] for row in capture.values()):
        raise ValueError(f"{origin['spec']}: an exact kernel role failed its raw trace integrity or clock check")
    physical = {role: row["kernel"] for role, row in capture.items()}
    d_idle = complement(physical["decode"], left, right)
    pending = merged(((row["P_ready"], row["D_http_complete"]) for row in proxy), left, right)
    quiet = intersect(d_idle, complement(pending, left, right))
    joint_activity = timeline.all_intersections(physical["encoder"], physical["prefill"], d_idle)
    gate = matched_observer_check(epoch, origin, clients, capture)
    origin_time = min(row["scheduled_arrival"] for row in clients)
    in_roi = [row for row in clients if row.get("token_times") and left <= row["token_times"][0] <= row["token_times"][-1] < right]
    tokens = {row["runtime_request_id"]: row["input_tokens_manifest"] for row in clients}
    p_completion_tokens = [tokens[row["runtime_request_id"]] for row in proxy if left <= row["P_ready"] < right]
    e_completions = sum(left <= row["E_complete"] < right for row in proxy)
    n_read = None
    if nixl and nixl["whole_cohort_READ_accounted"]:
        reads = merged((span for row in proxy for span in row.get("D_native_nixl_READ_possible", ())), left, right)
        n_read = intervals_ms(intersect(quiet, complement(reads, left, right)))
    p_state = timeline.scheduler_states(paths["prefill"], left, right)
    if p_state is not None:
        observed = p_state.pop("intervals")
        p_state["observed_fraction"] = elapsed(observed["observed"]) / duration
        if p_state["compute_ready_field_observed"]:
            ready = observed["known_compute_ready_waiting"]
            p_state["P_ready_fraction"] = elapsed(ready) / duration
            p_state["P_ready_and_D_kernel_proxy_quiet_fraction"] = elapsed(intersect(ready, quiet)) / duration
    queue_samples = queue_summaries(source_path(epoch / "raw", f"{origin['tag']}_scheduler.jsonl"), (left, right))
    if queue_samples is not None:
        queue_samples["sample_count"] = len(queue_samples.pop("samples"))
    idle = intervals_ms(d_idle)
    idle.pop("interval_ms")
    return {"case": origin, "evidence": "TRACE_ANALYSIS" if gate["passed"] else "DIAGNOSTIC",
            "decode_activity_unit": "concurrent_cuda_kernels",
            "scope": "Exact role kernel unions within the common physical ROI. D proxy quiet bounds outstanding requests.",
            "observer_gate": gate, "full_role_collection_anchors_monotonic_seconds": coverage,
            "collector_integrity": {role: {key: value for key, value in row.items() if key not in {"kernel", "copy"}}
                                    for role, row in capture.items()},
            "canonical_ROI_seconds_after_first_arrival": [point - origin_time for point in nominal],
            "observed_ROI_seconds_after_first_arrival": [left - origin_time, right - origin_time],
            "canonical_ROI_coverage_fraction": duration / (nominal[1] - nominal[0]), "ROI_seconds": duration,
            "E_P_D_kernel_busy_fraction": {role: elapsed(spans) / duration for role, spans in physical.items()},
            "E_P_kernel_busy_and_D_kernel_idle_fraction": elapsed(joint_activity) / duration,
            "D_kernel_idle_proxy_quiet_fraction": elapsed(quiet) / duration,
            "D_kernel_idle": idle, "D_kernel_idle_proxy_quiet": intervals_ms(quiet),
            "D_kernel_idle_proxy_quiet_outside_NIXL": n_read,
            "D_DMA_activity_traced": capture["decode"]["copy"] is not None,
            "prefill_input_tokens_per_second": sum(p_completion_tokens) / duration,
            "prefill_completed_requests_per_second": len(p_completion_tokens) / duration,
            "encoder_completed_requests_per_second": e_completions / duration,
            "decode_SLO_wholly_in_ROI": {"passes": sum(decode_pass(row) for row in in_roi), "requests": len(in_roi)},
            "P_scheduler_state": p_state, "host_queue_samples": queue_samples}


def diagnostic_profile(root, origin, clients, proxy):
    epoch = root / origin["epoch"]
    profiles = epoch / "raw/profiles" / origin["tag"]
    nixl = join_nixl(read_nixl_witness(epoch), proxy, len(clients))
    paths = {role: profiles / role for role in timeline.ROLES}
    if any((paths["decode"] / name).is_file() for name in ("graphs.csv", "graphs.csv.gz")):
        left, right = timeline.arrival_window(clients)
        graph = read_envelopes(paths["decode"], left, right)
        upstream = {role: timeline.kernel_capture(paths[role], left, right) for role in ("encoder", "prefill")}
        if not all(value["collector_ok"] and value["clock_ok"] for value in upstream.values()):
            raise ValueError(f"{origin['spec']}: incomplete E/P physical capture or misaligned clock")
        gate = matched_observer_check(epoch, origin, clients, {
            **upstream, "decode": {"collector_ok": True, "clock_ok": True, "activity_unit": graph["evidence_unit"]}})
        pending = merged(((row["P_ready"], row["D_http_complete"]) for row in proxy), left, right)
        quiet = complement(pending, left, right)
        graph_free = complement(graph["graph_envelope_union"], left, right)
        eligible = intersect(graph_free, quiet)
        measured = {"case": origin, "evidence": "TRACE_ANALYSIS" if gate["passed"] else "DIAGNOSTIC",
                    "observer_gate": gate, "decode_activity_unit": graph["evidence_unit"],
                    "scope": graph["scope"], "graph_records": graph["graph_records"], "ROI_seconds": right - left,
                    "E_P_kernel_busy_fraction": {role: elapsed(value["kernel"]) / (right - left)
                                                 for role, value in upstream.items()},
                    "D_graph_envelope_fraction": elapsed(graph["graph_envelope_union"]) / (right - left),
                    "D_outside_graph_and_proxy_quiet_fraction": elapsed(eligible) / (right - left),
                    "D_outside_graph_and_proxy_quiet": intervals_ms(eligible),
                    "E_P_kernel_busy_and_D_graph_clear_proxy_quiet_fraction": elapsed(
                        intersect(intersect(upstream["encoder"]["kernel"], upstream["prefill"]["kernel"]), eligible)
                    ) / (right - left)}
        if nixl and nixl["whole_cohort_READ_accounted"]:
            reads = merged((span for row in proxy for span in row.get("D_native_nixl_READ_possible", ())), left, right)
            measured["D_graph_clear_proxy_quiet_outside_accounted_READ_fraction"] = (
                elapsed(intersect(eligible, complement(reads, left, right))) / (right - left))
        return measured
    if origin["campaign"] == "current":
        return current_exact_profile(epoch, origin, clients, proxy, paths, nixl)
    trace = timeline.analyze_window(paths, clients, proxy, keep_intervals=True,
                                    whole_cohort_native_READ_accounted=bool(nixl and nixl["whole_cohort_READ_accounted"]))
    if trace is None or set(trace["captures"]) != set(paths) or not all(
        value["collector_ok"] and value["clock_ok"] for value in trace["captures"].values()
    ):
        raise ValueError(f"{origin['spec']}: incomplete physical capture or misaligned clock")
    intervals = trace["intervals"]
    idle = intervals_ms(intervals["D_kernel_idle"])
    idle.pop("interval_ms")
    return {"case": origin, "evidence": "DIAGNOSTIC", "decode_activity_unit": "concurrent_cuda_kernels",
            "scope": "Existing physical profiler captures failed the 5% observer guard or used unqualified auxiliary observation.",
            "ROI_seconds": trace["duration_seconds"],
            "E_P_D_kernel_busy_fraction": {role: trace["fraction_of_window"][role] for role in timeline.ROLES},
            "D_kernel_idle": idle,
            "D_kernel_idle_proxy_quiet": intervals_ms(intervals["D_kernel_idle_proxy_request_free"]),
            "D_kernel_idle_proxy_quiet_outside_NIXL": intervals_ms(
                intervals["D_kernel_idle_proxy_free_outside_accounted_native_READ"])
            if "D_kernel_idle_proxy_free_outside_accounted_native_READ" in intervals else None,
            "P_ready_possibly_concurrent": trace["retrospective_quantum_fit"]}


def read_failed_graph(root, spec, archived_failure_receipt=None):
    _, epoch, tag = parse_spec(spec)
    folder = root / epoch
    stem = tag.removesuffix("_profile")
    cfg = next(row for row in timeline.read_jsonl(folder / "configs/cohorts.jsonl") if row["tag"] == tag)
    counts = {}
    receipts = []
    for label in ("prime", "clean_before", "profile", "clean_after"):
        path = source_path(folder / "raw", f"{stem}_{label}_requests.jsonl")
        counts[label] = dict(Counter(row["status"] for row in timeline.read_jsonl(path))) if path else None
        if path:
            receipts.append(payload_receipt(path))
    decode = folder / "logs/decode.log"
    encoder = folder / "logs/encoder.log"
    if not decode.is_file() or not encoder.is_file():
        if archived_failure_receipt is None:
            raise FileNotFoundError(f"{spec}: preserve raw role logs or pass --graph-failure-receipt")
        matches = [row for row in archived_failure_receipt["failures"] if row["case"] == spec]
        if len(matches) != 1 or matches[0]["statuses"] != counts or matches[0]["planned"] != cfg["count"]:
            raise ValueError(f"{spec}: archived graph failure receipt does not match the raw cohort")
        return matches[0]
    d_log = decode.read_text(errors="replace")
    e_log = encoder.read_text(errors="replace")
    capture_point = d_log.find("Capturing CUDA graphs (decode, FULL)")
    rpc_point = d_log.find('POST /campaign/profile/begin HTTP/1.1" 200 OK')
    crash_point = d_log.find("Segfault encountered")
    d_profile = folder / "raw/profiles" / tag / "decode"
    crash_line = d_log.rfind("\n", 0, crash_point) + 1 if crash_point >= 0 else 0
    phase = d_log[crash_line:crash_point + 820] if crash_point >= 0 else ""
    expected = ("!!!!!!! Segfault encountered !!!!!!!", 'File "<unknown>", line 0, in cuGraphLaunch',
                'File "<unknown>", line 0, in cudaGraphLaunch',
                'File "<unknown>", line 0, in at::cuda::CUDAGraph::replay()')

    def marker(log, text):
        match = next((line for line in log.splitlines() if text in line), None)
        if not match:
            return None
        stamp = re.search(r"\b\d{2}-\d{2} \d{2}:\d{2}:\d{2}\b", match)
        return {"timestamp_from_log": stamp.group(0) if stamp is not None else None, "excerpt": text}

    guard = "MemoryError: Live EC backing storage exceeds the frozen12GiB research bound. Do not free in-flight tensors."
    excerpts = {"D_profile_begin": marker(d_log, 'POST /campaign/profile/begin HTTP/1.1" 200 OK'),
                "D_native_stack": [line for line in expected if line in phase],
                "D_exit": marker(d_log, "MPClient: engine core exited unexpectedly; starting cleanup"),
                "E_secondary_guard": marker(e_log, guard)}
    return {"case": spec, "evidence": "DIAGNOSTIC", "planned": cfg["count"], "statuses": counts,
            "D_graph_instantiation_preceded_begin_RPC": 0 <= capture_point < rpc_point,
            "D_begin_RPC_accepted_before_crash": 0 <= rpc_point < crash_point,
            "D_native_graph_launch_segfault": crash_point >= 0 and "cuGraphLaunch" in d_log[crash_point:crash_point + 800],
            "E_EC_storage_safety_guard": "Live EC backing storage exceeds the frozen12GiB research bound" in e_log,
            "D_graph_csv_present": (d_profile / "graphs.csv").is_file(),
            "D_ROI_present": (d_profile / "actual_ROI.json").is_file(),
            "diagnostic_excerpts": excerpts,
            "sources": [payload_receipt(decode), payload_receipt(encoder), *receipts]}


def make_reduction(root, training, evaluation, physical, reserves, minimum_pending,
                   fresh_root=None, fresh_evaluation=(), fresh_profiles=(), fresh_failed_graphs=(),
                   archived_failure_receipt=None):
    if len(set(evaluation)) != len(evaluation) or set(evaluation) & set(training):
        raise ValueError("Train and evaluation case lists must be disjoint and evaluation cases unique")
    trained, histories, proxies = [], defaultdict(list), {}
    for spec in training:
        origin, requests, (left, right), _, _ = load_case(root, spec, proxies)
        handoffs = sorted(req.prefill_ready for req in requests if left <= req.prefill_ready < right)
        gaps = [(later - earlier) * 1000 for earlier, later in zip(handoffs, handoffs[1:])]
        histories[origin["family"]].extend(gaps)
        trained.append({**origin, "calibration_inter_handoff_gaps": len(gaps)})
    # An empirical lower quantile is a causal input when frozen on separate
    # cases. It is a predictor, with no worst-case safety interpretation.
    calibration = {family: {"training_gaps": len(gaps),
                            "p10_native_inter_P_ready_ms": sorted(gaps)[int(.10 * (len(gaps) - 1))] if gaps else None}
                   for family, gaps in histories.items()}
    evaluated, decisions = [], []
    evaluation_panel = [(root, spec, "previous") for spec in evaluation] + [
        (fresh_root, spec, "current") for spec in fresh_evaluation]
    for folder, spec, campaign in evaluation_panel:
        origin, requests, window, _, _ = load_case(folder, spec, proxies)
        origin["campaign"] = campaign
        rows = decision_rows(requests, window)
        cadence = calibration.get(origin["family"], {}).get("p10_native_inter_P_ready_ms")
        evaluated.append({"case": origin, **case_analysis(requests, window, rows, cadence, reserves, minimum_pending)})
        decisions.extend({"case": spec, **row} for row in rows)
    profiled = []
    physical_panel = [(root, spec, "previous") for spec in physical] + [
        (fresh_root, spec, "current") for spec in fresh_profiles]
    for folder, spec, campaign in physical_panel:
        origin, _, _, clients, proxy = load_case(folder, spec, proxies)
        origin["campaign"] = campaign
        profiled.append(diagnostic_profile(folder, origin, clients, proxy))
    return {"training": trained, "calibration": calibration, "evaluation": evaluated, "profiles": profiled,
            "graph_failures": [read_failed_graph(fresh_root, spec, archived_failure_receipt)
                               for spec in fresh_failed_graphs],
            "contract": {"quantum_ms": QUANTA_MS, "reserves_ms": reserves, "minimum_P_outstanding": minimum_pending,
                         "policy_input": "Observable E-ready and proxy events at or before each candidate timestamp.",
                         "deadline_label": "Next observed native P-ready on the unchanged baseline, including the candidate's native copy.",
                         "scheduler_progress": "Per-request P progress and completion predictions are absent on these baseline proxy traces.",
                         "CPC": "Past GPU intervals remain diagnostic per f5d6d70d."}}, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", action="append", help="Override training with FAMILY:EPOCH/TAG")
    parser.add_argument("--evaluate", action="append", help="Override evaluation with FAMILY:EPOCH/TAG")
    parser.add_argument("--profile", action="append", help="Override diagnostic profiles with FAMILY:EPOCH/TAG")
    parser.add_argument("--fresh-root", type=Path, help="The current campaign's raw root")
    parser.add_argument("--fresh-evaluate", action="append", default=[], help="Append current clean FAMILY:EPOCH/TAG")
    parser.add_argument("--fresh-profile", action="append", default=[], help="Append current physical FAMILY:EPOCH/TAG")
    parser.add_argument("--fresh-failed-graph", action="append", default=[], help="Record a failed graph FAMILY:EPOCH/TAG")
    parser.add_argument("--graph-failure-receipt", type=Path, help="Sanitized receipt for when role logs are archived by digest")
    parser.add_argument("--without-profiles", action="store_true")
    parser.add_argument("--reserve-ms", action="append", type=float)
    parser.add_argument("--min-P-outstanding", type=int, default=2)
    args = parser.parse_args()
    reserves = tuple(args.reserve_ms or (10., 25.))
    if min(reserves) < 0 or args.min_P_outstanding < 0:
        parser.error("Reserves and minimum outstanding must be nonnegative")
    if (args.fresh_evaluate or args.fresh_profile or args.fresh_failed_graph) and args.fresh_root is None:
        parser.error("Specify --fresh-root for current campaign observations")
    failure_receipt = timeline.read_json(args.graph_failure_receipt) if args.graph_failure_receipt is not None else None
    if failure_receipt is not None and failure_receipt.get("schema") != 1:
        parser.error("Unsupported diagnostic graph failure receipt schema")
    profiles = () if args.without_profiles else args.profile or PROFILES
    reduction, rows = make_reduction(args.raw_root, args.train or TRAIN, args.evaluate or EVALUATE,
                                    profiles, reserves, args.min_P_outstanding,
                                    args.fresh_root, args.fresh_evaluate, args.fresh_profile, args.fresh_failed_graph,
                                    failure_receipt)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "idle_feasibility.json").write_text(json.dumps(reduction, indent=2, allow_nan=False) + "\n")
    if reduction["graph_failures"]:
        (args.out / "graph_failure_receipt.json").write_text(
            json.dumps({"schema": 1, "failures": reduction["graph_failures"]}, indent=2, allow_nan=False) + "\n")
    if rows:
        with (args.out / "decision_inputs_and_outcomes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    qualification = Counter(row["evidence"] for row in reduction["profiles"])
    print(f"{len(reduction['evaluation'])} held-out clean cases, {len(rows)} observed E-ready decisions, "
          f"physical observations {dict(qualification)}")


if __name__ == "__main__":
    main()
