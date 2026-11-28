"""Evaluate fresh client-facing online traces and historical same-budget DP3."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import unicodedata

from ..mllm_epd_prefill_heavy_modality_discovery.reduce_epd import (
    DECODE_LIMITS, TTFT_LIMIT_MS, payload_receipt, source_path,
)
from ..mllm_epd_prefill_heavy_modality_discovery.timeline import distribution, read_jsonl
from .cohorts import PRIMARY_QPS


HISTORICAL_DP3 = ("dp3-retry", "dp3-b2048")
METRICS = ("TTFT_ms", "TPOT_ms", "ITL_ms", "E2E_ms")
QUANTILES = ("p50", "p95", "p99", "max")
NATIVE_ROUTES = {"native", "epd", "native-epd"}
COMPARISON = ("cohort_id", "count", "qps", "output", "arrival", "manifest_sha256", "cache_gib")
STRICT_RATIO = 1.05
DECODE_MIN_PERCENT = 95


def read_optional(path):
    return json.loads(path.read_bytes()) if path.is_file() else None


def json_lines(path):
    opening = gzip.open if path.suffix == ".gz" else open
    with opening(path, "rt") as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except ValueError:
                if line.endswith("\n"):
                    raise


def response_text_digest(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def compact_client(row):
    fields = {"runtime_request_id", "request_id", "index", "scheduled_arrival", "finish", "finish_reason",
              "status", "prompt_SHA256", "input_tokens_manifest", "source", "image_count", "output_tokens",
              "token_ids", "prompt_token_ids_SHA256", "normalized_response_text_SHA256", *METRICS}
    chosen = {key: value for key, value in row.items() if key in fields}
    if (tokens := row.get("prompt_token_ids")) is not None:
        canonical = json.dumps(tokens, separators=(",", ":")).encode()
        chosen["prompt_token_ids_SHA256"] = hashlib.sha256(canonical).hexdigest()
    if row.get("status") == "MEASURED_ONLINE" and isinstance(text := row.get("text"), str):
        chosen["normalized_response_text_SHA256"] = response_text_digest(text)
    return chosen


def proxy_routes(epoch):
    receipts = {}
    for path in sorted((epoch / "raw").glob("proxy_*.jsonl*")):
        if not path.name.endswith((".jsonl", ".jsonl.gz")):
            continue
        for row in json_lines(path):
            fields = ("status", "deflected", "encoder_assisted", "input_tokens", "E_complete")
            receipts.setdefault(row["runtime_request_id"], []).append({key: row.get(key) for key in fields})
    return receipts


def successful(row, output):
    tokens = row.get("token_ids", ())
    if (row.get("status") != "MEASURED_ONLINE" or not tokens
            or row.get("output_tokens") != len(tokens) or (output is not None and len(tokens) != output)
            or len(row.get("ITL_ms", ())) != len(tokens) - 1):
        return False
    times = [row.get(key) for key in ("TTFT_ms", "TPOT_ms", "E2E_ms")] + row["ITL_ms"]
    return all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in times)


def measure(rows, meta, completion, arrival, routes):
    planned = meta["count"]
    served = [row for row in rows if successful(row, meta["output"])]
    complete = (completion.get("planned_count") == planned
                and completion.get("completed") == len(served)
                and len(rows) == len(served) == planned == len({row["runtime_request_id"] for row in rows})
                and {row["index"] for row in rows} == set(range(planned)))
    duration = max((row.get("finish", row["scheduled_arrival"]) for row in rows), default=0)
    duration -= completion["monotonic_start"]
    duration = duration if duration > 0 else None
    ttft = tpot = itl = decode = joint = 0
    for row in served:
        a = row["TTFT_ms"] <= TTFT_LIMIT_MS
        b = row["TPOT_ms"] <= DECODE_LIMITS["TPOT_ms"]
        c = max(row["ITL_ms"], default=0) <= DECODE_LIMITS["ITL_ms"]
        ttft += a
        tpot += b
        itl += c
        decode += b and c
        joint += a and b and c
    metrics = {key: distribution(row[key] for row in served) for key in METRICS if key != "ITL_ms"}
    metrics["ITL_ms"] = distribution(gap for row in served for gap in row["ITL_ms"])
    total_tokens = sum(row["output_tokens"] for row in served)
    proxies = [routes[row["runtime_request_id"]][0] for row in served
               if len(routes.get(row["runtime_request_id"], ())) == 1
               and routes[row["runtime_request_id"]][0]["status"] == "MEASURED_ONLINE"]
    local = [row for row in proxies if row["deflected"] is True]
    return {
        "planned": planned, "recorded": len(rows), "completed": len(served),
        "failed_or_missing": planned - len(served), "complete": complete,
        "observed_status_counts": dict(Counter(row["status"] for row in rows)),
        "finish_reasons": dict(Counter(row.get("finish_reason", "missing") for row in served)),
        "duration_seconds": duration, "output_tokens": total_tokens,
        "input_tokens": distribution(row["input_tokens_manifest"] for row in served),
        "output_length": distribution(row["output_tokens"] for row in served),
        "ITL_samples": sum(len(row["ITL_ms"]) for row in served), "metrics_ms": metrics,
        "attainment_counts": {"TTFT": ttft, "TPOT": tpot, "every_ITL": itl,
                              "absolute_decode": decode, "joint": joint},
        "request_throughput_req_s": len(served) / duration if duration else None,
        "output_throughput_tok_s": total_tokens / duration if duration else None,
        "decode_qualified_goodput_req_s": decode / duration if duration else None,
        "joint_goodput_req_s": joint / duration if duration else None,
        "proxy_routes_joined": len(proxies), "D_local_completed_requests": len(local),
        "D_local_prompt_tokens": sum(row["input_tokens"] or 0 for row in local),
        "E_assisted_completed_requests": sum(row["encoder_assisted"] is True for row in proxies),
        "arrival_receipt_records": len(arrival["trace"]) if arrival else None,
    }


def expected_online(plan):
    if plan.get("physical"):
        return {}
    result = {}
    for case in plan.get("cases", []):
        if "stage" in case or "teacher_probe_index" in case:
            continue
        group = case["tag"]
        suffixes = ("_clean_before", "_profile", "_clean_after") if case.get("profile") else ("",)
        for suffix in suffixes:
            tag = group + suffix
            result[tag] = {**case, "tag": tag, "group": group,
                           "observer": "cupti" if suffix == "_profile" else "clean"}
    return result


def arm_role(epoch, setup):
    if setup.get("observer", "clean") != "clean":
        return "PROFILE"
    if ("graph" in epoch.lower() or setup["tag"].endswith(("_clean_before", "_clean_after"))):
        return "OBSERVER_CLEAN_CONTROL"
    if setup.get("route") == "text":
        return "TEXT_CONTROL"
    return "SERVING"


def load_online(root, scope, names=None, only_tags=None):
    planned = {number: read_optional(root / "dataset" / f"primary_very_long_n{number}_arrival.json")
               for number in (24, 64, 128, 256)} if scope == "FRESH" else {}
    epochs = ((root / name for name in names) if names is not None
              else sorted({path.parent.parent for pattern in ("*/configs/cohorts.jsonl", "*/configs/frozen_plan.json")
                           for path in root.glob(pattern)}))
    result = []
    for epoch in epochs:
        ledger = epoch / "configs/cohorts.jsonl"
        setups = list(read_jsonl(ledger)) if ledger.is_file() else []
        if only_tags:
            setups = [setup for setup in setups if setup["tag"] in only_tags]
        actual = {setup["tag"] for setup in setups}
        if scope == "FRESH":
            plan = read_optional(epoch / "configs/frozen_plan.json") or {}
            setups.extend(setup for tag, setup in expected_online(plan).items() if tag not in actual)
        if not setups:
            continue
        environment = read_optional(epoch / "state/environment.json") or {}
        model = read_optional(epoch / "state/model_download_complete.json") or {}
        routes = proxy_routes(epoch) if scope == "FRESH" else {}
        for order, meta in enumerate(setups, 1):
            tag = meta["tag"]
            role = arm_role(epoch.name, meta)
            observer = None
            if role in {"PROFILE", "OBSERVER_CLEAN_CONTROL"}:
                observer = "CUDA_GRAPH_CUPTI" if "graph" in epoch.name.lower() else "CUPTI_KERNEL"
            base = {"scope": scope, "epoch": epoch.name, "tag": tag, "episode_order": order,
                    "measurement_role": role, "instrumentation_family": observer,
                    "setup": meta, "model_revision": model.get("revision"),
                    "GPU_UUID_configuration": environment.get("authorized_uuid_by_gpu")}
            arrival = read_optional(epoch / "configs" / f"{tag}_arrival.json")
            completion = read_optional(epoch / "state" / f"{tag}_complete.json")
            raw = source_path(epoch / "raw", tag + "_requests.jsonl")
            if raw is None or completion is None:
                state = "NOT_RUN" if tag not in actual else "PENDING"
                if (epoch / "state/complete.json").is_file():
                    state = "NOT_RUN"
                if (epoch / "state/failure.json").is_file():
                    state = "NOT_RUN_TERMINATED"
                result.append((base | {"evidence": state}, [], arrival))
                continue
            rows = [compact_client(row) for row in json_lines(raw)]
            for row in rows:
                matching = routes.get(row["runtime_request_id"], ())
                if len(matching) == 1 and matching[0]["status"] == "MEASURED_ONLINE":
                    row["D_local"] = matching[0]["deflected"] is True
                    row["E_complete"] = matching[0]["E_complete"]
            record = measure(rows, meta, completion, arrival, routes)
            frozen = planned.get(meta["count"])
            if frozen and meta.get("cohort_id") == "long" and meta["qps"] == frozen["QPS"]:
                record["matches_frozen_primary_arrivals"] = bool(
                    arrival and arrival["trace"] == frozen["trace"]
                    and original_schedule_matches({row["index"]: row for row in rows}, frozen))
            evidence = "DIAGNOSTIC" if meta.get("observer", "clean") != "clean" else "MEASURED_ONLINE"
            result.append((base | record | {"evidence": evidence, "raw_receipt": payload_receipt(raw),
                           "producer_declared_raw_sha256": completion.get("raw_SHA256")}, rows, arrival))
    return result


def relative(candidate, baseline):
    return candidate / baseline - 1 if candidate is not None and baseline is not None and baseline > 0 else None


def original_schedule_matches(rows, arrival):
    if not arrival or 0 not in rows or len(arrival["trace"]) != len(rows):
        return False
    origin = rows[0]["scheduled_arrival"]
    return all(
        (row := rows.get(point["index"])) is not None and row["request_id"] == point["request_id"]
        and abs(row["scheduled_arrival"] - origin - point["offset_seconds"]) <= 2e-6
        for point in arrival["trace"])


def exact_pair(before, after):
    baseline, left, left_arrival = before
    candidate, right, right_arrival = after
    setup, test = baseline["setup"], candidate["setup"]
    aa = {row["index"]: row for row in left}
    bb = {row["index"]: row for row in right}
    comparable = left_arrival and right_arrival and len(left_arrival["trace"]) == len(right_arrival["trace"])
    arrivals = bool(comparable and left_arrival.get("seed") == right_arrival.get("seed")
                    and left_arrival.get("serial", False) == right_arrival.get("serial", False)
                    and all(a["request_id"] == b["request_id"] and a["index"] == b["index"]
                            and abs(a["offset_seconds"] - b["offset_seconds"]) <= 2e-6
                            for a, b in zip(left_arrival["trace"], right_arrival["trace"], strict=True)))
    planned = setup["count"]
    complete = baseline["complete"] and candidate["complete"] and aa.keys() == bb.keys() == set(range(planned))
    contract = all(setup.get(key) == test.get(key) for key in COMPARISON)
    gpu = baseline["GPU_UUID_configuration"]
    same_compute = (bool(gpu) and gpu == candidate["GPU_UUID_configuration"] and bool(baseline["model_revision"])
                    and baseline["model_revision"] == candidate["model_revision"])
    geometry = (same_compute
                and all(setup.get(key) == test.get(key) for key in ("p_budget", "d_budget", "ec_policy")))
    rows = [(aa[i], bb[i]) for i in sorted(aa.keys() & bb.keys())]
    fields = ("request_id", "prompt_SHA256", "input_tokens_manifest", "image_count")
    identity = all(all(a[field] == b[field] for field in fields) for a, b in rows)
    scheduled_offsets_match = original_schedule_matches(aa, left_arrival) and original_schedule_matches(bb, right_arrival)
    prompt_exact = sum(a.get("prompt_token_ids_SHA256") == b.get("prompt_token_ids_SHA256")
                       and a.get("prompt_token_ids_SHA256") is not None for a, b in rows)
    output_exact = sum(a["token_ids"] == b["token_ids"] and bool(a["token_ids"]) for a, b in rows)
    text_observed = [(a["normalized_response_text_SHA256"], b["normalized_response_text_SHA256"])
                     for a, b in rows if "normalized_response_text_SHA256" in a and "normalized_response_text_SHA256" in b]
    divergences = []
    for a, b in rows:
        expected, observed = a["token_ids"], b["token_ids"]
        if not expected or not observed or expected == observed:
            continue
        difference = next((i for i, (left_id, right_id) in enumerate(zip(expected, observed))
                           if left_id != right_id), min(len(expected), len(observed)))
        divergences.append({"request_index": a["index"], "request_id": a["request_id"],
                            "candidate_D_local": b.get("D_local"), "first_difference_zero_based": difference,
                            "native_emitted_tokens": len(expected), "candidate_emitted_tokens": len(observed)})
    deltas = {field: relative(candidate["metrics_ms"][field][q], baseline["metrics_ms"][field][q])
              for field, q in (("TTFT_ms", "p95"), ("TPOT_ms", "p95"), ("ITL_ms", "p99"), ("E2E_ms", "p95"))}
    for key in ("joint_goodput_req_s", "decode_qualified_goodput_req_s"):
        deltas[key] = relative(candidate[key], baseline[key])
    absolute = candidate["attainment_counts"]["absolute_decode"]
    reference = baseline["attainment_counts"]["absolute_decode"]
    eligible = (complete and arrivals and scheduled_offsets_match and contract and identity and geometry
                and prompt_exact == planned and baseline["evidence"] == candidate["evidence"] == "MEASURED_ONLINE")
    gate = {
        "TPOT_p95_at_most_plus_5pct": deltas["TPOT_ms"] is not None and deltas["TPOT_ms"] <= STRICT_RATIO - 1,
        "ITL_p99_at_most_plus_5pct": deltas["ITL_ms"] is not None and deltas["ITL_ms"] <= STRICT_RATIO - 1,
        "absolute_Decode_attainment_at_least_95pct": absolute * 100 >= planned * DECODE_MIN_PERCENT,
        "absolute_Decode_attainment_at_least_native": absolute >= reference,
    }
    return {
        "baseline": name(baseline), "candidate": name(candidate), "planned": planned,
        "paired_requests_recorded": len(rows), "both_complete": complete, "planned_arrivals_match": arrivals,
        "recorded_scheduled_offsets_match": scheduled_offsets_match,
        "same_workload_and_output_contract": contract, "same_GPU_and_model": same_compute,
        "same_GPU_model_budget_and_transport": geometry,
        "same_original_request_geometry": identity, "full_prompt_exact_requests": prompt_exact,
        "full_output_exact_requests": output_exact, "same_server_start": baseline["epoch"] == candidate["epoch"],
        "paired_response_text_receipts": len(text_observed),
        "normalized_response_text_exact_count": sum(a == b for a, b in text_observed),
        "D_local_paired_requests": sum(b.get("D_local", False) for _, b in rows),
        "D_local_full_output_exact_requests": sum(b.get("D_local", False) and a["token_ids"] == b["token_ids"]
                                                  and bool(a["token_ids"]) for a, b in rows),
        "output_divergence_positions": divergences,
        "strict_gate_eligible": bool(eligible), "strict_decode_gates": gate,
        "strict_decode_verdict": ("PASS" if all(gate.values()) else "FAIL") if eligible else "INELIGIBLE",
        "candidate_over_baseline_relative_change": deltas, "scored_task_quality_noninferiority": "NOT_RUN",
    }


def pair_fresh(arms):
    clean = [arm for arm in arms if arm[0]["evidence"] == "MEASURED_ONLINE"
             and arm[0]["measurement_role"] == "SERVING"]
    native = [arm for arm in clean if arm[0]["setup"].get("route", "native") in NATIVE_ROUTES]
    non_native = [arm for arm in clean if arm[0]["setup"].get("route", "native") not in NATIVE_ROUTES | {"dp3"}]
    pairs = []
    for candidate in non_native:
        setup = candidate[0]["setup"]
        matched = [arm for arm in native
                   if all(arm[0]["setup"].get(key) == setup.get(key) for key in COMPARISON)]
        same_start = [arm for arm in matched if arm[0]["epoch"] == candidate[0]["epoch"]]
        pairs.extend(exact_pair(arm, candidate) for arm in same_start or matched)
    return pairs


def matched_dp3(arms):
    clean = [arm for arm in arms if arm[0]["evidence"] == "MEASURED_ONLINE"
             and arm[0]["measurement_role"] == "SERVING"]
    pairs = []
    for candidate in clean:
        setup = candidate[0]["setup"]
        if setup.get("route") != "dp3":
            continue
        for baseline in clean:
            native = baseline[0]["setup"]
            if (native.get("route", "native") not in NATIVE_ROUTES
                    or any(native.get(key) != setup.get(key) for key in COMPARISON)):
                continue
            pair = exact_pair(baseline, candidate)
            if pair["same_original_request_geometry"] and pair["planned_arrivals_match"] and pair["same_GPU_and_model"]:
                pairs.append(pair)
    return pairs


def text_replay_controls(arms):
    ready = [arm for arm in arms if arm[0]["measurement_role"] == "TEXT_CONTROL"
             and arm[0]["evidence"] == "MEASURED_ONLINE"]
    by_tag = {arm[0]["tag"]: arm for arm in arms if arm[0]["evidence"] == "MEASURED_ONLINE"
              and arm[0]["measurement_role"] == "SERVING" and arm[0]["setup"].get("route") in NATIVE_ROUTES}
    results = []
    for row, clients, arrivals in ready:
        trace = (arrivals or {}).get("source")
        if not isinstance(trace, dict) or trace.get("kind") != "observed_e_ready":
            continue
        native = by_tag.get(trace["tag"])
        ready_order = sorted((item["E_complete"], item["runtime_request_id"]) for item in native[1]
                             if item.get("E_complete") is not None) if native else []
        event_offsets = [stamp - ready_order[0][0] for stamp, _ in ready_order]
        replay_offsets = trace["offsets"]
        native_match = (
            len(event_offsets) == len(replay_offsets) == row["planned"]
            and [runtime_id for _, runtime_id in ready_order] == trace["request_ids"]
            and all(abs(a - b) <= 2e-6 for a, b in zip(event_offsets, replay_offsets, strict=True)))
        text_pos = {item["index"]: item for item in clients}
        match_schedule = original_schedule_matches(text_pos, arrivals)
        poisson = [arm for arm in ready if arm[0]["setup"].get("arrival") == "poisson"
                   and arm[0]["setup"]["manifest_sha256"] == row["setup"]["manifest_sha256"]
                   and arm[0]["planned"] == row["planned"]]
        record = {
            "text_replay": name(row), "replay_independent_source_sha256": trace["source_SHA256"],
            "native_source": name(native[0]) if native else None,
            "native_source_event_count": len(event_offsets),
            "observed_finite_native_ready_rate_req_s": (
                (len(event_offsets) - 1) / event_offsets[-1] if len(event_offsets) > 1 and event_offsets[-1] > 0 else None),
            "replayed_source_ready_events_match": native_match,
            "actual_replay_planned_schedule_matches": match_schedule,
            "original_source_ready_gap_ms": distribution(1000 * (b - a) for a, b in zip(event_offsets, event_offsets[1:])),
            "actual_native_and_text_inputs_use_same_model": bool(
                native and row["model_revision"] and row["model_revision"] == native[0]["model_revision"]),
        }
        if poisson:
            peer, peer_clients, _ = poisson[0]
            pool = {item["index"]: item for item in peer_clients}
            identity = ("request_id", "prompt_SHA256", "input_tokens_manifest", "prompt_token_ids_SHA256")
            matched_rows = [(pool[index], text_pos[index]) for index in sorted(pool.keys() & text_pos.keys())
                            if all(pool[index].get(field) == text_pos[index].get(field) for field in identity)]
            record.update({
                "text_poisson": name(peer), "aligned_text_inputs": len(matched_rows),
                "text_poisson_nominal_offered_QPS": peer["setup"]["qps"],
                "whole_token_sequences_equal": sum(bool(a["token_ids"]) and a["token_ids"] == b["token_ids"] for a, b in matched_rows),
                "relative_replay_qps_vs_text_poisson": relative(row["setup"]["qps"], peer["setup"]["qps"]),
                "paired_output_exact_scope": "Same controlled text inputs only, with a different offered arrival stream.",
            })
        results.append(record)
    return results


def independent_stages(root):
    measurements = []
    for path in sorted(root.glob("*/configs/frozen_plan.json")):
        epoch = path.parent.parent
        for case in read_optional(path).get("cases", []):
            if "stage" not in case:
                continue
            receipt = epoch / "analysis" / f"{case['tag']}.json"
            summary = read_optional(receipt)
            if summary is None:
                continue
            fields = ("evidence", "stage", "batch_one", "requests", "tokens", "seconds",
                      "requests_per_second", "tokens_per_second", "cohorts", "scope", "observer_qualification")
            row = {key: summary.get(key) for key in fields}
            row.update(epoch=epoch.name, tag=case["tag"], receipt=payload_receipt(receipt))
            measurements.append(row)
    return measurements


def persistence(arms, prior_native):
    observations = []
    for row, _, _ in prior_native + arms:
        setup = row["setup"]
        if (row.get("complete") and row["evidence"] == "MEASURED_ONLINE" and row["measurement_role"] == "SERVING"
                and setup.get("route") in NATIVE_ROUTES and setup["cohort_id"] == "long"
                and setup["qps"] == PRIMARY_QPS and setup["output"] == 16
                and row.get("matches_frozen_primary_arrivals") is not False
                and setup["count"] in (24, 64, 128, 256)):
            observations.append(row)
    return observations


def name(row):
    return f"{row['epoch']}/{row['tag']}"


def flatten(row):
    out = {key: row.get(key) for key in (
        "scope", "epoch", "tag", "evidence", "measurement_role", "instrumentation_family", "episode_order",
        "planned", "recorded", "completed",
        "complete", "failed_or_missing", "duration_seconds", "ITL_samples", "request_throughput_req_s",
        "output_throughput_tok_s", "decode_qualified_goodput_req_s", "joint_goodput_req_s",
        "proxy_routes_joined", "D_local_completed_requests", "D_local_prompt_tokens", "E_assisted_completed_requests",
        "matches_frozen_primary_arrivals",
    )}
    out.update({key: row["setup"].get(key) for key in ("route", "cohort_id", "count", "qps", "output", "p_budget", "d_budget")})
    out.update({f"{key}_pass": row.get("attainment_counts", {}).get(key)
                for key in ("TTFT", "TPOT", "every_ITL", "absolute_decode", "joint")})
    for metric in METRICS:
        out.update({f"{metric}_{key}": row.get("metrics_ms", {}).get(metric, {}).get(key) for key in QUANTILES})
    return out


def main():
    from .online_report import markdown, plot

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    fresh = load_online(args.raw_root, "FRESH")
    prior = load_online(args.prior_root, "HISTORICAL_DP3", HISTORICAL_DP3)
    old_n24 = load_online(args.prior_root, "HISTORICAL_NATIVE", ("deflection-inline",), {"inline_native_n24"})
    pairs = pair_fresh(fresh)
    topology = matched_dp3(fresh)
    text_controls = text_replay_controls(fresh)
    stages = independent_stages(args.raw_root)
    census = persistence(fresh, old_n24)
    rows, historical = ([row for row, _, _ in arms] for arms in (fresh, prior))
    args.out.mkdir(parents=True, exist_ok=True)
    output = {
        "SLO_ms": {"TTFT_ms": TTFT_LIMIT_MS, **DECODE_LIMITS},
        "strict_relative_maximum_ratio": STRICT_RATIO, "strict_absolute_minimum_percent": DECODE_MIN_PERCENT,
        "fresh": rows, "historical_DP3": historical, "pairs": pairs, "fresh_matched_DP3": topology,
        "text_ready_replay_controls": text_controls, "independent_stage_receipts": stages,
        "native_fixed16_persistence": census,
    }
    (args.out / "online_evidence.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    past_persistence = [row for row in census if row["scope"] == "HISTORICAL_NATIVE"]
    if flat := [flatten(row) for row in rows + historical + past_persistence]:
        with (args.out / "online_cases.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, list(flat[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(flat)
    rendered = plot(rows, historical, pairs, topology, census, args.out / "figures") if args.figures else []
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown(rows, historical, pairs, topology, text_controls, stages, census, rendered))
    print(f"{len(rows)} fresh arms, {len(historical)} historical arms, "
          f"{len(pairs)} strict-pair comparisons and {len(topology)} fresh DP3 controls")


if __name__ == "__main__":
    main()
