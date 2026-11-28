"""Reduce selected serial H200 readiness probes with matched recipient controls."""

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median


ARMS = ("late_p", "early_p", "late_d", "early_d")
SCOPE = ("Selected serial Fixed16 hardware probes. All arms retain the original image-complete cut. "
         "P continues locally and then performs a full P→D KV handoff. D continues locally. "
         "No online queueing, admission, steady traffic or Decode-protection conclusion.")


def digest(raw):
    return sha256(raw).hexdigest()


def ids_digest(ids):
    return digest(json.dumps(ids, separators=(",", ":")).encode())


def load_lines(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def percentile(items, q):
    values = sorted(items)
    ix = (len(values) - 1) * q
    lower, upper = math.floor(ix), math.ceil(ix)
    return values[lower] + (ix - lower) * (values[upper] - values[lower])


def ms(terminal, initial):
    return (terminal - initial) / 1e6


def visual_overlap(spans, end):
    return sum(max(0, min(end, right) - left) for left, right in spans)


def fixture_spans(fixture):
    spans = fixture.get("image_visual_spans")
    if spans is None:
        spans = fixture.get("request", {}).get("image_visual_spans")
    require(isinstance(spans, list) and len(spans) == fixture["total_images"],
            f"Need original image spans for {fixture['case_id']}. Use the full frozen case file.")
    prior = 0
    for left, right in spans:
        require(prior < left < right < fixture["total_tokens"], "Invalid original visual span")
        prior = right
    prefix = fixture["prefix_images"]
    require(0 < prefix < len(spans) and spans[prefix - 1][1] + 1 == fixture["cut_tokens"]
            <= spans[prefix][0], f"Invalid image-complete cut for {fixture['case_id']}")
    return spans


def reduce_trace(events, record, role):
    end = 0
    latest = record["recipient_submit_ns"]
    trace = []
    require(bool(events), "Missing recipient LM retirement trace")
    limit = record["prompt_tokens"]
    for row in events:
        before, after, count = row["before"], row["retired_rows"], row["scheduled_rows"]
        stamp, resume = row["retire_ns"], row.get("resume_ns")
        require(row["role"] == role and row["cut"] == record["cut"]
                and row["key"] == record["key"], "Retirement identity or cut mismatch")
        require(latest <= stamp and before == end and 0 <= before < after <= limit
                and count > 0 and after == min(limit, before + count), "Noncontiguous LM retirement trace")
        require(not before < record["cut"] < after, "LM chunk crossed the frozen cut")
        if before >= record["cut"]:
            require(resume is not None and record["prefix_published_ns"] <= resume <= stamp,
                    "Suffix retirement without its all-published resume")
        trace.append((after, stamp))
        end, latest = after, stamp
    require(len({event["request_id"] for event in events}) == 1, "Recipient request identity changed")
    require(end == limit and any(rows == record["cut"] for rows, _ in trace),
            "Recipient trace did not complete the prescribed prefix and full prompt")
    return trace


def retired_before(trace, bound):
    if bound is None:
        return None
    return max((rows for rows, stamp in trace if stamp < bound), default=0)


def get_image_witness(raw):
    folder = raw / "image_ready"
    source = sorted(folder.glob("events_*.jsonl")) + sorted(folder.glob("manifests_*.jsonl"))
    events, manifests = defaultdict(list), defaultdict(list)
    for path in source:
        for row in load_lines(path):
            if "image_identifier" in row:
                events[row["image_identifier"]].append(row)
            else:
                manifests[row["request_key"]].append(row)
    return source, events, manifests


def reduce_image_witness(record, trace, spans, observer):
    sources, events, manifests = observer
    if not sources:
        return None, None
    matches = manifests.get(record["key"], [])
    if not matches:
        return {"complete": False, "reason": "REQUEST_MANIFEST_NOT_OBSERVED"}, None
    require(len(matches) == 1, "An early trial has multiple encoder image-ready manifests")
    manifest, = matches
    ids = manifest["ordered_image_identifiers"]
    require(len(ids) == len(set(ids)) == len(spans) and manifest["image_spans"] == spans
            and manifest["lm_input_tokens"] == record["prompt_tokens"], "Encoder image-ready manifest drift")
    selected = []
    for key, (left, right) in zip(ids, spans, strict=True):
        found = events.get(key, [])
        if len(found) != 1 or found[0]["status"] != "CUDA_EVENT_COMPLETE":
            return {"complete": False, "reason": "IMAGE_EVENT_MISSING_OR_INCOMPLETE"}, ids
        row, = found
        if row["embedding_rows"] != right - left:
            raise ValueError("Original encoder image-ready event row count differs")
        selected.append(row)
    require(len({row["producer_pid"] for row in selected}) == 1, "Producer changed within a trial")
    require(all(record["start_ns"] <= r["save_begin_ns"] <= r["completion_lower_bound_ns"]
                <= r["completion_upper_bound_ns"] and r["save_begin_ns"] <= r["save_returned_ns"]
                for r in selected), "Invalid producer CUDA event bounds")
    all_lower = max(r["completion_lower_bound_ns"] for r in selected)
    all_upper = max(r["completion_upper_bound_ns"] for r in selected)
    prefix = selected[:record["prefix_images"]]
    prefix_event_lower = max(r["completion_lower_bound_ns"] for r in prefix)
    prefix_event_upper = max(r["completion_upper_bound_ns"] for r in prefix)
    prefix_available_upper = max(max(r["completion_upper_bound_ns"],r["save_returned_ns"]) for r in prefix)
    proven = retired_before(trace, all_lower)
    return {"complete": True, "images_joined": len(selected), "producer_pid": selected[0]["producer_pid"],
            "conservative_E_GPU_prefix_lead_ms": ms(all_lower, prefix_event_upper),
            "E_GPU_prefix_lead_upper_bound_ms": ms(all_upper, prefix_event_lower),
            "conservative_published_prefix_to_E_GPU_all_complete_ms": ms(all_lower, prefix_available_upper),
            "E_GPU_all_ready_bound_width_ms": ms(all_upper, all_lower),
            "retired_rows_before_all_E_GPU_ready_lower_bound": proven,
            "visual_rows_before_all_E_GPU_ready_lower_bound": visual_overlap(spans, proven)}, ids


def get_ipc_records(raw):
    files = {role: sorted(raw.glob(f"ec_cuda_ipc_{role}_*.jsonl")) for role in ("prefill", "decode")}
    records = {role: sorted((row for source in paths for row in load_lines(source)), key=lambda item: item["begin"])
               for role, paths in files.items()}
    return [path for paths in files.values() for path in paths], records


def reduce_ipc_witness(record, first_token, role, spans, ipc, expected_ids):
    source, records = ipc
    if not source:
        return None
    start = record["recipient_submit_ns"] / 1e9
    finish = first_token / 1e9
    selected = [row for row in records[role] if start <= row["begin"] <= row["end"] <= finish]
    parts = [part for row in selected for part in row["clone_shapes"]]
    if not parts:
        return {"complete": False, "reason": "NO_IPC_ROWS_IN_SERIAL_WINDOW"}
    ids = [part["key"] for part in parts]
    require(len(ids) == len(set(ids)) == len(spans)
            and (expected_ids is None or ids == expected_ids), "Live transport image identity changed")
    require(all(row["keys"] == [item["key"] for item in row["clone_shapes"]]
                and row["logical_bytes"] == sum(item["logical_bytes"] for item in row["clone_shapes"])
                and row["clone_complete_before_ACK"] for row in selected), "IPC transport receipts differ")
    require(all(part["first_source_row"] == 0 and part["clone_shape"] == [right-left,3584]
                for part,(left,right) in zip(parts,spans,strict=True)), "Unexpected live IPC clone shape")
    prefix = []
    images = 0
    for row in selected:
        if images == record["prefix_images"]:
            break
        prefix.append(row)
        images += row["images"]
    require(images == record["prefix_images"], "IPC group straddled the frozen image boundary")
    return {"complete": True, "joined_exact_E_image_ids": expected_ids is not None,
            "received_images": len(ids), "prefix_images": images, "prefix_batches": len(prefix),
            "prefix_logical_bytes": sum(row["logical_bytes"] for row in prefix),
            "prefix_export_RPC_ms": sum(row["export_RPC_ms"] for row in prefix),
            "prefix_import_clone_sync_host_ms": sum(row["import_clone_sync_host_ms"] for row in prefix),
            "prefix_completion_ACK_host_ms": sum(row["completion_ACK_host_ms"] for row in prefix),
            "prefix_copy_first_begin_from_submit_ms": (prefix[0]["begin"] - start) * 1000,
            "prefix_copy_last_end_from_submit_ms": (prefix[-1]["end"] - start) * 1000}


def reduce_trial(record, events, fixture, spans, output, observer, ipc):
    role = "prefill" if record["arm"].endswith("_p") else "decode"
    require(record["key"] == digest(record["tag"].encode()), "Unverified request key")
    require(record["fixture"] == fixture["case_id"] and record["cut"] == fixture["cut_tokens"]
            and record["prefix_images"] == fixture["prefix_images"]
            and record["prompt_tokens"] == fixture["total_tokens"]
            and record["published_items"] == fixture["total_images"], "Frozen fixture geometry changed")
    start, prefix, all_observed, ack = (record[k] for k in ("start_ns", "prefix_published_ns",
                                       "all_published_ns", "e_ack_ns"))
    lower = record.get("all_not_yet_published_ns")
    submit, d_submit, p_done = (record[k] for k in ("recipient_submit_ns", "d_submit_ns", "p_done_ns"))
    require(start <= prefix <= all_observed and start <= ack and prefix <= submit <= d_submit,
            "Invalid image publication or submission timestamps")
    require(lower is None or start <= lower <= all_observed, "Invalid lower publication witness")
    if record["arm"].startswith("late"):
        require(submit >= ack, "A late control submitted before E HTTP ACK")
    if role == "prefill":
        require(p_done is not None and submit <= p_done <= d_submit and all_observed <= d_submit,
                "P→D full-KV route chronology changed")
    else:
        require(p_done is None, "D-local arm unexpectedly used P")
    trace = reduce_trace(events, record, role)
    if role == "prefill":
        require(trace[-1][1] <= p_done, "P returned before complete LM retirement")

    ids, token_gaps = record["output_token_ids"], record["ITL_ms"]
    ttft, tpot = record["TTFT_ms"], record["TPOT_ms"]
    require(len(ids) == output and len(token_gaps) == output - 1 and all(type(v) is int for v in ids)
            and all(math.isfinite(v) and v >= 0 for v in [ttft, tpot, *token_gaps])
            and abs(tpot - sum(token_gaps) / (output - 1)) < .001,
            "Invalid explicit output-token and timing contract")
    first_token = start + round(ttft * 1e6)
    require(first_token + 1000 >= d_submit, "Impossible first-token chronology")
    if role == "decode":
        require(first_token + 1000 >= trace[-1][1], "First output precedes D prompt retirement")

    ack_rows = retired_before(trace, ack)
    observation_rows = retired_before(trace, all_observed)
    proven_rows = retired_before(trace, lower)
    image_witness, identity = reduce_image_witness(record, trace, spans, observer)
    ipc_witness = reduce_ipc_witness(record, first_token, role, spans, ipc, identity)
    if record["arm"].startswith("late"):
        require(ack_rows == 0, "Late control retired rows before E HTTP ACK")
    return {
        "fixture": record["fixture"], "arm": record["arm"], "repeat": record["repeat"], "tag": record["tag"],
        "cut": record["cut"], "prompt_rows": record["prompt_tokens"],
        "prefix_visual_rows": visual_overlap(spans, record["cut"]),
        "recipient_chunk_rows": [event["scheduled_rows"] for event in events],
        "retired_rows_before_E_ACK": ack_rows,
        "visual_rows_before_E_ACK": visual_overlap(spans, ack_rows),
        "retired_rows_before_all_descriptor_observation": observation_rows,
        "visual_rows_before_all_descriptor_observation": visual_overlap(spans, observation_rows),
        "retired_rows_before_last_missing_descriptor_observation": proven_rows,
        "visual_rows_before_last_missing_descriptor_observation": (
            visual_overlap(spans, proven_rows) if proven_rows is not None else None),
        "E_HTTP_elapsed_ms": ms(ack, start),
        "prefix_descriptor_to_all_observation_ms": ms(all_observed, prefix),
        "prefix_descriptor_to_E_ACK_ms": ms(ack, prefix),
        "recipient_submission_lead_over_E_ACK_ms": ms(ack, submit),
        "first_retirement_lead_over_E_ACK_ms": ms(ack, trace[0][1]),
        "prefix_complete_lead_over_E_ACK_ms": ms(ack, next(t for rows, t in trace if rows == record["cut"])),
        "P_completion_to_D_submit_ms": (ms(d_submit, p_done) if p_done is not None else None),
        "P_HTTP_elapsed_ms": (ms(p_done, submit) if p_done is not None else None),
        "kv_route": "full_P_to_D" if p_done is not None else "D_local",
        "D_submit_to_first_token_ms": ms(first_token, d_submit),
        "TTFT_ms": ttft, "TPOT_ms": tpot, "ITL_p99_ms": percentile(token_gaps, .99),
        "ITL_max_ms": max(token_gaps), "decode_absolute_pass": tpot <= 33 and max(token_gaps) <= 34,
        "prompt_token_ids_SHA256": record["prompt_token_ids_SHA256"],
        "output_token_ids_SHA256": ids_digest(ids),
        "observed_text_SHA256": record.get("normalized_response_text_SHA256",
                                           record.get("response_text_SHA256", record.get("output_text_SHA256"))),
        "E_image_ready_observer": image_witness,
        "IPC_host_enclosure": ipc_witness,
    }


def make_pair(group):
    lp, ep, ld, ed = (group[arm] for arm in ARMS)
    base = {"fixture": lp["fixture"], "repeat": lp["repeat"], "cut": lp["cut"]}
    for label, metric in (("TTFT", "TTFT_ms"), ("TPOT", "TPOT_ms"), ("ITL_p99", "ITL_p99_ms")):
        gain_p, gain_d = lp[metric] - ep[metric], ld[metric] - ed[metric]
        base.update({f"{label}_P_late_minus_early_ms": gain_p,
                     f"{label}_D_late_minus_early_ms": gain_d,
                     f"{label}_DiD_D_early_gain_minus_P_early_gain_ms": gain_d - gain_p})
    base.update(late_P_minus_late_D_TTFT_ms=lp["TTFT_ms"] - ld["TTFT_ms"],
                early_P_minus_early_D_TTFT_ms=ep["TTFT_ms"] - ed["TTFT_ms"],
                P_matched_recipient_chunk_shapes=lp["recipient_chunk_rows"] == ep["recipient_chunk_rows"],
                D_matched_recipient_chunk_shapes=ld["recipient_chunk_rows"] == ed["recipient_chunk_rows"],
                early_P_D_recipient_chunk_shapes=ep["recipient_chunk_rows"] == ed["recipient_chunk_rows"],
                four_arm_prompt_identity=len({r["prompt_token_ids_SHA256"] for r in group.values()}) == 1,
                four_arm_output_identity=len({r["output_token_ids_SHA256"] for r in group.values()}) == 1,
                P_output_identity=lp["output_token_ids_SHA256"] == ep["output_token_ids_SHA256"],
                D_output_identity=ld["output_token_ids_SHA256"] == ed["output_token_ids_SHA256"])
    text = [r["observed_text_SHA256"] for r in group.values()]
    base["four_arm_observed_text_identity"] = len(set(text)) == 1 if all(text) else None
    require(base["four_arm_prompt_identity"], "Original prompt digest differs across paired arms")
    return base


def pair_summary(pairs):
    if not pairs:
        return {"count": 0}
    keys = [key for key in pairs[0] if key.endswith("_ms")]
    return {"count": len(pairs),
            "medians_ms": {key: median(row[key] for row in pairs) for key in keys},
            "P_matching_shapes": sum(row["P_matched_recipient_chunk_shapes"] for row in pairs),
            "D_matching_shapes": sum(row["D_matched_recipient_chunk_shapes"] for row in pairs),
            "P_vs_D_early_matching_shapes": sum(row["early_P_D_recipient_chunk_shapes"] for row in pairs),
            "four_arm_output_identity": sum(row["four_arm_output_identity"] for row in pairs),
            "P_output_identity": sum(row["P_output_identity"] for row in pairs),
            "D_output_identity": sum(row["D_output_identity"] for row in pairs),
            "four_arm_observed_text_pairs": sum(row["four_arm_observed_text_identity"] is not None for row in pairs),
            "four_arm_observed_text_identity": sum(row["four_arm_observed_text_identity"] is True for row in pairs)}


def summarize(run, cases, plan, allow_partial):
    fixtures = json.loads(cases.read_bytes())["cases"]
    config = json.loads(plan.read_bytes())
    count = config["readiness"].get("repetitions", 3)
    output = config["cases"][0]["output"]
    require(bool(fixtures) and output == 16 and count > 0,
            "Plan differs from the fixed serial trial")
    cases_by_id = {f["case_id"]: f for f in fixtures}
    require(len(cases_by_id) == len(fixtures), "Duplicated fixture ID")
    spans = {k: fixture_spans(f) for k, f in cases_by_id.items()}
    from .readiness_followup_cases import read_original

    original_prompt_digests = {k: ids_digest(read_original(f["request"])[0].tolist())
                               for k, f in cases_by_id.items() if "request" in f}
    raw = run / "raw"
    log = raw / "ready_trials.jsonl"
    trace_files = sorted(raw.glob("ready_retired_*.jsonl"))
    observer = get_image_witness(raw)
    ipc = get_ipc_records(raw)
    events = defaultdict(list)
    for path in trace_files:
        for row in load_lines(path):
            events[row["key"]].append(row)
    rows = list(load_lines(log))
    expected = []
    for rep in range(count):
        for index, fixture in enumerate(fixtures):
            shift = (rep + index) % len(ARMS)
            expected.extend((f"ready_{rep}_{index}_{arm}", rep, fixture["case_id"], arm)
                            for arm in ARMS[shift:] + ARMS[:shift])
    observed = [(r["tag"], r["repeat"], r["fixture"], r["arm"]) for r in rows]
    require(observed == expected[:len(observed)], "Trial ordering or paired selection differs from frozen plan")
    failures, trials = [], []
    group = defaultdict(dict)
    gaps_by_arm = defaultdict(list)
    for row in rows:
        key = (row["repeat"], row["fixture"])
        if row.get("error_type"):
            failures.append({k: row[k] for k in ("tag", "fixture", "repeat", "arm", "error_type")})
            continue
        trial = reduce_trial(row, events.get(row["key"], []), cases_by_id[row["fixture"]],
                             spans[row["fixture"]], output, observer, ipc)
        if original := original_prompt_digests.get(row["fixture"]):
            require(original == trial["prompt_token_ids_SHA256"], "Live prompt differs from frozen processor token IDs")
        group[key][row["arm"]] = trial
        trials.append(trial)
        gaps_by_arm[row["arm"]].extend(row["ITL_ms"])
    complete = len(rows) == len(expected) and not failures
    require(allow_partial or complete, "Incomplete physical epoch. Use --allow-partial to surface failed/missing arms.")
    pairs = [make_pair(result) for _, result in sorted(group.items()) if set(result) == set(ARMS)]
    primes = []
    if config["readiness"].get("prime"):
        prime_file = raw / "ready_prime.jsonl"
        if prime_file.exists():
            primes = list(load_lines(prime_file))
        expected_primes = [(f"ready_prime_{i}_{arm}", fixture["case_id"], arm)
                           for i, fixture in enumerate(fixtures) for arm in ("late_p", "late_d")]
        require([(r["tag"], r["fixture"], r["arm"]) for r in primes] == expected_primes[:len(primes)]
                and (allow_partial or len(primes) == len(expected_primes)), "Priming schedule differs from the plan")
    evidence_sources = [cases, plan, log, *trace_files, *observer[0], *ipc[0]]
    if primes:
        evidence_sources.append(prime_file)
    aggregates = {}
    for arm in ARMS:
        items = [r for r in trials if r["arm"] == arm]
        if items:
            physical = [r["E_image_ready_observer"] for r in items if r["E_image_ready_observer"] is not None
                        and r["E_image_ready_observer"]["complete"]]
            ipc_enclosures = [r["IPC_host_enclosure"]["prefix_import_clone_sync_host_ms"] for r in items
                              if r["IPC_host_enclosure"] is not None and r["IPC_host_enclosure"]["complete"]]
            aggregates[arm] = {"complete": len(items),
                               "TTFT_p50_ms": median(r["TTFT_ms"] for r in items),
                               "TTFT_p95_ms": percentile((r["TTFT_ms"] for r in items), .95),
                               "TPOT_p95_ms": percentile((r["TPOT_ms"] for r in items), .95),
                               "ITL_p99_ms": percentile(gaps_by_arm[arm], .99),
                               "request_ITL_p99_p50_ms": median(r["ITL_p99_ms"] for r in items),
                               "ITL_max_ms": max(r["ITL_max_ms"] for r in items),
                               "D_submit_to_first_p50_ms": median(r["D_submit_to_first_token_ms"] for r in items),
                               "absolute_decode_pass": sum(r["decode_absolute_pass"] for r in items),
                               "any_visual_rows_before_E_ACK": sum(r["visual_rows_before_E_ACK"] > 0 for r in items),
                               "visual_rows_before_E_ACK": sum(r["visual_rows_before_E_ACK"] for r in items),
                               "any_visual_rows_before_all_descriptor_observation": sum(
                                   r["visual_rows_before_all_descriptor_observation"] > 0 for r in items),
                               "proven_before_all_available_rows": sum(
                                   r["visual_rows_before_last_missing_descriptor_observation"] or 0 for r in items),
                               "trials_with_lower_publication_witness": sum(
                                   r["visual_rows_before_last_missing_descriptor_observation"] is not None for r in items),
                               "E_event_complete_joins": len(physical),
                               "positive_conservative_E_GPU_lead_trials": sum(
                                   r["conservative_E_GPU_prefix_lead_ms"] > 0 for r in physical),
                               "visual_rows_before_all_E_GPU_ready_lower_bound": sum(
                                   r["visual_rows_before_all_E_GPU_ready_lower_bound"] for r in physical),
                               "IPC_prefix_import_clone_sync_host_p50_ms": median(ipc_enclosures) if ipc_enclosures else None}
    return {"evidence": "MEASURED_PHYSICAL", "scope": SCOPE, "expected_arms": len(expected),
            "attempted_arms": len(rows), "completed_arms": len(trials), "verified_complete": complete,
            "planned_repeats_per_fixture": count, "fixed_output_tokens": output,
            "excluded_priming_arms": len(primes),
            "complete_matched_quartets": len(pairs), "source_SHA256": {
                p.name: digest(p.read_bytes()) for p in evidence_sources},
            "exact_original_prompt_matches": sum(r["fixture"] in original_prompt_digests for r in trials),
            "paired_summary": {
                "all": pair_summary(pairs),
                "first_repeat": pair_summary([r for r in pairs if r["repeat"] == 0]),
                "subsequent_repeats": pair_summary([r for r in pairs if r["repeat"] > 0]),
                "by_fixture": {case_id: pair_summary([r for r in pairs if r["fixture"] == case_id])
                               for case_id in cases_by_id},
            },
            "interpretation": {
                "publication": "First positive all-descriptor poll is an upper observation bound. A retirement before it "
                    "may have followed true publication. A retirement before the last probe that actually saw a "
                    "missing final descriptor is conservative proof for that run when the probe exists.",
                "early_before_ACK": "Actual scheduler completion before E's full HTTP response. Includes host ACK latency.",
                "difference_in_differences": "(late D − early D) − (late P − early P) per matched fixture and repeat. "
                    "All comparisons include legitimate serial overhead. Raw P/D contrast also changes the full KV route.",
                "full_KV": "P completion→D submit is host/orchestration gap. D submit→first token on P includes "
                    "NIXL and native D processing. Individual NIXL DMA duration is outside this reduction.",
                "ipc_host": "Prefix clone/sync uses measured serial recipient host intervals. These can include "
                    "producer readiness waits, CUDA IPC mapping, clone enqueue, synchronization and CPU scheduling. "
                    "They are not isolated memory copy or bandwidth measurements.",
                "E_event": "Each successful producer CUDA event query contributes a readiness interval. "
                    "All-image lower minus prefix upper bounds the lead from below. All-image upper minus "
                    "prefix lower bounds it from above. "
                    "Observed event diagnostics do not qualify profiler overhead.",
                "decode": "Per-request ITL from emitted token timestamps. Serial trials cannot establish protection of "
                    "another active Decode request. Request-level p99 is distinct from a pooled-token p99.",
                "repeat_strata": "All completed quartets are primary. First and subsequent repeats are descriptive "
                    "cold-start sensitivity and do not represent independent server starts.",
            }, "failures": failures, "not_run": [x[0] for x in expected[len(rows):]],
            "arms": aggregates, "trials": trials, "pairs": pairs}


def draw(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(dict.fromkeys(r["fixture"] for r in data["trials"]))
    paired = defaultdict(list)
    for pair in data["pairs"]:
        paired[pair["fixture"]].append(pair)
    observed_gpu = any((r["E_image_ready_observer"] or {}).get("complete") for r in data["trials"])
    figure, axes = plt.subplots(1, 3 if observed_gpu else 2, figsize=(15.4 if observed_gpu else 11, 4.5),
                                layout="constrained")
    for i, (role, color) in enumerate((("P", "#31668f"), ("D", "#ad5e35"))):
        role_key = role.lower()
        offset = -.12 if i == 0 else .12
        xx, gain, ack, observed = [], [], [], []
        for index, case in enumerate(order):
            arm = [r for r in data["trials"] if r["fixture"] == case and r["arm"] == f"early_{role_key}"]
            if arm:
                used = [r["visual_rows_before_E_ACK"] for r in arm]
                noticed = [r["visual_rows_before_all_descriptor_observation"] for r in arm]
                xx.append(index + offset)
                ack.append(median(used))
                observed.append(median(noticed))
                if observed_gpu:
                    measured = [r for r in arm if (r["E_image_ready_observer"] or {}).get("complete")
                                and (r["IPC_host_enclosure"] or {}).get("complete")]
                    if measured:
                        x = index + offset
                        shown = (median(r["prefix_descriptor_to_all_observation_ms"] for r in measured),
                                 median(r["IPC_host_enclosure"]["prefix_import_clone_sync_host_ms"] for r in measured))
                        bounds = [median(r["E_image_ready_observer"][key] for r in measured)
                                  for key in ("conservative_E_GPU_prefix_lead_ms", "E_GPU_prefix_lead_upper_bound_ms")]
                        with_label = index == 0 and role == "P"
                        axes[2].scatter(x, shown[0], marker="s", facecolors="none", edgecolors=color,
                                        label="All−prefix descriptor polls" if with_label else None)
                        axes[2].scatter(x, shown[1], marker="^", color=color,
                                        label="Prefix IPC clone/sync host" if with_label else None)
                        axes[2].vlines(x, *bounds, lw=2.5, color=color,
                                       label="E CUDA event lead bounds" if with_label else None)
            for result in paired[case]:
                gain.append((index + offset, result[f"TTFT_{role}_late_minus_early_ms"]))
        if gain:
            axes[0].scatter(*zip(*gain), color=color, alpha=.7, label=f"{role} recipient")
        axes[1].scatter(xx, ack, color=color, marker="o", label=f"{role}, before E HTTP ACK")
        axes[1].scatter(xx, observed, facecolors="none", edgecolors=color, marker="s",
                        label=f"{role}, before all-descriptor observation")
    axes[0].axhline(0, color=".45", lw=1)
    axes[0].set(ylabel="Paired TTFT gain, late − early (ms)", title="Matched recipient comparisons")
    if any(abs(r[f"TTFT_{role}_late_minus_early_ms"]) > 1000 for r in data["pairs"] for role in ("P", "D")):
        axes[0].set_yscale("symlog", linthresh=100)
        axes[0].set_ylabel("Paired TTFT gain, late − early (ms, symlog)")
    targets = {r["fixture"]: r["prefix_visual_rows"] for r in data["trials"]}
    axes[1].scatter(range(len(order)), [targets[case] for case in order], marker="_", s=138, color=".52",
                    label="Visual rows in prescribed prefix")
    max_rows = max(max(targets.values()), *(r["visual_rows_before_all_descriptor_observation"] for r in data["trials"]))
    axes[1].set_ylim(-.05 * max_rows, 1.05 * max_rows)
    axes[1].set(ylabel="Median visual rows retired", title="Actual prefix completion timing")
    if observed_gpu:
        axes[2].set(ylabel="Sample median (ms)", title="Descriptor lead and IPC enclosure")
    for axis in axes:
        axis.set_xticks(range(len(order)), labels=order, rotation=30, ha="right")
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(frameon=False, fontsize=8, loc="best")
        axis.grid(axis="y", alpha=.2)
    figure.suptitle("Original image-complete causal prefixes on H200 · selected serial probes")
    figure.savefig(out)
    plt.close(figure)
    out.write_text("\n".join(line.rstrip() for line in out.read_text().splitlines()) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--figure", action="store_true")
    args = parser.parse_args()
    data = summarize(args.run, args.cases, args.plan, args.allow_partial)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "readiness_followup_summary.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    if rows := data["pairs"]:
        with (args.out / "readiness_followup_pairs.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    if args.figure and data["trials"]:
        figures = args.out / "figures"
        figures.mkdir(exist_ok=True)
        draw(data, figures / "readiness_followup_paired.svg")
    print(f"Completed {data['completed_arms']}/{data['expected_arms']} probes, "
          f"{data['complete_matched_quartets']} matched quartets; complete={data['verified_complete']}")


if __name__ == "__main__":
    main()
