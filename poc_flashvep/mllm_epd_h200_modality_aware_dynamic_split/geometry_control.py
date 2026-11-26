"""Correctness-only Native and late N8 comparison with byte-verified E outputs."""

import asyncio
from collections import Counter
from contextlib import ExitStack
import hashlib
from pathlib import Path
import time

from .correctness import (
    ROOT, ROLES, _request, boundary_variant, contiguous, digest, ownership_empty, read, require,
    rows_in, rpc, single, write,
)


async def issue_cohort(rows, url, tag, cuts, native, natural=False, lead=False):
    require(not (ROOT / "raw" / f"{tag}_requests.jsonl").exists(), "Geometry run ID already exists")
    start = time.monotonic()
    requests = []
    for index, (row, cut) in enumerate(zip(rows, cuts, strict=True)):
        if native or (lead and index != 0):
            choice = "native"
        else:
            choice = "split" if index % 2 == 0 else "continue"
        requests.append(_request(row, url, tag, cut, choice, index, start + index * .005,
                                 e_cohort=tag, natural=natural, e_lead=(index == 0) if lead else None))
    return await asyncio.gather(*requests)


def capture_cohort(rows, url, tag, cuts, native, natural=False, lead=False):
    contracts = {
        f"{tag}_{index:04d}": {
            "prefix_tokens": cut, "full_prompt_tokens": row["total_input_tokens"],
            "prefill_full_prompt": True,
        }
        for index, (row, cut) in enumerate(zip(rows, cuts, strict=True))
    }
    opened = []
    try:
        for port, role in ROLES:
            rpc(port, "batch_prefix/begin", {
                "name": f"{tag}_{role}", "contracts": contracts,
                "allow_decode": role == "D", "capture_full_prompt": True,
            })
            opened.append((port, role))
        output = asyncio.run(issue_cohort(rows, url, tag, cuts, native, natural=natural, lead=lead))
    finally:
        for port, role in reversed(opened):
            closed = rpc(port, "batch_prefix/end", {})
            write(ROOT / "state" / f"{tag}_{role}_probe_close.json", closed)
    ownership_empty(tag, "cuda_ipc")
    captures = {
        role: read(ROOT / "raw" / "batch_prefix" / f"{tag}_{role}" / "manifest.json")
        for _, role in ROLES
    }
    return output, captures


def e_signatures(ec_path, base):
    key = hashlib.sha256(base.encode()).hexdigest()
    manifest = read(ec_path / "processed_prompts" / key / "manifest.json")
    return [
        read(ec_path / identifier / "cuda_ipc_ready.json")["original_embedding_signature"]
        for identifier in manifest["image_hashes"]
    ]


def query_receipt(captures, base, p_end, d_begin, full, cut=None):
    rows = {
        role: [query for event in record["events"] for query in event["requests"]
               if query["base"] == base and query["phase"] == "prefill"]
        for role, record in captures.items()
    }
    return {
        "P": contiguous(rows["P"], 0, p_end, cut),
        "D": contiguous(rows["D"], d_begin, full),
    }


def compare_state(paths, cut, length, native_prefix, late_prefix):
    import torch
    from safetensors import safe_open

    from ..mllm_epd_prefill_heavy_token_split.numerical import embedding_metrics

    def update(stat, actual, expected):
        exact = torch.equal(actual, expected)
        if exact:
            relative_l2, cosine, finite = 0.0, 1.0, True
        else:
            metric = embedding_metrics(actual, expected)
            relative_l2, cosine, finite = metric["relative_L2"], metric["cosine"], metric["finite"]
        good = finite and relative_l2 <= .02 and cosine >= .999
        stat["layer_segments"] += 1
        stat["strict_equal"] += exact
        stat["within_tolerance"] += good
        stat["max_relative_L2"] = max(stat["max_relative_L2"], relative_l2)
        stat["min_cosine"] = min(stat["min_cosine"], cosine)

    with ExitStack() as stack:
        files = {}
        for name, receipt in paths.items():
            path = Path(receipt.get("path") or receipt["tensor_file"])
            expect = receipt.get("SHA256") or receipt["tensor_SHA256"]
            require(digest(path) == expect, f"{name} final-prompt receipt failed SHA verification")
            files[name] = stack.enter_context(safe_open(path, framework="pt", device="cpu"))
        fields = {
            name: {
                key: torch.equal(tensor.get_tensor(key), files["teacher"].get_tensor(key))
                for key in ("prompt_token_ids", "full_prompt_mrope")
            }
            for name, tensor in files.items() if name != "teacher"
        }
        comparisons = {
            "native_P_vs_serial": ("native_P", "teacher"),
            "native_D_vs_serial": ("native_D", "teacher"),
            "late_D_vs_serial": ("late_D", "teacher"),
            "late_D_vs_native_D": ("late_D", "native_D"),
        }
        if "late_P" in files:
            comparisons["late_P_vs_serial"] = ("late_P", "teacher")
            comparisons["late_P_vs_native_P"] = ("late_P", "native_P")
        tolerances = {
            label: {side: {"layer_segments": 0, "strict_equal": 0, "within_tolerance": 0,
                           "max_relative_L2": 0., "min_cosine": 1.} for side in ("prefix", "suffix")}
            for label in comparisons
        }
        imports = {"native": [], "late_fallback": []}
        for layer in range(28):
            key = f"prompt_KV_{layer}"
            states = {name: tensor.get_tensor(key) for name, tensor in files.items()}
            for name, (a, b) in comparisons.items():
                update(tolerances[name]["prefix"], states[a][:cut], states[b][:cut])
                update(tolerances[name]["suffix"], states[a][cut:length], states[b][cut:length])
            imports["native"].append(torch.equal(states["native_P"][:native_prefix],
                                                 states["native_D"][:native_prefix]))
            if "late_P" in states:
                imports["late_fallback"].append(torch.equal(states["late_P"][:late_prefix],
                                                            states["late_D"][:late_prefix]))
        return fields, tolerances, imports


def tolerance_pass(tolerances):
    return all(row["layer_segments"] == row["within_tolerance"] == 28 for row in tolerances.values())


def known_natural_cuts(rows, native_tag, capture, targets):
    """Choose at most one exact target per request using only the earlier Native probe."""
    chosen, source = [], []
    p_events = capture["P"]["events"]
    for index, (row, target) in enumerate(zip(rows, targets, strict=True)):
        base = f"{native_tag}_{index:04d}"
        length = row["total_input_tokens"]
        candidates = {
            item["start"] + item["query_tokens"] for event in p_events for item in event["requests"]
            if item["base"] == base and item["phase"] == "prefill"
        }
        qualified = [prefix for prefix in candidates if .05 <= (length - prefix) / length <= .30]
        if qualified:
            chosen.append(min(qualified, key=lambda prefix: (abs(prefix - target), prefix)))
            source.append("OBSERVED_NATIVE_CHUNK")
        else:
            chosen.append(target)
            source.append("NO_ELIGIBLE_NATIVE_CHUNK_ORIGINAL_TARGET")
    return chosen, source


def shared_decode_steps(capture, lead_base):
    lead_batches = []
    for event in capture["D"]["events"]:
        decoding = {item["base"] for item in event["requests"] if item["phase"] == "decode"}
        if lead_base in decoding and len(decoding) > 1:
            lead_batches.append(len(decoding))
    return dict(shared_steps=len(lead_batches), max_decode_batch=max(lead_batches, default=0))


def run_geometry_control(rows, tag, url, teacher_url, ec_path, *, ratio=.10, natural=False,
                         targets=None, use_native_cuts=True, lead=False):
    """Verify serialized original E inputs, then burst P/D with both policies."""
    require(len(rows) == 8 and len({row["request_id"] for row in rows}) == 8, "Use eight unique frozen rows")
    require(ratio in (.05, .10, .20, .30), "Use a frozen nominal cutoff")
    variant = boundary_variant(natural)
    nominal = [round(row["total_input_tokens"] * (1 - ratio)) for row in rows]
    cuts = nominal if targets is None else [
        value if value is not None else default for value, default in zip(targets, nominal, strict=True)
    ]
    if lead:
        require(variant == "natural_hold" and targets is not None and targets[0] is not None,
                "Specify the previously reached lead boundary for the natural-hold control")
    require(len(cuts) == len(rows) and all(0 < cut < row["total_input_tokens"]
            for row, cut in zip(rows, cuts, strict=True)), "Invalid geometry target cuts")
    target_sources = ["FROZEN_NOMINAL_TARGET"] * len(rows)
    if lead:
        target_sources = ["PREDECLARED_SINGLETON_CHUNK"] + ["NATIVE_FOLLOWER"] * (len(rows) - 1)
    ec_path = Path(ec_path)
    tags = {role: f"{tag}_{role}" for role in ("teacher", "native", "late")}
    if variant:
        native, n_capture = capture_cohort(rows, url, tags["native"], cuts, native=True, lead=lead)
        if use_native_cuts and not lead:
            cuts, target_sources = known_natural_cuts(rows, tags["native"], n_capture, cuts)
    baseline = [
        single.capture(row, f"{tags['teacher']}_{index}", teacher_url, [(18841, "P")], teacher_prefix=cut)
        for index, (row, cut) in enumerate(zip(rows, cuts, strict=True))
    ]
    if not variant:
        native, n_capture = capture_cohort(rows, url, tags["native"], cuts, native=True)
    late, d_capture = capture_cohort(rows, url, tags["late"], cuts, native=False, natural=natural, lead=lead)
    known = {f"{tags['late']}_{index:04d}" for index in range(len(rows))}
    offers = [
        item for path in (ROOT / "raw").glob("late_prefill_cut_*.jsonl") for item in rows_in(path)
        if item["request_id"].startswith(tags["native"] + "_") or item["request_id"] in known
    ]
    by_offer = {item["request_id"]: item for item in offers}
    require(len(offers) == len(by_offer) and set(by_offer) <= known,
            "Native must have zero offers and each late request may offer once")
    require(variant or set(by_offer) == known, "Each fixed-cut late request must reach its clamped boundary")
    skips = [
        item for path in (ROOT / "raw").glob("late_prefill_natural_skips_*.jsonl") for item in rows_in(path)
        if item["request_id"] in known
    ] if variant else []
    cases = []
    for index, (row, cut) in enumerate(zip(rows, cuts, strict=True)):
        length = row["total_input_tokens"]
        base = {role: f"{value}_{index:04d}" for role, value in tags.items()}
        base["teacher"] = f"{tags['teacher']}_{index}_0000"
        offer = by_offer.get(base["late"])
        if offer is not None:
            require(offer["prefix_tokens"] == cut and bool(offer.get("natural")) == bool(variant)
                    and bool(offer.get("hold")) == (variant == "natural_hold"),
                    "Observed a different P boundary or policy from the frozen cut")
        split = bool(offer and offer["accepted"])
        e_values = {role: e_signatures(ec_path, runtime_id) for role, runtime_id in base.items()}
        e_same = len(e_values["teacher"]) == row["image_count"] > 0 and all(
            e_values[role] == e_values["teacher"] for role in ("native", "late"))
        native_rows = query_receipt(n_capture, base["native"], length, length - 1, length)
        p_end, d_begin = (cut, cut) if split else (length, length - 1)
        late_rows = query_receipt(d_capture, base["late"], p_end, d_begin, length, cut if offer else None)
        files = {"teacher": read(ROOT / "raw" / "llm_reference" /
                 f"{tags['teacher']}_{index}_P" / "manifest.json")}
        for arm, capture in (("native", n_capture), ("late", d_capture)):
            for _, role in ROLES:
                if arm == "late" and split and role == "P":
                    continue
                files[f"{arm}_{role}"] = capture[role]["full_prompt_tensor_files"][base[arm]]
        fields, numbers, imports = compare_state(files, cut, length, length - 1, d_begin)
        if split:
            p, d = (d_capture[role]["snapshots"][base["late"]] for role in ("P", "D"))
            imports["late_suffix"] = [a == b for a, b in zip(p["layers"], d["layers"], strict=True)]
            fields["late_split_prefix"] = {
                "prompt_token_ids": p["prefix_ids"] == d["prefix_ids"],
                "full_prompt_mrope": p["prefix_mrope"] == d["prefix_mrope"],
                "full_input_retained": p["prompt_tokens"] == length,
            }
        paired_greedy = native[index]["token_ids"] == late[index]["token_ids"]
        native_serial = all(tolerance_pass(numbers[f"native_{role}_vs_serial"]) for role in ("P", "D"))
        late_serial = tolerance_pass(numbers["late_D_vs_serial"])
        numerically_matched = tolerance_pass(numbers["late_D_vs_native_D"])
        orig = all(all(check.values()) for check in fields.values())
        transfers = all(all(checks) and len(checks) == 28 for checks in imports.values() if checks)
        coverage = all((*native_rows.values(), *late_rows.values()))
        no_recompute = coverage and (offer is None or offer["prefill_recomputations"] == 0)
        state_passed = e_same and orig and transfers and numerically_matched and no_recompute
        cases.append({
            "request_id": row["request_id"], "prefix_tokens": cut, "input_length": length, "offer": offer,
            "cut_source": target_sources[index], "actual_offloaded_tokens": length - cut if split else 0,
            "identical_serial_E": e_same, "encoder_signatures": e_values, "original_fields": fields,
            "native_rows": native_rows, "late_rows": late_rows, "actual_exact_KV_import": imports,
            "numerical_checks": numbers, "native_vs_serial_tolerance_pass": native_serial,
            "late_vs_serial_tolerance_pass": late_serial, "late_vs_native_tolerance_pass": numerically_matched,
            "late_vs_native_exact_greedy": paired_greedy,
            "native_vs_serial_exact_greedy": native[index]["token_ids"] == baseline[index]["token_ids"],
            "late_vs_serial_exact_greedy": late[index]["token_ids"] == baseline[index]["token_ids"],
            "state_passed": state_passed, "passed": state_passed and paired_greedy,
        })
    hist = {arm: {role: dict(Counter(event["actual_batch"] for event in capture[role]["events"]))
                  for _, role in ROLES}
            for arm, capture in (("native", n_capture), ("late", d_capture))}
    concurrent = all(max(hist[arm][role], default=0) > 1 for arm in hist for _, role in ROLES)
    accepted = sum(offer["accepted"] for offer in offers)
    shared_decode = {arm: shared_decode_steps(capture, f"{tags[arm]}_0000")
                     for arm, capture in (("native", n_capture), ("late", d_capture))} if lead else None
    lead_pass = not lead or (accepted == 1 and cases[0]["actual_offloaded_tokens"] > 0
                            and all(value["shared_steps"] > 0 for value in shared_decode.values()))
    witnessed_mixed = 0 < accepted < len(rows) and concurrent and lead_pass
    passed = witnessed_mixed and all(case["passed"] for case in cases)
    result = {
        "ratio": ratio, "natural_boundary": variant, "lead_boundary_control": lead, "passed": passed,
        "evidence": "CORRECTNESS_PASS" if passed else "DIAGNOSTIC",
        "state_gate_passed": witnessed_mixed and all(case["state_passed"] for case in cases),
        "native_pass": all(case["native_vs_serial_tolerance_pass"] for case in cases),
        "late_serial_pass": all(case["late_vs_serial_tolerance_pass"] for case in cases),
        "verified_same_E": sum(case["identical_serial_E"] for case in cases),
        "accepted": accepted, "continued": len(rows) - accepted, "actual_P_D_concurrency": concurrent,
        "no_offer_continuations": len(rows) - len(offers), "natural_skips": skips,
        "actual_batch_histograms": hist, "lead_shared_decode": shared_decode, "cases": cases,
        "scope": ("Correctness-only serial E lock and post-E cohort barrier. "
                  "Native remains uncut; the late arm evaluates real idle reservation. P/D run continuously. "
                  "The paired tolerance and exact greedy checks use byte-verified E embeddings, "
                  "distinct from each arm's disagreement with its serial teacher. "
                  "Batch histograms record the actual independent scheduler geometry."
                  + (" Both arms admit the lead to P before its seven followers, released at the lead P "
                     "terminal response. The D ledger must show the lead decoding with a Native follower. "
                     "This stagger is a correctness fixture." if lead else "")),
    }
    write(ROOT / "analysis" / f"{tag}_geometry_control.json", result)
    return result
