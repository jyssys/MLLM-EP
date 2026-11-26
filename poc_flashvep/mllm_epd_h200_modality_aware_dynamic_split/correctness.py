"""GPU-synchronized correctness receipts for the late-bound runtime."""

import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import time

from ..mllm_epd_prefill_heavy_token_split import batch_token_correctness_control as batch_check
from ..mllm_epd_prefill_heavy_token_split import native_common_control as native_check
from ..mllm_epd_prefill_heavy_token_split import token_correctness_control as single
from ..mllm_epd_prefill_heavy_token_split.batch_prefix_probe import tensor_signature
from ..mllm_epd_prefill_heavy_token_split.client import one, payload
from ..mllm_epd_prefill_heavy_token_split.common import ROOT, digest, write
from ..mllm_epd_prefill_heavy_token_split.ipc_ec_control import ownership_empty
from ..mllm_epd_prefill_heavy_token_split.llm_probe_control import rpc


ROLES = ((18841, "P"), (18842, "D"))


def read(path):
    return json.loads(path.read_text())


def rows_in(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def require(value, description):
    if not value:
        raise RuntimeError(description)


def boundary_variant(natural):
    if natural is False:
        return None
    if natural is True or natural == "natural":
        return "natural"
    if natural in ("hold", "natural_hold"):
        return "natural_hold"
    raise ValueError("Unknown natural boundary variant")


async def _request(row, url, tag, cut, choice, index, arrival, e_cohort=None, natural=False, e_lead=None):
    import aiohttp

    headers = {
        "x-campaign-late-cut": str(cut),
        "x-campaign-late-nonce": os.environ["MLLM_EPD_RUN_NONCE"],
    }
    if choice in ("continue", "native"):
        headers["x-campaign-late-choice"] = choice
    if e_cohort is not None:
        headers["x-campaign-late-e-cohort"] = e_cohort
    if e_lead is not None:
        require(e_cohort is not None, "The correctness lead requires a cohort")
        headers["x-campaign-late-e-lead"] = "1" if e_lead else "0"
    variant = boundary_variant(natural)
    if variant:
        headers["x-campaign-late-boundary"] = variant
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False, headers=headers) as session:
        record = await one(session, url, row, payload(row, tag, None), arrival, tag, index)
    require(record["status"] == "MEASURED_ONLINE", f"Correctness request failed: {record.get('error')}")
    return record


async def request(row, url, tag, cut, choice, natural=False):
    require(not (ROOT / "raw" / f"{tag}_requests.jsonl").exists(), "Correctness run ID already exists")
    record = await _request(row, url, tag, cut, choice, 0, time.monotonic(), natural=natural)
    ownership_empty(tag, "cuda_ipc")
    return record


def capture_late(row, url, tag, cut, choice, natural=False):
    if choice == "continue":
        operation = "llm_probe"
        probe_args = {"max_logits": 16, "include_decode_inputs": True}
    else:
        operation = "batch_prefix"
        probe_args = {"contracts": {
            f"{tag}_0000": {"prefix_tokens": cut, "full_prompt_tokens": row["total_input_tokens"],
                            "prefill_full_prompt": True},
        }}
    opened = []
    try:
        for port, role in ROLES:
            body = dict(probe_args, name=f"{tag}_{role}")
            if operation == "batch_prefix":
                body.update(allow_decode=role == "D", capture_full_prompt=role == "D")
            rpc(port, f"{operation}/begin", body)
            opened.append((port, role))
        return asyncio.run(request(row, url, tag, cut, choice, natural=natural))
    finally:
        for port, role in reversed(opened):
            write(ROOT / "state" / f"{tag}_{role}_probe_close.json", rpc(port, f"{operation}/end", {}))


def witnessed_offer(tag, cut, choice, natural=False):
    offers = [record for path in (ROOT / "raw").glob("late_prefill_cut_*.jsonl") for record in rows_in(path)
              if record["request_id"] == f"{tag}_0000"]
    variant = boundary_variant(natural)
    if not offers and variant is not None:
        return None
    require(len(offers) == 1, "The correctness request must make exactly one real P decision")
    offer = offers[0]
    require(offer["prefix_tokens"] == cut and offer["accepted"] == (choice == "split"),
            f"Correctness test exercised a different route: {offer}")
    require(bool(offer.get("natural")) == bool(variant)
            and bool(offer.get("hold")) == (variant == "natural_hold"),
            "Wrong P boundary policy in correctness result")
    return offer


def contiguous(events, begin, end, cut=None):
    cursor = begin
    requested_cut_reached = cut is None
    engine_ids = set()
    for event in events:
        start = event["start"]
        if start != cursor or start + event["query_tokens"] > end:
            return False
        cursor = start + event["query_tokens"]
        engine_ids.add(event["request_id"])
        requested_cut_reached |= cursor == cut
    return cursor == end and requested_cut_reached and len(engine_ids) == 1


def check_split(tag, teacher_tag, row, prefix):
    import torch
    from safetensors.torch import load_file

    base = f"{tag}_0000"
    full = row["total_input_tokens"]
    contracts = {base: dict(prefix_tokens=prefix, full_prompt_tokens=full, prefill_full_prompt=True)}
    handoff = batch_check.validate(tag, contracts, include_decode=True)
    teacher_meta = read(ROOT / "raw" / "llm_reference" / f"{teacher_tag}_P" / "manifest.json")
    require(digest(Path(teacher_meta["tensor_file"])) == teacher_meta["tensor_SHA256"], "Teacher digest mismatch")
    teacher = load_file(teacher_meta["tensor_file"])
    aligned_chunks = {item["start"]: item for item in teacher_meta["events"] if item["start"] < full}
    inputs = []
    records = {}
    for _, role in ROLES:
        path = ROOT / "raw" / "batch_prefix" / f"{tag}_{role}" / "manifest.json"
        receipt = read(path)
        records[role] = receipt
        for event in receipt["events"]:
            for query in event["requests"]:
                if query["base"] != base or query["phase"] != "prefill":
                    continue
                reference = aligned_chunks.get(query["start"])
                require(reference is not None and reference["query_tokens"] == query["query_tokens"],
                        "N1 teacher and observed P/D chunk rows differ")
                for field in ("positions", "input_ids", "inputs_embeds"):
                    name = f"{field}_{reference['index']}"
                    expected = tensor_signature(teacher[name]) if name in teacher else None
                    inputs.append(dict(role=role, field=field, start=query["start"],
                                       strict=query.get(field) == expected))
    full_record = records["D"]["full_prompt_tensor_files"][base]
    require(digest(Path(full_record["path"])) == full_record["SHA256"], "D final-prompt digest mismatch")
    destination = load_file(full_record["path"])
    strict_layers = [bool(torch.equal(destination[f"prompt_KV_{i}"], teacher[f"prompt_KV_{i}"]))
                     for i in range(28)]
    field_checks = {
        "full_prompt_ids": tensor_signature(teacher["prompt_token_ids"][:prefix])
                          == records["P"]["snapshots"][base]["prefix_ids"],
        "full_prompt_mrope": tensor_signature(teacher["full_prompt_mrope"][..., :prefix])
                            == records["P"]["snapshots"][base]["prefix_mrope"],
        "p_original_full_input_retained": records["P"]["snapshots"][base]["prompt_tokens"] == full,
    }
    return dict(
        passed=handoff["passed"] and all(item["strict"] for item in inputs)
            and all(strict_layers) and all(field_checks.values()),
        exact_imported_prefix_layers=handoff["checks"][0]["imported_layers_strict"],
        exact_d_final_prompt_layers=strict_layers, exact_chunk_inputs=inputs, original_fields=field_checks,
        actual_batch_histograms=handoff["actual_batch_histograms"],
        teacher_tensor_SHA256=teacher_meta["tensor_SHA256"], d_tensor_SHA256=full_record["SHA256"],
    )


def check_continuation(tag, teacher_tag, row, cut, baseline, observed, offer):
    record = native_check.compare_capture(row, teacher_tag, tag, baseline, observed)
    worker = read(ROOT / "raw" / "llm_reference" / f"{tag}_P" / "manifest.json")
    continued = contiguous(worker["events"], 0, row["total_input_tokens"], cut)
    return dict(
        passed=record["passed"] and continued and offer["prefill_recomputations"] == 0,
        exact_p_final_prompt_layers=[layer["P_teacher_strict"] for layer in record["layers"]],
        exact_imported_prefix_layers=[layer["imported_prefix_strict"] for layer in record["layers"]],
        native_last_row_layer_tolerances=[layer["last_query_tolerance_pass"] for layer in record["layers"]],
        no_repeated_query_rows_same_engine_request=continued,
    )


def run_case(row, tag, cut, url, teacher_url, choice="split", *, natural=False):
    """One serial, same-engine E+P teacher against accepted or rejected late decision."""
    require(choice in ("split", "continue"), "Unknown correctness route")
    require(0 < cut < row["total_input_tokens"], "Correctness cut is outside the original prompt")
    teacher = f"{tag}_teacher"
    baseline = single.capture(row, teacher, teacher_url, [(18841, "P")], teacher_prefix=cut)
    observed = capture_late(row, url, tag, cut, choice, natural=natural)
    offer = witnessed_offer(tag, cut, choice, natural=natural)
    variant = boundary_variant(natural)
    if offer is None:
        skipped = [record for path in (ROOT / "raw").glob("late_prefill_natural_skips_*.jsonl")
                   for record in rows_in(path) if record["request_id"] == f"{tag}_0000"]
        result = {
            "evidence": "DIAGNOSTIC", "passed": False, "case": tag, "request_id": row["request_id"],
            "route": choice, "natural_boundary": variant, "reason": "NATURAL_BOUNDARY_NOT_WITNESSED",
            "requested_prefix_tokens": cut, "full_prompt_tokens": row["total_input_tokens"],
            "native_path_completed": True, "observed_output_tokens": observed["output_tokens"],
            "exact_greedy_tokens": observed["token_ids"] == baseline["token_ids"],
            "natural_skips": skipped,
            "scope": "The original request completed, but no naturally retired candidate was offered.",
        }
        write(ROOT / "analysis" / f"{tag}_late_correctness.json", result)
        return result
    checks = (check_split(tag, teacher, row, cut) if choice == "split" else
              check_continuation(tag, teacher, row, cut, baseline, observed, offer))
    greedy = observed["token_ids"] == baseline["token_ids"]
    passed = checks["passed"] and greedy
    result = {
        "evidence": "CORRECTNESS_PASS" if passed else "DIAGNOSTIC",
        "passed": passed, "case": tag, "request_id": row["request_id"], "route": choice,
        "natural_boundary": variant,
        "prefix_tokens": cut, "full_prompt_tokens": row["total_input_tokens"],
        "exact_greedy_tokens": greedy, "teacher_output_tokens": baseline["output_tokens"],
        "observed_output_tokens": observed["output_tokens"], "offer": offer, "checks": checks,
        "scope": "Single serial GPU state comparison. CPU correctness copies invalidate latency.",
    }
    write(ROOT / "analysis" / f"{tag}_late_correctness.json", result)
    return result


async def request_mixed(rows, url, tag, cuts):
    require(not (ROOT / "raw" / f"{tag}_requests.jsonl").exists(), "Correctness run ID already exists")
    start = time.monotonic()
    return await asyncio.gather(*(
        _request(row, url, tag, cut, "split" if index % 2 == 0 else "continue", index,
                 start + (0.0 if index == 0 else .2 + index * .02))
        for index, (row, cut) in enumerate(zip(rows, cuts, strict=True))
    ))


def run_mixed(rows, tag, url, teacher_url, ratio=.10):
    """N8 mixed original requests with one-cut proposals and explicit continuation controls."""
    from safetensors.torch import load_file

    from ..mllm_epd_prefill_heavy_token_split.numerical import embedding_metrics

    require(len(rows) == 8 and len({row["request_id"] for row in rows}) == 8, "Expected eight unique rows")
    lengths = [row["total_input_tokens"] for row in rows]
    cuts = [round(length * (1 - ratio)) for length in lengths]
    bases = [f"{tag}_{index:04d}" for index in range(len(rows))]
    contracts = {base: {"prefix_tokens": cut, "full_prompt_tokens": length, "prefill_full_prompt": True}
                 for base, cut, length in zip(bases, cuts, lengths, strict=True)}
    teacher = [
        single.capture(row, f"{tag}_teacher_{index}", teacher_url, [(18841, "P")], teacher_prefix=cut)
        for index, (row, cut) in enumerate(zip(rows, cuts, strict=True))
    ]
    opened = []
    try:
        for port, role in ROLES:
            rpc(port, "batch_prefix/begin", dict(name=f"{tag}_{role}", contracts=contracts,
                allow_decode=role == "D", capture_full_prompt=role == "D"))
            opened.append((port, role))
        observed = asyncio.run(request_mixed(rows, url, tag, cuts))
    finally:
        for port, role in reversed(opened):
            write(ROOT / "state" / f"{tag}_{role}_probe_close.json", rpc(port, "batch_prefix/end", {}))
    ownership_empty(tag, "cuda_ipc")

    records = {role: read(ROOT / "raw" / "batch_prefix" / f"{tag}_{role}" / "manifest.json")
               for _, role in ROLES}
    offered = [item for path in (ROOT / "raw").glob("late_prefill_cut_*.jsonl") for item in rows_in(path)
               if item["request_id"] in contracts]
    require(len(offered) == len(bases) and {offer["request_id"] for offer in offered} == set(bases),
            "Expected one real P decision for every mixed request")
    offers = {offer["request_id"]: offer for offer in offered}
    events = {role: {base: [item for event in records[role]["events"] for item in event["requests"]
                            if item["base"] == base and item["phase"] == "prefill"] for base in bases}
              for _, role in ROLES}
    cases = []
    for index, base in enumerate(bases):
        offer = offers[base]
        cut, length = cuts[index], lengths[index]
        split = offer["accepted"]
        prefix_ok = None
        if split:
            a, b = (records[role]["snapshots"][base] for role in ("P", "D"))
            prefix_ok = a["prefix_ids"] == b["prefix_ids"] and a["prefix_mrope"] == b["prefix_mrope"] and (
                a["layers"] == b["layers"] and len(a["layers"]) == 28)
        source = read(ROOT / "raw" / "llm_reference" / f"{tag}_teacher_{index}_P" / "manifest.json")
        final = records["D"]["full_prompt_tensor_files"][base]
        require(digest(Path(source["tensor_file"])) == source["tensor_SHA256"], "Teacher digest mismatch")
        require(digest(Path(final["path"])) == final["SHA256"], "D final-prompt digest mismatch")
        expected, actual = load_file(source["tensor_file"]), load_file(final["path"])
        checks = []
        for layer in range(28):
            key = f"prompt_KV_{layer}"
            for side, first, last in (("prefix", 0, cut), ("suffix", cut, length)):
                metric = embedding_metrics(actual[key][first:last], expected[key][first:last])
                ok = metric["finite"] and metric["relative_L2"] <= .02 and metric["cosine"] >= .999
                checks.append(dict(layer=layer, segment=side, tolerance_pass=ok, **metric))
        p_end, d_begin = (cut, cut) if split else (length, length - 1)
        p_coverage = contiguous(events["P"][base], 0, p_end, cut)
        d_coverage = contiguous(events["D"][base], d_begin, length)
        valid = ((not split or prefix_ok) and p_coverage and d_coverage and
                 offer["prefill_recomputations"] == 0 and all(check["tolerance_pass"] for check in checks))
        cases.append(dict(
            request_id=rows[index]["request_id"], offer=offer, exact_prefix_import=prefix_ok,
            P_query_coverage=p_coverage, D_query_coverage=d_coverage, numerical_tolerance_checks=checks,
            exact_greedy_tokens=observed[index]["token_ids"] == teacher[index]["token_ids"], passed=valid,
        ))

    splits = sum(offer["accepted"] for offer in offered)
    passed = 0 < splits < len(rows) and all(item["passed"] for item in cases)
    result = dict(
        evidence="CORRECTNESS_PASS" if passed else "DIAGNOSTIC", passed=passed,
        cases=cases, accepted=splits, continued=len(rows) - splits,
        actual_batch_histograms={role: dict(Counter(event["actual_batch"] for event in value["events"]))
                                 for role, value in records.items()},
        scope=("Concurrent state probe with no timing claims. Actual P→D import is byte exact for split "
               "requests. Each D final-cache segment uses preregistered BI0 tolerance across different "
               "encoder and LLM batch geometries. Greedy exactness is reported separately."),
    )
    write(ROOT / "analysis" / f"{tag}_late_correctness.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--cut", type=int, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18847")
    parser.add_argument("--teacher-url", default="http://127.0.0.1:18848")
    parser.add_argument("--choice", choices=("split", "continue"), default="split")
    parser.add_argument("--natural", action="store_true",
                        help="Witness the exact target cut only if naturally scheduled")
    parser.add_argument("--natural-hold", action="store_true",
                        help="Hold only the next chunk after the stock scheduler issues the target")
    args = parser.parse_args()
    require(args.tag.isidentifier(), "Use an identifier for the correctness run tag")
    row = read(args.manifest)["requests"][args.index]
    result = run_case(row, args.tag, args.cut, args.url, args.teacher_url, args.choice,
                      natural="hold" if args.natural_hold else args.natural)
    print(json.dumps({"case": args.tag, "passed": result["passed"]}), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
