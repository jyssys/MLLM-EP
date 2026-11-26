"""Same-cut H200 whole-image and slice-only CUDA IPC crossover."""

import argparse
import asyncio
from collections import defaultdict
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import median

from ..mllm_epd_h200_modality_aware_dynamic_split.calibration import nvlink, observe_reader, trial, write_records
from ..mllm_epd_h200_modality_aware_dynamic_split.calibration_main import INVARIANCE_ROLES, run_calibration
from ..mllm_epd_h200_modality_aware_dynamic_split.modality import cut_features, inventory, read, require, write
from .embedding_rows import SLICE_KEY


SEED = 2026112609
MODES = ("whole", "slice")
REFERENCE_INDEX = 1
REFERENCE_CUT = 7804
REFERENCE_LENGTH = 7823


def plan(calibration):
    cases = {(cell["index"], cell["prefix_tokens"]): cell for cell in calibration["plan"]["cells"]
             if cell["series"] == "transfer" and cell["arm"] in ("image_boundary", "intra_image")}
    all_rows = calibration["Calibration32"]
    reference = all_rows[REFERENCE_INDEX]
    require(reference["input_tokens"] == REFERENCE_LENGTH
            and reference["images"][0]["span"] == [15, 7805], "Required 7823/19 case changed")
    strict = [(REFERENCE_INDEX, REFERENCE_CUT)]
    for cut in range(7802, 7808):
        cases.setdefault((REFERENCE_INDEX, cut), {"index": REFERENCE_INDEX,
                         "arm": "one_token_edge_sweep", "request_id": reference["request_id"],
                         **cut_features(reference, cut)})
    for index in (1, 20):
        row = all_rows[index]
        for ratio in (.05, .10, .20, .30):
            cut = round(row["input_tokens"] * (1 - ratio))
            cases.setdefault((index, cut), {"index": index, "arm": f"ratio_{ratio:g}",
                             "request_id": row["request_id"], **cut_features(row, cut)})
            if index == 20:
                strict.append((index, cut))
    return list(cases.values()), [cases[key] for key in strict]


def cost_plan(calibration):
    cases = []
    ratios = (0, .01, .025, .05, .10, .20, .30)
    for index in (1, 20):
        row = calibration["Calibration32"][index]
        candidates = {round(row["input_tokens"] * (1 - ratio)): f"ratio_{ratio:g}" for ratio in ratios}
        if index == REFERENCE_INDEX:
            candidates.update({row["input_tokens"] - size: f"tokens_{size}" for size in (19, 64, 128)})
        cases.extend({"index": index, "arm": label, "request_id": row["request_id"], **cut_features(row, cut)}
                     for cut, label in candidates.items())
    return cases


def attach_observed_clones(records, clone_logs, calibration_rows):
    for record in records:
        sources = calibration_rows[record["index"]]["images"]
        cut = record["prefix_tokens"]
        for role in ("prefill", "decode"):
            events = {}
            shapes = {}
            for key in record["image_keys"]:
                for event_role, event in clone_logs[key]:
                    if role == event_role:
                        events[id(event)] = event
                        entries = [shape for shape in event["clone_shapes"] if shape["key"] == key]
                        require(len(entries) == 1 and key not in shapes, "Unexpected repeated EC clone")
                        shapes[key] = entries[0]
                        require(event["clone_complete_before_ACK"], "Missing clone-completion ACK")
            expected = set()
            for key, source in zip(record["image_keys"], sources, strict=True):
                start, end = source["span"]
                if (role == "prefill" and start < cut) or (role == "decode" and end > cut):
                    expected.add(key)
                    row_offset = max(0, cut - start) if role == "decode" and record["mode"] == "slice" else 0
                    shape = shapes.get(key)
                    num_rows = end - start
                    require(shape is not None and shape["first_source_row"] == row_offset
                            and shape["source_shape"] == [num_rows, 3584]
                            and shape["clone_shape"] == [num_rows - row_offset, 3584]
                            and shape["logical_bytes"] == (num_rows - row_offset) * 3584 * 2,
                            f"EC clone interval differs from the frozen original token span for {record['trial']}")
            require(set(shapes) == expected, "Observed clone set does not match token ownership")
            record[f"{role}_clones"] = list(shapes.values())
            record[f"{role}_clone_operations"] = len(shapes)
            record[f"{role}_logical_bytes"] = sum(shape["logical_bytes"] for shape in shapes.values())
            record[f"{role}_clone_resident_bytes"] = sum(shape["resident_storage_bytes"] for shape in shapes.values())
            record[f"{role}_clone_sync_host_ms"] = sum(event["import_clone_sync_host_ms"] for event in events.values())
            record[f"{role}_descriptor_operations"] = sum(event["inline_images"] + bool(event["export_RPC_ms"])
                                                         for event in events.values())
            record[f"{role}_redirected_inline_images"] = sum(event["redirected_inline_images"]
                                                            for event in events.values())


async def checked_trial(session, run, row, cell, identifier, ec_path, mode):
    from ..mllm_epd_prefill_heavy_token_split.batch_token_correctness_control import post, validate

    contract = {"prefix_tokens": cell["prefix_tokens"], "full_prompt_tokens": row["total_input_tokens"]}
    opened = []
    try:
        for role, port in (("P", 18841), ("D", 18842)):
            body = {"name": f"{identifier}_{role}", "contracts": {identifier: contract}}
            if role == "D":
                body.update(contracts={identifier: dict(contract, prefill_full_prompt=True)},
                            allow_decode=True, capture_full_prompt=True)
            await post(session, port, "/campaign/batch_prefix/begin", body)
            opened.append(port)
        result = await trial(session, row, cell, identifier, ec_path,
                             decode_params={SLICE_KEY: mode == "slice"}, natural_decode=True)
    finally:
        for port in reversed(opened):
            await post(session, port, "/campaign/batch_prefix/end", {})
    prefix = validate(identifier, {identifier: contract}, include_decode=True)
    d_path = run / "raw/batch_prefix" / f"{identifier}_D" / "manifest.json"
    receipt = read(d_path)
    inputs = [{key: query.get(key) for key in ("start", "query_tokens", "positions", "inputs_embeds", "input_ids")}
              for event in receipt["events"] for query in event["requests"] if query["phase"] == "prefill"]
    query_geometry = [{"actual_batch": event["actual_batch"], "actual_query_tokens": event["actual_query_tokens"],
                       "graph_mode": event["cudagraph_mode"], "batch_descriptor": event["batch_descriptor"]}
                      for event in receipt["events"]]
    return result | {"mode": mode, "tensor_file": receipt["full_prompt_tensor_files"][identifier],
                     "exact_imported_kv_all_layers": prefix["checks"][0]["imported_layers_strict"],
                     "d_query_inputs": inputs, "d_query_geometry": query_geometry,
                     "d_prefix_kv": receipt["snapshots"][identifier]["layers"]}


def compare_correctness(whole, sliced):
    import torch
    from safetensors import safe_open

    from ..mllm_epd_prefill_heavy_token_split.common import digest

    torch.set_num_threads(2)
    a_path, b_path = (Path(row["tensor_file"]["path"]) for row in (whole, sliced))
    require(digest(a_path) == whole["tensor_file"]["SHA256"]
            and digest(b_path) == sliced["tensor_file"]["SHA256"], "Correctness tensor checksum mismatch")
    with safe_open(a_path, framework="pt", device="cpu") as a, safe_open(b_path, framework="pt", device="cpu") as b:
        require(set(a.keys()) == set(b.keys()) == {"prompt_token_ids", "full_prompt_mrope",
                *(f"prompt_KV_{layer}" for layer in range(28))}, "Incomplete strict prompt state")
        final_kv = []
        for layer in range(28):
            key = f"prompt_KV_{layer}"
            left, right = a.get_tensor(key), b.get_tensor(key)
            final_kv.append(left.shape == right.shape and left.dtype == right.dtype
                            and bool(torch.equal(left.view(torch.uint8), right.view(torch.uint8))))
        mrope = bool(torch.equal(a.get_tensor("full_prompt_mrope"), b.get_tensor("full_prompt_mrope")))
        prompt_ids = bool(torch.equal(a.get_tensor("prompt_token_ids"), b.get_tensor("prompt_token_ids")))
    conditions = {
        "exact_imported_prefix_kv_in_both_arms": all(whole["exact_imported_kv_all_layers"])
                                                and all(sliced["exact_imported_kv_all_layers"]),
        "exact_imported_prefix_kv_between_arms": whole["d_prefix_kv"] == sliced["d_prefix_kv"],
        "exact_final_kv_all28": all(final_kv),
        "exact_prompt_mrope": mrope,
        "exact_prompt_ids": prompt_ids,
        "exact_d_query_inputs": whole["d_query_inputs"] == sliced["d_query_inputs"],
        "same_d_query_geometry": whole["d_query_geometry"] == sliced["d_query_geometry"],
        "exact_natural_greedy": whole["token_ids"] == sliced["token_ids"]
                                and whole["finish_reason"] == sliced["finish_reason"],
    }
    return {"evidence": "CORRECTNESS_PASS" if all(conditions.values()) else "DIAGNOSTIC",
            "passed": all(conditions.values()), "checks": conditions, "exact_final_kv_layers": final_kv,
            "request_id": whole["request_id"], "input_tokens": whole["input_tokens"],
            "prefix_tokens": whole["prefix_tokens"], "suffix_tokens": whole["suffix_tokens"],
            "token_counts": {row["mode"]: len(row["token_ids"]) for row in (whole, sliced)},
            "arms": {row["mode"]: row for row in (whole, sliced)}}


async def shared_media_regression(session, run, row, geometry, ec_path):
    """Submit the same physical image twice with distinct live D row intervals."""
    from ..mllm_epd_h200_modality_aware_dynamic_split.calibration import get
    from ..mllm_epd_prefill_heavy_token_split.batch_token_correctness_control import finish, post, prepare

    records = []
    prepared = []
    for cut in (7802, REFERENCE_CUT):
        tag = f"same_media_{cut}"
        original = await prepare(session, row, tag, cut)
        original["continuation"]["kv_transfer_params"][SLICE_KEY] = True
        key = sha256(tag.encode()).hexdigest()
        manifest = read(ec_path / "processed_prompts" / key / "manifest.json")
        prepared.append(original)
        records.append({"trial": tag, "prefix_tokens": cut, "image_keys": manifest["image_hashes"],
                        "index": REFERENCE_INDEX, "mode": "slice"})
    require(len({key for record in records for key in record["image_keys"]}) == 2,
            "Same media produced overlapping request keys")
    before, = await get(session, 18840, "/campaign/ec_ipc/state")
    require(before["live_images"] == 2, "Both original-image exports must overlap in lifetime")
    names = {record["base"]: {"prefix_tokens": record["prefix"], "full_prompt_tokens": REFERENCE_LENGTH,
                             "prefill_full_prompt": True} for record in prepared}
    await post(session, 18842, "/campaign/batch_prefix/begin", {"name": "same_media_D", "contracts": names})
    try:
        completed = await asyncio.gather(*(finish(session, item) for item in prepared))
    finally:
        await post(session, 18842, "/campaign/batch_prefix/end", {})
    after, = await get(session, 18840, "/campaign/ec_ipc/state")
    require(after["live_images"] == after["outstanding_exports"] == 0, "Overlapping media did not retire")
    attach_observed_clones(records, observe_reader(run), geometry["Calibration32"])
    observation = read(run / "raw/batch_prefix/same_media_D/manifest.json")
    return {"evidence": "ISOLATED_GPU_MEASUREMENT", "same_frozen_request_id": row["request_id"],
            "concurrent_d_submissions": 2, "e_live_images_before_d": before["live_images"],
            "final_e_live_images": after["live_images"], "final_e_outstanding_exports": after["outstanding_exports"],
            "prepared_d_max_batch": max(event["actual_batch"] for event in observation["events"]),
            "d_first_tokens": [item["response"]["choices"][0]["token_ids"][0] for item in completed],
            "records": records, "scope": "Separate isolated ownership regression, not online policy or latency evidence."}


def pair_results(records):
    groups = defaultdict(dict)
    for record in records:
        groups[(record["index"], record["prefix_tokens"], record["repeat"])][record["mode"]] = record
    rows = []
    for (_, cut, repeat), arms in groups.items():
        whole, sliced = (arms[mode] for mode in MODES)
        require(whole["prefill_logical_bytes"] == sliced["prefill_logical_bytes"],
                "P accounting changed under D-only transport selection")
        same_nixl = whole["nixl_D_counters"] is not None and sliced["nixl_D_counters"] is not None
        same_nixl = (whole["nixl_D_counters"]["bytes"] == sliced["nixl_D_counters"]["bytes"]
                     if same_nixl else None)
        require(same_nixl is not False, "Matched pair has different physical NIXL telemetry")
        row = {"request_id": whole["request_id"], "index": whole["index"], "prefix_tokens": cut,
               "input_tokens": whole["input_tokens"], "suffix_tokens": whole["suffix_tokens"], "repeat": repeat,
               "causal_attention_pairs_per_layer": whole["causal_attention_pairs_per_layer"],
               "prefix_blocks_128": whole["prefix_blocks_128"], "same_nixl_bytes": same_nixl,
               "input_E_prompt_unchanged": True, "exact_first_greedy_token": whole["first_token"] == sliced["first_token"]}
        for key in ("decode_logical_bytes", "decode_clone_resident_bytes",
                    "decode_clone_operations", "decode_descriptor_operations",
                    "decode_clone_sync_host_ms", "D_first_token_http_ms", "producer_host_marker_lifetime_max_ms",
                    "producer_buffer_storage_snapshot_bytes"):
            row[f"whole_{key}"] = whole[key]
            row[f"slice_{key}"] = sliced[key]
            row[f"saved_{key}"] = whole[key] - sliced[key]
        for arm, source in arms.items():
            counters = source["NVLink_D_stage_device_delta"] or {}
            for counter in ("E_TX_bytes", "D_RX_bytes"):
                row[f"{arm}_D_stage_{counter}"] = counters.get(counter)
        for counter in ("E_TX_bytes", "D_RX_bytes"):
            before = row[f"whole_D_stage_{counter}"]
            after = row[f"slice_D_stage_{counter}"]
            row[f"saved_D_stage_{counter}"] = before - after if before is not None and after is not None else None
        rows.append(row)
    return rows


async def run_epoch(run, calibration_manifest, main_manifest, ec_path, device_indices, repeats,
                    *, inline_redirect, cost_only=False, strict_only=False):
    import aiohttp

    from ..mllm_epd_prefill_heavy_token_split.batch_token_correctness_control import post

    require(inline_redirect, "Ablation requires the same inline descriptor policy in both arms")
    geometry = inventory(main_manifest, calibration_manifest)
    calibration = read(calibration_manifest)["requests"]
    cells, strict = (cost_plan(geometry), []) if cost_only else plan(geometry)
    if strict_only:
        cells = strict
    write(run / "configs/same_cut_plan.json", {"seed": SEED, "repeats": repeats, "cells": cells, "strict": strict})
    records, correctness, matched = [], [], []
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        await post(session, 18840, "/campaign/ec_ipc/inline_policy", {"enabled": True})
        reference = next(cell for cell in cells if cell["index"] == REFERENCE_INDEX
                         and cell["prefix_tokens"] == REFERENCE_CUT)
        for mode in MODES:
            await trial(session, calibration[reference["index"]], reference, f"same_cut_warm_{mode}", ec_path,
                        decode_params={SLICE_KEY: mode == "slice"})
        if not strict_only:
            path = run / "raw/same_cut_trials.jsonl"
            with path.open("x", buffering=1) as journal, nvlink(device_indices) as sample:
                for repeat in range(repeats):
                    for ordinal, cell in enumerate(random.Random(SEED + repeat).sample(cells, len(cells))):
                        order = MODES if (ordinal + repeat) % 2 == 0 else MODES[::-1]
                        for mode in order:
                            identifier = f"same_cut_{repeat}_{ordinal:02d}_{mode}"
                            value = await trial(session, calibration[cell["index"]], cell, identifier, ec_path, sample,
                                                decode_params={SLICE_KEY: mode == "slice"})
                            record = value | {"mode": mode, "repeat": repeat}
                            records.append(record)
                            journal.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(f"same-cut repeat {repeat + 1}/{repeats}: {len(cells)} pairs", flush=True)
            attach_observed_clones(records, observe_reader(run), geometry["Calibration32"])
            matched = pair_results(records)
            write(run / "analysis/same_cut_trials.json", records)
            write_records(run / "analysis/same_cut_pairs.csv", matched)
            write(run / "analysis/same_cut_pairs.json", matched)
        for index, cell in enumerate(strict):
            result = {}
            for mode in MODES:
                identifier = f"same_cut_correctness_{index}_{mode}"
                result[mode] = await checked_trial(session, run, calibration[cell["index"]], cell, identifier,
                                                   ec_path, mode)
            attach_observed_clones(result.values(), observe_reader(run), geometry["Calibration32"])
            comparison = compare_correctness(result["whole"], result["slice"])
            correctness.append(comparison)
            write(run / "analysis" / f"same_cut_correctness_{index}.json", comparison)
            print(f"same-cut strict case {index + 1}/{len(strict)}: {comparison['passed']}", flush=True)
        if strict:
            shared = await shared_media_regression(session, run, calibration[REFERENCE_INDEX], geometry, ec_path)
            write(run / "analysis/same_media_regression.json", shared)

    key = "saved_D_stage_E_TX_bytes"
    contributors = [row[key] for row in matched if row[key] is not None and row["saved_decode_logical_bytes"] > 0]
    write(run / "analysis/same_cut_summary.json", {
        "evidence": "ISOLATED_GPU_MEASUREMENT", "timing_pairs": len(matched),
        "crossing_image_pairs": sum(row["saved_decode_logical_bytes"] > 0 for row in matched),
        "strict_transport_checks": len(correctness), "strict_transport_passes": sum(row["passed"] for row in correctness),
        "strict_scope": "Whole vs slice at the same imported-KV cut. This checks transport equivalence, not split-vs-Native or mixed serving correctness.",
        "nvlink_scope": "Device-wide data deltas in isolated D stage, including handoff traffic. No per-tensor physical attribution.",
        "timing_scope": "HTTP and clone/ACK host enclosures, not GPU-exclusive time.",
        "crossing_pairs_with_nvlink": len(contributors),
        "crossing_median_E_TX_saved_bytes": median(contributors) if contributors else None,
        "reference_L7823_suffix19": [row for row in matched if row["index"] == REFERENCE_INDEX
                                    and row["prefix_tokens"] == REFERENCE_CUT],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-repo", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--wall-seconds", type=int, default=2700)
    parser.add_argument("--all-burners-off", action="store_true")
    parser.add_argument("--preissue-crossing-ec", action="store_true")
    parser.add_argument("--numeric-regime", choices=INVARIANCE_ROLES, default="bi0",
                        help="P/D numerical regime. eager keeps BI0 and the existing E runtime")
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--cost-only", action="store_true",
                          help="Calibrate ratios from 0 to 30 percent and tiny suffixes without tensor probes")
    variants.add_argument("--strict-only", action="store_true",
                          help="Run transported-row and Full28 state validation without the timing sweep")
    parser.set_defaults(inline_redirect=True)
    args = parser.parse_args()
    run_calibration(args, measure_epoch=partial(run_epoch, cost_only=args.cost_only, strict_only=args.strict_only),
                    tag="slice-ablation",
                    decode_environment={"MLLM_EPD_SLICE_AWARE_EC": "1"}, freeze_auxiliary=True,
                    numeric_regime=args.numeric_regime, preissue_crossing=args.preissue_crossing_ec)


if __name__ == "__main__":
    main()
