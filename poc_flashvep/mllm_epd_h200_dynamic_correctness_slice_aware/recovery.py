"""Replay the previous H200 divergences without changing the serving gate."""

import asyncio
from contextlib import ExitStack
from pathlib import Path
import time

from ..mllm_epd_h200_modality_aware_dynamic_split.correctness import (
    ROOT, ROLES, _request, digest, ownership_empty, read, require, rows_in, rpc, write,
)
from ..mllm_epd_h200_modality_aware_dynamic_split.geometry_control import e_signatures, issue_cohort


# Frozen from a53d707e's paired Native runs. The held target is the sole
# intended split while both arms decode the same eight original requests.
FIXTURES = (
    ("interior", 31, 24275, 0, "yubo2333__MMLongBench-Doc__688", 13, (
        3862, 525, 1378, 26131, 5435, 311, 26183, 82, 2718, 287, 304, 279, 2197,
    )),
    ("image_edge", 19, 15929, 6, "TIGER-Lab__Mantis-Eval__wc_25", 23, (
        5349, 8, 8278, 2168, 220, 16, 5711, 2155, 9666, 448, 4911, 19029, 271,
        69769, 25, 4654, 220, 16, 374, 264, 32719, 2482, 448,
    )),
)
FOLLOWERS = (2, 0, 5, 6, 12, 4, 9)


async def issue_serial(rows, cuts, url, label, native):
    outputs = []
    for index, (row, cut) in enumerate(zip(rows, cuts, strict=True)):
        route = "native" if native or index else "split"
        outputs.append(await _request(row, url, label, cut, route, index, time.monotonic(),
                                      natural=False if native or index else "hold"))
    return outputs


def observe(rows, cuts, url, ec, tag, *, native, serial=False, target=None):
    ids = [f"{tag}_{index:04d}" for index in range(len(rows))]
    require(not (ROOT / "raw" / f"{tag}_requests.jsonl").exists(), "Diagnostic run ID already exists")
    opened = []
    try:
        for port, role in ROLES:
            targets = {} if target is None or role == "P" else {ids[target[0]]: target[1]}
            rpc(port, "greedy_probe/begin", {"name": f"{tag}_{role}", "ids": ids, "targets": targets})
            opened.append(port)
        if serial:
            output = asyncio.run(issue_serial(rows, cuts, url, tag, native))
        else:
            output = asyncio.run(issue_cohort(rows, url, tag, cuts, native, "hold", lead=True))
    finally:
        for port in reversed(opened):
            rpc(port, "greedy_probe/end", {})
    ownership_empty(tag, "cuda_ipc")
    result = {role: read(ROOT / "raw" / "greedy_probe" / f"{tag}_{role}" / "manifest.json") for _, role in ROLES}
    offers = [event for path in (ROOT / "raw").glob("late_prefill_cut_*.jsonl") for event in rows_in(path)
              if event["request_id"] in ids]
    result.update(
        tag=tag, output_ids=[row["token_ids"] for row in output],
        E=[e_signatures(ec, base) for base in ids], offers=offers,
    )
    return result


def trace_geometry(run, role, index):
    base = f"{run['tag']}_{index:04d}"
    geometry = []
    for event in run[role]["events"]:
        target = next((row for row in event["requests"] if row["base"] == base), None)
        if target is None:
            continue
        geometry.append({
            "target": [target[key] for key in ("start", "query_tokens", "prompt_tokens", "position")],
            "batch": [[row[key] for key in ("start", "query_tokens", "prompt_tokens")] for row in event["requests"]],
            "padding": event["padded_query_rows"], "graph": event["graph_mode"],
        })
    return geometry


def compare_natural(a, b, rows):
    comparisons = []
    for index, (request, left, right) in enumerate(zip(rows, a["output_ids"], b["output_ids"], strict=True)):
        first = next((step for step, (x, y) in enumerate(zip(left, right)) if x != y), min(len(left), len(right)))
        p_a, p_b = (trace_geometry(run, "P", index) for run in (a, b))
        d_a, d_b = (trace_geometry(run, "D", index) for run in (a, b))
        d_at_first_difference = [next((event for event in geometry
                                      if event["target"][0] + event["target"][1] - event["target"][2] == first), None)
                                 for geometry in (d_a, d_b)] if left != right else []
        comparisons.append({
            "id": request["request_id"], "greedy_exact": left == right, "E_byte_equal": a["E"][index] == b["E"][index],
            "left_output_count": len(left), "right_output_count": len(right),
            "first_distinct_token": first if left != right else None,
            "P_geometry_equal": p_a == p_b, "D_geometry_equal": d_a == d_b,
            "P_target_query_rows": [[event["target"][1] for event in geometry] for geometry in (p_a, p_b)],
            "D_geometry_at_first_difference": d_at_first_difference,
            "D_batch_max": [max((len(event["batch"]) for event in geometry), default=0) for geometry in (d_a, d_b)],
        })
    return comparisons


def compare_teacher(a, b, index, expected_prefix, requested_step):
    import torch
    from safetensors import safe_open

    target_names = [f"{run['tag']}_{index:04d}" for run in (a, b)]
    steps = []
    with ExitStack() as stack:
        tensors = []
        for run in (a, b):
            receipt = run["D"]["receipt"]
            tensor_path = Path(receipt["path"])
            require(digest(tensor_path) == receipt["sha256"], "Teacher tensor receipt mismatch")
            tensors.append(stack.enter_context(safe_open(tensor_path, framework="pt", device="cpu")))
        for step in range(requested_step + 1):
            keys = [f"{name}.{step}" for name in target_names]
            aa, bb = (run["D"]["samples"].get(key) for run, key in zip((a, b), keys, strict=True))
            if aa is None or bb is None:
                break
            if step:
                require(aa["input_token_id"] == bb["input_token_id"] == expected_prefix[step-1],
                        "Teacher replay did not reach the requested causal prefix")
            require(aa.get("mrope_position_ids") == bb.get("mrope_position_ids"),
                    "Teacher replay changed the target's actual MRoPE position")
            a_sections = {key[len(keys[0]) + 1:] for key in tensors[0].keys() if key.startswith(keys[0] + ".")}
            b_sections = {key[len(keys[1]) + 1:] for key in tensors[1].keys() if key.startswith(keys[1] + ".")}
            require(a_sections == b_sections and {"final_hidden", "logits"} <= a_sections,
                    "Teacher probes exposed different tensor sections")
            differences = {}
            for section in sorted(a_sections):
                left, right = (file.get_tensor(f"{key}.{section}") for file, key in zip(tensors, keys, strict=True))
                require(left.shape == right.shape, "Teacher replay changed the selected tensor shape")
                delta = left.float() - right.float()
                differences[section] = {
                    "exact": bool(torch.equal(left, right)),
                    "max_abs": float(delta.abs().max().item()),
                    "relative_L2": float((delta.norm() / left.float().norm().clamp_min(1e-12)).item()),
                }
            geometry = []
            for run, token in ((a, aa), (b, bb)):
                event = run["D"]["events"][token["event"]]
                row = next(item for item in event["requests"] if item["base"] == target_names[len(geometry)])
                geometry.append({"actual_batch": len(event["requests"]), "actual_query_rows": event["actual_query_rows"],
                                 "padded_query_rows": event["padded_query_rows"], "target_position": row["position"],
                                 "graph_mode": event["graph_mode"], "query_rows": row["query_tokens"]})
            steps.append({"step": step, "native": aa, "candidate": bb, "tensor_differences": differences,
                          "geometry": geometry, "greedy_differs": aa["greedy_id"] != bb["greedy_id"]})
    return {
        "first_greedy_difference": next((value["step"] for value in steps if value["greedy_differs"]), None),
        "requested_step": requested_step, "steps": steps,
        "validated_through_requested_step": len(steps) == requested_step + 1,
    }


def run_recovery(rows, url, ec, numeric_regime, selections=("interior", "image_edge")):
    cases = []
    for name, lead, cut, target, request_id, step, prefix in FIXTURES:
        if name not in selections:
            continue
        selected = [rows[i] for i in (lead, *FOLLOWERS)]
        require(selected[target]["request_id"] == request_id, "Calibration recovery target changed")
        cuts = [cut] + [round(row["total_input_tokens"] * .9) for row in selected[1:]]
        stem = "recovery_" + name
        serial_a = observe(selected, cuts, url, ec, stem + "_serial_a", native=True, serial=True)
        serial_b = observe(selected, cuts, url, ec, stem + "_serial_b", native=True, serial=True)
        native_a = observe(selected, cuts, url, ec, stem + "_mixed_a", native=True)
        native_b = observe(selected, cuts, url, ec, stem + "_mixed_b", native=True)
        late = observe(selected, cuts, url, ec, stem + "_mixed_late", native=False)
        plan = {
            "steps": step + 1, "prefix": list(prefix),
            "layer_steps": [step] if numeric_regime.endswith("eager") else [],
        }
        result = {
            "name": name, "request_id": request_id, "target_row": target, "cut": cut, "numeric_regime": numeric_regime,
            "serial_AA": compare_natural(serial_a, serial_b, selected),
            "mixed_AA": compare_natural(native_a, native_b, selected),
            "late_vs_native": compare_natural(native_a, late, selected), "unforced_late_offers": late["offers"],
            "evidence": "DIAGNOSTIC",
        }
        try:
            teacher_a = observe(selected, cuts, url, ec, stem + "_teacher_a", native=True, target=(target, plan))
            teacher_b = observe(selected, cuts, url, ec, stem + "_teacher_b", native=False, target=(target, plan))
            result.update(
                teacher=compare_teacher(teacher_a, teacher_b, target, prefix, step),
                teacher_E_byte_equal=teacher_a["E"][target] == teacher_b["E"][target],
                teacher_late_offers=teacher_b["offers"],
            )
        except Exception as error:
            result["teacher"] = {"steps": [], "first_greedy_difference": None, "evidence": "FAILED_EXECUTION",
                                 "error": f"{type(error).__name__}: {error}"}
        write(ROOT / "analysis" / f"{stem}.json", result)
        cases.append(result)
        print("CORRECTNESS_RECOVERY", name, numeric_regime, result["teacher"]["first_greedy_difference"],
              sum(row["greedy_exact"] for row in result["late_vs_native"]), flush=True)
    write(ROOT / "analysis" / "correctness_recovery.json", {"cases": cases})
