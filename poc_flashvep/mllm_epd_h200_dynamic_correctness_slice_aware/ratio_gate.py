"""Apply the frozen N1 and active Mixed-N8 gates to named cut fixtures."""

import hashlib
from pathlib import Path

from ..mllm_epd_h200_modality_aware_dynamic_split.correctness import (
    ROOT, capture_late, digest, read, rows_in, run_case, write,
)
from ..mllm_epd_h200_modality_aware_dynamic_split.geometry_control import run_geometry_control
from .clone_receipt import clonelist
from .recovery import FOLLOWERS


# The larger cuts are original retired chunks and approximate the nominal ratios.
# The small cuts test the new forced policy and an actual same-cut transport fixture.
RATIO_FIXTURES = {
    "small64": (1, 7823 - 64, .01, False),
    "small128": (4, 5032 - 128, .025, False),
    "small19": (1, 7804, .01, False),
    "small64_fastjoin": (1, 7823 - 64, .01, False),
    "small19_fastjoin": (1, 7804, .01, False),
    "long64_fastjoin": (20, 17334 - 64, .01, False),
    "long128_fastjoin": (20, 17334 - 128, .025, False),
    "natural05": (20, 16083, .05, "hold"),
    "natural10": (31, 24275, .10, "hold"),
    "natural20": (19, 15929, .20, "hold"),
    "natural30": (26, 8192, .30, "hold"),
}


def d_clones(ec, runtime_id, row, cut):
    request_key = hashlib.sha256(runtime_id.encode()).hexdigest()
    manifest = read(ec / "processed_prompts" / request_key / "manifest.json")
    return clonelist(ROOT / "raw", manifest["image_hashes"], row["image_visual_spans"], cut)


def compare_native_resume(row, url, ec, cut, resumed_tag):
    """Compare unchanged Native EPD with the actual same-request resume route."""
    import torch
    from safetensors import safe_open

    from ..mllm_epd_h200_modality_aware_dynamic_split.geometry_control import e_signatures

    native_tag = "native_no_hold_resume_control"
    native = capture_late(row, url, native_tag, cut, "native")
    resumed, = rows_in(ROOT / "raw" / f"{resumed_tag}_requests.jsonl")
    results = {}
    for role in ("P", "D"):
        a, b = [
            read(ROOT / "raw/llm_reference" / f"{tag}_{role}" / "manifest.json")
            for tag in (native_tag, resumed_tag)
        ]
        paths = [Path(record["tensor_file"]) for record in (a, b)]
        if any(digest(path) != record["tensor_SHA256"] for path, record in zip(paths, (a, b), strict=True)):
            raise RuntimeError("Native EPD control tensor receipt mismatch")
        shapes = [
            [[event[key] for key in ("start", "query_tokens", "batch_descriptor")] for event in record["events"]]
            for record in (a, b)
        ]
        with safe_open(paths[0], framework="pt", device="cpu") as x, safe_open(
            paths[1], framework="pt", device="cpu"
        ) as y:
            fields = {
                key: torch.equal(x.get_tensor(key), y.get_tensor(key)) for key in ("prompt_token_ids", "full_prompt_mrope")
            }
            layers = sum(torch.equal(x.get_tensor(f"prompt_KV_{i}"), y.get_tensor(f"prompt_KV_{i}")) for i in range(28))
        results[role] = {"same_geometry": shapes[0] == shapes[1], "prompt_field_exactness": fields,
                         "prompt_KV_exact_layers": layers, "same_tensor_file": a["tensor_SHA256"] == b["tensor_SHA256"],
                         "query_geometry": shapes}
    return {
        "evidence": "DIAGNOSTIC", "native": native_tag, "continuation": resumed_tag,
        "Paired_Greedy_exact": native["token_ids"] == resumed["token_ids"],
        "output_counts": [native["output_tokens"], resumed["output_tokens"]],
        "E_embedding_exact": e_signatures(ec, native_tag + "_0000") == e_signatures(ec, resumed_tag + "_0000"),
        "actual_geometry_and_state": results,
    }


def run_ratio_gates(rows, url, teacher, ec, numeric_regime, names, slice_aware=False):
    packets = []
    for name in names:
        index, cut, nominal, natural = RATIO_FIXTURES[name]
        candidate = rows[index]
        label = f"ratio_{name}"
        packet = {
            "name": name, "request_id": candidate["request_id"], "cut": cut, "nominal_ratio": nominal,
            "actual_ratio": 1 - cut / candidate["total_input_tokens"], "numeric_regime": numeric_regime,
        }
        for test, label_suffix in (("N1", "_n1"), ("N8", "_n8")):
            try:
                if test == "N1":
                    packet[test] = run_case(candidate, label + label_suffix, cut, url, teacher, natural=natural)
                else:
                    # The original short-answer lead can finish before a baseline
                    # follower reaches D. The alternate fixture uses the 543-token
                    # Calibration row for a faster genuine Native join in both arms.
                    follower_indices = (11, *FOLLOWERS[1:]) if name.endswith("_fastjoin") else FOLLOWERS
                    followers = [rows[i] for i in follower_indices if i != index]
                    if len(followers) < 7:
                        followers.append(rows[13])
                    packet[test] = run_geometry_control(
                        [candidate, *followers], label + label_suffix, url, teacher, ec, ratio=nominal, natural=natural,
                        lead=True, use_native_cuts=False, targets=[cut] + [None] * 7,
                    )
            except Exception as error:
                packet[test] = {"passed": False, "evidence": "FAILED_EXECUTION",
                                "error": f"{type(error).__name__}: {error}"}
        if slice_aware:
            for stage, rid in (("N1", f"{label}_n1_0000"), ("N8", f"{label}_n8_late_0000")):
                if packet[stage]["passed"] or stage == "N8" and packet[stage].get("accepted") == 1:
                    try:
                        transport = d_clones(ec, rid, candidate, cut)
                    except Exception as error:
                        transport = {"verified_range": False, "error": f"{type(error).__name__}: {error}"}
                    packet[stage]["D_transport"] = transport
        packet["passed"] = all(packet[test]["passed"] for test in ("N1", "N8")) and (
            not slice_aware or all(packet[test]["D_transport"]["verified_range"] for test in ("N1", "N8"))
        )
        packets.append(packet)
        write(ROOT / "analysis" / "ratio_gates.json", {"cases": packets})
        print("RATIO_STRICT_GATE", name, nominal, packet["actual_ratio"],
              packet["N1"]["passed"], packet["N8"]["passed"], packet["passed"], flush=True)
    try:
        continued = run_case(rows[31], "rejected_natural_hold_continuation", 24275, url, teacher,
                             choice="continue", natural="hold")
    except Exception as error:
        continued = {"passed": False, "evidence": "FAILED_EXECUTION",
                     "error": f"{type(error).__name__}: {error}"}
    report = {"cases": packets, "natural_hold_continuation": continued}
    if numeric_regime == "eager" and not continued["passed"]:
        try:
            report["native_epd_resume_control"] = compare_native_resume(
                rows[31], url, ec, 24275, "rejected_natural_hold_continuation",
            )
        except Exception as error:
            report["native_epd_resume_control"] = {"evidence": "FAILED_EXECUTION", "error": f"{type(error).__name__}: {error}"}
    write(ROOT / "analysis" / "ratio_gates.json", report)
    print("CONTINUATION_STRICT_GATE", continued["passed"], flush=True)
