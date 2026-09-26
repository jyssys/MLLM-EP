"""Build immutable Part-1 analysis artifacts from measured runs.

No model execution happens here.  Every latency value is read from a measured
artifact and every unsupported claim is emitted as NOT_RUN/UNVALIDATED.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from poc_mllm_ep_foundation.constants import HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_EXPERTS, TOP_K
from poc_mllm_ep_foundation.io import atomic_json
from poc_mllm_ep_foundation.virtual_ep.cost import validate


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _walls(root: Path) -> dict[str, float]:
    rows = _json(root / "driver.dp_rank0.json")["records"]
    return {str(row["request_id"]): float(row["wall_ms"]) for row in rows if row["measured"]}


def build_profile(profile: Path, clean: Path, historical_baseline: Path) -> tuple[dict, list[dict]]:
    deep: list[dict] = []
    for rank, name in enumerate(sorted(glob.glob(str(profile / "raw_live" / "rank[0-9].jsonl")))):
        for row in _jsonl(Path(name)):
            if row["measured"]:
                row["rank"] = rank
                deep.append(row)

    full: list[dict] = []
    for rank, name in enumerate(sorted(glob.glob(str(profile / "raw_live" / "full_rank*.jsonl")))):
        seen: set[tuple[int, str, int]] = set()
        for row in _jsonl(Path(name)):
            if not row["measured"]:
                continue
            key = (int(row["wave"]), str(row["kind"]), int(row["layer"]))
            if key in seen:  # reject DP padding invocations within the same wave
                continue
            seen.add(key)
            row["rank"] = rank
            full.append(row)

    event = {(int(x["wave"]), int(x["layer"]), str(x["kind"]), int(x["rank"])): float(x["ms"])
             for x in full}
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in deep:
        grouped[(str(row["request_id"]), int(row["wave"]), int(row["layer"]))].append(row)

    request_parts: dict[str, Counter] = defaultdict(Counter)
    shape_rows: list[dict] = []
    for (request_id, wave, layer), ranks in sorted(grouped.items()):
        ranks = sorted(ranks, key=lambda x: int(x["rank"]))
        critical = max(ranks, key=lambda x: event[(wave, layer, "moe_total", int(x["rank"]))])
        critical_rank = int(critical["rank"])
        moe_ms = event[(wave, layer, "moe_total", critical_rank)]
        dispatch = float(critical["dispatch"]["ms"])
        expert = float(critical["expert"]["ms"])
        combine = float(critical["combine"]["ms"])
        layout_residual = max(0.0, moe_ms - dispatch - expert - combine)
        attention = max(event[(wave, layer, "attention", int(row["rank"]))] for row in ranks)
        decoder = max(event[(wave, layer, "decoder_layer", int(row["rank"]))] for row in ranks)
        decoder_other = max(0.0, decoder - attention - moe_ms)
        request_parts[request_id].update({
            "attention_ms": attention,
            "router_layout_residual_ms": layout_residual,
            "dispatch_ms": dispatch,
            "expert_ms": expert,
            "combine_ms": combine,
            "decoder_other_ms": decoder_other,
            "routed_ms": moe_ms,
        })
        histogram = np.asarray([row["total_assignments"] for row in ranks], dtype=float)
        shape_rows.append({
            "observation_id": f"{request_id}:{layer}",
            "request_id": request_id,
            "layer": layer,
            "assignments": float(histogram.sum()),
            "max_rank_assignments": float(histogram.max()),
            "remote_unique_rows": float(sum(int(row["dispatched_rows"]) for row in ranks)),
            "active_experts": float(sum(sum(int(v) > 0 for v in row["expert_histogram"]) for row in ranks)),
            "dispatch_ms": dispatch,
            "expert_ms": expert,
            "combine_ms": combine,
            "routed_ms": moe_ms,
            "critical_rank": critical_rank,
            "evidence": "MEASURED_TRUE_EP4_CUDA_EVENT_HIGH_OVERHEAD_PASS",
        })

    prof, base = _walls(profile), _walls(clean)
    common = sorted(set(prof) & set(base))
    ratios = [prof[key] / base[key] - 1.0 for key in common]
    historical = _json(historical_baseline)
    by_case: dict[str, list[float]] = defaultdict(list)
    for row in historical["rows"]:
        by_case[str(row["case"])].append(float(row["wall_ms"]))
    clean_medians = {key: float(statistics.median(values)) for key, values in by_case.items()}
    aggregate = {
        key: float(statistics.median([parts[key] for parts in request_parts.values()]))
        for key in next(iter(request_parts.values()))
    }
    payload = {
        "evidence": "MEASURED_TRUE_EP4",
        "physical_gpus": [4, 5, 6, 7],
        "parallelism": {"tp": 2, "dp": 2, "sp": 1, "pp": 1, "ep": 4},
        "backend": "deepep_high_throughput+triton_sparse_top8",
        "clean_input_ready_wall_medians_ms": clean_medians,
        "diagnostic_component_medians_ms": aggregate,
        "diagnostic_profile_overhead_median_fraction": float(statistics.median(ratios)),
        "diagnostic_profile_overhead_weighted_fraction": float(sum(prof.values()) / sum(base.values()) - 1.0),
        "profiling_gate": "DIAGNOSTIC_HIGH_OVERHEAD" if sum(prof.values()) / sum(base.values()) - 1.0 > 0.10 else "PASS",
        "component_warning": (
            "The full-category pass is diagnostic because overhead exceeds 10%; clean wall is the denominator. "
            "DeepEP subevents remain measured CUDA spans, not an E2E sum."
        ),
        "requests": {key: dict(value) for key, value in sorted(request_parts.items())},
    }
    return payload, shape_rows


def build_calibration(rows: list[dict], output: Path) -> dict:
    ids = sorted({str(row["request_id"]) for row in rows})
    calibration_ids, heldout_ids = ids[:7], ids[7:]
    observations = [row for row in rows if row["request_id"] in calibration_ids]
    heldout = [row for row in rows if row["request_id"] in heldout_ids]
    signature = {
        "model_revision": "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c",
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_experts": NUM_EXPERTS,
        "top_k": TOP_K,
        "backend": "deepep_high_throughput",
        "topology": "tp2_dp2_sp1_pp1_ep4",
    }
    calibration = {
        "schema_version": "virtual_ep_cost.v1",
        "signature": signature,
        "observations": observations,
        "heldout_truth": {row["observation_id"]: {key: row[key] for key in
                          ("dispatch_ms", "expert_ms", "combine_ms", "routed_ms")}
                          for row in heldout},
        "split": {
            "unit": "request",
            "calibration_request_ids": calibration_ids,
            "heldout_request_ids": heldout_ids,
            "calibration_requests": len(calibration_ids),
            "heldout_requests": len(heldout_ids),
            "required_minimum_met": False,
            "reason": "current matched profiling cohort has 11 requests, below requested 16/8 split",
        },
        "status": "CALIBRATED_SMALL_COHORT",
    }
    replay = {
        "schema_version": "virtual_ep_replay.v1",
        "target_ep": 4,
        "evidence": "MEASURED_TRUE_EP4_HELDOUT_SHAPES",
        "rows": heldout,
    }
    atomic_json(output / "cost_calibration.json", calibration)
    atomic_json(output / "heldout_replay.json", replay)
    result = validate(output / "cost_calibration.json", output / "heldout_replay.json",
                      output / "heldout_validation.json")
    result["minimum_split_gate"] = "FAIL_7_CALIBRATION_4_HELDOUT"
    result["overall_status"] = "SIM_UNVALIDATED"
    atomic_json(output / "heldout_validation.json", result)
    return result


def build_quality(root: Path) -> dict:
    reference = {row["request_id"]: row for row in _jsonl(root / "quality/sanity_reference_tp4/results.jsonl")}
    ep = {row["request_id"]: row for row in _jsonl(root / "quality/sanity_physical_ep4/results.jsonl")}
    ids = sorted(set(reference) & set(ep))
    exact = sum(reference[key]["output_token_ids"] == ep[key]["output_token_ids"] for key in ids)
    return {
        "matched_requests": len(ids),
        "exact_token_sequence": exact,
        "exact_token_sequence_fraction": exact / len(ids),
        "ep_empty_outputs": sum(not ep[key]["output_text"].strip() for key in ids),
        "ep_prompt_leaks": sum(bool(ep[key]["prompt_in_output"]) for key in ids),
        "image_plumbing_controls": {
            "blank_control": ep["blank_control"]["output_text"],
            "wrong_image_control": ep["wrong_image_control"]["output_text"],
        },
        "status": "IMAGE_PIPELINE_PASS_NUMERICAL_PARITY_PARTIAL",
        "caveat": "Manual image-dependent sanity is not MMStar/ChartQA benchmark reproduction.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("poc_mllm_ep_foundation/artifacts"))
    args = parser.parse_args()
    output = args.root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    profile, rows = build_profile(
        args.root / "profiling/ep4_component_matched_fixed_retry_20260924",
        args.root / "profiling/ep4_clean_matched_20260924",
        Path("poc_spatial_basis_ep/artifacts/20260924_spatial_basis_ep/baseline/baseline.json"),
    )
    atomic_json(output / "full_model_profile.json", profile)
    validation = build_calibration(rows, output)
    quality = build_quality(args.root)
    atomic_json(output / "quality_bridge.json", quality)
    atomic_json(output / "part1_analysis_summary.json", {
        "profile": profile,
        "validation": validation,
        "quality": quality,
    })
    print(json.dumps({"profile_gate": profile["profiling_gate"],
                      "simulator": validation["overall_status"],
                      "quality": quality["status"]}, indent=2))


if __name__ == "__main__":
    main()
