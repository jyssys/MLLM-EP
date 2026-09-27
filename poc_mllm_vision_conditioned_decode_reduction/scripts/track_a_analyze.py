from __future__ import annotations

import argparse
import json
from pathlib import Path

from poc_mllm_vision_conditioned_decode_reduction.track_a import (
    load_capture_groups,
    oracle_frontier,
)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _render(result: dict, integrity: dict) -> str:
    lines = [
        "# Track A — Oracle headroom",
        "",
        f"Status: **{result['status']}**",
        "",
        "The frozen long-reasoning route traces contain expert IDs and router weights, "
        "but no per-path expert output vectors. Route overlap alone cannot establish "
        "whether re-routing is semantically safe.",
        "",
        "## Capture integrity",
        "",
        f"- Shards: {integrity['shards']}",
        f"- Complete top-8 token/layer groups: {integrity['complete_top8_groups']}",
        f"- Unmatched receiver rows: {integrity['unmatched_receiver_rows']}",
        f"- Ambiguous receiver rows: {integrity['ambiguous_receiver_rows']}",
        f"- Incomplete groups: {integrity['incomplete_top8_groups']}",
        f"- Discarded non-decode router rows: {integrity['filtered_non_decode_router_rows']}",
        f"- Receiver shards with filename/canonical-step mismatch: {integrity['step_mismatch_shards']}",
        f"- Receiver rows with filename/canonical-step mismatch: {integrity['step_mismatch_rows']}",
        "",
    ]
    if result["status"] != "MEASURED_BOUNDED_VECTOR_ORACLE":
        lines += [
            "## Gate",
            "",
            "`NOT_EVALUATED`: a bounded diagnostic vector capture is required. "
            "No Track-A quality or physical runtime stage is authorized from the "
            "available routing histograms.",
            "",
        ]
        return "\n".join(lines)
    lines += [
        "## Bounded oracle frontier",
        "",
        "| Requested path reduction | Actual paths | Active-union reduction | Mean output cosine | P90 weighted contribution rel-L2 | P90 mixture rel-L2 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["frontier"]:
        lines.append(
            f"| {100*row['requested_reduction']:.1f}% | {100*row['actual_path_reduction']:.2f}% "
            f"| {100*row['active_expert_union_reduction']:.2f}% "
            f"| {row['mean_expert_output_cosine']:.6f} "
            f"| {row['p90_weighted_contribution_relative_l2']:.6f} "
            f"| {row['p90_mixture_relative_l2']:.6f} |"
        )
    lines += [
        "",
        f"Stage-A local-vector gate: **{'PASS' if result['gate_pass'] else 'FAIL'}**.",
        "",
        "This oracle is not a full-model quality result and cannot authorize timing "
        "unless the separate natural-generation quality frontier also passes.",
        "",
    ]
    return "\n".join(lines)


def _write_stage_boundary_reports(report_dir: Path, artifact_dir: Path, result: dict) -> None:
    measured = result["status"] == "MEASURED_BOUNDED_VECTOR_ORACLE"
    gate_pass = bool(result["gate_pass"])
    if not measured:
        quality_status = "NOT_RUN_VECTOR_CAPTURE_UNAVAILABLE"
        runtime_status = "NOT_RUN_STAGE_A_UNRESOLVED"
        ep8_status = "NOT_RUN_NO_MODIFIED_ROUTE_MASK"
        verdict = "A_RUNTIME_BLOCKED"
        reason = (
            "The pre-existing traces have no Expert_e(h) vectors. The first shared "
            "bounded capture smoke failed before model load because the local Hugging "
            "Face snapshot was rejected for missing repository-only metadata files."
        )
    elif not gate_pass:
        quality_status = "NOT_RUN_STAGE_A_ORACLE_FAILED"
        runtime_status = "NOT_RUN_BY_SPEC_GATE"
        ep8_status = "STRUCTURAL_EP8_PROJECTION_ONLY_NOT_RUN_NO_PROMOTED_POLICY"
        verdict = "A_INSUFFICIENT_CONSOLIDATION_HEADROOM"
        reason = "The bounded output-similarity oracle did not pass its frozen local-vector gate."
    else:
        quality_status = "PENDING_REQUIRED_FULL_MODEL_QUALITY_FRONTIER"
        runtime_status = "NOT_RUN_PENDING_QUALITY_GATE"
        ep8_status = "STRUCTURAL_EP8_PROJECTION_ONLY_PENDING_FROZEN_ROUTE_MASK"
        verdict = "UNRESOLVED_STAGE_A_PASS"
        reason = "The local-vector oracle passed; full-model quality must run before physical timing."

    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    stages = {
        "quality_frontier": {
            "status": quality_status,
            "policies": ["A0", "A1", "A2", "A3", "A4"],
            "reason": reason,
            "heldout_threshold_tuning": False,
            "evidence_label": "NOT_MEASURED" if not quality_status.startswith("PENDING") else "PENDING",
        },
        "ep2_ep4_runtime": {
            "status": runtime_status,
            "ep2_tpot_gain": None,
            "ep4_tpot_gain": None,
            "simulator_substitution": False,
            "reason": reason,
            "evidence_label": "NOT_RUN",
        },
        "ep8_structural": {
            "status": ep8_status,
            "latency": None,
            "tpot": None,
            "e2e": None,
            "reason": reason,
            "evidence_label": "STRUCTURAL_EP8_PROJECTION_ONLY",
        },
        "result": {
            "oracle_status": result["status"],
            "oracle_gate_pass": gate_pass,
            "quality_status": quality_status,
            "runtime_status": runtime_status,
            "ep8_status": ep8_status,
            "track_a_verdict": verdict,
            "reason": reason,
        },
    }
    for name, payload in stages.items():
        _atomic_json(artifact_dir / f"{name}.json", payload)

    (report_dir / "track_a_quality_frontier.md").write_text(
        "# Track A — Quality frontier\n\n"
        f"Status: **{quality_status}**\n\n{reason}\n\n"
        "No local vector similarity result is reported as task accuracy. A0–A4 "
        "natural long-reasoning evaluation is permitted only after the Stage-A oracle gate.\n"
    )
    (report_dir / "track_a_ep2_ep4_runtime.md").write_text(
        "# Track A — TRUE EP2 / EP4 runtime\n\n"
        f"Status: **{runtime_status}**\n\n{reason}\n\n"
        "EP2_TPOT_GAIN: N/A\n\nEP4_TPOT_GAIN: N/A\n\n"
        "No simulator value is substituted for physical timing.\n"
    )
    (report_dir / "track_a_ep8_structural.md").write_text(
        "# Track A — EP8 structural replay\n\n"
        f"Status: **{ep8_status}**\n\n{reason}\n\n"
        "Evidence boundary: `STRUCTURAL_EP8_PROJECTION_ONLY`. Latency, TPOT, "
        "E2E, and speedup are intentionally N/A.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    groups, integrity = load_capture_groups(args.capture_dir)
    result = oracle_frontier(groups)
    payload = {**result, "capture_integrity": integrity}
    _atomic_json(args.artifact, payload)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_render(result, integrity))
    _write_stage_boundary_reports(args.report.parent, args.artifact.parent, result)
    print(json.dumps({
        "status": result["status"],
        "groups": len(groups),
        "gate_pass": result["gate_pass"],
    }, indent=2))


if __name__ == "__main__":
    main()
