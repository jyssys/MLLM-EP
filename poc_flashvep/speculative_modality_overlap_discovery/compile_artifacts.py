"""Compile fresh partial-output measurements into the discovery artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--result-dir", type=Path, required=True); ap.add_argument("--code-dir", type=Path, required=True)
    args = ap.parse_args(); result, code = args.result_dir, args.code_dir
    summary = read_csv(result / "partial_quality_summary.csv")
    measured = []
    for row in summary:
        measured.append({"candidate": f"{row['method']}_{row['variant']}", "scope": "partial expert output", "modality": row["modality"], "tokens": row["tokens"], "mean_cosine": row["cosine_mean"], "median_l2": row["relative_l2_p50"], "mean_l2": row["relative_l2_mean"], "p90_l2": row["relative_l2_p90"], "quality_pass_rate": row["pass_rate"], "mean_selected_k": row["selected_k_mean"], "mean_router_mass": row["router_mass_mean"], "evidence": "FRESH_LIVE_QWEN3VL_DEEPEP", "notes": "Expert-output-level only; no next-layer/logit propagation."})
    diagnostics = [
        {"candidate": "specmoe_affinity_same_expert_mean", "scope": "output-affinity surrogate", "modality": "visual+text", "tokens": "473", "mean_cosine": "see surrogate_summary.csv", "median_l2": "UNKNOWN", "mean_l2": "see surrogate_summary.csv", "p90_l2": "see surrogate_summary.csv", "quality_pass_rate": "UNKNOWN", "mean_selected_k": "top4/6/7", "mean_router_mass": "preserved", "evidence": "FRESH_LIVE_OUTPUT_DIAGNOSTIC", "notes": "Same-expert output mean; hidden/weight affinity not captured; not a SpecMoE novelty claim."},
        {"candidate": "spatial_neighbor", "scope": "output-affinity surrogate", "modality": "visual", "tokens": "447", "mean_cosine": "see surrogate_summary.csv", "median_l2": "UNKNOWN", "mean_l2": "see surrogate_summary.csv", "p90_l2": "see surrogate_summary.csv", "quality_pass_rate": "UNKNOWN", "mean_selected_k": "top4/6/7", "mean_router_mass": "preserved", "evidence": "FRESH_LIVE_OUTPUT_DIAGNOSTIC", "notes": "Nearest same-expert visual token excluding self; no 2-D grid metadata in compact capture."},
        {"candidate": "cross_layer_residual", "scope": "cross-layer", "modality": "all", "tokens": "0", "mean_cosine": "UNKNOWN", "median_l2": "UNKNOWN", "mean_l2": "UNKNOWN", "p90_l2": "UNKNOWN", "quality_pass_rate": "UNKNOWN", "mean_selected_k": "N/A", "mean_router_mass": "N/A", "evidence": "NOT_CAPTURED", "notes": "No hidden state at successive residual boundaries."},
    ]
    pareto = measured + diagnostics
    write_csv(code / "QUALITY_PARETO.csv", pareto); write_csv(result / "QUALITY_PARETO.csv", pareto)
    oracle = []
    for row in summary:
        candidate = f"{row['method']}_{row['variant']}"
        quality = float(row["pass_rate"])
        # A downstream overlap claim is only eligible when both modalities
        # retain >=95% of tokens within the preregistered quality gate.
        oracle.append({"candidate": candidate, "scope": "one MoE boundary", "provisional_ready_ms": "UNKNOWN", "exact_ready_ms": "UNKNOWN", "created_window_ms": "UNKNOWN", "eligible_downstream_ms": "UNKNOWN", "perfect_e2e_oracle_pct": "0" if quality < .95 else "0 (no expert work omitted at mass99)", "feasible_e2e_oracle_pct": "0", "quality_gate": "PASS" if quality >= .95 else "FAIL", "headroom_gate": "KILL" if quality < .95 else "NO_SAVED_WORK", "evidence": "FRESH_LIVE_QWEN3VL_DEEPEP", "notes": "No downstream timing was captured; quality failure or k≈8 prevents a non-zero safe window."})
    oracle.append({"candidate": "all_quality_safe_partial", "scope": "cross-modality", "provisional_ready_ms": "UNKNOWN", "exact_ready_ms": "UNKNOWN", "created_window_ms": "0", "eligible_downstream_ms": "UNKNOWN", "perfect_e2e_oracle_pct": "0", "feasible_e2e_oracle_pct": "0", "quality_gate": "PASS only at mass99/top8", "headroom_gate": "KILL", "evidence": "FRESH_LIVE_QWEN3VL_DEEPEP", "notes": "The only >=95% pass operating point across visual and text is effectively exact top-8; no overlap headroom."})
    write_csv(code / "OVERLAP_ORACLE.csv", oracle); write_csv(result / "OVERLAP_ORACLE.csv", oracle)
    for src in (code / "SPECMOE_OVERLAP_MATRIX.md", code / "PRIOR_ART_MATRIX.md"):
        shutil.copy2(src, result / src.name)
    workload = json.loads((result / "manifest.json").read_text())
    workload["capture_evidence"] = {"fresh_live": True, "capture_attempts": ["conda_no_deep_ep_failed", "deepep_without_hook_pass", "deepep_with_worker_hook_pass"], "hook": "explicit worker-local instrumentation", "raw_files": len(list((result / "raw").glob("*.npz")))}
    (result / "workload_manifest.json").write_text(json.dumps(workload, indent=2) + "\n")
    gate = {"status": "SEARCH_SPACE_NO_GO", "quality_gate": "FAIL for top1/top2/top4/top6/top7 and mass<=95 on text; mass99 retains all 8", "overlap_gate": "KILL before downstream speculation", "qwen3_vl_fresh_samples": int(workload.get("policy", {}).get("samples_per_category", 0) * 3), "next_layer_or_logits": "NOT_CAPTURED", "killed_branches": ["partial completion", "router-mass threshold below exact", "output-affinity surrogate", "spatial neighbour", "speculative overlap"], "evidence_boundary": "fresh live Qwen3-VL DeepEP expert-output capture; no model-output equivalence or E2E speedup claim"}
    (result / "gate_summary.json").write_text(json.dumps(gate, indent=2) + "\n")
    # Keep a compact copy of the capture command/log outside the raw tensors.
    for suffix in (".profiling_command.txt", ".capture.log"):
        src = Path(str(result) + suffix)
        if src.exists(): shutil.copy2(src, result / suffix.lstrip("."))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__": main()
