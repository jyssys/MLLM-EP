#!/usr/bin/env python3
"""Case/restart-level analysis; all E2E figures remain projected oracles."""

from __future__ import annotations

import csv
import json
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PRIOR = ROOT.parent / "poc_refinegemm"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def med(rows: list[float]) -> float:
    return statistics.median(rows)


def nsight_profile(name: str) -> dict:
    path = RESULTS / f"nsys_{name}.sqlite"
    con = sqlite3.connect(path)
    counts = {name: (count, avg) for name, count, avg in con.execute(
        "select s.value, count(*), avg((k.end-k.start)/1000.0) "
        "from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id "
        "where s.value in ('device_kernel','prepare_grouped_gemm_data',"
        "'elementwise_kernel') group by s.value"
    ).fetchall()}
    con.close()
    expected = {"device_kernel": 26, "prepare_grouped_gemm_data": 26,
                "elementwise_kernel": 39}
    if any(counts[k][0] != expected[k] for k in expected):
        raise RuntimeError(f"unexpected nsys kernels: {counts}")
    return {
        "device_kernel_us": counts["device_kernel"][1],
        "elementwise_us": counts["elementwise_kernel"][1],
        "grouped_descriptor_prep_us": counts["prepare_grouped_gemm_data"][1],
        "device_kernel_calls": counts["device_kernel"][0],
        "elementwise_calls": counts["elementwise_kernel"][0],
        "grouped_prep_calls": counts["prepare_grouped_gemm_data"][0],
        "boundary_gpu_us_per_full_stage": 3 * counts["elementwise_kernel"][1],
        "repeated_prep_gpu_us_per_full_stage": 2 * counts[
            "prepare_grouped_gemm_data"][1],
    }


def load_cases(scope: str) -> list[dict]:
    stage = defaultdict(list)
    components = defaultdict(list)
    attrs: dict[str, dict] = {}
    for restart in range(1, 4):
        directory = RESULTS / f"{scope}_all_r{restart}"
        for target, filename in [
            (stage, "ACTIVE_SUPPORT_RAW.csv"),
            (components, "MLP_COMPONENT_RAW.csv"),
        ]:
            for row in read_csv(directory / filename):
                key = row["case_id"]
                attrs[key] = {k: row[k] for k in ("task", "phase", "scope",
                                                   "total_rows", "active_experts",
                                                   "inactive_experts", "m_e_cv")}
                target[(key, restart, row["operation"])].append(
                    float(row["latency_ms"]))
    if any(len(samples) != 40 for samples in (*stage.values(), *components.values())):
        raise RuntimeError("incomplete independent repetitions")
    output = []
    for key, meta in attrs.items():
        times = {}
        for mapping in (stage, components):
            operations = {operation for case, _, operation in mapping if case == key}
            for operation in operations:
                times[operation] = med([med(mapping[(key, restart, operation)])
                                        for restart in range(1, 4)])
        base = times["torch_grouped_full64_prebuilt"]
        active = times["torch_grouped_active_prebuilt_oracle"]
        profile = nsight_profile("dense" if scope == "dense" else "compacted")
        clean_boundary = times["swiglu"] + times["route_weight"]
        gpu_boundary = (profile["boundary_gpu_us_per_full_stage"] +
                        profile["repeated_prep_gpu_us_per_full_stage"]) / 1000.0
        # O2: impossible zero-cost boundary, including useful activation
        # arithmetic; deliberately an upper bound, NOT a credible kernel target.
        o2 = min(base, clean_boundary + profile[
            "repeated_prep_gpu_us_per_full_stage"] / 1000.0)
        # O3: only a fraction of the GPU kernel envelope, avoiding double
        # counting materialization and preserving GEMM FLOPs and activation.
        o3 = min(o2, gpu_boundary, base) * 0.60
        # The actual gate/up HBM output writes have no device counter. This
        # separate impossible sensitivity adds a standalone full-intermediate
        # memcpy including its own launch, intentionally double-counting
        # SwiGLU reads and activated output. NOT a production-causal saving.
        memory_proxy = max(0.01875, 0.01390 * int(meta["total_rows"]) / 1024)
        o2_plus_memory = min(base, o2 + memory_proxy)
        o1 = min(base, max(0.0, base - active))
        # Never count offset rebuild with descriptor shortening, or copy
        # materialization alongside standalone activation timings.
        o4 = min(base, o1 + o3)
        output.append({
            "case_id": key, **meta, "restarts": 3, "repetitions_per_restart": 40,
            "torch_grouped_full64_ms": base,
            "torch_grouped_active_only_optimistic_ms": active,
            "vllm_fused_ms": times["vllm_fused"],
            "vllm_fused_default_warning": "yes",
            "gate_up_ms": times["gate_up_full64_prebuilt"],
            "down_ms": times["down_full64_prebuilt"],
            "swiglu_ms": times["swiglu"],
            "route_weight_ms": times["route_weight"],
            "clean_separate_boundary_ms": clean_boundary,
            "offset_gpu_rebuild_extra_ms": max(
                0, times["torch_grouped_full64_gpu_cumsum"] - base),
            "offset_cpu_list_rebuild_extra_ms": max(
                0, times["torch_grouped_full64_cpu_list_offsets"] - base),
            "active_only_optimistic_saved_ms": o1,
            "perfect_boundary_zero_cost_saved_ms": o2,
            "perfect_boundary_plus_memcpy_sensitivity_ms": o2_plus_memory,
            "extra_memcpy_proxy_includes_double_counting": "yes",
            "credible_boundary_saved_ms": o3,
            "combined_credible_saved_ms": o4,
            "combined_credible_expert_stage_pct": 100 * o4 / base,
            "o2_perfect_expert_stage_pct": 100 * o2 / base,
            "gate_up_output_roundtrip_analytic_bytes": int(meta["total_rows"]) * 8192,
            "activated_output_roundtrip_analytic_bytes": int(meta["total_rows"]) * 4096,
            "materialization_hbm_direct_measured": "no",
            "nsight_elementwise_gpu_us": profile["elementwise_us"],
            "nsight_descriptor_prep_gpu_us": profile[
                "grouped_descriptor_prep_us"],
        })
    return sorted(output, key=lambda row: row["case_id"])


def synthetic() -> list[dict]:
    stage = defaultdict(list)
    for restart in range(1, 4):
        rows = read_csv(RESULTS / f"support_r{restart}" / "ACTIVE_SUPPORT_RAW.csv")
        for row in rows:
            key = (int(row["total_rows"]), int(row["active_experts"]),
                   row["geometry"], restart, row["operation"])
            stage[key].append(float(row["latency_ms"]))
    output = []
    keys = sorted({key[:3] for key in stage})
    for total, active, geometry in keys:
        times = {
            operation: med([
                med(stage[(total, active, geometry, restart, operation)])
                for restart in range(1, 4)])
            for operation in ("torch_grouped_full64_prebuilt",
                              "torch_grouped_active_prebuilt_oracle",
                              "vllm_fused")
        }
        output.append({
            "total_rows": total, "active_experts": active,
            "inactive_experts": 64 - active, "geometry": geometry,
            "restarts": 3, "repetitions_per_restart": 40,
            "grouped_full_ms": times["torch_grouped_full64_prebuilt"],
            "active_only_oracle_ms": times[
                "torch_grouped_active_prebuilt_oracle"],
            "optimistic_descriptor_saved_pct": 100 * (
                times["torch_grouped_full64_prebuilt"] -
                times["torch_grouped_active_prebuilt_oracle"]) /
                times["torch_grouped_full64_prebuilt"],
            "vllm_fused_ms": times["vllm_fused"],
            "useful_routed_rows_fixed": "yes",
            "useful_expert_arithmetic_flops_fixed": "yes",
        })
    return output


def request_oracles(all_cases: list[dict]) -> list[dict]:
    breakdown = read_csv(PRIOR / "OPERATOR_BREAKDOWN.csv")
    breakdown_pie = {
        row["dataset"]: float(row["clean_e2e_upper_bound_percent"]) / 100.0
        for row in breakdown if row["component"] == "expert"
    }
    # Independent RefineGEMM replay matched same BF16 contract, winner grouped:
    prior_grouped_operator_gain = {"gsm8k": 0.2208, "humaneval": 0.2052}
    walls = defaultdict(list)
    for row in read_csv(PRIOR / "E2E_RESULTS.csv"):
        if row["repeat"] in ("1", "2", "3") and row["dataset"] in breakdown_pie:
            walls[row["dataset"]].append(float(row["wall_seconds"]))
    result = []
    for scope in ("dense", "compacted"):
        for task in breakdown_pie:
            cases = [c for c in all_cases if c["task"] == task and
                     (c["scope"] == "measured_dense_runtime") == (scope == "dense")]
            if not cases:
                raise RuntimeError((scope, task))
            # Summation weighted by whole-stage GPU time, NOT unweighted mean
            # of tiny late cases. Corpus includes sampled real owner-rank
            # invocations, not exhaustive clean request event totals.
            denom = sum(c["torch_grouped_full64_ms"] for c in cases)
            sums = {
                "O1_active_only": sum(c["active_only_optimistic_saved_ms"]
                                      for c in cases),
                "O2_perfect_boundary": sum(c[
                    "perfect_boundary_zero_cost_saved_ms"] for c in cases),
                "O2_plus_all_materialization_unrealistic": sum(c[
                    "perfect_boundary_plus_memcpy_sensitivity_ms"]
                    for c in cases),
                "O3_credible_boundary": sum(c["credible_boundary_saved_ms"]
                                           for c in cases),
                "O4_combined_credible": sum(c["combined_credible_saved_ms"]
                                           for c in cases),
            }
            pie = breakdown_pie[task]
            operator_gain = prior_grouped_operator_gain[task]
            projected_post_grouped_pie = pie * (1 - operator_gain) / (
                1 - pie * operator_gain)
            for oracle, saved in sums.items():
                stage_pct = 100 * saved / denom
                # Amdahl optimistic full-request mapping: measured prior
                # observer pie / clean E2E, assumed perfect feasible grouped
                # replacement, representative replay distribution. This is
                # NOT an observed request-level gain.
                result.append({
                    "task": task, "scope": scope, "oracle": oracle,
                    "real_owner_rank_cases": len(cases),
                    "restart_unit": 3, "clean_prior_bct_seconds_median":
                    med(walls[task]), "observer_attributed_production_expert_pie_pct":
                    100 * pie, "prior_grouped_operator_replay_gain_pct":
                    100 * operator_gain, "hypothetical_post_grouped_expert_pie_pct":
                    100 * projected_post_grouped_pie,
                    "measured_replay_expert_stage_saved_pct": stage_pct,
                    "projected_optimistic_request_e2e_upper_bound_pct":
                    projected_post_grouped_pie * stage_pct,
                    "evidence": ("future-known compacted worklist sensitivity; "
                                 "post-Epoch latency NOT measured" if scope == "compacted"
                                 else "dense real owner-rank replay; projected "
                                 "full-request ceiling NOT measured"),
                })
    return result


def summarize_metadata() -> list[dict]:
    timings = defaultdict(list)
    for restart in range(1, 4):
        for row in read_csv(RESULTS / f"dense_all_r{restart}" /
                            "METADATA_PREP_RAW.csv"):
            timings[(int(row["active_experts"]), restart,
                     row["operation"])].append(float(
                         row["host_visible_latency_us"]))
    operations = {key[2] for key in timings}
    return [{
        "active_experts": active, "operation": name, "restart_median_us":
        med([med(timings[(active, restart, name)])
             for restart in range(1, 4)]),
        "scope": ("standalone host-visible CPU/GPU synchronization benchmark; "
                  "NOT removable production request latency")
    } for active in (1, 2, 4, 8, 16, 32, 64) for name in sorted(operations)]


def materialization_sensitivity() -> list[dict]:
    copies = defaultdict(list)
    for restart in range(1, 4):
        for row in read_csv(RESULTS / f"dense_all_r{restart}" /
                            "COPY_BANDWIDTH_RAW.csv"):
            copies[(int(row["total_rows"]), restart)].append(float(
                row["latency_ms"]))
    return [{
        "rows": rows,
        "gate_up_roundtrip_bytes": rows * 8192,
        "activated_intermediate_roundtrip_bytes": rows * 4096,
        "total_analytic_roundtrip_bytes": rows * 12288,
        "separate_copy_gpu_latency_ms": med([
            med(copies[(rows, restart)]) for restart in range(1, 4)]),
        "hardware_io_counter_available": "no",
        "evidence": ("standalone buffer memcpy proxy includes its own launch; "
                     "not existing HBM GEMM traffic or independently saved work")
    } for rows in (64, 128, 256, 512, 1024)]


def gpu_time_log() -> list[dict]:
    output = []
    for scope in ("support", "dense_all", "compacted_all"):
        for restart in range(1, 4):
            audit_file = RESULTS / f"{scope}_r{restart}" / "AUDIT.json"
            audit = json.loads(audit_file.read_text())
            output.append({
                "completion_kst_from_file_mtime": datetime.fromtimestamp(
                    audit_file.stat().st_mtime).astimezone().isoformat(),
                "session": scope, "restart": restart,
                "physical_gpu_used": audit["physical_gpu"],
                "visible_gpus": audit["cuda_visible_devices"],
                "uuid": audit["physical_uuid"],
                "case_count": audit.get("case_count", "see benchmark output"),
                "measured_kernel_loop_seconds_excludes_import_load": audit[
                    "wall_seconds"],
                "independent_process": "yes",
                "burn_during_measurement": "no",
            })
    return output


def nsight_kernels() -> list[dict]:
    output = []
    for scope in ("dense", "compacted"):
        entry = nsight_profile(scope)
        for name, call_field, duration_field in (
            ("grouped_GEMM", "device_kernel_calls", "device_kernel_us"),
            ("grouped_descriptor_prep", "grouped_prep_calls",
             "grouped_descriptor_prep_us"),
            ("elementwise", "elementwise_calls", "elementwise_us"),
        ):
            output.append({
                "scope": scope,
                "kernel": name,
                "calls_13_forward_passes": entry[call_field],
                "gpu_mean_duration_us": entry[duration_field],
                "observer": "Nsight Systems GPU kernel; not clean request E2E",
                "raw_trace_ignored_from_git": f"results/nsys_{scope}.nsys-rep",
            })
    return output


def phase_summary(cases: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in cases:
        groups[(row["scope"], row["task"], row["phase"])].append(row)
    result = []
    for (scope, task, phase), rows in sorted(groups.items()):
        denom = sum(r["torch_grouped_full64_ms"] for r in rows)
        per = {}
        for target in ("gate_up_ms", "down_ms", "swiglu_ms",
                       "route_weight_ms", "active_only_optimistic_saved_ms",
                       "credible_boundary_saved_ms"):
            per[target.replace("_ms", "_pct_of_stage")] = (
                100 * sum(r[target] for r in rows) / denom)
        result.append({
            "scope": scope, "task": task, "phase": phase, "cases": len(rows),
            "total_operator_stage_ms": denom,
            "mean_active_experts": statistics.mean(
                int(r["active_experts"]) for r in rows),
            "mean_inactive_experts": statistics.mean(
                int(r["inactive_experts"]) for r in rows),
            "mean_local_expert_rows": statistics.mean(
                int(r["total_rows"]) for r in rows),
            **per,
            "components_evidence": "standalone components; not jointly additive",
        })
    return result


def main() -> None:
    for name in ("dense", "compacted"):
        nsight_profile(name)
    controls = synthetic()
    real = load_cases("dense") + load_cases("compacted")
    write_csv(ROOT / "SYNTHETIC_CONTROLS.csv", controls)
    write_csv(ROOT / "REAL_REFINEMENT_MAPPING.csv", real)
    write_csv(ROOT / "REQUEST_ORACLES.csv", request_oracles(real))
    write_csv(ROOT / "METADATA_SUMMARY.csv", summarize_metadata())
    write_csv(ROOT / "MATERIALIZATION_SENSITIVITY.csv",
              materialization_sensitivity())
    write_csv(ROOT / "GPU_TIME_LOG.csv", gpu_time_log())
    write_csv(ROOT / "NSIGHT_KERNELS.csv", nsight_kernels())
    write_csv(ROOT / "PHASE_COST_SUMMARY.csv", phase_summary(real))
    checks = []
    for scope in ("dense", "compacted"):
        for restart in range(1, 4):
            checks.extend(read_csv(RESULTS / f"{scope}_all_r{restart}" /
                                   "CORRECTNESS.csv"))
    metrics = {
        "synthetic_control_cases": len(controls),
        "dense_real_cases": len([r for r in real if r["scope"] ==
                                 "measured_dense_runtime"]),
        "compacted_real_cases": len(real) - len([
            r for r in real if r["scope"] == "measured_dense_runtime"]),
        "min_cosine_fused_vs_grouped": min(float(c["cosine"]) for c in checks),
        "max_relative_l2_fused_vs_grouped": max(float(c["relative_l2"])
                                               for c in checks),
        "nsight_dense": nsight_profile("dense"),
        "nsight_compacted": nsight_profile("compacted"),
    }
    (ROOT / "ANALYSIS_SUMMARY.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
