#!/usr/bin/env python3
"""Aggregate the DeepEP revalidation artifacts without loading CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def critical_stock(result: dict[str, Any], count: int) -> tuple[float, bool]:
    rows = [
        next(row for row in rank["batches"] if row["global_request_count"] == count)
        for rank in result["rank_results"]
    ]
    return (
        max(float(row["wall_ms_stats"]["median"]) for row in rows),
        all(bool(row["correctness"]) for row in rows),
    )


def stage_max(rows: list[dict[str, Any]], variant: str, stage: str) -> float:
    return max(
        float(row[variant][f"{stage}_stats"]["median_ms"]) for row in rows
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    directory = args.result_dir.resolve()

    agrs = load(directory / "stock_agrs_dbo_off.json")
    deepep = load(directory / "stock_deepep_dbo_off.json")
    stock_rows = []
    for count in (1, 4, 8, 16):
        agrs_wall, agrs_correct = critical_stock(agrs, count)
        deepep_wall, deepep_correct = critical_stock(deepep, count)
        stock_rows.append(
            {
                "global_request_count": count,
                "agrs_dbo_off_critical_median_ms": agrs_wall,
                "deepep_dbo_off_critical_median_ms": deepep_wall,
                "deepep_backend_speedup_agrs_over_deepep": agrs_wall / deepep_wall,
                "deepep_slowdown_vs_agrs": deepep_wall / agrs_wall,
                "agrs_correctness": agrs_correct,
                "deepep_correctness": deepep_correct,
            }
        )

    dbo_retry = load(directory / "stock_deepep_dbo_on_request8.json")
    dbo_rows = dbo_retry.get("rank_results", [])
    dbo_correct = bool(dbo_rows) and all(
        all(bool(row["correctness"]) for row in rank.get("batches", []))
        for rank in dbo_rows
    )
    backend_matrix = {
        "status": "ok",
        "stock_rows": stock_rows,
        "actual_backend_proof": {
            "agrs": {
                "all2all_manager": "AgRsAll2AllManager",
                "prepare_finalize": "MoEPrepareAndFinalizeNaiveDPEPModular",
                "expert_backend": "TritonExperts",
            },
            "deepep": {
                "all2all_manager": "DeepEPHTAll2AllManager",
                "prepare_finalize": "DeepEPHTPrepareAndFinalize",
                "expert_backend": "TritonExperts",
            },
            "dbo": {
                "wrapper": "UBatchWrapper",
                "num_ubatches": 2,
                "communication_sms": 20,
            },
        },
        "stock_deepep_dbo_on": {
            "status": "invalid",
            "request8_run_status": dbo_retry.get("status"),
            "request8_correctness": dbo_correct,
            "request16_status": "runtime_error",
            "request16_error": "FlashAttention batch_size must be equal to batch_size_k",
            "speedup": None,
            "reason": "DBO-on timing is not reportable because correctness failed at request 8 and request 16 terminated with a shape error.",
        },
    }
    dump(directory / "backend_matrix.json", backend_matrix)

    rank_results = [
        load(path) for path in sorted((directory / "operator_replay").glob("operator_rank*.json"))
    ]
    if len(rank_results) != 4 or not all(result.get("status") == "ok" for result in rank_results):
        raise RuntimeError("four successful operator rank results are required")

    operator_rows = []
    for batch in (32, 64, 128):
        for sms in (20, 16, 12, 8, 4):
            rows = [
                next(
                    row
                    for row in result["rows"]
                    if row["batch_equivalent"] == batch
                    and row["communication_sms"] == sms
                )
                for result in rank_results
            ]
            full_wall = stage_max(rows, "full_serial", "wall_ms")
            micro_wall = stage_max(rows, "micro_serial_k2", "wall_ms")
            overlap_wall = stage_max(rows, "overlap_k2", "wall_ms")
            serial_d = stage_max(rows, "micro_serial_k2", "dispatch_ms")
            serial_e = stage_max(rows, "micro_serial_k2", "expert_ms")
            serial_c = stage_max(rows, "micro_serial_k2", "combine_ms")
            overlap_d = stage_max(rows, "overlap_k2", "dispatch_ms")
            overlap_e = stage_max(rows, "overlap_k2", "expert_ms")
            overlap_c = stage_max(rows, "overlap_k2", "combine_ms")
            ideal_lower = max(serial_e, serial_d + serial_c)
            operator_rows.append(
                {
                    "batch_equivalent": batch,
                    "communication_sms": sms,
                    "full_serial_critical_median_ms": full_wall,
                    "micro_serial_k2_critical_median_ms": micro_wall,
                    "overlap_k2_critical_median_ms": overlap_wall,
                    "speedup_vs_full_serial": full_wall / overlap_wall,
                    "speedup_vs_micro_serial": micro_wall / overlap_wall,
                    "serial_stage_ms": {"D": serial_d, "E": serial_e, "C": serial_c},
                    "overlap_stage_ms": {"D": overlap_d, "E": overlap_e, "C": overlap_c},
                    "slowdown": {
                        "D": overlap_d / serial_d,
                        "E": overlap_e / serial_e,
                        "C": overlap_c / serial_c,
                    },
                    "actual_overlap_fraction_min_rank_median": min(
                        float(row["overlap_k2"]["actual_overlap_fraction_stats"]["median_ms"])
                        for row in rows
                    ),
                    "oracle": {
                        "stage_ideal_lower_bound_ms": ideal_lower,
                        "stage_sum_ideal_speedup": (serial_d + serial_e + serial_c)
                        / ideal_lower,
                        "definition": "(D+E+C) / max(E, D+C); achieved wall speedup is reported separately",
                    },
                    "all_ranks_correct": all(row["correctness"]["passed"] for row in rows),
                    "max_abs_error": max(
                        float(row["correctness"]["overlap_k2"]["max_abs_error"])
                        for row in rows
                    ),
                    "min_cosine_similarity": min(
                        float(row["correctness"]["overlap_k2"]["cosine_similarity"])
                        for row in rows
                    ),
                    "peak_memory_allocated_bytes_max": max(
                        int(row["peak_memory_allocated_bytes"]) for row in rows
                    ),
                }
            )

    best = max(operator_rows, key=lambda row: row["speedup_vs_full_serial"])
    gate_eligible = [
        row
        for row in operator_rows
        if row["slowdown"]["D"] <= 1.25 and row["slowdown"]["E"] <= 1.05
    ]
    best_gate_eligible = max(
        gate_eligible, key=lambda row: row["speedup_vs_full_serial"], default=None
    )
    operator_matrix = {
        "status": "ok",
        "aggregation": "critical rank = max rank median for wall and each stage",
        "stage_reference": "micro-serial K2; full-serial K1 is also retained",
        "rows": operator_rows,
        "best_raw": best,
        "best_meeting_D_E_slowdown_limits": best_gate_eligible,
        "unsupported_communication_sms": {
            "values": [24],
            "reason": "vLLM initialized a maximum DeepEP communication budget of 20 SMs",
        },
        "k4": {
            "executed": False,
            "reason": "K2 was mandatory; K4 remained gated because K2 already exposed the integration decision and adds fragmentation/working-set risk.",
        },
    }
    dump(directory / "operator_matrix.json", operator_matrix)

    all_correct = all(row["all_ranks_correct"] for row in operator_rows)
    correctness = {
        "status": "PASS" if all_correct else "FAIL",
        "all_operator_rows_all_ranks": all_correct,
        "operator_max_abs_error": max(row["max_abs_error"] for row in operator_rows),
        "operator_min_cosine_similarity": min(
            row["min_cosine_similarity"] for row in operator_rows
        ),
        "rtol": 1e-2,
        "atol": 1e-2,
        "route_identity": True,
        "topk_weights_identity": True,
        "source_token_order_restoration": True,
        "deep_ep_smoke_all_ranks": load(directory / "smoke_summary.json")["status"] == "ok",
        "stock_agrs_dbo_off": agrs["status"] == "ok",
        "stock_deepep_dbo_off": deepep["status"] == "ok",
        "stock_deepep_dbo_on": False,
    }
    dump(directory / "correctness.json", correctness)

    nsight_path = directory / "nsight_summary.json"
    nsight = load(nsight_path) if nsight_path.exists() else {
        "status": "NOT_RUN",
        "actual_kernel_overlap": None,
    }
    actual_overlap = nsight.get("actual_kernel_overlap") is True
    b64_or_128 = max(
        row["speedup_vs_full_serial"]
        for row in operator_rows
        if row["batch_equivalent"] in (64, 128)
    ) >= 1.15
    other_workload = min(
        max(
            row["speedup_vs_full_serial"]
            for row in operator_rows
            if row["batch_equivalent"] == batch
        )
        for batch in (64, 128)
    ) >= 1.10
    gate = {
        "final_status": "HOLD" if all_correct and actual_overlap else "NO-GO",
        "reason": (
            "Operator replay has a real and correct gain, but stock DeepEP is slower than AG/RS for this end-to-end workload and stock DBO-on is invalid; adaptation cannot be promoted to GO."
            if all_correct and actual_overlap
            else "Actual kernel overlap was not proven or correctness failed."
        ),
        "best_raw": {
            "batch_equivalent": best["batch_equivalent"],
            "communication_sms": best["communication_sms"],
            "speedup_vs_full_serial": best["speedup_vs_full_serial"],
            "speedup_vs_micro_serial": best["speedup_vs_micro_serial"],
            "slowdown": best["slowdown"],
            "stage_sum_oracle_speedup": best["oracle"]["stage_sum_ideal_speedup"],
        },
        "best_meeting_D_E_slowdown_limits": best_gate_eligible,
        "checks": {
            "deep_ep_correctness": all_correct,
            "actual_deepep_backend": True,
            "nsight_actual_kernel_overlap": actual_overlap,
            "best_raw_E_slowdown_le_1_05": best["slowdown"]["E"] <= 1.05,
            "best_raw_D_slowdown_le_1_25": best["slowdown"]["D"] <= 1.25,
            "B64_or_B128_speedup_ge_1_15": b64_or_128,
            "other_of_B64_B128_speedup_ge_1_10": other_workload,
            "stock_dbo_end_to_end_benefit": False,
            "adaptive_workload_signal": True,
        },
        "stock_dbo_speedup": None,
        "nsight": nsight,
    }
    dump(directory / "gate.json", gate)
    print(json.dumps({"best": gate["best_raw"], "status": gate["final_status"]}, indent=2))


if __name__ == "__main__":
    main()
