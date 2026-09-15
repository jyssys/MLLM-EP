#!/usr/bin/env python3
"""Fail-fast checks for the RefineGEMM evidence package."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    for scope in ("dense", "compacted"):
        assert len(list(ROOT.glob(f"kernel2_{scope}_r*/EXISTING_EXPERT_KERNELS.csv"))) == 3

    compact = pd.read_csv(ROOT / "EXPERT_SHAPE_SUMMARY.csv")
    global_rows = compact[compact["rank"].eq(-1)]
    assert (global_rows["assignments"] == 8 * global_rows["fresh_m"]).all()

    correctness = pd.read_csv(ROOT / "KERNEL_CORRECTNESS.csv")
    assert correctness.relative_l2.max() <= 5e-3
    assert correctness.cosine.min() >= 0.9999

    kernels = pd.read_csv(ROOT / "EXISTING_EXPERT_KERNELS.csv")
    real = kernels[kernels.geometry.eq("real")]
    pivot = real.pivot_table(
        index=["scope", "restart", "case_id"], columns="backend", values="latency_ms"
    )
    hybrid = [column for column in pivot if column.startswith("hybrid_")]
    assert len(pivot) == 102
    assert (pivot.torch_grouped <= pivot.vllm_fused).all()
    assert not (pivot[hybrid].min(axis=1) < pivot[["torch_grouped", "vllm_fused"]].min(axis=1)).any()

    oracles = pd.read_csv(ROOT / "REFINEGEMM_ORACLES.csv")
    novel = oracles[oracles.oracle.str.startswith("O")]
    assert novel.request_e2e_gain_pct.max() < 5.0

    required = [
        "OPERATOR_BREAKDOWN.csv", "EXPERT_SHAPE_SUMMARY.csv", "EXPERT_ROW_TRACE.parquet",
        "SINGLE_EXPERT_SWEEP.csv", "EXPERT_REPLAY_SUMMARY.csv",
        "EXISTING_EXPERT_KERNELS.csv", "REFINEGEMM_ORACLES.csv",
        "REFINEGEMM_MICROBENCH.csv", "E2E_RESULTS.csv", "ATTEMPT_LOG.csv",
    ]
    for name in required:
        assert (ROOT / name).is_file(), name
    reports = [
        "00_environment_and_baseline.md", "01_operator_breakdown.md",
        "02_expert_row_trace.md", "03_refinement_regime_atlas.md",
        "04_existing_kernel_envelope.md", "05_single_expert_sweep.md",
        "06_whole_invocation_replay.md", "07_switching_oracle.md",
        "08_hybrid_oracle.md", "09_refinegemm_design.md",
        "10_refinegemm_correctness.md", "11_refinegemm_microbench.md",
        "12_end_to_end_integration.md", "13_prior_art_and_novelty.md",
        "final_decision.md",
    ]
    for name in reports:
        assert (ROOT / "reports" / name).is_file(), name
    print("RefineGEMM invariants: PASS")


if __name__ == "__main__":
    main()
