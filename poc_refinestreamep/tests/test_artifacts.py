from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_required_artifacts_exist():
    required = [
        "M_CROSSOVER.csv", "CAPACITY_MATRIX.csv", "INFLIGHT_RESULTS.csv",
        "REFINEMENT_STREAMS.jsonl", "SERVING_RESULTS.csv", "SERVING_ORACLES.csv",
        "ATTEMPT_LOG.csv", "reports/final_decision.md",
    ]
    assert all((ROOT / name).exists() for name in required)
    assert len(list((ROOT / "figures").glob("*.png"))) >= 17


def test_crossover_and_real_range():
    c = pd.read_csv(ROOT / "M_CROSSOVER.csv")
    p = c.pivot_table(index=["global_m", "routing"], columns="policy",
                      values="transaction_ms").reset_index()
    assert (p[p.global_m <= 4096].low_latency < p[p.global_m <= 4096].normal).all()
    assert (p[p.global_m == 8192].normal < p[p.global_m == 8192].low_latency).all()
    stream = pd.read_json(ROOT / "REFINEMENT_STREAMS.jsonl", lines=True)
    assert stream.global_m.max() == 1024


def test_inflight_correctness_and_no_normal_reversal():
    d = pd.read_csv(ROOT / "INFLIGHT_RESULTS.csv")
    real = d[d.workload == "real_age_offset"]
    assert real.max_relative_l2.max() < 1e-3
    q16 = real[real.q == 16].set_index("policy")
    assert q16.loc["ll_two_slot", "waves_per_s"] > q16.loc["ll_serial", "waves_per_s"]
    assert q16.loc["ll_two_slot", "waves_per_s"] > q16.loc["normal_serial", "waves_per_s"]


def test_kernel_gate_fails():
    d = pd.read_csv(ROOT / "SERVING_ORACLES.csv")
    o4 = d[(d.policy == "o4_credible_refinestreamep") & (d.arrival == "poisson")]
    assert o4.throughput_gain_vs_ll_pct.max() < 10
    assert o4.p99_reduction_vs_ll_pct.max() < 8
