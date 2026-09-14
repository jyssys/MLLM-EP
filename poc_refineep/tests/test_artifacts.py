from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_shape_atlas_is_complete_and_validated():
    atlas = pd.read_csv(ROOT / "REFINEMENT_EP_SHAPES.csv")
    assert len(atlas) == 4650
    assert set(atlas.dataset) == {"gsm8k", "humaneval"}
    assert (atlas.token_expert_pairs == atlas.fresh_m * 8).all()
    assert (atlas.epoch_status == "sensitivity_only_not_measured_epoch").all()


def test_replay_contains_real_and_controlled_cases():
    replay = pd.read_json(ROOT / "EP_SHAPE_REPLAY.jsonl", lines=True)
    assert (replay.case_kind == "real").sum() == 25
    assert (replay.case_kind == "controlled").sum() == 15
    summary = pd.read_csv(ROOT / "EP_SHAPE_SUMMARY.csv")
    assert len(summary) == 40
    assert {1.0, 2.0, 4.0}.issubset(set(summary.destination_fanout_mean))


def test_existing_path_results_are_paired_and_correct():
    bench = pd.read_csv(ROOT / "EXISTING_KERNEL_BENCH.csv")
    restarts = pd.read_csv(ROOT / "EXISTING_KERNEL_BENCH_RESTARTS.csv")
    assert len(bench) == 25 * 3
    assert restarts.restart.nunique() == 5
    assert len(restarts) == 25 * 3 * 5
    assert bench.assignment_count_match.all()
    assert bench.max_relative_l2.max() < 0.001
    pivot = bench.pivot(index="case_id", columns="policy", values="comm_semantics_ms")
    assert (pivot.low_latency < pivot.normal_fresh).all()


def test_cuda_gate_was_not_crossed():
    oracle = pd.read_csv(ROOT / "HEADROOM_ORACLE.csv")
    o3 = oracle[oracle.scenario == "O3_CREDIBLE_REFINEEP_TARGET"]
    assert set(o3.dataset) == {"gsm8k", "humaneval"}
    assert (o3.incremental_gain_vs_o0_percent < 5).all()
    microbench = pd.read_csv(ROOT / "REFINEEP_MICROBENCH.csv")
    assert microbench.status.iloc[0] == "NOT_RUN_GATE_BELOW_12_PERCENT"
