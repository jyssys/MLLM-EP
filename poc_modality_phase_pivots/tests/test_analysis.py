import math

from poc_modality_phase_pivots.analyze import critical_rows, scheduling_oracle


def test_critical_rows_selects_rank_max_and_deduplicates_flush_copies():
    docs = [
        {"rows": [
            {"phase": "dispatch", "iteration": 0, "x": 1.0},
            {"phase": "dispatch", "iteration": 0, "x": 1.0},
        ]},
        {"rows": [{"phase": "dispatch", "iteration": 0, "x": 2.0}]},
    ]
    got = critical_rows(docs, ["phase", "iteration"], ["x"])
    assert got == [{"phase": "dispatch", "iteration": 0, "x": 2.0}]


def test_perfect_schedule_oracle_has_overlap_but_zero_eta_does_not():
    stages = []
    for modality in ["vision", "text"]:
        for layer in range(48):
            for stage, value in [("attention", 1.0), ("moe", 1.0)]:
                stages.append({"modality": modality, "layer": layer,
                               "stage": stage, "duration_ms": value})
    pairs = [
        {"modality": modality, "phase": "moe", "eta_median": 0.0}
        for modality in ["vision", "text"]
    ]
    got = {x["policy"]: x for x in scheduling_oracle(stages, pairs)}
    # Two alternating chains expose 95 pairable transitions out of 192 units.
    assert math.isclose(got["perfect phase-aware oracle"]["gain_pct"],
                        100 * 95 / 192)
    assert math.isclose(got["contention-corrected layer oracle"]["gain_pct"], 0.0)
    assert math.isclose(got["simple eta>=0.2 heuristic"]["gain_pct"], 0.0)
