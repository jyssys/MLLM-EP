"""Synthetic timestamps test definitions only, never experimental evidence."""

from .metrics import compliance, summarize


def record(intervals, reason="frontend_stop", status="COMPLETE"):
    return dict(
        status=status,
        token_times=[1, 2, 3],
        output_ids=[4, 5, 2],
        TTFT_s=0.5,
        ITLs_s=intervals,
        finish_reason=reason,
    )


def test_one_bad_token_fails_request_max():
    assert not compliance(record([0.01, 0.201]), 1, 0.2)["joint_pass"]


def test_capped_answer_is_not_complete_quality():
    r = record([0.01, 0.01], reason="length")
    assert compliance(r, 1, 0.2)["joint_pass"]
    assert not compliance(r, 1, 0.2, True)["joint_pass"]


def test_timeout_not_silently_excluded():
    assert not compliance(record([0.01], status="TIMEOUT"), 1, 0.2)["joint_pass"]


def test_ttft_and_tbt_separate():
    r = record([0.01, 0.01])
    result = compliance(r, 0.1, 0.2)
    assert result["TBT_observed_all_token_pass"] and not result["TTFT_pass"]


def test_zero_tokens_is_not_success():
    r = record([])
    r["token_times"] = []
    assert not compliance(r, 1, 1)["joint_pass"]


def test_one_token_empty_turn_is_not_natural_joint_success():
    r = record([])
    r["token_times"] = [1]
    r["output_ids"] = [2]
    assert compliance(r, 1, 1)["joint_pass"]
    assert not compliance(r, 1, 1, True)["joint_pass"]


def test_all_offered_and_goodput_denominators():
    requests = []
    for i, reason in enumerate(("frontend_stop", "length", "timeout")):
        r = record([0.01], reason, "TIMEOUT" if reason == "timeout" else "COMPLETE")
        r.update(
            scheduled_arrival=float(i),
            completion=12.0 + i,
            E2E_s=12.0,
            first_GPU_submission=float(i) + 0.2,
        )
        requests.append(r)
    result = summarize(
        dict(
            requests=requests,
            start=0.0,
            end=14.0,
            arrival_window_s=10.0,
            queue_series=[dict(time=float(i)) for i in range(10)],
        ),
        1,
        0.2,
        True,
    )
    assert result["offered"] == 3
    assert result["joint_pass"] == 1
    assert result["joint_attainment"] == 1 / 3
    assert result["timeouts"] == 1
    assert result["goodput_offered_window_with_declared_drain"] == 0.1
    assert result["goodput_completed_within_arrival_window"] == 0.0
    assert not result["stable_under_preregistered_finite_window_rule"]
