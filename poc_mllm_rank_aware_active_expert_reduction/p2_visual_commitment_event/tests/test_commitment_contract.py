from poc_mllm_rank_aware_active_expert_reduction.p2_visual_commitment_event.scripts.analyze_commitment import _event


def test_persistent_event_rule() -> None:
    rows = [
        {"position": index + 1, "jsd_nats": 0.1 if index < 10 else 0.001}
        for index in range(30)
    ]
    event = _event(rows, {"threshold": 0.01, "width": 3, "minimum_position": 5})
    assert event is not None
    assert event["position"] >= 11
    assert event["delta_jsd"] > 0
