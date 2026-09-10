import pytest

from poc_rankfanout.rankfanout.oracle import amdahl_ttft_oracle, compute_oracles
from poc_rankfanout.rankfanout.trace_schema import AlignmentError


def rows():
    out = []
    for backend, values in {"agrs": [10.0, 2.0], "deepep_ht": [4.0, 8.0]}.items():
        for layer, value in enumerate(values):
            out.append({
                "backend": backend,
                "workload_id": "w",
                "request_id": "r",
                "phase": "prefill",
                "step_id": 0,
                "layer_id": layer,
                "num_tokens": 128,
                "expert_placement_hash": "same",
                "route_hash": f"route-{layer}",
                "moe_total_ms": value,
            })
    return out


def test_oracle_granularity():
    out = compute_oracles(rows())
    assert out["best_static_ms"] == 12.0
    assert out["per_invocation_oracle_ms"] == 6.0
    assert out["per_step_oracle_ms"] == 12.0
    assert out["per_request_oracle_ms"] == 12.0
    assert out["per_invocation_improvement_pct"] == 50.0


def test_amdahl():
    out = amdahl_ttft_oracle(
        ttft_static_ms=100.0,
        replaceable_static_ms=40.0,
        replaceable_oracle_ms=20.0,
    )
    assert out["projected_ttft_oracle_ms"] == 80.0
    assert out["projected_ttft_improvement_pct"] == 20.0


def test_alignment_rejects_missing_row():
    broken = rows()[:-1]
    with pytest.raises(AlignmentError):
        compute_oracles(broken)


def test_alignment_rejects_route_mismatch():
    broken = rows()
    broken[-1]["route_hash"] = "wrong"
    with pytest.raises(AlignmentError):
        compute_oracles(broken)
