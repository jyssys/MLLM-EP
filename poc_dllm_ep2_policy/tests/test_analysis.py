import math
import json
from pathlib import Path

import torch


RESULT = (Path(__file__).resolve().parents[1] / "results" /
          "ep2_20260911_170000")


def best_static(table):
    backends = sorted({backend for row in table for backend in row})
    return min(sum(row[backend] for row in table) for backend in backends)


def perfect_dynamic(table):
    return sum(min(row.values()) for row in table)


def active_bin(ratio):
    if ratio > 0.67:
        return "early"
    if ratio > 0.33:
        return "middle"
    return "late"


def threshold_policy(rows, threshold, high_backend, low_backend):
    return sum(row[high_backend] if row["active_ratio"] >= threshold
               else row[low_backend] for row in rows)


def test_oracle_math():
    rows = [{"a": 10.0, "b": 12.0}, {"a": 20.0, "b": 8.0}]
    assert best_static(rows) == 20.0
    assert perfect_dynamic(rows) == 18.0
    assert math.isclose(1 - perfect_dynamic(rows) / best_static(rows), 0.1)


def test_active_bins():
    assert active_bin(1.0) == "early"
    assert active_bin(0.67) == "middle"
    assert active_bin(0.33) == "late"


def test_true_ep2_ownership():
    audits = [json.loads((RESULT / "runtime_audit" /
                          f"audit_naive_rank{rank}.json").read_text())
              for rank in range(2)]
    assert {(item["tp_size"], item["dp_size"], item["ep_size"])
            for item in audits} == {(1, 2, 2)}
    assert all(item["local_expert_count"] == 32 for item in audits)
    assert audits[0]["local_expert_ids"] == list(range(32))
    assert audits[1]["local_expert_ids"] == list(range(32, 64))


def test_expert_work_conservation():
    path = RESULT / "scaling/naive_run1/scaling_naive_rank0.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows
    for row in rows:
        assert row["global_m"] == 2 * row["local_m"]
        assert row["routed_assignments"] == row["global_m"] * row["top_k"]


def test_active_position_accounting_and_physical_work():
    path = (RESULT / "trajectories/gen64_batch4_cacheoff_trace" /
            "trajectory_gen64_batch4_cacheoff_trace_rank0.json")
    payload = json.loads(path.read_text())
    assert payload["batch_size"] == 4
    assert all(row["masked_before"] - row["masked_after"] == row["newly_accepted"]
               for row in payload["iterations"])
    assert min(row["masked_before"] for row in payload["iterations"]) == 1
    assert {row["model_positions"] for row in payload["iterations"]} == {512}


def test_backend_correctness():
    naive = torch.load(RESULT / "runtime_audit/audit_naive_output.pt",
                       weights_only=True)
    ht = torch.load(RESULT / "runtime_audit_deepep_ht" /
                    "audit_deepep_high_throughput_output.pt", weights_only=True)
    assert torch.equal(naive, ht)


def test_generated_output_stability():
    payloads = []
    for path in sorted((RESULT / "request_runs_warm5").glob("*/*rank0.json")):
        payloads.append(json.loads(path.read_text())["output_ids"])
    assert len(payloads) == 6
    assert all(item == payloads[0] for item in payloads)


def test_generated_oracle_is_bounded():
    summary = json.loads((RESULT / "analysis/summary.json").read_text())
    assert summary["perfect_dynamic_ms"] <= summary["best_static_ms"]
    assert 0 <= summary["threshold_recovery_pct"] <= 100
    assert summary["request_oracle_gain_pct_optimistic"] < 5


def test_threshold_policy_uses_only_current_active_state():
    rows = [
        {"active_ratio": 0.8, "ht": 4.0, "ll": 9.0},
        {"active_ratio": 0.2, "ht": 7.0, "ll": 3.0},
    ]
    assert threshold_policy(rows, 0.5, "ht", "ll") == 7.0


def test_required_plot_set_exists():
    plots = list((RESULT / "analysis/plots").glob("*.png"))
    assert len(plots) == 11
