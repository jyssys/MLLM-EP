import csv
import json
from pathlib import Path


POC = Path(__file__).resolve().parents[1]
ROOT = Path((POC / "CURRENT_RESULT_ROOT").read_text().strip())


def test_required_reports_exist():
    assert len(list((POC / "reports").glob("trackA_*.md"))) == 7
    assert len(list((POC / "reports").glob("trackB_*.md"))) == 8
    assert (POC / "reports/cross_track_comparison.md").is_file()
    assert (POC / "reports/final_decision.md").is_file()


def test_track_a_is_below_kill_gate():
    rows = list(csv.DictReader((ROOT / "analysis/trackA_compression_oracles.csv").open()))
    assert rows
    assert max(float(row["payload_sensitive_gross_e2e_oracle_pct"]) for row in rows) < 1.0
    assert all(row["gate"] == "KILL" for row in rows)


def test_both_quality_sweeps_complete():
    assert len(list((ROOT / "trackB/policy_sweep/gsm8k/r3").glob("policy_*.json"))) == 18
    assert len(list((ROOT / "trackB/policy_sweep/humaneval/r1").glob("policy_*.json"))) == 18
    pareto = list(csv.DictReader((ROOT / "analysis/trackB_speed_quality_pareto.csv").open()))
    assert {row["task"] for row in pareto} == {"gsm8k", "humaneval"}


def test_stage_local_load_audits_are_four_rank():
    for topology, pp_size in (("pp2_ep2", 2), ("pp4", 4)):
        rows = [
            json.loads(path.read_text())
            for path in (ROOT / f"trackB/{topology}_load_audit").glob("load_audit_rank*.json")
        ]
        assert len(rows) == 4
        assert {row["global_rank"] for row in rows} == {0, 1, 2, 3}
        assert all(row["pp_size"] == pp_size for row in rows)
        assert all(row["physical_gpu"] in {4, 5, 6, 7} for row in rows)


def test_launchers_only_expose_allowed_physical_gpus():
    for path in (POC / "scripts").glob("*.sh"):
        text = path.read_text()
        if "CUDA_VISIBLE_DEVICES" in text:
            assert "CUDA_VISIBLE_DEVICES=4,5,6,7" in text
