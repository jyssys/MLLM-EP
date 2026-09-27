from __future__ import annotations

from pathlib import Path


def test_ep8_report_generator_forbids_latency_claims() -> None:
    source = Path(__file__).parents[1] / "scripts" / "analyze.py"
    text = source.read_text()
    assert '"evidence_label": "STRUCTURAL_EP8_PROJECTION_ONLY"' in text
    assert '"latency_tpot_e2e_speedup": "NOT_COMPUTED_PROHIBITED"' in text
