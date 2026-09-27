from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


REQUIRED_REPORTS = (
    "baseline_readiness.md",
    "request_pool.md",
    "oracle_headroom.md",
    "policy_workingset_comparison.md",
    "mllm_specific_increment.md",
    "true_ep4_runtime.md",
    "gain_vs_decode_length.md",
    "final_summary.md",
)

FINAL_FIELDS = (
    "BASELINE_STATUS",
    "REQUEST_POOL_STATUS",
    "ORACLE_HEADROOM",
    "TEXT_POLICY_HEADROOM",
    "FULL_PREFILL_ELDR_STYLE_HEADROOM",
    "VISION_SEPARATED_HEADROOM",
    "VISION_INCREMENT_VS_FULL",
    "VISION_INCREMENT_VS_TEXT",
    "TRUE_EP4_TPOT_GAIN_VS_FIFO",
    "TRUE_EP4_TPOT_GAIN_VS_FULL",
    "FIXED128_E2E_GAIN",
    "RUNTIME_PARITY",
    "VERDICT",
    "NEXT_ACTION",
)


def load(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    artifacts = args.root / "artifacts"
    reports = args.root / "reports"

    pool = load(artifacts / "frozen_request_pool.json")
    assignments = load(artifacts / "frozen_wave_assignments.json")
    metrics = load(artifacts / "working_set_metrics.json")
    result = load(artifacts / "stage_a_result.json")
    runtime = load(artifacts / "clean_timing.json")
    pair_ids = {row["pair_id"] for row in pool["anchors"]}
    image_ids = {row["original_image_sha256"] for row in pool["anchors"]}
    assert len(pair_ids) == 20 == len(image_ids)
    assert sum(row["coscheduling_split"] == "discovery" for row in pool["anchors"]) == 12
    assert sum(row["coscheduling_split"] == "heldout" for row in pool["anchors"]) == 8
    assert all(row["natural_output_token_count"] >= 128 for row in pool["anchors"])
    assert all(row["natural_finish_reason"] == "stop" for row in pool["anchors"])

    for policy, splits in assignments["policies"].items():
        for split, waves in splits.items():
            expected = {
                row["pair_id"]
                for row in pool["anchors"]
                if row["coscheduling_split"] == split
            }
            flat = [pair for wave in waves for pair in wave]
            assert all(len(wave) == 4 for wave in waves), policy
            assert len(flat) == len(set(flat)), policy
            assert set(flat) == expected, policy

    heldout = assignments["policies"]
    assert heldout["S2_TEXT"]["heldout"] == heldout["S3_FULL"]["heldout"]
    assert heldout["S3_FULL"]["heldout"] == heldout["S4_SEPARATED"]["heldout"]
    assert heldout["S1_LENGTH"]["heldout"] == heldout["S4_SEPARATED"]["heldout"]
    assert metrics["heldout"]["1-128"]["S4_SEPARATED"]["mean_active_expert_union"] == metrics["heldout"]["1-128"]["S3_FULL"]["mean_active_expert_union"]
    assert result["verdict"] == "INSUFFICIENT_BATCHING_HEADROOM"
    assert result["gate"]["pass"] is False
    assert runtime["status"] == "NOT_RUN_STAGE_A_GATE_FAILED"
    assert runtime["full_model_executions"] == 0

    with gzip.open(artifacts / "source_pool_trace.jsonl.gz", "rt") as handle:
        source_rows = sum(1 for line in handle if line.strip())
    assert source_rows > 0

    final = (reports / "final_summary.md").read_text()
    for name in REQUIRED_REPORTS:
        assert (reports / name).is_file(), name
    for field in FINAL_FIELDS:
        assert f"{field}:" in final, field
    print(
        json.dumps(
            {
                "status": "PASS",
                "requests": len(pair_ids),
                "source_trace_rows": source_rows,
                "reports": len(REQUIRED_REPORTS),
                "stage_b_executed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
