#!/usr/bin/env python3
"""Fit endpoint-aware EP communication scenarios from true-EP2 DeepEP data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _fit(cases: list[dict], operation: str, statistic: str) -> dict:
    x = np.asarray(
        [case["max_endpoint_remote_activation_bytes"] for case in cases],
        dtype=np.float64,
    )
    y = np.asarray(
        [case[operation][f"{statistic}_ms"] for case in cases], dtype=np.float64
    )
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    intercept = max(0.0, float(intercept))
    slope = max(float(slope), 1e-12)
    prediction = intercept + slope * x
    return {
        "startup_ms": intercept,
        "gb_per_s": 1.0 / (slope * 1e6),
        # EP2 has one possible remote peer, so it cannot identify an
        # independent multi-peer coefficient. Keep it zero rather than invent
        # one; endpoint/payload P10--P90 scenarios express measured uncertainty.
        "peer_ms": 0.0,
        "sync_ms": 0.0,
        "fit_mae_ms": float(np.mean(np.abs(prediction - y))),
        "fit_points": int(len(x)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--validation-verdict",
        choices=("EP2-CALIBRATED", "EP2-STRUCTURAL-ONLY", "EP2-VALIDATION-FAILED"),
        default="EP2-STRUCTURAL-ONLY",
    )
    args = parser.parse_args()
    document = json.loads(args.calibration.read_text())
    cases = document["cases"]
    mapping = {
        "endpoint_optimistic": "p10",
        "ep2_calibrated_base": "median",
        "concurrency_stressed": "p90",
    }
    scenarios = {}
    for name, statistic in mapping.items():
        dispatch = _fit(cases, "dispatch", statistic)
        combine = _fit(cases, "combine", statistic)
        provenance = {
            "source": str(args.calibration),
            "measured_topology": "true DP1/TP1/SP1/EP2 on GPUs 0,1",
            "measurement": f"actual DeepEP normal {statistic}",
            "payload_axis": "max endpoint logical BF16 activation bytes",
            "multi_peer_coefficient": (
                "not identifiable from EP2; set to zero, not fitted or invented"
            ),
            "validation_verdict": args.validation_verdict,
            "replicated_state_bridge_allgather": "excluded",
        }
        scenarios[name] = {
            "dispatch": {
                key: dispatch[key]
                for key in ("startup_ms", "gb_per_s", "peer_ms", "sync_ms")
            },
            "combine": {
                key: combine[key]
                for key in ("startup_ms", "gb_per_s", "peer_ms", "sync_ms")
            },
            "fit_diagnostics": {"dispatch": dispatch, "combine": combine},
            "provenance": provenance,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "model_type": "NVSwitch endpoint alpha-beta envelope",
                "scenarios_are_not_confidence_intervals": True,
                "scenarios": scenarios,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
