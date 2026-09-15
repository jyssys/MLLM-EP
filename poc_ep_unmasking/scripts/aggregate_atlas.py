#!/usr/bin/env python3
"""Build the contract's cross-task machine-readable atlas artifacts."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ("gsm8k", "gsm8k8_atlas_r1", True),
    ("humaneval", "humaneval8_atlas_r1", False),
)
for filename in ("CONFIDENCE_SLACK.csv", "ONE_STEP_ORACLES.csv"):
    frames = []
    for task, directory, complete in SOURCES:
        frame = pd.read_csv(ROOT / "results" / directory / filename)
        frame.insert(0, "task", task)
        frame.insert(1, "trajectory_complete", int(complete))
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(ROOT / filename, index=False)

frames = []
for task, directory, complete in SOURCES:
    frame = pd.read_parquet(ROOT / "results" / directory / "TOKEN_EP_SIGNATURES.parquet")
    frame.insert(0, "task", task)
    frame.insert(1, "trajectory_complete", int(complete))
    frames.append(frame)
pd.concat(frames, ignore_index=True).to_parquet(
    ROOT / "TOKEN_EP_SIGNATURES.parquet", index=False
)
