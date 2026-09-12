#!/usr/bin/env python3
"""Generate cumulative low-sensitivity sets for the interaction-safe O2 probe."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


root = Path(__file__).resolve().parents[1]
baseline = json.loads((root / "plans/baseline.json").read_text())[0]


def read(path: str, mode: str) -> pd.DataFrame:
    frame = pd.read_csv(root / path)
    frame = frame[frame["mode"] == mode].copy()
    frame = frame[~frame["layers"].fillna("").astype(str).str.contains(",")]
    frame["layer"] = frame["layers"].astype(int)
    frame["trajectory_divergence"] = 1.0 - frame["final_sequence_exact"] / 32.0
    frame["score_flip_fraction"] = frame["per_sample_score_flips"] / 32.0
    frame["rank_score"] = (
        frame["live_top1_flip_fraction"].fillna(1.0)
        + frame["accepted_set_change_fraction"].fillna(1.0)
        + frame["trajectory_divergence"]
        + frame["score_flip_fraction"]
    )
    return frame


gsm_deep = read("CAUSAL_DEEP_DIAGNOSTICS_GSM8K.csv", "layer_bypass")
gsm_stale = read("CAUSAL_STALE_ROUTED_GSM8K.csv", "stale_routed")
gsm_routed = read("CAUSAL_ROUTED_BYPASS_GSM8K.csv", "routed_bypass")
human_routed = read("CAUSAL_ROUTED_BYPASS_HUMANEVAL.csv", "routed_bypass")
routed = pd.concat([gsm_routed, human_routed], ignore_index=True)
routed = (
    routed[routed["layer"] > 0]
    .groupby(["phase", "layer"], as_index=False)["rank_score"]
    .mean()
)

sources = {
    "layer_bypass": gsm_deep,
    "routed_bypass": routed,
    "stale_routed": gsm_stale[gsm_stale["layer"] > 0],
}
policies = [baseline]
rankings = {}
for mode, frame in sources.items():
    rankings[mode] = {}
    for phase in ("early", "middle", "late"):
        ordered = (
            frame[frame["phase"] == phase]
            .sort_values(["rank_score", "layer"])["layer"]
            .astype(int)
            .tolist()
        )
        rankings[mode][phase] = ordered
        for count in (2, 4, 8, 16):
            layers = sorted(ordered[:count])
            policies.append(
                {
                    "name": f"greedy_{mode}_{phase}_k{count:02d}",
                    "mode": mode,
                    "layers": ",".join(str(layer) for layer in layers),
                    "phases": phase,
                    "waves": "",
                    "capture_phases": phase,
                }
            )

(root / "plans/greedy_oracle.json").write_text(json.dumps(policies, indent=2) + "\n")
(root / "GREEDY_LAYER_RANKINGS.json").write_text(json.dumps(rankings, indent=2) + "\n")
print(json.dumps({"policies": len(policies), "rankings": rankings}))
