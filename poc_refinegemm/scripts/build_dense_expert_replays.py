#!/usr/bin/env python3
"""Extract actual dense-runtime M_e vectors from the prior EP4 trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(values: list[int]) -> dict:
    active = [x for x in values if x]
    mean = float(np.mean(active)) if active else 0.0
    std = float(np.std(active)) if active else 0.0
    return {
        "assignments": sum(values),
        "active_experts": len(active),
        "median_m_e": float(np.median(active)) if active else 0.0,
        "p90_m_e": float(np.percentile(active, 90)) if active else 0.0,
        "max_m_e": max(active, default=0),
        "tiny_le4_active_fraction": sum(x <= 4 for x in active) / len(active) if active else 0.0,
        "large_ge16_active_fraction": sum(x >= 16 for x in active) / len(active) if active else 0.0,
        "m_e_cv": std / mean if mean else 0.0,
        "inactive_fraction": sum(x == 0 for x in values) / len(values),
    }


def control(values: list[int], high: bool) -> list[int]:
    total, active = sum(values), max(1, sum(x > 0 for x in values))
    out = [0] * len(values)
    if not high:
        for i in range(active):
            out[i] = total // active + int(i < total % active)
    else:
        tiny = min(total, max(1, active // 2))
        for i in range(tiny):
            out[i] = 1
        remainder = total - tiny
        for i in range(active - tiny):
            out[tiny + i] = remainder // max(1, active - tiny) + int(i < remainder % max(1, active - tiny))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-wave", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = pd.read_csv(args.layer_wave)
    row_records, summary_records, candidates = [], [], []
    for _, row in source.iterrows():
        hist = [int(x) for x in json.loads(row["full_histogram"])]
        for rank in range(4):
            values = hist[rank * 64 : (rank + 1) * 64]
            meta = {
                "task": row["dataset"], "wave": int(row["wave"]),
                "layer": int(row["layer"]), "phase": row["phase"], "rank": rank,
                "physical_m": int(row["physical_rows"]), "fresh_m": int(row["physical_rows"]),
                "worklist_scope": "measured_dense_runtime", **metrics(values),
            }
            summary_records.append(meta)
            candidates.append({**meta, "m_e_vector": values})
            for local, value in enumerate(values):
                row_records.append({
                    "task": row["dataset"], "wave": int(row["wave"]), "layer": int(row["layer"]),
                    "phase": row["phase"], "rank": rank, "expert_local": local,
                    "expert_global": rank * 64 + local, "m_e": value,
                    "worklist_scope": "measured_dense_runtime",
                })
    args.output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_records)
    summary.to_csv(args.output / "DENSE_EXPERT_SHAPE_SUMMARY.csv", index=False)
    pd.DataFrame(row_records).to_parquet(args.output / "DENSE_EXPERT_ROW_TRACE.parquet", index=False)

    chosen = []
    frame = pd.DataFrame([{k: v for k, v in x.items() if k != "m_e_vector"} for x in candidates])
    for (_, _), group in frame.groupby(["task", "phase"]):
        order = group.sort_values(["assignments", "m_e_cv"])
        for q in (0.1, 0.5, 0.9):
            chosen.append(candidates[int(order.iloc[round((len(order) - 1) * q)].name)])
    with (args.output / "EXPERT_REPLAY_DENSE.jsonl").open("w") as stream:
        for index, item in enumerate(chosen):
            base = {k: v for k, v in item.items() if k != "m_e_vector"}
            for geometry, vector in (
                ("real", item["m_e_vector"]),
                ("low_heterogeneity_control", control(item["m_e_vector"], False)),
                ("high_heterogeneity_control", control(item["m_e_vector"], True)),
            ):
                out = {
                    "case_id": f"dense_{index:03d}_{geometry}", "geometry": geometry,
                    **base, **metrics(vector), "m_e_vector": vector,
                    "same_total_assignment_control_group": f"dense_{index:03d}",
                }
                stream.write(json.dumps(out) + "\n")
    print(json.dumps({"summary_rows": len(summary_records), "expert_rows": len(row_records), "selected_real": len(chosen)}))


if __name__ == "__main__":
    main()
