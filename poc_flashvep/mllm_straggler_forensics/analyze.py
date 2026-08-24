"""Fixed-policy analysis for matched-work MLLM EP straggler forensics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


N_TOLERANCE = 0.05
G_TOLERANCE = 0
Q_TOLERANCE = 2
FORENSIC_GAP = 0.15
BOOTSTRAPS = 10_000
SEED = 20260824


def _dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _match_cross(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(["layer", "rank", "token_bucket"], sort=True):
        vision = group[group.modality == "vision"].sort_values("request_id").reset_index(drop=True)
        text = group[group.modality == "text"].sort_values("request_id").reset_index(drop=True)
        cost = np.full((len(vision), len(text)), 1e9)
        for i, left in vision.iterrows():
            dn = abs(text.total_assignments - left.total_assignments) / np.maximum(text.total_assignments, left.total_assignments)
            dg = abs(text.active_experts - left.active_experts)
            dq = abs(text.effective_tiles - left.effective_tiles)
            valid = (dn <= N_TOLERANCE) & (dg <= G_TOLERANCE) & (dq <= Q_TOLERANCE)
            cost[i, valid] = dn[valid] + 0.01 * dq[valid] + np.arange(len(text))[valid] * 1e-10
        ii, jj = linear_sum_assignment(cost)
        for i, j in zip(ii, jj):
            if cost[i, j] >= 1e9:
                continue
            v, t = vision.iloc[i], text.iloc[j]
            rows.append({
                "layer": int(key[0]), "rank": int(key[1]), "token_bucket": key[2],
                "vision_request_id": v.request_id, "text_request_id": t.request_id,
                "vision_n": int(v.total_assignments), "text_n": int(t.total_assignments),
                "vision_g": int(v.active_experts), "text_g": int(t.active_experts),
                "vision_q": int(v.effective_tiles), "text_q": int(t.effective_tiles),
                "vision_ms": float(v.expert_median_ms), "text_ms": float(t.expert_median_ms),
                "relative_n_error": float(abs(v.total_assignments - t.total_assignments) / max(v.total_assignments, t.total_assignments)),
                "relative_latency_residual": float(v.expert_median_ms / t.expert_median_ms - 1),
                "absolute_latency_gap_ms": float(v.expert_median_ms - t.expert_median_ms),
                "vision_histogram": v.expert_histogram, "text_histogram": t.expert_histogram,
            })
    return pd.DataFrame(rows)


def _critical(frame: pd.DataFrame) -> dict[tuple[str, str, int], int]:
    result = {}
    for key, group in frame.groupby(["modality", "request_id", "layer"]):
        result[key] = int(group.loc[group.expert_median_ms.idxmax(), "rank"])
    return result


def _bootstrap(pairs: pd.DataFrame) -> tuple[float, float]:
    grouped = pairs.groupby("vision_request_id").relative_latency_residual.mean()
    values = grouped.to_numpy(); rng = np.random.default_rng(SEED)
    samples = rng.choice(values, (BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.percentile(samples, [2.5, 97.5]))


def _within_count(frame: pd.DataFrame, modality: str) -> int:
    count = 0
    local = frame[frame.modality == modality]
    for _, group in local.groupby(["layer", "rank", "token_bucket"]):
        rows = list(group.itertuples())
        for i, left in enumerate(rows):
            for right in rows[i + 1:]:
                dn = abs(left.total_assignments - right.total_assignments) / max(left.total_assignments, right.total_assignments)
                valid = dn <= N_TOLERANCE and left.active_experts == right.active_experts and abs(left.effective_tiles - right.effective_tiles) <= Q_TOLERANCE
                gap = max(left.expert_median_ms, right.expert_median_ms) / min(left.expert_median_ms, right.expert_median_ms) - 1
                count += bool(valid and gap >= FORENSIC_GAP)
    return count


def prepare(source: Path, result: Path) -> None:
    result.mkdir(parents=True, exist_ok=False); figures = result / "figures"; figures.mkdir()
    frame = pd.read_csv(source / "per_rank_shape_latency.csv")
    pairs = _match_cross(frame)
    if pairs.empty:
        raise RuntimeError("no preregistered N/G/Q matched pairs")
    critical = _critical(frame)
    pairs["vision_is_actual_critical"] = [row.rank == critical[("vision", row.vision_request_id, row.layer)] for row in pairs.itertuples()]
    pairs["text_is_actual_critical"] = [row.rank == critical[("text", row.text_request_id, row.layer)] for row in pairs.itertuples()]
    pairs.to_csv(result / "matched_pairs.csv", index=False)
    forensic = pairs[abs(pairs.relative_latency_residual) >= FORENSIC_GAP].copy()
    vision_slow = forensic[forensic.relative_latency_residual >= FORENSIC_GAP]
    if vision_slow.empty:
        selected = forensic.iloc[(abs(forensic.relative_latency_residual).sort_values()).index[len(forensic)//2]] if not forensic.empty else pairs.iloc[abs(pairs.relative_latency_residual).idxmax()]
    else:
        # Fixed representative: nearest the median positive forensic gap.
        target = vision_slow.relative_latency_residual.median()
        selected = vision_slow.loc[(vision_slow.relative_latency_residual - target).abs().idxmin()]
    selection = {
        "policy": "one-to-one Hungarian within layer/rank/bucket; |dN|<=5%, exact G, |dQ|<=2; representative nearest median positive >=15% gap",
        "selected_cross_modality_pair": {key: (value.item() if hasattr(value, "item") else value) for key, value in selected.to_dict().items()},
    }
    _dump(result / "selection.json", selection)
    ci = _bootstrap(pairs)
    summary = {
        "fixed_policy": {"n_tolerance": N_TOLERANCE, "g_tolerance": G_TOLERANCE, "q_tolerance": Q_TOLERANCE, "forensic_gap": FORENSIC_GAP},
        "source": str(source), "matched_pairs": len(pairs),
        "matched_vision_requests": int(pairs.vision_request_id.nunique()), "matched_text_requests": int(pairs.text_request_id.nunique()),
        "mean_relative_vision_residual": float(pairs.relative_latency_residual.mean()),
        "median_relative_vision_residual": float(pairs.relative_latency_residual.median()),
        "source_request_clustered_ci95": list(ci), "vision_slower_fraction": float((pairs.relative_latency_residual > 0).mean()),
        "forensic_cross_pairs": len(forensic), "forensic_vision_slower_pairs": len(vision_slow),
        "within_vision_forensic_pairs": _within_count(frame, "vision"), "within_text_forensic_pairs": _within_count(frame, "text"),
        "vision_actual_critical_frequency": float(pairs.vision_is_actual_critical.mean()),
        "text_actual_critical_frequency": float(pairs.text_is_actual_critical.mean()),
        "selected": selection["selected_cross_modality_pair"],
    }
    _dump(result / "summary_preliminary.json", summary)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(100 * pairs.relative_latency_residual, bins=30, color="tab:blue", alpha=.8); axes[0].axvline(0, color="black", lw=.8)
    axes[0].set(xlabel="Vision - Text latency residual (%)", ylabel="Matched pairs", title="All fixed-policy N/G/Q matches")
    axes[1].scatter(pairs.text_ms, pairs.vision_ms, s=16, alpha=.7); lo=min(pairs.text_ms.min(),pairs.vision_ms.min()); hi=max(pairs.text_ms.max(),pairs.vision_ms.max())
    axes[1].plot([lo,hi],[lo,hi],color="black",lw=.8); axes[1].set(xlabel="Text expert latency (ms)",ylabel="Vision expert latency (ms)",title="One-to-one matched work")
    fig.tight_layout(); fig.savefig(figures / "plot1_matched_vision_text_latency.png", dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("source", type=Path); parser.add_argument("result", type=Path)
    args = parser.parse_args(); prepare(args.source, args.result)


if __name__ == "__main__":
    main()
