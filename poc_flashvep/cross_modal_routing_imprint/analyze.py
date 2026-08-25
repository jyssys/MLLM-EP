"""Preregistered held-out-image analysis for cross-modal routing imprint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXPERTS = 128
SOURCE_LAYERS = (0, 4, 8, 12)
TARGET_LAYERS = (16, 20, 24, 28, 32, 36, 40, 44, 47)
TOPK = 8
FOLDS = 6
RIDGE_ALPHA = 1.0
SEED = 20260825


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _mass(routes: np.ndarray) -> np.ndarray:
    counts = np.bincount(routes.reshape(-1), minlength=EXPERTS)[:EXPERTS].astype(float)
    return counts / counts.sum()


def _jsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(jensenshannon(np.clip(a, 1e-12, None), np.clip(b, 1e-12, None), base=2) ** 2)


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pred = np.clip(pred, 0, None)
    pred /= np.maximum(pred.sum(axis=1, keepdims=True), 1e-12)
    return {
        "r2": float(r2_score(y.reshape(-1), pred.reshape(-1))),
        "jsd": float(np.mean([_jsd(a, b) for a, b in zip(y, pred)])),
        "top1": float(np.mean(np.argmax(y, axis=1) == np.argmax(pred, axis=1))),
        "topk": float(np.mean([
            len(set(np.argsort(a)[-TOPK:]) & set(np.argsort(b)[-TOPK:])) / TOPK
            for a, b in zip(y, pred)
        ])),
    }


def _oof(x: np.ndarray, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]],
         mode: str, shuffle_seed: int = 0) -> np.ndarray:
    pred = np.zeros_like(y)
    rng = np.random.default_rng(shuffle_seed)
    for train, test in folds:
        if mode == "mean":
            pred[test] = y[train].mean(axis=0)
            continue
        local_x = x[train].copy()
        if mode == "shuffle":
            local_x = local_x[rng.permutation(len(local_x))]
        model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
        model.fit(local_x, y[train])
        pred[test] = model.predict(x[test])
    return pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args()
    result = args.result_dir
    manifest = json.loads((result / "manifest.json").read_text())
    samples = sorted(manifest["samples"], key=lambda row: row["sample_id"])
    visual = np.zeros((len(samples), 48, EXPERTS))
    text = np.zeros_like(visual)
    integrity = []
    for index, meta in enumerate(samples):
        path = result / next(row["array_file"] for rank in (0, 1)
                             for row in json.loads((result / f"capture.dp{rank}.json").read_text())["records"]
                             if row["sample_id"] == meta["sample_id"])
        arrays = np.load(path)
        routes = arrays["routed_experts"].astype(int)
        token_ids = arrays["prompt_token_ids"]
        start, end = meta["visual_span"]
        language_start, language_end = meta["post_visual_span"]
        language_positions = np.arange(language_start, language_end)
        if len(language_positions) == 0:
            raise AssertionError(f"{meta['sample_id']}: empty post-visual language span")
        for layer in range(48):
            visual[index, layer] = _mass(routes[start:end, layer])
            text[index, layer] = _mass(routes[language_positions, layer])
        integrity.append({"sample_id": meta["sample_id"], "tokens": len(token_ids),
                          "vision_tokens": end - start,
                          "language_tokens": len(language_positions),
                          "route_min": int(routes.min()), "route_max": int(routes.max())})

    policy = {
        "source_layers": list(SOURCE_LAYERS), "target_layers": list(TARGET_LAYERS),
        "long_range_distance": 24, "folds": FOLDS, "ridge_alpha": RIDGE_ALPHA,
        "topk": TOPK, "seed": SEED,
        "go": "long-range visual R2 beats mean and shuffled by >=0.10 and does so for >=50% long-range cells",
        "hold": "long-range visual R2 or JSD improves over both baselines by >=0.02 in >=25% cells; otherwise NO-GO",
    }
    _write_json(result / "analysis_policy.json", policy)
    _write_json(result / "capture_integrity.json", integrity)
    folds = list(KFold(FOLDS, shuffle=True, random_state=SEED).split(samples))
    rows = []
    matrices: dict[str, np.ndarray] = {name: np.full((len(SOURCE_LAYERS), len(TARGET_LAYERS)), np.nan)
                                      for name in ("visual", "mean", "shuffle", "early_text")}
    for si, source in enumerate(SOURCE_LAYERS):
        for ti, target in enumerate(TARGET_LAYERS):
            y = text[:, target]
            predictions = {
                "visual": _oof(visual[:, source], y, folds, "ridge"),
                "mean": _oof(visual[:, source], y, folds, "mean"),
                "early_text": _oof(text[:, source], y, folds, "ridge"),
            }
            shuffled = [_metrics(y, _oof(visual[:, source], y, folds, "shuffle", SEED + seed))
                        for seed in range(10)]
            for model, prediction in predictions.items():
                metric = _metrics(y, prediction)
                matrices[model][si, ti] = metric["r2"]
                rows.append({"source_layer": source, "target_layer": target,
                             "distance": target - source, "model": model, **metric})
            shuffle_metric = {key: float(np.mean([item[key] for item in shuffled]))
                              for key in shuffled[0]}
            matrices["shuffle"][si, ti] = shuffle_metric["r2"]
            rows.append({"source_layer": source, "target_layer": target,
                         "distance": target - source, "model": "shuffle", **shuffle_metric})
    frame = pd.DataFrame(rows)
    frame.to_csv(result / "predictive_metrics.csv", index=False)
    figures = result / "figures"
    figures.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.7), sharey=True)
    for ax, name in zip(axes, ("visual", "mean", "shuffle", "early_text"), strict=True):
        image = ax.imshow(matrices[name], aspect="auto", cmap="coolwarm", vmin=-.5, vmax=.5)
        ax.set_title(name); ax.set_xticks(range(len(TARGET_LAYERS)), TARGET_LAYERS)
        ax.set_yticks(range(len(SOURCE_LAYERS)), SOURCE_LAYERS); ax.set_xlabel("language target layer")
    axes[0].set_ylabel("early source layer")
    fig.colorbar(image, ax=axes, label="held-out CV R²", shrink=.85)
    fig.suptitle("Early routing → later language-routing predictability")
    fig.subplots_adjust(left=.06, right=.94, bottom=.16, top=.82, wspace=.12)
    fig.savefig(figures / "plot1_cross_layer_predictability.png", dpi=200)
    plt.close(fig)

    persistence = frame.pivot_table(index="distance", columns="model", values="r2", aggfunc="mean")
    persistence.to_csv(result / "persistence.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in ("visual", "mean", "shuffle", "early_text"):
        if name in persistence:
            ax.plot(persistence.index, persistence[name], marker="o", label=name)
    ax.axhline(0, color="black", linewidth=.8); ax.set(xlabel="layer distance", ylabel="mean CV R²",
        title="Routing-imprint persistence by layer distance")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(figures / "plot2_imprint_persistence.png", dpi=200)
    plt.close(fig)

    long = frame[frame.distance >= policy["long_range_distance"]].pivot_table(
        index=["source_layer", "target_layer"], columns="model", values=["r2", "jsd"])
    r2_adv = long[("r2", "visual")] - np.maximum(long[("r2", "mean")], long[("r2", "shuffle")])
    jsd_adv = np.minimum(long[("jsd", "mean")], long[("jsd", "shuffle")]) - long[("jsd", "visual")]
    go_fraction = float(np.mean(r2_adv >= .10))
    hold_fraction = float(np.mean((r2_adv >= .02) | (jsd_adv >= .02)))
    if float(r2_adv.max()) >= .10 and go_fraction >= .50:
        status = "GO"
    elif (float(r2_adv.max()) >= .02 or float(jsd_adv.max()) >= .02) and hold_fraction >= .25:
        status = "HOLD"
    else:
        status = "NO-GO"
    best_index = r2_adv.idxmax()
    best = frame[(frame.source_layer == best_index[0]) & (frame.target_layer == best_index[1])]
    summary = {
        "PREDICTIVE_IMPRINT": status,
        "samples": len(samples), "fixed_edge": manifest["fixed_edge"],
        "vision_token_counts": sorted(set(row["vision_tokens"] for row in samples)),
        "language_token_counts": sorted(set(item["language_tokens"] for item in integrity)),
        "best_long_range": {"source_layer": int(best_index[0]), "target_layer": int(best_index[1]),
                            "r2_advantage": float(r2_adv.loc[best_index]),
                            "metrics": best.set_index("model")[["r2", "jsd", "top1", "topk"]].to_dict("index")},
        "long_range_r2_advantage_mean": float(r2_adv.mean()),
        "long_range_jsd_advantage_mean": float(jsd_adv.mean()),
        "go_cell_fraction": go_fraction, "hold_cell_fraction": hold_fraction,
        "CAUSAL_IMPRINT": "NOT-RUN" if status == "NO-GO" else "PENDING",
        "COMPRESSION_HEADROOM": "NOT-RUN" if status == "NO-GO" else "PENDING",
    }
    _write_json(result / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
