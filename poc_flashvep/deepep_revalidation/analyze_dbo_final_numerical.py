"""Summarize the final DBO raw-logit comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    rows = {"off": [], "on": []}
    for mode in rows:
        for rank in (0, 1):
            stem = args.result_dir / f"distinct_red_dbo_{mode}.dp_rank{rank}"
            metadata = json.loads(Path(f"{stem}.json").read_text())
            arrays = np.load(Path(f"{stem}.npz"))
            for index, run in enumerate(metadata["runs"]):
                rows[mode].append({
                    "rank": rank,
                    "repetition": index,
                    "tokens": run["generated_token_ids"],
                    "logits": arrays[f"run_{index}"].astype(np.float64),
                })

    reference = rows["off"][0]
    first_divergences = []
    for run in rows["on"]:
        divergence = next(
            (i for i, pair in enumerate(zip(reference["tokens"], run["tokens"]))
             if pair[0] != pair[1]),
            None,
        )
        first_divergences.append({
            "rank": run["rank"], "repetition": run["repetition"],
            "zero_based_step": divergence,
            "one_based_position": None if divergence is None else divergence + 1,
            "off_token": None if divergence is None else reference["tokens"][divergence],
            "on_token": None if divergence is None else run["tokens"][divergence],
        })

    comparable = [run for run in rows["on"] if run["tokens"][:3] == reference["tokens"][:3]]
    off = reference["logits"][3]
    on = comparable[0]["logits"][3]
    difference = on - off
    off_top = np.argsort(off)[-10:][::-1]
    on_top = np.argsort(on)[-10:][::-1]

    def top(values: np.ndarray, indices: np.ndarray) -> list[dict[str, float | int]]:
        return [{"token_id": int(i), "logit": float(values[i])} for i in indices]

    summary = {
        "final_status": "HOLD",
        "generated_tokens": {
            mode: [{k: v for k, v in run.items() if k != "logits"} for run in mode_rows]
            for mode, mode_rows in rows.items()
        },
        "first_divergences": first_divergences,
        "common_prefix_for_primary_comparison": reference["tokens"][:3],
        "primary_divergence_zero_based_step": 3,
        "primary_divergence_one_based_position": 4,
        "top10": {"off": top(off, off_top), "on": top(on, on_top)},
        "token_logits": {
            "498": {"off": float(off[498]), "on": float(on[498])},
            "697": {"off": float(off[697]), "on": float(on[697])},
        },
        "top1_top2_margin": {
            "off": float(off[off_top[0]] - off[off_top[1]]),
            "on": float(on[on_top[0]] - on[on_top[1]]),
        },
        "full_vocabulary_error": {
            "max_absolute": float(np.max(np.abs(difference))),
            "mean_absolute": float(np.mean(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference * difference))),
            "cosine_similarity": float(np.dot(off, on) / (np.linalg.norm(off) * np.linalg.norm(on))),
            "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(off)),
        },
        "repeatability": {
            "off_generated_tokens_deterministic": len({tuple(run["tokens"]) for run in rows["off"]}) == 1,
            "on_generated_tokens_deterministic": len({tuple(run["tokens"]) for run in rows["on"]}) == 1,
            "off_step3_max_abs_vs_first": [float(np.max(np.abs(run["logits"][3] - off))) for run in rows["off"]],
            "on_same_prefix_step3_max_abs_vs_first": [float(np.max(np.abs(run["logits"][3] - on))) for run in comparable],
        },
        "decision": "Logit ordering is numerically sensitive, but DBO-on is not run-to-run token deterministic; HOLD under the explicit gate.",
    }
    (args.result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
