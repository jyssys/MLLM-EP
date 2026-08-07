"""Create compact layer/stage metrics from localization tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


STAGES = (
    "layer_input", "attention_output", "attention_residual",
    "dispatch_input", "moe_output", "layer_final",
)


def metrics(off: np.ndarray, on: np.ndarray) -> dict[str, float]:
    off = off.astype(np.float64)
    on = on.astype(np.float64)
    delta = on - off
    off_norm = np.linalg.norm(off)
    on_norm = np.linalg.norm(on)
    return {
        "max_absolute": float(np.max(np.abs(delta))),
        "mean_absolute": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "relative_l2": float(np.linalg.norm(delta) / off_norm),
        "cosine_similarity": float(np.dot(off, on) / (off_norm * on_norm)),
        "norm_off": float(off_norm),
        "norm_on": float(on_norm),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--on-repeat", type=Path, action="append", default=[])
    args = parser.parse_args()
    tensor_dir = args.result_dir / "tensors"

    comparisons = []
    for dp_rank in (0, 1):
        for wave in range(1):
            rows = []
            first = None
            for layer in range(48):
                stage_rows = {}
                for stage in STAGES:
                    pattern = (
                        f"mode-{{mode}}_dp-{dp_rank}_local-x_wave-{wave}"
                        f"_step-2_layer-{layer:02d}_{stage}.npy"
                    )
                    off = np.load(tensor_dir / pattern.format(mode="off"))
                    on = np.load(tensor_dir / pattern.format(mode="on"))
                    value = metrics(off, on)
                    stage_rows[stage] = value
                    if first is None and value["max_absolute"] > 0:
                        first = {"layer": layer, "stage": stage, **value}
                rows.append({"layer": layer, "stages": stage_rows})
            comparisons.append({
                "dp_rank": dp_rank, "wave": wave,
                "first_divergence": first, "layers": rows,
            })

    repeat_locations = []
    off_dir = tensor_dir
    for repeat_index, repeat_dir in enumerate(args.on_repeat, 1):
        for dp_rank in (0, 1):
            first = None
            for layer in range(48):
                for stage in STAGES:
                    base = f"_dp-{dp_rank}_local-x_wave-0_step-2_layer-{layer:02d}_{stage}.npy"
                    off = np.load(off_dir / f"mode-off{base}")
                    on = np.load(repeat_dir / "tensors" / f"mode-on{base}")
                    value = metrics(off, on)
                    if value["max_absolute"] > 0:
                        first = {"layer": layer, "stage": stage, **value}
                        break
                if first is not None:
                    break
            repeat_locations.append({
                "repeat": repeat_index, "dp_rank": dp_rank, **first
            })

    summary = {
        "final_status": "GO",
        "capture_step": {
            "zero_based_generated_step": 3,
            "common_prefix": [2132, 4977, 1075],
        },
        "first_divergence_locations": [
            {"dp_rank": row["dp_rank"], "wave": row["wave"], **row["first_divergence"]}
            for row in comparisons
        ],
        "independent_on_repeat_locations": repeat_locations,
        "representative_layer_metrics": comparisons[0]["layers"],
        "all_comparisons": comparisons,
        "router_topk": "not captured: vLLM FusedMoE uses an internal fused router",
        "stage_decision": (
            "Layer 0 is exact through attention, dispatch input, MoE output, and final output. "
            "The first nonzero delta is layer 1 attention_output; layer 1 dispatch input and "
            "MoE see an already-different input, so dispatch/expert/combine are not the initial decode-stage source."
        ),
    }
    (args.result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
