#!/usr/bin/env python3
from pathlib import Path

import pandas as pd


root = Path(__file__).resolve().parents[1]
stability = pd.read_csv(root / "LAYER_PHASE_STABILITY.csv")
assert len(stability) == 192
assert set(stability["dataset"]) == {"gsm8k", "humaneval"}
assert not (stability["relative_l2_update_median"] < 0.05).any()

routed = pd.read_csv(root / "DECISION_SENSITIVITY_COST.csv")
routed = routed[routed["mode"] == "routed_bypass"]
assert len(routed) == 192
assert routed["live_top1_flip_fraction"].notna().all()

cross = pd.read_csv(root / "CROSS_TASK_ORACLE.csv")
assert not (cross["criterion"] == "final_trajectory_exact").any()
best_benchmark = cross[cross["criterion"] == "benchmark_vector_safe"].sort_values(
    "median_optimistic_e2e_percent"
).iloc[-1]
assert best_benchmark["mode"] == "stale_routed"
assert best_benchmark["median_optimistic_e2e_percent"] < 9.0
assert (
    0.5
    * (
        best_benchmark["gsm8k_post_epoch_percent"]
        + best_benchmark["humaneval_post_epoch_percent"]
    )
    < 5.0
)

gpu_script = (root / "scripts/run_ep4_sensitivity.sh").read_text()
assert "CUDA_VISIBLE_DEVICES=0,1,2,3" in gpu_script
assert "--gpu 0,1,2,3" in gpu_script
print("all invariants passed")
