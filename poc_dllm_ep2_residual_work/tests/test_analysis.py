from pathlib import Path
import sys

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poc_dllm_ep2_residual_work.scripts.analyze_temporal_work import (  # noqa: E402
    destination_code,
    rel_l2,
)
from poc_dllm_ep2_residual_work.scripts.finalize_analysis import (  # noqa: E402
    epoch_fresh_sequence,
)


def test_destination_code_ep2():
    ids = torch.tensor([[0, 31], [32, 63], [3, 45]])
    assert destination_code(ids).tolist() == [1, 2, 3]


def test_relative_l2_identity_and_change():
    x = torch.tensor([[1.0, 2.0]])
    assert rel_l2(x, x).item() == 0.0
    assert rel_l2(torch.zeros_like(x), x).item() == 1.0


def test_epoch_fresh_lane_includes_previous_live_and_periodic_refresh():
    frame = pd.DataFrame({
        "iteration_id": [0, 1, 2, 3, 4, 5, 6],
        "physical_m": [128] * 7,
        "masked_before": [64, 63, 60, 55, 50, 44, 30],
    })
    assert epoch_fresh_sequence(frame) == [128, 64, 63, 60, 55, 128, 44]
