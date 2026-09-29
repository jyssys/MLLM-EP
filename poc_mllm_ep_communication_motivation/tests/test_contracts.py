from __future__ import annotations

import torch

from poc_mllm_ep_communication_motivation.contract import ALLOWED_GPUS, EXPERTS_PER_RANK
from poc_mllm_ep_communication_motivation.instrumentation import _route_stats


def test_gpu_scope_and_owner_contract() -> None:
    assert ALLOWED_GPUS == (2, 3, 6, 7)
    assert EXPERTS_PER_RANK == 32


def test_destination_dedup_payload() -> None:
    ids = torch.tensor([[0, 1, 33, 34], [64, 65, 66, 67]])
    visual = torch.tensor([True, False])
    text = ~visual
    result = _route_stats(ids, visual, text, None)
    assert result["visual"]["pairs"] == 4
    assert result["visual"]["unique_token_destination_rows_per_rank"] == [1, 1, 0, 0]
    assert result["text"]["unique_token_destination_rows_per_rank"] == [0, 0, 1, 0]
    assert result["visual"]["dispatch_payload_bytes"] == 2 * 2048 * 2


def test_special_rows_do_not_leak_into_text() -> None:
    ids = torch.tensor([[0], [32], [64]])
    visual = torch.tensor([True, False, False])
    text = torch.tensor([False, True, False])
    result = _route_stats(ids, visual, text, None)
    assert result["special_other"]["tokens"] == 1
    assert result["text"]["tokens"] == 1
