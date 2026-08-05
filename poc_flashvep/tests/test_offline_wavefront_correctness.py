from __future__ import annotations

import torch

from poc_flashvep.offline_wavefront.capture_schema import (
    SCHEMA_VERSION,
    validate_capture,
)
from poc_flashvep.offline_wavefront.workload_builder import (
    build_repeated_workload,
    rank_slice,
)


def test_capture_route_ownership_and_repeat_order() -> None:
    tokens, hidden_size, top_k = 4, 3, 2
    hidden = torch.arange(tokens * hidden_size, dtype=torch.bfloat16).reshape(
        tokens, hidden_size
    )
    ids = torch.tensor([[0, 33], [65, 127], [31, 32], [64, 96]])
    weights = torch.full((tokens, top_k), 0.5, dtype=torch.float32)
    capture = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "original_token_count": tokens,
            "hidden_size": hidden_size,
            "top_k": top_k,
            "ep_size": 4,
            "local_experts_per_rank": 32,
        },
        "post_attention_hidden": hidden,
        "topk_expert_ids": ids,
        "topk_weights": weights,
        "destination_rank": ids // 32,
        "local_expert_id": ids % 32,
    }
    validate_capture(capture)
    workload = build_repeated_workload(hidden, ids, weights, 4)
    assert workload.token_count == 16
    assert torch.equal(workload.hidden[:tokens], workload.hidden[tokens : 2 * tokens])
    reconstructed = torch.cat(
        [workload.hidden[rank_slice(workload.token_count, 4, rank)] for rank in range(4)]
    )
    torch.testing.assert_close(reconstructed, workload.hidden)


def test_microbatch_concatenation_restores_serial_order() -> None:
    values = torch.arange(32).reshape(16, 2)
    for microbatches in (2, 4):
        restored = torch.cat(list(values.chunk(microbatches, dim=0)), dim=0)
        assert torch.equal(restored, values)
