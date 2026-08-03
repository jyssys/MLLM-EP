import torch

from pipeline.dummy_moe import run_dummy_pipeline


def test_dummy_pipeline_runs_shapes_and_preserves_high_importance_tokens():
    attention = torch.zeros(6, 6)
    attention[0, 2:6] = torch.tensor([0.1, 0.2, 0.9, 0.3])
    attention[1, 2:6] = torch.tensor([0.1, 0.2, 0.9, 0.3])
    hidden = torch.tensor(
        [
            [1.0, 0.0],
            [1.1, 0.0],
            [9.0, 0.0],
            [1.2, 0.0],
        ]
    )
    expert_assignment = torch.tensor([[0], [0], [0], [1]])
    expert_to_gpu = torch.tensor([0, 1])

    result = run_dummy_pipeline(
        attention=attention,
        text_indices=[0, 1],
        vision_indices=[2, 3, 4, 5],
        hidden_states=hidden,
        expert_assignment=expert_assignment,
        expert_to_gpu=expert_to_gpu,
        straggler_threshold=2.0,
        rho=0.67,
        merge_similarity_threshold=0.9,
        apply_cap=False,
    )

    assert result["importance"].shape == (4,)
    assert result["output_before"].shape == hidden.shape
    assert result["output_after"].shape == hidden.shape
    assert not result["selection"]["candidate_mask"][2]
    assert torch.allclose(result["output_before"][2], result["output_after"][2])
    assert result["merge"].candidate_load_after[0] < result["merge"].candidate_load_before[0]

