import torch

from method2.merge import expert_aware_merge, simulate_identity_expert_combine


def test_tokens_for_different_experts_are_not_clustered_together():
    hidden = torch.tensor([[1.0, 0.0], [0.99, 0.01]])
    assignment = torch.tensor([[0], [1]])
    importance = torch.tensor([0.4, 0.4])
    candidates = torch.tensor([0, 1])

    result = expert_aware_merge(candidates, assignment, hidden, importance, similarity_threshold=0.1)

    assert [cluster.token_indices.tolist() for cluster in result.clusters_by_expert[0]] == [[0]]
    assert [cluster.token_indices.tolist() for cluster in result.clusters_by_expert[1]] == [[1]]


def test_weighted_average_is_proportional_to_importance():
    hidden = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    assignment = torch.tensor([[0], [0]])
    importance = torch.tensor([1.0, 3.0])

    result = expert_aware_merge(
        torch.tensor([0, 1]),
        assignment,
        hidden,
        importance,
        similarity_threshold=-1.0,
    )

    cluster = result.clusters_by_expert[0][0]
    assert torch.allclose(cluster.weights, torch.tensor([0.25, 0.75]))
    assert torch.allclose(cluster.representative, torch.tensor([7.5, 0.0]))


def test_merge_reduces_expert_input_count():
    hidden = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]])
    assignment = torch.tensor([[0], [0], [0]])
    importance = torch.tensor([0.1, 0.2, 0.3])

    result = expert_aware_merge(torch.tensor([0, 1, 2]), assignment, hidden, importance, similarity_threshold=0.9)

    assert result.candidate_load_before[0] == 3
    assert result.candidate_load_after[0] == 1


def test_merge_replaces_only_the_target_expert_route():
    hidden = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    assignment = torch.tensor([[0, 1], [0, 2]])
    importance = torch.tensor([1.0, 3.0])
    routing_weights = torch.tensor([[0.5, 0.5], [0.5, 0.5]])

    result = expert_aware_merge(
        torch.tensor([0, 1]),
        assignment,
        hidden,
        importance,
        target_experts=[0],
        similarity_threshold=-1.0,
    )
    before = simulate_identity_expert_combine(hidden, assignment, None, routing_weights)
    after = simulate_identity_expert_combine(hidden, assignment, result, routing_weights)

    assert torch.allclose(before, hidden)
    assert torch.allclose(after[0], torch.tensor([3.75, 0.0]))
    assert torch.allclose(after[1], torch.tensor([8.75, 0.0]))

