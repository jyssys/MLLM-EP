import torch

from method1.placement import (
    brute_force_optimal_max_load,
    compute_loads,
    contiguous_placement,
    lpt_placement,
)


def test_lpt_reduces_max_load_for_clustered_vision_heavy_experts():
    weights = torch.tensor([10.0, 9.0, 8.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    vanilla = contiguous_placement(num_experts=weights.numel(), num_gpus=4)
    vanilla_loads = compute_loads(weights, vanilla, num_gpus=4)

    result = lpt_placement(weights, num_gpus=4)

    assert result.gpu_loads.max() < vanilla_loads.max()


def test_lpt_satisfies_four_thirds_sanity_against_optimal_small_case():
    weights = torch.tensor([8.0, 7.0, 6.0, 5.0, 4.0])
    result = lpt_placement(weights, num_gpus=2)
    optimal, _ = brute_force_optimal_max_load(weights, num_gpus=2)

    assert float(result.gpu_loads.max()) <= (4.0 / 3.0) * optimal


def test_equal_weights_distribute_evenly():
    weights = torch.ones(8)

    result = lpt_placement(weights, num_gpus=4)

    assert sorted(result.gpu_loads.tolist()) == [2.0, 2.0, 2.0, 2.0]

