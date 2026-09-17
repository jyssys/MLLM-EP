import json

import numpy as np

from scripts.analyze_selective_refinement import read_generations
from selective_refinement.policy import PolicyConfig, choose_active_set


def test_warmup_and_age_force_refresh():
    masked = np.ones(8, dtype=bool)
    confidence = np.arange(8, dtype=float)
    ages = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    config = PolicyConfig(name="confidence_high", active_ratio=0.5, max_freeze_age=1)
    warm, forced = choose_active_set(masked, confidence, ages, config, block_iteration=0)
    assert warm.all()
    assert not forced.any()
    active, forced = choose_active_set(masked, confidence, ages, config, block_iteration=1)
    assert np.array_equal(np.flatnonzero(forced), np.arange(4, 8))
    assert active.sum() == 4


def test_ep_selection_prefers_complementary_token():
    masked = np.ones(3, dtype=bool)
    token_load = np.asarray([[4, 0, 0, 0], [0, 4, 0, 0], [3, 0, 0, 0]])
    base = np.asarray([8, 0, 0, 0])
    config = PolicyConfig(
        name="ep_load", active_ratio=1 / 3, max_freeze_age=4,
        target_ep=4, lambda_max=1.0, lambda_cv=1.0,
    )
    active, _ = choose_active_set(
        masked, np.zeros(3), np.zeros(3), config, block_iteration=1,
        token_rank_load=token_load, base_rank_load=base,
    )
    assert np.array_equal(np.flatnonzero(active), np.asarray([1]))


def test_generation_reader_includes_rebalanced_shards(tmp_path):
    worker = tmp_path / "generations_worker0.jsonl"
    rebalance = tmp_path / "generations_rebalance1.jsonl"
    worker.write_text(json.dumps({"sample_id": 0, "correct": True}) + "\n")
    rebalance.write_text(json.dumps({"sample_id": 127, "correct": False}) + "\n")

    rows = read_generations(tmp_path)

    assert set(rows) == {0, 127}
