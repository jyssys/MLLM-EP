from __future__ import annotations

import unittest

import torch

from poc_mllm_resolution_expert_tradeoff.contract import configured_max_pixels
from poc_mllm_resolution_expert_tradeoff.routing import apply_visual_k


class ContractTests(unittest.TestCase):
    def test_visual_k_noop_is_same_tensor(self) -> None:
        weights = torch.rand(3, 8)
        ids = torch.arange(24).reshape(3, 8)
        out_weights, out_ids = apply_visual_k(
            weights, ids, torch.tensor([True, False, True]), 8
        )
        self.assertIs(out_weights, weights)
        self.assertIs(out_ids, ids)

    def test_visual_k_only_changes_visual_rows_and_renormalizes(self) -> None:
        weights = torch.tensor([
            [0.01, 0.21, 0.10, 0.03, 0.30, 0.04, 0.12, 0.19],
            [0.08, 0.12, 0.15, 0.09, 0.18, 0.14, 0.11, 0.13],
        ])
        ids = torch.arange(16).reshape(2, 8)
        out_weights, out_ids = apply_visual_k(
            weights, ids, torch.tensor([True, False]), 4
        )
        self.assertEqual(out_ids[0].tolist(), [4, 1, 7, 6, -1, -1, -1, -1])
        self.assertTrue(torch.allclose(out_weights[0].sum(), torch.tensor(1.0)))
        self.assertTrue(torch.equal(out_weights[1], weights[1]))
        self.assertTrue(torch.equal(out_ids[1], ids[1]))
        self.assertEqual(int(out_ids[0].ge(0).sum()), 4)

    def test_resolution_cap_never_changes_reference_or_upsamples_below_floor(
        self,
    ) -> None:
        self.assertIsNone(configured_max_pixels(1_000_000, 1.0))
        self.assertEqual(configured_max_pixels(1_000_000, 0.5), 500_000)
        self.assertEqual(configured_max_pixels(70_000, 0.25), 65_536)
