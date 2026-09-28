import unittest

from poc_mllm_rank_aware_active_expert_reduction.contract import expert_owner


class ContractTest(unittest.TestCase):
    def test_owner_full_coverage(self) -> None:
        for ep, per_rank in ((2, 64), (4, 32), (8, 16)):
            owners = [expert_owner(expert, ep) for expert in range(128)]
            self.assertEqual(set(owners), set(range(ep)))
            self.assertTrue(all(owners.count(rank) == per_rank for rank in range(ep)))

    def test_ep8_is_structural_mapping(self) -> None:
        self.assertEqual(expert_owner(0, 8), 0)
        self.assertEqual(expert_owner(15, 8), 0)
        self.assertEqual(expert_owner(16, 8), 1)
        self.assertEqual(expert_owner(127, 8), 7)


if __name__ == "__main__":
    unittest.main()
