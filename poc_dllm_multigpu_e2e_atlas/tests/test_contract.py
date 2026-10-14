import unittest
from poc_dllm_multigpu_e2e_atlas.contract import *


class ContractTests(unittest.TestCase):
    def test_mask(self):self.assertEqual(MASK,'4,5,6,7')

    def test_frozen_ids_and_lengths(self):
        self.assertEqual(len(rows()),8);self.assertEqual(len(rows('held_out')),24)
        self.assertTrue(set(r['request_id'] for r in rows()).isdisjoint(r['request_id'] for r in rows('held_out')))
        for row in rows():
            for prefix in PREFIXES:self.assertEqual(len(token_ids(row,prefix)),prefix)


if __name__=='__main__':unittest.main()
