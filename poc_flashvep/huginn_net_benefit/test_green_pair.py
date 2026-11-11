import unittest
from .green_pair import group_indices, checked


class DisjointGroupTests(unittest.TestCase):
    def test_eight_SM_prompt_leaves_fifteen_distinct_groups(self):
        decode, prompt = group_indices([8]*16, 8)
        self.assertEqual(len(decode), 15)
        self.assertEqual(prompt, [15])
        self.assertFalse(set(decode) & set(prompt))

    def test_merged_groups_still_come_from_one_split(self):
        decode, prompt = group_indices([8]*16, 32)
        self.assertEqual(sum([8]*len(decode)), 96)
        self.assertEqual(prompt, [12,13,14,15])

    def test_invalid_or_all_prompt_partition_is_rejected(self):
        for value in (0, True, 7, 132, 128, 136):
            with self.assertRaises(ValueError):
                group_indices([8]*16, value)

    def test_driver_errors_do_not_select_a_different_geometry(self):
        with self.assertRaises(RuntimeError):
            checked((1, None))
        self.assertEqual(checked((0, 4)), 4)
        self.assertEqual(checked((0, 4, 8)), (4, 8))


if __name__ == '__main__':
    unittest.main()
