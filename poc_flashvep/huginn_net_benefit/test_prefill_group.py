import unittest
from .prefill_group import bounded_count


class PrefillGroupTest(unittest.TestCase):
    def test_bounded_prelude_and_recurrent(self):
        stages = {'prelude', 'prompt_recurrent'}
        self.assertEqual(bounded_count(32, {'prelude'}, stages, 8), 8)
        self.assertEqual(bounded_count(0, {'prelude'}, stages, 8), 0)
        self.assertEqual(bounded_count(24, {'prompt_recurrent'}, stages, 8), 8)

    def test_native_other_phases_unchanged(self):
        stages = {'prelude', 'prompt_recurrent'}
        for phases in ({'decode'}, {'prompt_recurrent', 'decode'}, {'atomic_prompt'}, set()):
            self.assertEqual(bounded_count(32, phases, stages, 8), 32)


if __name__ == '__main__':
    unittest.main()
