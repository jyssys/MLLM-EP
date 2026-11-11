"""Guard the exact fixed-depth bulk transition used by the Runtime extension."""
import unittest
from types import SimpleNamespace as S
from .fixed_decode import group_limit, advance_before_native_update


def request(depth=0, threshold=1., minimum=32, maximum=32):
    return S(loops_done=depth, sampling_params=S(min_loops=minimum,
        max_loops=maximum, exit_threshold=threshold))


class FixedDecodeTests(unittest.TestCase):
    def test_all_remaining_loops_are_explicit(self):
        self.assertEqual(group_limit([request(), request()], 32, 32), 32)

    def test_mixed_depth_stops_at_first_valid_boundary(self):
        self.assertEqual(group_limit([request(4), request(12)], 32, 32), 20)

    def test_quantum_cap_is_preserved(self):
        self.assertEqual(group_limit([request(4)], 32, 8), 8)

    def test_no_early_exit_configuration_is_accepted(self):
        with self.assertRaises(ValueError):
            group_limit([request(threshold=.5)], 32, 32)

    def test_native_final_increment_reaches_exact_depth(self):
        r = request(4)
        advance_before_native_update(r, 28, 32)
        self.assertEqual(r.loops_done, 31)
        r.loops_done += 1
        self.assertEqual(r.loops_done, 32)

    def test_excess_execution_is_rejected(self):
        with self.assertRaises(ValueError):
            advance_before_native_update(request(4), 29, 32)


if __name__ == '__main__':
    unittest.main()
