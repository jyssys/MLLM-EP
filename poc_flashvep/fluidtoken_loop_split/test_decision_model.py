import unittest
from .decision_model import QueuedRequest, SplitCost, choose, decode_pressure, prefill_pressure


class Decisions(unittest.TestCase):
    def test_pressure_uses_cumulative_service_and_deadline(self):
        queue = [QueuedRequest(0, .3, 1), QueuedRequest(.1, .4, 1)]
        self.assertAlmostEqual(prefill_pressure(queue, .5), .7 / .6)

    def test_kv_cannot_count_reserved_twice(self):
        with self.assertRaises(ValueError):
            decode_pressure(8, 4, 10, 1, 4)
        self.assertEqual(decode_pressure(3, 2, 10, 1, 4)['free_blocks'], 5)

    def test_slack_and_memory_fallback(self):
        full = SplitCost(32, 1., 0., .01, 0)
        partial = SplitCost(31, .8, .03, .02, 2)
        args = dict(candidates=[partial, full], full=full, p_queue_service_s=1,
                    d_queue_service_s=0, d_iteration_s=.1, token_age_s=.01,
                    itl_limit_s=.2, tpot_margin_s=.17, free_blocks=2, pressure=2)
        self.assertIs(choose(**args)[0], partial)
        self.assertIs(choose(**dict(args, free_blocks=1))[0], full)
        self.assertIs(choose(**dict(args, token_age_s=.09))[0], full)
        self.assertIs(choose(**dict(args, has_continuation=True))[0], full)

    def test_no_gpu_speedup_claim(self):
        # CPU arithmetic tests do not validate the predictor or the runtime.
        self.assertEqual(prefill_pressure([], 0), 0)


if __name__ == '__main__':
    unittest.main()
