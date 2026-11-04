import unittest
from types import SimpleNamespace as Namespace
from .loop_burst import remaining_burst


def item(start=0, end=2048, depth=0, cut=31, prefix=2048, origin=0):
    boundaries = [[start, end]] + [[start, start] for _ in range(31)]
    request = Namespace(prompt_token_ids=range(2048), prefill_window_origin=origin,
        ltp_prefix=prefix, arc_export_cut=cut, num_preluded_tokens=end,
        prefill_depth_ranges=boundaries)
    return Namespace(request=request, token_start=start, token_count=end - start, depth=depth)


class BurstTests(unittest.TestCase):
    def test_whole_prompt_stops_at_the_export_cut(self):
        self.assertEqual(remaining_burst(item(), 32), 31)

    def test_token_prefix_uses_full_depth(self):
        self.assertEqual(remaining_burst(item(end=1984, prefix=1984, cut=32), 32), 32)

    def test_imported_suffix_retains_its_nonzero_origin(self):
        self.assertEqual(remaining_burst(item(start=1981, origin=1981, cut=32), 32), 32)

    def test_partial_chunk_does_not_bypass_wavefront_dependencies(self):
        self.assertIsNone(remaining_burst(item(end=1024), 32))

    def test_advanced_future_frontier_prevents_a_burst(self):
        value = item()
        value.request.prefill_depth_ranges[2][0] = 16
        self.assertIsNone(remaining_burst(value, 32))


if __name__ == '__main__':
    unittest.main()
