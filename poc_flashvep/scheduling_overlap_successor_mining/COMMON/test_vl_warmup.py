"""CPU-only actual warmup selector regression for new real-image families."""
import unittest
from run_vl_operation_trace import warmup_pool


class WarmupSelectionTest(unittest.TestCase):
    def test_original_family_unchanged(self):
        rows = [{"family": "single_real_image", "images": ["real.png"], "id": i}
                for i in range(8)]
        self.assertEqual(warmup_pool(rows), rows)

    def test_large_matched_pool_has_two_requests_per_dp(self):
        rows = [{"family": family, "images": images, "id": i}
                for family, images in [("matched_natural_text_short", []),
                                       ("four_real_charts_short", ["chart.png"]*4)]
                for i in range(8)]
        pool = warmup_pool(rows)
        for rank in (0, 1):
            selected = pool[rank*2:(rank+1)*2]
            self.assertEqual(len(selected), 2)
            self.assertTrue(all(r["images"] for r in selected))


if __name__ == "__main__":
    unittest.main()
