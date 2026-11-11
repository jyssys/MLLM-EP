import unittest
from .analyze import accounting,union_length,percentile
from .output_analysis import idle_windows


class AccountingTests(unittest.TestCase):
    def test_signed_delivery_delta_is_not_extra_gpu_work(self):
        row=accounting(90,95,80)
        self.assertEqual(row,dict(model_enclosure_ms=80,between_enclosure_ms=15,output_observation_delta_ms=-5))
        self.assertEqual(sum(row.values()),90)

    def test_union_does_not_add_overlapping_enclosures(self):
        self.assertEqual(union_length([(0,5),(3,8),(12,14)]),10)

    def test_percentile_is_request_level_interpolation(self):
        self.assertEqual(percentile([0,10,20],.95),19)

    def test_physical_windows_use_union_and_clipped_denominator(self):
        union,idle=idle_windows([(-3,2),(1,4),(6,8),(7,20)],0,10)
        self.assertEqual(union,[(0,4),(6,10)])
        self.assertEqual(idle,[(4,6)])

    def test_empty_kernel_capture_does_not_imply_a_safe_window(self):
        union,idle=idle_windows([],1,6)
        self.assertEqual(union,[])
        self.assertEqual(idle,[(1,6)])
        # The helper exposes physical inactivity only. No safety label exists.
