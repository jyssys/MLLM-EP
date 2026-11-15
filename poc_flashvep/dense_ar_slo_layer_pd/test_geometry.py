"""Keep heterogeneous prompt work atomic and budget accounting exact."""
import unittest
from .geometry import prefix_group,macro_frontier_ready,source_queued_inside_group


class GeometryTest(unittest.TestCase):
    def test_long_group_fits(self):self.assertEqual(prefix_group(range(4),[12288]*4,4,32000),2)
    def test_short_group_uses_all_slots(self):self.assertEqual(prefix_group(range(4),[2048]*4,4,32000),4)
    def test_mixed_rows(self):self.assertEqual(prefix_group(range(4),[2048,12288,8192,12288],4,32000),3)
    def test_equal_length_calibration_group(self):self.assertEqual(prefix_group(range(4),[2048,2048,8192,8192],4,32000,True),2)
    def test_no_prompt_splitting(self):
        with self.assertRaises(AssertionError):prefix_group([0],[12288],1,8192)
    def test_split_macro_reproduces_old_starvation_predicate(self):
        members=['q0','q1'];states={'q0':'finished','q1':'D_HOST_PENDING'}
        self.assertFalse(all(states.get(rid)=='D_HOST_PENDING' for rid in members))
        self.assertTrue(macro_frontier_ready(members,states))
    def test_a_partial_source_frontier_is_not_ready(self):
        self.assertFalse(macro_frontier_ready(['q0','q1'],{'q0':'D_HOST_PENDING','q1':'prefill'}))
    def test_source_queue_does_not_depend_on_a_destination_wait_guard(self):
        self.assertEqual(source_queued_inside_group(8,4),4)
        self.assertEqual(source_queued_inside_group(3,4),0)
        self.assertEqual(source_queued_inside_group(16,4),12)


if __name__=='__main__':unittest.main()
