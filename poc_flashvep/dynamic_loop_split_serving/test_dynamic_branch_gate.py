import copy
import unittest
from .dynamic_branch_gate import branch_reached


class DynamicBranchTests(unittest.TestCase):
    def setUp(self):
        self.work = dict(completed=2, completed_offloaded_serving_requests=2,
            measured_D_prompt_rows_from_active_counter=8192,
            requests=[dict(partial=True, cut=3, prefix=4096) for _ in range(2)])

    def test_exact_output_with_no_offload_does_not_pass_accept(self):
        work = copy.deepcopy(self.work)
        work['completed_offloaded_serving_requests'] = 0
        self.assertFalse(branch_reached('loop', 'accept', 3, 2, work, 0))

    def test_accept_requires_selected_key_and_active_rows(self):
        self.assertTrue(branch_reached('loop', 'accept', 3, 2, self.work, 0))
        self.assertFalse(branch_reached('loop', 'accept', 2, 2, self.work, 0))
        self.work['measured_D_prompt_rows_from_active_counter'] = 0
        self.assertFalse(branch_reached('loop', 'accept', 3, 2, self.work, 0))

    def test_reject_requires_measured_frontier_rejection_and_zero_D_work(self):
        self.work.update(completed_offloaded_serving_requests=0,
            measured_D_prompt_rows_from_active_counter=0,
            requests=[dict(partial=False, cut=4, prefix=4096) for _ in range(2)])
        self.assertTrue(branch_reached('loop', 'reject', 3, 2, self.work, 2))
        self.assertFalse(branch_reached('loop', 'reject', 3, 2, self.work, 0))


if __name__ == '__main__':
    unittest.main()
