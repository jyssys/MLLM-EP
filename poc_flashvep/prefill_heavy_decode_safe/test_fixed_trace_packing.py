"""Check accounting with synthetic times, not GPU evidence."""
import unittest
from .fixed_trace_packing import pack


class PackingTests(unittest.TestCase):
    def test_partial_prompt_tiles_do_not_count_as_a_loop(self):
        value=pack([(0.,.002),(1.,1.002)],2.,{1:0.},3)
        self.assertEqual(value['completed_full_loops'],0)
        self.assertEqual(value['unfinished_partial_tiles'],2)

    def test_dependency_can_span_multiple_windows(self):
        value=pack([(0.,.002),(1.,1.002),(2.,2.003)],2.,{1:0.},3)
        self.assertEqual(value['completed_full_loops'],1)
        self.assertEqual(value['executed_tiles'],3)

    def test_not_ready_work_cannot_use_an_earlier_window(self):
        value=pack([(0.,.010)],2.,{1:.020},1)
        self.assertEqual(value['completed_full_loops'],0)


if __name__=='__main__':unittest.main()
