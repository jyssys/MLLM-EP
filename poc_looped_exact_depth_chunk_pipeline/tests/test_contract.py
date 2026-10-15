import unittest
from poc_looped_exact_depth_chunk_pipeline.scoring import final_answer,aime_integer
from poc_looped_exact_depth_chunk_pipeline.dag import diagonal_order,verify_order,schedule,micro_isomorphism


class ScoringTests(unittest.TestCase):
    def test_box_boundary(self):
        self.assertEqual(final_answer(r'Work 17. Final $\boxed{\frac{1}{2}}$')[0],r'\frac{1}{2}')
        self.assertEqual(final_answer(r'\boxed{123} then \boxed{204}')[0],'204')
        self.assertIsNone(final_answer('Reasoning has 12 and 204.')[0])
        self.assertIsNone(final_answer(r'Incomplete \boxed{12')[0])
    def test_integer(self):
        self.assertEqual(aime_integer('004'),4)
        self.assertIsNone(aime_integer('1004'))
        self.assertIsNone(aime_integer('4+0'))
        self.assertIsNone(aime_integer('__import__("os")'))
    def test_causal_dependencies(self):
        for chunks in [1,2,4,16,64]:
            self.assertTrue(verify_order(diagonal_order(chunks)))
        with self.assertRaises(AssertionError):verify_order([(1,0),(0,0)])
        self.assertTrue(micro_isomorphism(8)['edge_sets_equal'])
    def test_bubble(self):
        end,rows=schedule([1.]*8,[0.]*8)
        self.assertEqual(end,11.)
        self.assertEqual(len(rows),32)
        end,_=schedule([1.]*8,[0.]*8,stages=2)
        self.assertEqual(end,18.)


if __name__=='__main__':unittest.main()
