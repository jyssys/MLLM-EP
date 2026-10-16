"""A problem cluster must retain its repeated sampled seeds."""
import unittest
from poc_ouro_reasoning_fidelity_multigpu.scripts.analyze_quality import problem_ci


class ProblemClusterTests(unittest.TestCase):
    def test_seeds_do_not_create_six_independent_problems(self):
        result=problem_ci([('a',1)]*3+[('b',0)]*3)
        self.assertEqual(result['independent_problems'],2)
        self.assertEqual(result['mean'],.5)
        self.assertEqual(result['CI95'],[0.,1.])

    def test_paired_problem_delta_keeps_its_sign(self):
        result=problem_ci([('a',1)]*3+[('b',-1)]*3)
        self.assertEqual(result['independent_problems'],2)
        self.assertEqual(result['mean'],0.)
        self.assertEqual(result['CI95'],[-1.,1.])


if __name__=='__main__':
    unittest.main()
