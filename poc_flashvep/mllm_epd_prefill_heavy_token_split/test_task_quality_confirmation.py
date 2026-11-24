"""Context dependence must remain intact during quality uncertainty calculation."""
import unittest
import numpy as np

from .task_quality_confirmation import context_components,quality_difference


class ContextQualityTest(unittest.TestCase):
    def row(self,identifier,image):
        return dict(request_id=identifier,images=[dict(SHA256=image,pixel_SHA256=image)],metadata={})

    def test_duplicate_context_questions_are_one_cluster(self):
        groups=context_components([self.row('a','shared'),self.row('b','shared'),self.row('c','different')])
        self.assertEqual(sorted(map(len,groups)),[1,2])

    def test_constant_paired_difference_stays_constant(self):
        actual=np.ones((3,3))*.9;reference=np.ones((3,3))
        result=quality_difference(actual,reference,[[0,1],[2]],samples=100)
        self.assertAlmostEqual(result['difference_pp'],-10)
        self.assertTrue(all(abs(x+10)<1e-10 for x in result['hierarchical_start_context_bootstrap95_CI_pp']))
        self.assertEqual(result['independent_start_pairs'],3)
        self.assertEqual(result['context_clusters'],2)


if __name__=='__main__':unittest.main()
