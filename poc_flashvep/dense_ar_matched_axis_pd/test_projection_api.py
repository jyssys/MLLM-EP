"""Reject a substituted projection API in the causal timing control."""
import inspect
import unittest
from . import isolated
from .isolated_serving_gemm import transformed_source


class ProjectionAPITests(unittest.TestCase):
    def test_legacy_and_serving_calls_are_not_labeled_identical(self):
        old = inspect.getsource(isolated.main)
        new = transformed_source()
        self.assertIn('operation=lambda x=x,weight=weight:torch.nn.functional.linear(x,weight)', old)
        self.assertNotIn('operation=lambda x=x,weight=weight:torch.nn.functional.linear(x,weight)', new)
        self.assertEqual(new.count('operation=lambda x=x,weight=weight:linear_batch_invariant(x,weight)'), 1)
        self.assertEqual(new.replace('linear_batch_invariant(x,weight)', 'torch.nn.functional.linear(x,weight)'), old)
        compile(new, 'projection_api_CPU_gate', 'exec')


if __name__ == '__main__':
    unittest.main()
