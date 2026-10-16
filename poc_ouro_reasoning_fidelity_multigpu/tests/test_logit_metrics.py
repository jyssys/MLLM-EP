"""Top-1 analysis must use the same tie rule as greedy argmax."""
import unittest
import numpy as np
from poc_ouro_reasoning_fidelity_multigpu.scripts.analyze_fidelity import logit_metrics

class LogitMetricTests(unittest.TestCase):
    def test_equal_maximum_uses_first_token_index(self):
        value=np.array([0.,0.,0.,1.,2.,2.])
        result=logit_metrics(value,value)
        self.assertEqual(result['reference_top5'][0],int(np.argmax(value)))
        self.assertEqual(result['actual_top5'][0],4)
        self.assertEqual(result['reference_margin'],0.)
        self.assertTrue(result['top1_equal'])

if __name__=='__main__':
    unittest.main()
