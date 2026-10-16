"""Diagnostic property, not a benchmark accuracy test. Preserve its failure."""
import unittest
from poc_ouro_reasoning_fidelity_multigpu.contract import ROOT,read

class RestartProperty(unittest.TestCase):
    def test_identical_engine_kwargs_and_seed_reproduce_whole_output(self):
        record=read(ROOT/'results/quality_engine_restart_reproducibility.json')
        self.assertTrue(record['engine_kwargs_identical'])
        self.assertTrue(record['all_equal'],'Fresh-session outputs differ. This test does not establish a sampling-distribution bug.')

if __name__=='__main__':
    unittest.main()
