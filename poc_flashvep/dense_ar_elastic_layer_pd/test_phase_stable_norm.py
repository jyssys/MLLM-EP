"""Check phase frontiers without a GPU or a future arrival trace."""
import unittest
from .phase_stable_norm import phase_runs


class PhaseRunsTest(unittest.TestCase):
    def test_mixed(self):self.assertEqual(phase_runs([8000,6144],[1,512],8000),[(0,1,'DECODE'),(1,513,'PREFILL')])
    def test_native_token_reuse(self):self.assertEqual(phase_runs([8000],[1],8000),[(0,1,'DECODE')])
    def test_prompt_only(self):self.assertEqual(phase_runs([0,0],[8000,8000],8000),[(0,16000,'PREFILL')])
    def test_frontier(self):self.assertEqual(phase_runs([7998],[4],8000),[(0,2,'PREFILL'),(2,4,'DECODE')])


if __name__=='__main__':unittest.main()
