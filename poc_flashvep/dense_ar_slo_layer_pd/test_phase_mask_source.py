"""Preserve the scalar body and both original launch widths."""
import unittest
from .phase_mask_norm import CUDA,KERNEL,ORIGINAL,HEADER


class MaskSourceTests(unittest.TestCase):
    def test_body_has_only_a_uniform_phase_return(self):
        expected=ORIGINAL.split('\nvoid phase_norm(')[0]
        actual=KERNEL.replace('const int num_tokens, const int hidden_size, const int32_t* phases, int expected) {\n  if (phases[blockIdx.x] != expected) return;',HEADER)
        self.assertEqual(actual,expected)
    def test_original_thread_counts_remain_separate(self):
        self.assertIn('<<<x.size(0),256,0,stream>>>',CUDA)
        self.assertIn('<<<x.size(0),1024,0,stream>>>',CUDA)


if __name__=='__main__':unittest.main()
