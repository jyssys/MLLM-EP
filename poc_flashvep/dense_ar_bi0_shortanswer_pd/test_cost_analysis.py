import unittest
from .cost_analysis import category,exclusive


class CostAccounting(unittest.TestCase):
    def test_flash_cutlass_is_not_gemm(self):
        self.assertEqual(category('_ZN7cutlass_FlashAttnFwdSm90'),'Attention_named')
    def test_primitive_identity_precedes_generic_name(self):
        name='nvjet_tst_192x128'
        self.assertEqual(category(name),'Other_unclassified')
        self.assertEqual(category(name,{name}),'GEMM_API_identified')
    def test_exclusive_overlap_is_not_double_counted(self):
        result=exclusive([(0,10,'A'),(5,15,'B'),(6,9,'B')])
        self.assertEqual(result,dict(A=5e-6,B=5e-6,Cross_category_overlap=5e-6))


if __name__=='__main__':unittest.main()
