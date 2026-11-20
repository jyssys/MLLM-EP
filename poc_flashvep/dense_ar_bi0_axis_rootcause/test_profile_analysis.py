"""Check that interval arithmetic does not count overlap twice."""
import unittest
from .profile_analysis import union
from ..dense_ar_bi0_shortanswer_pd.cost_analysis import exclusive,category


class ProfileArithmetic(unittest.TestCase):
    def test_union(self):
        self.assertEqual(union([(0,4),(2,6),(7,9),(3,3)]),8)

    def test_exclusive(self):
        value=exclusive([(0,4000000,'GEMM'),(2000000,6000000,'Attention')])
        self.assertEqual(value,{'GEMM':2.,'Cross_category_overlap':2.,'Attention':2.})

    def test_attention_before_gemm_identity(self):
        name='cutlass_flash_fwd_sm90'
        self.assertEqual(category(name,{name}),'Attention_named')


if __name__=='__main__':unittest.main()
