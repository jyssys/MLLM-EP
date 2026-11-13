import unittest
from .enclosure_analysis import execution_class


class ClassificationTests(unittest.TestCase):
    def test_cached_one_query_is_full_decode_not_suffix(self):
        row=dict(role='decode',prefilling=True,cut=28,layer_mode=None,query_tokens=4,requests=['q0','q1','q2','q3'])
        self.assertEqual(execution_class(row),('D_FULL36_DECODE',36))
    def test_true_full_prompt_suffix(self):
        row=dict(role='decode',prefilling=True,cut=28,layer_mode='MIGRATE',query_tokens=32000,requests=['q0','q1','q2','q3'])
        self.assertEqual(execution_class(row),('D_SUFFIX',28))
    def test_native_prefix_ignores_unused_configuration_cut(self):
        row=dict(role='prefill',prefilling=True,cut=28,layer_mode=None,query_tokens=8000,requests=['q0'])
        self.assertEqual(execution_class(row),('P_PREFIX',36))
    def test_colocated_semantic_prefill_and_decode(self):
        row=dict(role='replica0',prefilling=True,cut=28,layer_mode=None,query_tokens=8000,requests=['q0'])
        self.assertEqual(execution_class(row),('COLOCATED_PREFILL',36))
        row['prefilling']=False;row['query_tokens']=1
        self.assertEqual(execution_class(row),('COLOCATED_FULL36_DECODE',36))


if __name__=='__main__':unittest.main()
