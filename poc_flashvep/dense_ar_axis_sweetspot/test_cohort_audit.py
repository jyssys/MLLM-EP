import json
import unittest
from .cohort_audit import validate_shapes


def shape(requests,prefill,queries):
    return json.dumps(dict(requests=requests,prefilling=prefill,query_tokens=queries,padded_query_tokens=queries))


class CohortTests(unittest.TestCase):
    def test_prep1_decode4_passes(self):
        result=validate_shapes('Token',4,{shape(1,True,7111):72},
            {shape(1,True,889):72,shape(4,False,4):270},72,16)
        self.assertEqual(result['decode']['full36_decode4'],270)

    def test_decode2_cannot_pass_as_decode4(self):
        with self.assertRaises(AssertionError):
            validate_shapes('Layer',4,{shape(1,True,8000):72},
                {shape(1,True,8000):72,shape(2,False,2):540},72,16)

    def test_prefill2_cannot_pass_as_prep1(self):
        with self.assertRaises(AssertionError):
            validate_shapes('Token',4,{shape(1,True,7000):72},
                {shape(2,True,2000):36,shape(4,False,4):270},72,16)
