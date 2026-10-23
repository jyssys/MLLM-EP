import unittest
from ..cost_predictor import ServiceLookup,path_features

class LookupTests(unittest.TestCase):
    def test_empirical_path_cost_and_calibration_slack_are_separate(self):
        model=ServiceLookup(dict(tables=[dict(path='staged_core',rows=512,G=2,depth=0,
            context_bin=2048,median_ms=20.,q95_ms=23.)],slack_ms={'staged_core':4.}))
        self.assertEqual(model.raw('staged_core',512,2,2000,0),(20.,True))
        self.assertEqual(model.estimate('staged_core',512,2,2000,0),(24.,True))
        self.assertFalse(model.estimate('staged_core',1024,2,8000,0)[1])
        self.assertFalse(model.estimate('staged_core',512,3,2000,0)[1])

    def test_mixed_is_not_pure_prefill_cost(self):
        row=dict(spans=[dict(phase='prefilling_recurrent',G=1,depth=0),dict(phase='recurrent',G=1,depth=2)],
            actual_rows=513,context_max=2048)
        features=path_features(row)
        self.assertEqual(features['path'],'mixed_core')
        self.assertEqual(features['decode_rows'],1)

if __name__=='__main__':
    unittest.main()
