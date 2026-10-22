"""Feature selection tests with synthetic definition fixtures only."""
import unittest
from .predictor import feature_dict,vector,key

class LeakageTests(unittest.TestCase):
    def fixture(self):
        return dict(spans=[dict(tokens=128,phase='prefilling_recurrent')],actual_G=2,context_max=1024,
            actual_total_rows=128,actual_decode_rows=0,N=4,graph_entries=1,staged_bytes=1024,policy='grouped')

    def test_future_labels_do_not_change_features(self):
        a=self.fixture()
        b=dict(a,target_ms=1000000,deadline_violation_afterwards=True,future_outputs=[1,2,3],
            completed_token_residuals=[dict(time_to_completed_next_token_ms=999999)])
        self.assertEqual(feature_dict(a),feature_dict(b))
        self.assertEqual(vector(a),vector(b))
        self.assertEqual(key(a),key(b))

if __name__=='__main__':
    unittest.main()
