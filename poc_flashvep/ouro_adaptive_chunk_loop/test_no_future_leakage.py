"""Feature selection tests with synthetic definition fixtures only."""
import unittest
from .predictor import feature_dict,vector,key
from .decode_prediction import vector as decode_vector,key as decode_key

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

    def test_completion_labels_do_not_change_decode_features(self):
        a=dict(policy='grouped',phase='recurrent',remaining_R=2,context=2048,N=4,active_requests=3,
            elapsed_since_last_token_ms=30,pending_prefill_tokens=2048,staged_prompt_tokens=512,
            configured_C=512,actual_G=2,graph_entries=3,host_load1=2,load_missing=0,pending_submission_events=1)
        b=dict(a,target_ms=999999,future_output_ids=[1,2],future_arrivals=[3,4])
        self.assertEqual(decode_vector(a),decode_vector(b))
        self.assertEqual(decode_key(a),decode_key(b))

if __name__=='__main__':
    unittest.main()
