"""Keep request-weighted and token-weighted failure fractions separate."""
import unittest

from .main100_result_figures import failure_averages
from .token_confirmation_analysis import CATEGORIES


class FailureFigureTest(unittest.TestCase):
    def rows(self):
        return [dict(block=block,method=method,mode=mode,category=category,
            requests=50 if category=='JOINT_PASS' else 50 if category=='TTFT_FAIL' else 0,
            request_denominator=100,output_tokens=10 if category=='JOINT_PASS' else 90 if category=='TTFT_FAIL' else 0,
            total_output_tokens=100)
            for block in range(3) for mode in ['natural','replay']
            for method in ['NativeStrong','NativeCommon','Static10','Idle10'] for category in CATEGORIES]

    def test_token_weight_does_not_equal_request_fraction(self):
        rows=failure_averages(self.rows());passed=next(r for r in rows if r['category']=='JOINT_PASS')
        self.assertEqual(passed['start_mean_request_percent'],50.)
        self.assertEqual(passed['start_mean_token_percent'],10.)

    def test_missing_independent_start_is_not_silently_pooled(self):
        with self.assertRaises(AssertionError):failure_averages(self.rows()[1:])


if __name__=='__main__':unittest.main()
