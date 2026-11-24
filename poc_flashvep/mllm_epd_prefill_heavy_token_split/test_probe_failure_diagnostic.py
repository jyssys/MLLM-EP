"""Request-level failures cannot be replaced with a pooled ITL percentile."""
import unittest

from .probe_failure_diagnostic import failure_counts
from .token_confirmation_analysis import CATEGORIES


class ProbeFailureTest(unittest.TestCase):
    def fixture(self):
        return [dict(category=category,requests=1,request_denominator=8,output_tokens=2,total_output_tokens=16)
            for category in CATEGORIES]

    def test_overlap_is_counted_once_per_joint_gate(self):
        row=failure_counts(self.fixture())
        self.assertEqual(row['TTFT_pass_requests'],4)
        self.assertEqual(row['ITL_pass_requests'],4)
        self.assertEqual(row['TPOT_pass_requests'],4)
        self.assertEqual(row['DECODE_pass_requests'],2)
        self.assertEqual(row['JOINT_pass_requests'],1)
        self.assertEqual(row['DECODE_qualified_tokens'],4)

    def test_category_count_and_tokens_conserve_denominators(self):
        rows=self.fixture();rows[0]['requests']=2
        with self.assertRaises(AssertionError):failure_counts(rows)
        rows=self.fixture();rows[0]['output_tokens']=3
        with self.assertRaises(AssertionError):failure_counts(rows)

    def test_duplicate_category_does_not_pass_eight_row_check(self):
        rows=self.fixture();rows[-1]['category']=rows[0]['category']
        with self.assertRaises(AssertionError):failure_counts(rows)


if __name__=='__main__':unittest.main()
