"""Check the actual period averaging and transport evidence selection."""
import unittest

from .transport_policy_analysis import start_policy_means,paired_delta,import_groups
from .transport_policy_control import policy_order


class PolicyAnalysisTest(unittest.TestCase):
    def fixture(self):
        return [dict(block=block,mode=mode,period=period,inline=inline,value=100*block+period)
            for block in range(3) for mode in ['natural','replay'] for period,inline in enumerate(policy_order(block))]

    def test_two_periods_are_averaged_before_independent_start_inference(self):
        result=start_policy_means(self.fixture(),['value'])
        self.assertEqual(len(result),12)
        for row in result:
            self.assertEqual(row['value'],100*row['block']+1.5)
            self.assertEqual(row['independent_initializations'],1)
            self.assertTrue(row['periods_are_not_independent_starts'])

    def test_a_missing_period_cannot_be_hidden_by_pooling_requests(self):
        with self.assertRaises(AssertionError):start_policy_means(self.fixture()[1:],['value'])

    def test_zero_reference_host_delta_is_valid_without_a_ratio(self):
        result=paired_delta([0,1,2],[0,0,0])
        self.assertEqual(result['mean_delta'],1)
        self.assertEqual(result['independent_start_pairs'],3)
        self.assertEqual(result['empirical_resamples'],27)

    def group(self):
        return dict(begin=2.,end=3.,producer_pid=4,consumer_pid=5,clone_complete_before_ACK=True,
            ack_mode='file',images=7,inline_images=7,export_RPC_ms=0.)

    def test_groups_outside_the_cohort_are_not_attributed_to_its_requests(self):
        row=self.group();outside=dict(row,begin=11.,end=12.)
        self.assertEqual(import_groups([row,outside],1.,10.,True),[row])
        row['export_RPC_ms']=1.
        with self.assertRaises(AssertionError):import_groups([row],1.,10.,True)

    def test_crossing_or_wrong_policy_groups_are_rejected_not_clipped(self):
        row=self.group()
        with self.assertRaises(AssertionError):import_groups([row],2.5,10.,True)
        with self.assertRaises(AssertionError):import_groups([row],1.,10.,False)


if __name__=='__main__':unittest.main()
