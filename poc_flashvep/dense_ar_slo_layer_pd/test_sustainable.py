"""Protect fixed-window denominators and queue stability gates."""
import unittest
from .sustainable import assess_rows


class SustainableTests(unittest.TestCase):
    def fixture(self):
        cfg=dict(name='fixture',request_caps=[16]*360,fixed_measurement=dict(warm_seconds=60.,
            measurement_seconds=240.,target_joint_attainment=.95,completion_rate_fraction=.98,
            outstanding_slope_max=.01,outstanding_slope_upper_max=.025,queue_block_seconds=30.))
        offered=[dict(sample_id=i,arrival=float(i)) for i in range(360)]
        rows=[dict(sample_id=i,arrival=float(i),finish=i+.5,first_token=i+.2,output_tokens=16,
            ttft_ms=200.,tpot_ms=20.,e2e_ms=500.,
            token_events=[dict(count=j+1,time=i+.2+j*.02) for j in range(16)]) for i in range(360)]
        return cfg,rows,offered
    def test_stable(self):
        cfg,rows,offered=self.fixture();r=assess_rows(cfg,rows,offered,0.)
        self.assertEqual(r['measurement_arrivals'],240)
        self.assertEqual(r['raw_completed_output_tokens_s'],16.)
        self.assertTrue(r['SLO_regimes']['Decode_priority']['qualified'])
    def test_unfinished_remains_denominator(self):
        cfg,rows,offered=self.fixture();rows=[r for r in rows if not 60<=r['sample_id']<90]
        r=assess_rows(cfg,rows,offered,0.)
        self.assertEqual(r['unfinished_measurement_arrivals'],30)
        self.assertEqual(r['SLO_regimes']['Balanced']['arrival_cohort_joint_attainment'],210/240)
        self.assertFalse(r['SLO_regimes']['Balanced']['qualified'])
    def test_warm_completion_counts_but_late_completion_does_not(self):
        cfg,rows,offered=self.fixture();rows[59]['finish']=60.1;rows[299]['finish']=300.1
        r=assess_rows(cfg,rows,offered,0.)
        self.assertEqual(r['measurement_completions'],240)
        self.assertEqual(r['measurement_arrivals'],240)
    def test_growing_queue_is_not_sustainable(self):
        cfg,rows,offered=self.fixture()
        for r in rows:r['finish']+=r['sample_id']*.4
        r=assess_rows(cfg,rows,offered,0.)
        self.assertFalse(r['SLO_regimes']['Balanced']['qualified'])
        self.assertIn('COMPLETION_RATE_BELOW_ARRIVALS',r['SLO_regimes']['Balanced']['rejections'])
    def test_coalesced_events_count_actual_token_increment(self):
        cfg,rows,offered=self.fixture()
        for r in rows:r['token_events']=[dict(count=1,time=r['arrival']+.2),dict(count=16,time=r['arrival']+.5)]
        r=assess_rows(cfg,rows,offered,0.)
        self.assertEqual(r['actual_token_events_in_window'],16.)
    def test_bounded_periodic_queue_does_not_create_growth_uncertainty(self):
        cfg,rows,offered=self.fixture();cfg['fixed_measurement']['measurement_seconds']=600.
        cfg['request_caps']=[16]*720
        offered=[dict(sample_id=i,arrival=float(i)) for i in range(720)]
        rows=[]
        for i in range(720):
            delay=5. if (i//30)%2 else .5
            rows.append(dict(sample_id=i,arrival=float(i),finish=i+delay+.3,first_token=i+delay,
                output_tokens=16,ttft_ms=delay*1000,tpot_ms=20.,e2e_ms=delay*1000+300,
                token_events=[dict(count=j+1,time=i+delay+j*.02) for j in range(16)]))
        result=assess_rows(cfg,rows,offered,0.)
        self.assertLessEqual(result['stability']['empirical_slope_upper_95'],.025)


if __name__=='__main__':unittest.main()
