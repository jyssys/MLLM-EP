"""A derived mean and an actual no-probe cohort are different observations."""
import unittest

from .probe_reference_variation import control_rows,control_pairs,CONTROLS,METRICS


class ProbeReferenceTest(unittest.TestCase):
    def fixture(self):
        return [dict(arm=arm,mode=mode,block=block,decode_pass_percent=94 if arm=='NativeStrong' and block==0 else 98,
            accepted_probes=0,**{metric:10 for metric in METRICS})
            for arm in CONTROLS+['ResidentNoProbeMean'] for mode in ['natural','replay'] for block in range(3)]

    def test_derived_mean_is_excluded_from_actual_cohort_counts(self):
        rows=control_rows(self.fixture())
        self.assertEqual(len(rows),24)
        self.assertEqual(sum(not r['absolute_decode_95_pass'] for r in rows),2)
        self.assertNotIn('ResidentNoProbeMean',{r['arm'] for r in rows})

    def test_control_pairs_preserve_independent_start_unit(self):
        rows=control_pairs(self.fixture())
        self.assertEqual(len(rows),40)
        self.assertTrue(all(r['independent_start_pairs']==3 for r in rows))
        self.assertEqual(sum(r['derived_control_is_not_another_start'] for r in rows),16)


if __name__=='__main__':unittest.main()
