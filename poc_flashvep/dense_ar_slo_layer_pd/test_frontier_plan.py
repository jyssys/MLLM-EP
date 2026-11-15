"""Test readiness, exact request coverage and non-preemptive retirement."""
import unittest
from .frontier_plan import simulate


class FrontierPlanTests(unittest.TestCase):
    def run_plan(self,**changes):
        args=dict(now=0.,source_ms=800.,source_frame_ms=lambda n:100.*n,slots=4,
            maximum_destination=8,caps=[16]*8,available=0.,cost=lambda n:(20.,80.,2.,20.))
        args.update(changes);return simulate(**args)
    def test_streaming_first_cohort_does_not_wait_for_last_prefix(self):
        r=self.run_plan()
        self.assertAlmostEqual(r['source_frontiers'][0],.4)
        self.assertAlmostEqual(r['source_frontiers'][7],.8)
        self.assertLess(r['first_outputs'][0],r['source_ready'])
        self.assertEqual([c['batch'] for c in r['cohorts']],[4,4])
    def test_known_busy_destination_can_collect_a_larger_batch(self):
        r=self.run_plan(available=1.)
        self.assertEqual([c['batch'] for c in r['cohorts']],[8])
        self.assertGreaterEqual(r['first_suffix_start'],1.)
    def test_whole_source_guard_matches_its_actual_runtime_contract(self):
        r=self.run_plan(wait_whole_source=True)
        self.assertEqual([c['batch'] for c in r['cohorts']],[8])
        self.assertAlmostEqual(r['first_suffix_start'],.82)
    def test_decode_is_not_preempted(self):
        r=self.run_plan();a,b=r['cohorts']
        self.assertGreaterEqual(b['handoff_start'],a['retire'])
    def test_every_known_request_is_processed_once(self):
        r=self.run_plan(caps=[16]*10)
        self.assertEqual(sorted(i for c in r['cohorts'] for i in c['indices']),list(range(10)))
    def test_heterogeneous_caps_charge_the_actual_selected_cohort(self):
        r=self.run_plan(caps=[16]*4+[64]*4)
        self.assertAlmostEqual(r['cohorts'][0]['decode_ms'],300.)
        self.assertAlmostEqual(r['cohorts'][1]['decode_ms'],1260.)


if __name__=='__main__':unittest.main()
