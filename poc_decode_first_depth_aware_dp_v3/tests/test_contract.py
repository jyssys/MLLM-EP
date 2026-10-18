"""CPU tests for information access, immutable data, and ownership constraints."""
import unittest
from poc_decode_first_depth_aware_dp_v3.contract import *

class Contract(unittest.TestCase):
    def test_selection(self):
        r=read(ROOT/'artifacts/dynamic_baseline_selection.json')
        self.assertEqual(r['decision'],'INCUMBENT_CONTINUE')
        self.assertEqual((r['q'],r['min_depth'],r['max_depth']),(.5,2,4))
        self.assertFalse(r['synthetic_replay'])

    def test_frozen_prior(self):
        r=read(ROOT/'artifacts/continuity.json')
        self.assertEqual(r['base_commit'],BASE_COMMIT)
        self.assertEqual(r['prior_quality_verdict'],'QUALITY_FAIL')
        for source,digest in r['sources'].items():self.assertEqual(sha(OLD_ROOT/source),digest)

    def test_cohort(self):
        r=read(ROOT/'artifacts/discovery_frozen.json')['rows']
        self.assertEqual(len(r),128)
        self.assertEqual(len({x['id'] for x in r}),128)
        for domain in ['gsm8k','math500','alpaca','sharegpt']:
            values=[v for v in r if v['dataset']==domain]
            self.assertEqual(len(values),32)
            self.assertEqual(sum(v['split']=='calibration' for v in values),8)

    def test_destination_replicas(self):
        self.assertEqual(GPUS,[4,5,6,7])
        self.assertEqual(MASK,'4,5,6,7')

    def test_simulator_progress(self):
        from poc_decode_first_depth_aware_dp_v3.simulator import CostModel,simulate_rank
        model=CostModel({s:dict(coefficients=[1.]+[0.]*(n-1),minimum_ms=1.)
            for s,n in [('prefill',4),('recurrent',5),('coda',2)]})
        requests=[dict(id=str(i),prompt_tokens=8,depths=[2,2],arrival_ms=0,order=i,cap=3) for i in range(3)]
        result=simulate_rank(requests,model)
        self.assertEqual(len(result['outputs']),3)
        self.assertEqual(result['makespan_ms'],7)
        self.assertEqual(sum(s['stage']=='recurrent' for s in result['stages']),4)

    def test_information_boundary(self):
        plan=read(ROOT/'artifacts/live_clean_plan.json')
        for values in plan['causal_features'].values():
            self.assertEqual(set(values),{'prompt_tokens','expected_output_tokens','expected_work_ms'})
        for case in plan['cases']:
            self.assertEqual(len(case['ids']),len(set(case['ids'])))

    def test_report_decision(self):
        path=ROOT/'artifacts/final_decision.json'
        if not path.exists():self.skipTest('Final analysis is not complete.')
        d=read(path)
        self.assertEqual(d['source_selection'],'INCUMBENT_CONTINUE')
        self.assertEqual(d['quality'],'QUALITY_WARNING')
        self.assertEqual(d['RL'],'NOT_RUN')

    def test_published_json_fallback(self):
        import gzip,tempfile
        with tempfile.TemporaryDirectory(prefix='decode_dp_v3_test_') as directory:
            path=Path(directory)/'probe.json'
            with gzip.open(str(path)+'.gz','wt') as stream:json.dump({'valid':True},stream)
            self.assertEqual(read(path),{'valid':True})

if __name__=='__main__':unittest.main()
