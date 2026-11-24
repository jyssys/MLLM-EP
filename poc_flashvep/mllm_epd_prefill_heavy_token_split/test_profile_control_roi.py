"""Check the actual CPC control path with CPU-only serving and RPC fixtures."""
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class ProfileControlROITest(unittest.IsolatedAsyncioTestCase):
    async def test_short_observer_keeps_all_three_full_cohort_runs(self):
        from . import profile_control as control
        from .common import write
        with TemporaryDirectory() as directory:
            root=Path(directory);manifest=root/'manifest.json';reference=root/'reference.json'
            rows=[dict(request_id=f'request-{i}') for i in range(100)]
            write(manifest,dict(requests=rows));write(reference,{})
            for name in ['E','P','D']:write(root/'state'/f'server_{name}.json',dict(ready=True,finished=False))
            runs=[];calls=[]
            async def execute(requests,urls,run,qps,seed,counts,debug,**kwargs):
                runs.append(dict(name=run,ids=[r['request_id'] for r in requests],qps=qps,seed=seed,counts=counts))
                path=root/'raw'/f'{run}_requests.jsonl';path.parent.mkdir(exist_ok=True)
                for row in requests:
                    with path.open('a') as stream:stream.write(json.dumps(row)+'\n')
                    if run.endswith('_profile'):await asyncio.sleep(.002)
                return [dict(request_id=r['request_id'],status='MEASURED_ONLINE',output_tokens=5) for r in requests]
            def rpc(url,operation,body):calls.append((url,operation));return [dict(passed=True)]
            metrics={k:1. for k in ['duration_seconds','TTFT_p50_ms','TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms']}
            options=SimpleNamespace(manifest=manifest,replay_from=reference,server_names='E,P,D',
                order_seed=1,qps=3,seed=2,name='cpu_ROI_fixture',url='unused',worker_urls='E,P,D',
                observer_kind='profile',state_interval_ms=0,observe_client_host=False,
                freeze_client_static_heap=False,profile_completion_roi=[30,70])
            with patch.object(control,'ROOT',root),patch.object(control,'preserve'),\
                 patch.object(control,'resolve_manifest',return_value=manifest),\
                 patch.object(control,'load_replay_counts',return_value={r['request_id']:5 for r in rows}),\
                 patch.object(control,'execute',side_effect=execute),patch.object(control,'rpc',side_effect=rpc),\
                 patch.object(control,'summarize',return_value=metrics):
                await control.run(options)
            self.assertEqual(len(runs),3)
            self.assertTrue(all(len(r['ids'])==100 and r['qps']==3 for r in runs))
            self.assertEqual(runs[0]['ids'],runs[1]['ids']);self.assertEqual(runs[1]['ids'],runs[2]['ids'])
            self.assertEqual(calls,[('E','profile/begin'),('P','profile/begin'),('D','profile/begin'),
                ('D','profile/end'),('P','profile/end'),('E','profile/end')])
            audit=json.loads((root/'state/cpu_ROI_fixture_completion_roi.json').read_text())
            self.assertGreaterEqual(audit['begin_completed'],30);self.assertLess(audit['begin_completed'],70)
            self.assertGreaterEqual(audit['end_completed'],70)
            result=json.loads((root/'analysis/cpu_ROI_fixture_qualification.json').read_text())
            self.assertTrue(result['observer_gate_passed'])
            self.assertEqual(result['model_quality_gate'],'PENDING')


if __name__=='__main__':unittest.main()
