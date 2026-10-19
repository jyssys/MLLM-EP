import hashlib
import gzip
import tempfile
import unittest
from poc_depth_cdb_factorial.contract import *
from poc_depth_cdb_factorial.scripts.analyze import paired_ratio,parity

class ContractTests(unittest.TestCase):
    def test_prior_sources_unchanged(self):
        c=read(ROOT/'artifacts/continuity.json')
        for path,digest in c['sources'].items():self.assertEqual(sha(PROJECT/path),digest)
        self.assertEqual(sha(SPEC),c['spec_sha256'])

    def test_quality_counts(self):
        rows=read(ROOT/'artifacts/workload_frozen.json')['quality']
        self.assertEqual(len({r['id'] for r in rows}),256)
        for d in ['gsm8k','math500']:self.assertEqual(sum(r['dataset']==d for r in rows),128)

    def test_all_required_cells(self):
        cases=read(ROOT/'artifacts/speed_plan.json')['cases']
        self.assertEqual(len(cases),75)
        for C in [4,8,16,32,64]:
            for repeat in range(3):
                rows={r['arm']:r for r in cases if r['C']==C and r['repeat']==repeat}
                self.assertEqual(set(rows),set(ARMS))
                self.assertEqual(rows['A1']['placements'],rows['A3']['placements'])
                self.assertEqual(rows['A2']['placements'],rows['A4']['placements'])
                self.assertEqual(set(rows['A0']['placements'].values()),{0})

    def test_LPT_uses_only_work(self):
        self.assertEqual(lpt(['a','b','c','d'],dict(a=4,b=3,c=2,d=1)),dict(a=0,b=1,c=2,d=3))
        source=(ROOT/'contract.py').read_text().split('def lpt',1)[1]
        self.assertNotIn("['answer']",source)
        self.assertNotIn("['prompt']",source)

    def test_complete_precision_extension(self):
        d=read(ROOT/'artifacts/repeat_extension.json')
        self.assertEqual(d['total_repeats'],5)
        self.assertEqual(len(d['cases']),50)
        for C in [4,8,16,32,64]:
            for repeat in [3,4]:
                rows={r['arm']:r for r in d['cases'] if r['C']==C and r['repeat']==repeat}
                self.assertEqual(set(rows),set(ARMS))
                self.assertEqual(rows['A1']['placements'],rows['A3']['placements'])
                self.assertEqual(rows['A2']['placements'],rows['A4']['placements'])

    def test_real_native_branch(self):
        source=(ROOT/'scripts/worker.py').read_text()
        self.assertIn('engine._run_no_refill_loop(model_kwargs=None,arrivals=admission)',source)
        self.assertIn('engine._run_scheduler_loop(model_kwargs=None,arrivals=admission)',source)
        self.assertIn("if not engine.cdb_config.refill:assert stats['scheduler_refills']==0",source)

    def test_ratio_direction(self):
        self.assertEqual(paired_ratio([10,10,10],[20,20,20])['speedup'],.5)
        self.assertEqual(paired_ratio([10,10,10],[5,5,5])['speedup'],2.)

    def test_token_parity_length_penalty(self):
        a=[dict(id='x',output_ids=[1,2],depths=[dict(depth=2)])]
        b=[dict(id='x',output_ids=[1],depths=[])]
        r=parity(a,b)
        self.assertEqual(r['token_exact'],0)
        self.assertEqual(r['free_running_positional_token_agreement_pct'],50.)

    def test_masks(self):
        self.assertEqual(ARMS['A0']['gpus'],[4])
        for a in ['A1','A2','A3','A4']:self.assertEqual(ARMS[a]['gpus'],[4,5,6,7])

    def test_published_gzip_reader(self):
        with tempfile.TemporaryDirectory(prefix='factorial-read-test-') as directory:
            path=Path(directory)/'sample.json'
            Path(str(path)+'.gz').write_bytes(gzip.compress(b'{"sample": 1}',mtime=0))
            self.assertEqual(read(path),dict(sample=1))

    def test_published_gzip_case_discovery(self):
        from poc_depth_cdb_factorial.scripts import analyze as analysis
        previous=analysis.ROOT
        try:
            with tempfile.TemporaryDirectory(prefix='factorial-case-test-') as directory:
                analysis.ROOT=Path(directory)
                output=analysis.ROOT/'outputs/run'
                (output/'rank_0').mkdir(parents=True)
                (output/'replicas.json').write_text('{}')
                master=dict(case=dict(name='speed_case',ids=['x']))
                (output/'speed_case.json.gz').write_bytes(gzip.compress(json.dumps(master).encode(),mtime=0))
                rank=dict(outputs=[dict(id='x')])
                (output/'rank_0/speed_case.json.gz').write_bytes(gzip.compress(json.dumps(rank).encode(),mtime=0))
                result=list(analysis.records())
                self.assertEqual(len(result),1)
                self.assertEqual(result[0][4][0]['id'],'x')
        finally:analysis.ROOT=previous

if __name__=='__main__':unittest.main()
