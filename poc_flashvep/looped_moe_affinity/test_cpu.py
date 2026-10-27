"""Fast regression tests. These tests do not establish model quality."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess
import gzip
import numpy as np
from .common import write
from .affinity import slot_null,fit_transition,top_recall,set_information
from .count_prediction import rounded_counts
from .attribution_partition import partition,split_phases
from .attribution_report import choose_observer

class CPUChecks(unittest.TestCase):
    def test_concurrent_publish(self):
        with tempfile.TemporaryDirectory(prefix='affinity-write-regression-') as directory:
            path=Path(directory)/'state.json'
            def check(i):
                write(path,dict(index=i,payload='x'*(20+i%100)))
                return json.loads(path.read_text())
            with ThreadPoolExecutor(8) as pool:values=list(pool.map(check,range(200)))
            self.assertEqual(len(values),200)

    def test_null_preserves_slots_and_distinct_choices(self):
        rng=np.random.default_rng(16)
        x=np.array([rng.choice(25,2,replace=False) for _ in range(2000)])
        y=slot_null(x,rng)
        for k in [0,1]:np.testing.assert_array_equal(np.bincount(x[:,k],minlength=25),np.bincount(y[:,k],minlength=25))
        self.assertTrue(np.all(y[:,0]!=y[:,1]))
        self.assertFalse(np.array_equal(x,y))

    def test_multi_label_probabilities(self):
        x=np.array([[0,1],[0,2],[2,3],[3,1]]*10)
        y=(x+1)%5
        conditional,pop=fit_transition(x,y,5)
        np.testing.assert_allclose(conditional.sum(1),2)
        self.assertAlmostEqual(pop.sum(),2)
        self.assertTrue(np.all((conditional>=0)&(conditional<=1)))
        self.assertEqual(top_recall(np.eye(5)[y].sum(1),y,2),1.)
        mi,nmi=set_information(x,x,5)
        self.assertGreater(mi,0);self.assertAlmostEqual(nmi,1.)

    def test_count_rounding(self):
        p=np.array([.7,.6,.4,.3]);self.assertAlmostEqual(p.sum(),2)
        for n in [0,1,7,64,512]:
            value=rounded_counts(p*n,n*2)
            self.assertEqual(value.sum(),n*2)
            self.assertTrue(np.all(value>=0))

    def test_nested_attribution_has_no_double_count(self):
        result=partition([(1,9,'model'),(2,5,'attention'),(3,4,'router'),(6,8,'dispatch')],10)
        self.assertAlmostEqual(sum(result.values()),10)
        self.assertAlmostEqual(result['unattributed'],2)
        self.assertAlmostEqual(result['attention'],2)
        self.assertAlmostEqual(result['router'],1)
        self.assertAlmostEqual(result['dispatch'],2)
        self.assertAlmostEqual(result['model'],3)

    def test_attribution_clips_outside_regions(self):
        result=partition([(-2,2,'input'),(8,14,'output'),(10,12,'outside')],10)
        self.assertEqual(result,dict(input=2,unattributed=6,output=2))
        self.assertEqual(partition([],10),dict(unattributed=10))

    def test_attribution_rejects_invalid_timestamps(self):
        for window in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):partition([],window)
        for start,end in [(2,1),(float('nan'),3),(1,float('inf'))]:
            with self.assertRaises(ValueError):partition([(start,end,'invalid')],10)

    def test_phase_attribution_preserves_full_window(self):
        intervals=[(0,10,'parent'),(2,8,'attention'),(3,4,'router')]
        whole=partition(intervals,10)
        phases=split_phases(intervals,10,5)
        combined={key:sum(p['components_ms'].get(key,0) for p in phases.values()) for key in whole}
        self.assertEqual(whole,combined)
        self.assertEqual(sum(p['window_ms'] for p in phases.values()),10)
        for phase in phases.values():
            self.assertEqual(sum(phase['components_ms'].values()),phase['window_ms'])

    def test_equal_timestamp_regions_use_actual_nesting(self):
        values=[(1,9,'aaa_parent','site',0,0),(1,9,'zzz_child','site',0,1)]
        result=partition(values,10)
        self.assertEqual(result,dict(unattributed=2,zzz_child=8))

    def test_attribution_does_not_pool_failed_and_passing_observers(self):
        base=dict(model='CPU_TEST_NOT_AN_EXPERIMENT',ep=4,batch=1,prefix_tokens=128,policy='static',
            component='test_interval',mean_component_pct=100)
        failed=dict(base,detail='fine',timing_status='OBSERVER_ONLY_TIMING_GATE_FAILED',observer_overhead_mean_ratio=1.2)
        passed=dict(base,detail='coarse',timing_status='LOW_OVERHEAD_OBSERVER',observer_overhead_mean_ratio=1.02)
        self.assertEqual(choose_observer([failed,passed]),[passed])

    def test_scoped_publish_preserves_user_index(self):
        from . import publish
        with tempfile.TemporaryDirectory(prefix='affinity-git-safety-test-') as directory:
            root=Path(directory);repo=root/'repo';repo.mkdir();remote=root/'remote.git'
            def git(*args):
                return subprocess.check_output(['git',*args],cwd=repo,text=True,stderr=subprocess.DEVNULL)
            subprocess.run(['git','init','--bare',str(remote)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            git('init');git('config','user.name','Scoped Release Test');git('config','user.email','test@example.invalid')
            (repo/'study').mkdir();(repo/'study/data.txt').write_text('base study\n')
            (repo/'user.txt').write_text('base user\n')
            (repo/'.gitignore').write_text('study/release_state.json\n')
            git('add','.');git('commit','-m','Fixture base');git('switch','-c','unit')
            git('remote','add','origin',str(remote));base=git('rev-parse','HEAD').strip()
            (repo/'study/data.txt').write_text('completed study\n')
            (repo/'study/new.txt').write_text('new evidence\n')
            (repo/'user.txt').write_text('user staged work\n');git('add','user.txt')
            (repo/'untracked-user.txt').write_text('preserve this file\n')
            with patch.multiple(publish,REPO=repo,ROOT=repo/'study',SCOPE=['study'],BRANCH='unit',BASE=base):
                before=publish.unrelated_index_hash();publish.main();after=publish.unrelated_index_hash()
            self.assertEqual(before,after)
            self.assertEqual(git('show','HEAD:user.txt'),'base user\n')
            self.assertIn('user staged work',git('diff','--cached','--','user.txt'))
            self.assertEqual((repo/'untracked-user.txt').read_text(),'preserve this file\n')
            state=json.loads((repo/'study/release_state.json').read_text())
            self.assertEqual(state['status'],'COMMITTED_AND_PUSHED')
            self.assertEqual(state['commit'],state['remote_sha'])

    def test_native_resume_recovers_only_complete_cells(self):
        from . import native_resume
        with tempfile.TemporaryDirectory(prefix='affinity-native-resume-test-') as directory:
            root=Path(directory);source=root/'raw/source';target=root/'raw/target'
            source.mkdir(parents=True);target.mkdir()
            (source/'native_forward_b1_p128.rank0.json').write_text(json.dumps(dict(clean_ms=[1]*5)))
            complete=dict(warmup_exact=True,generated_ids=[[7]]*32)
            (source/'native_generation_b1_p128.rank0.jsonl').write_text((json.dumps(complete)+'\n')*3)
            for split in ['calibration','heldout']:
                (source/f'full_routes_b1_p128_{split}.json').write_text('{}')
                (source/f'full_routes_b1_p128_{split}.npz').write_bytes(b'CPU_FIXTURE_NOT_GPU_DATA')
            (source/'native_forward_b16_p2048.rank0.json').write_text(json.dumps(dict(clean_ms=[1]*5)))
            rows=[dict(batch=1,original_prefix_tokens=128,fixture='complete'),dict(batch=16,original_prefix_tokens=2048,fixture='incomplete')]
            with gzip.open(source/'decode_router_corpus.jsonl.gz','wt') as stream:
                stream.write(''.join(json.dumps(row)+'\n' for row in rows))
            before={p.name:p.read_bytes() for p in source.iterdir()}
            with patch.object(native_resume,'ROOT',root):
                with gzip.open(target/'decode_router_corpus.jsonl.gz','wt') as stream:
                    recovered=native_resume.inherit(source,target,stream)
            self.assertEqual(recovered,{(1,128)})
            self.assertFalse((target/'native_forward_b16_p2048.rank0.json').exists())
            self.assertEqual(before,{p.name:p.read_bytes() for p in source.iterdir()})
            with gzip.open(target/'decode_router_corpus.jsonl.gz','rt') as stream:
                self.assertEqual([json.loads(line) for line in stream],[rows[0]])

    def test_checkpoint_allows_only_its_verified_successor_release(self):
        from . import publish
        with tempfile.TemporaryDirectory(prefix='affinity-checkpoint-safety-test-') as directory:
            root=Path(directory);repo=root/'repo';repo.mkdir();remote=root/'remote.git'
            def git(*args):
                return subprocess.check_output(['git',*args],cwd=repo,text=True,stderr=subprocess.DEVNULL)
            subprocess.run(['git','init','--bare',str(remote)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            git('init');git('config','user.name','Checkpoint Test');git('config','user.email','test@example.invalid')
            study=repo/'study';study.mkdir();(study/'data.txt').write_text('base\n')
            (repo/'user.txt').write_text('user base\n')
            (repo/'.gitignore').write_text('study/*release_state.json\n')
            git('add','.');git('commit','-m','Fixture base');git('switch','-c','unit')
            git('remote','add','origin',str(remote));base=git('rev-parse','HEAD').strip()
            (repo/'user.txt').write_text('preserve staged work\n');git('add','user.txt')
            (study/'checkpoint.md').write_text('IN_PROGRESS: CPU fixture, not a measured experiment.\n')
            (study/'data.txt').write_text('completed checkpoint evidence\n')
            with patch.multiple(publish,REPO=repo,ROOT=study,SCOPE=['study'],BRANCH='unit',BASE=base):
                before=publish.unrelated_index_hash();publish.main(checkpoint=True)
                checkpoint=git('rev-parse','HEAD').strip()
                self.assertEqual(publish.reviewed_head(),checkpoint)
                self.assertEqual(before,publish.unrelated_index_hash())
                (study/'data.txt').write_text('later completed evidence\n');publish.main()
                self.assertNotEqual(git('rev-parse','HEAD').strip(),checkpoint)
                self.assertEqual(before,publish.unrelated_index_hash())
                with self.assertRaises(AssertionError):publish.reviewed_head()
            self.assertEqual(git('show','HEAD:user.txt'),'user base\n')
            state=json.loads((study/'release_state.json').read_text())
            self.assertEqual(state['kind'],'FINAL_RELEASE')
            self.assertEqual(state['commit'],state['remote_sha'])

    def test_checkpoint_archive_excludes_active_attempts(self):
        from . import checkpoint_bundle
        with tempfile.TemporaryDirectory(prefix='affinity-checkpoint-evidence-test-') as directory:
            root=Path(directory);closed=root/'raw/closed';active=root/'raw/active'
            closed.mkdir(parents=True);active.mkdir()
            (closed/'complete.json').write_text('{}')
            (closed/'metadata.rank0.json').write_text('{"fixture": "CLOSED_NOT_GPU_MEASURED"}')
            (active/'metadata.rank0.json').write_text('{"fixture": "ACTIVE_NOT_GPU_MEASURED"}')
            for name in ['research_state.json','post_state.json','attribution_campaign_state.json','finalization_state.json']:
                (root/name).write_text('{"status": "CPU_FIXTURE"}')
            with patch.object(checkpoint_bundle,'ROOT',root):checkpoint_bundle.main()
            manifest=json.loads((root/'quality/checkpoint_evidence_manifest.json').read_text())
            self.assertEqual(manifest['completed_attempts'],['closed'])
            self.assertTrue(all('/closed/' in row['path'] for row in manifest['sources']))
            with gzip.open(root/'archives/checkpoint_completed_numeric.jsonl.gz','rt') as stream:
                self.assertNotIn('ACTIVE_NOT_GPU_MEASURED',stream.read())

if __name__=='__main__':unittest.main()
