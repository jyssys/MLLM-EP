"""Finish analysis after GPU campaigns. Preserve incomplete jobs and failed attempts."""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from .common import ROOT,REPO,write

def run(module,*args):
    name=module.rsplit('.',1)[-1]
    path=ROOT/'runs/logs'/f'final_analysis_{name}.log'
    command=[sys.executable,'-m',module,*args]
    print('ANALYSIS',command,flush=True)
    with path.open('w') as output:subprocess.run(command,cwd=REPO,stdout=output,stderr=subprocess.STDOUT,check=True)

def ready():
    for name in ['post_state.json','attribution_campaign_state.json']:
        path=ROOT/name
        if path.exists() and json.loads(path.read_text())['status']=='FAILED':
            raise RuntimeError(f'Upstream campaign failed: {name}. Do not publish a complete result.')
    p=ROOT/'attribution_campaign_state.json'
    return p.exists() and json.loads(p.read_text())['status']=='COMPLETE'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    try:
        write(ROOT/'finalization_state.json',dict(status='WAITING_FOR_GPU_CAMPAIGNS',time=time.time()))
        while not ready():
            if not args.wait:raise RuntimeError('GPU campaigns remain incomplete. Do not manufacture a final report.')
            time.sleep(20)
        assert json.loads((ROOT/'research_state.json').read_text())['status']=='GPU_CAMPAIGN_COMPLETE_ANALYSIS_REQUIRED'
        assert json.loads((ROOT/'post_state.json').read_text())['status']=='GPU_CONTROLS_COMPLETE'
        os.sched_setaffinity(0,set(range(176,184)))
        write(ROOT/'finalization_state.json',dict(status='RUNNING',time=time.time()))
        for name in ['aggregate_quality','aggregate_timing','aggregate_generic','compare_baselines','paired_quality','ep_affinity',
                'kernel_aggregate','verify_fidelity']:
            run('poc_flashvep.looped_moe_affinity.'+name)
        run('poc_flashvep.looped_moe_affinity.affinity','--decode')
        for name in ['count_prediction','oracle_summary','aggregate_attribution','figures']:
            run('poc_flashvep.looped_moe_affinity.'+name)
        from .reporting import build
        snapshot=build()
        assert snapshot['completed_performance_cells']==90,snapshot
        assert snapshot['matched_native_ep_generation_cells']==90,snapshot
        assert snapshot['official_ep_tasks']==14,snapshot
        assert snapshot['native_comparison_rows']==180,snapshot
        assert snapshot['strong_generic_cells']==90,snapshot
        assert snapshot['heldout_nll_configurations']==3,snapshot
        documents=[ROOT/name for name in ['report.md','summary.md','go_no_go.md','accuracy_reproduction.md',
            'benchmark_summary.md','profiling.md','dependency_dag.md','oracle.md','failures.md',
            'implementation_scope.md','reproduce.md']]
        lint=[sys.executable,'/home/esjung/.codex/skills/asd-ste100/scripts/ste-lint.py',
            '--disable','synonym-rotation',*[str(p) for p in documents],'--json']
        with (ROOT/'quality/final_writing_check.json').open('w') as output:
            subprocess.run(lint,cwd=REPO,stdout=output,check=True)
        write(ROOT/'quality/writing_review_scope.json',dict(
            disabled_lexical_rule='synonym-rotation',
            reason='Domain nouns such as fixed placement and correct count do not name synonymous verbs. Preserve those technical terms.',
            scope='Structural lint plus manual terminology review. No certified STE compliance claim.'))
        run('poc_flashvep.looped_moe_affinity.evidence_bundle')
        with (ROOT/'runs/logs/final_cpu_tests.log').open('w') as output:
            subprocess.run([sys.executable,'-m','unittest','poc_flashvep.looped_moe_affinity.test_cpu','-v'],
                cwd=REPO,stdout=output,stderr=subprocess.STDOUT,check=True)
        subprocess.run(['/home/esjung/anaconda3/bin/ruff','check','--select','F821',
            'poc_flashvep/looped_moe_affinity'],cwd=REPO,check=True)
        from . import resources
        resources.restore(list(range(8)))
        write(ROOT/'analysis_complete.json',dict(status='COMPLETE_WITH_REPORTED_LIMITATIONS',time=time.time(),**snapshot))
        write(ROOT/'research_state.json',dict(status='EXPERIMENT_ANALYSIS_COMPLETE',time=time.time(),
            verdict=snapshot['verdict'],release_proof='release_state.json',
            warning='Read missing-comparator and cache-free limitations. No loop-aware E2E speedup is claimed.'))
        if args.publish:run('poc_flashvep.looped_moe_affinity.publish')
        write(ROOT/'finalization_state.json',dict(status='COMPLETE',time=time.time(),published=args.publish,**snapshot))
        print('COMPLETE',snapshot,flush=True)
    except BaseException as error:
        write(ROOT/'finalization_state.json',dict(status='FAILED',time=time.time(),error=repr(error),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
