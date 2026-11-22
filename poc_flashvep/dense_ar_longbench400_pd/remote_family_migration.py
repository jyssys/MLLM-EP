"""Move only future discovery dispatch. Keep all started measurements intact."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,preserve,digest
from .experiment import create
from .compute_family_tuning import action_extra
from .run import execute
from .qualification import audit
from .quality_gate import compare as quality
from .compute_throttle_qualification import no_mixed
from .metrics import summarize


MARKER='remote_family_lane0_resource_handoff_v2.json'


def move():
    preserve()
    assert not (ROOT/'state'/MARKER).exists()
    receipt=json.loads((ROOT/'state/background_remote_family_tuning_lane0_v1.json').read_text())
    pid=receipt['pid'];proc=Path('/proc')/str(pid)
    assert proc.stat().st_uid==os.getuid()
    assert b'poc_flashvep.dense_ar_longbench400_pd.remote_family_tuning\x000\0' in (proc/'cmdline').read_bytes()
    # A prior attempt failed this check before sending a signal. Wait only for
    # a future-dispatch lock boundary. Never stop a GPU child to create one.
    for _ in range(100):
        if 'locks_lock_inode_wait' in (proc/'wchan').read_text():break
        time.sleep(.1)
    else:raise AssertionError('No safe future-dispatch boundary is available')
    plan=json.loads((ROOT/'configs/remote_family_tuning_lane0_v1.json').read_text())
    stopped=False
    os.kill(pid,signal.SIGSTOP)
    try:
        for _ in range(100):
            if '\nState:\tT' in (proc/'status').read_text():break
            time.sleep(.01)
        else:raise AssertionError('The owned future-dispatch process did not stop')
        assert not (proc/'task'/str(pid)/'children').read_text().strip()
        unfinished=[];keep_original_pair=[]
        for action in plan['candidates']:
            label=action['label']
            names=[f'Q29_RemoteFamily_{label}_Natural{n}_v1' for n in (32,400)]
            names += [f'RemoteTune_Discovery200_q{q}_{label}_v1' for q in (1.8,2.8)]
            for name in names:
                raw=ROOT/'raw'/name
                if raw.exists():
                    result=raw/'dispatcher_result.json'
                    assert result.exists() and json.loads(result.read_text())['complete'],name
            done=all((ROOT/'raw'/name/'dispatcher_result.json').exists() for name in names[-2:])
            if not done:
                started=[(ROOT/'raw'/name).exists() for name in names[-2:]]
                if any(started):
                    keep_original_pair.extend(name for name,present in zip(names[-2:],started) if not present)
                else:
                    unfinished.append(action)
        assert unfinished
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT);stopped=True
        for _ in range(200):
            if not proc.exists() or not (proc/'cmdline').read_bytes():break
            time.sleep(.02)
        else:raise AssertionError('The old controller can still dispatch')
        write(ROOT/'state'/MARKER,dict(time=time.time(),old_controller=receipt,remaining=unfinished,
            keep_original_pair=keep_original_pair,
            original_plan_SHA256=digest(ROOT/'configs/remote_family_tuning_lane0_v1.json'),
            original_source_SHA256=plan['source_capsule_SHA256'],old_pair=[0,1],new_pair=[2,3],
            old_future_performance_status='NOT_RUN_RESOURCE_REASSIGNMENT',no_GPU_child_interrupted=True,
            scope='Stop only the owned controller while blocked on a pair lock. Every existing raw job completed. '
                'Keep completed proofs and previous performance unchanged. Finish a partial paired-QPS block on its original pair. '
                'Move both QPS points of each wholly unstarted action together. '
                'Equivalent H100 pair2/3 became free after compound discovery. New names bind the resource change.'))
        print('FUTURE_REMOTE_DISCOVERY_HANDOFF',len(unfinished),flush=True)
    finally:
        if not stopped and proc.exists():os.kill(pid,signal.SIGCONT)


def resume():
    marker=json.loads((ROOT/'state'/MARKER).read_text())
    original=json.loads((ROOT/'configs/remote_family_tuning_lane0_v1.json').read_text())
    capsule_path=next(p for p in (ROOT/'references/source_capsules').glob('remote_family_static_tuning_v1_20261121.zip'))
    assert digest(capsule_path)==marker['original_source_SHA256']
    results=[]
    retained=marker.get('keep_original_pair',[])
    if retained:
        manifest='remote_family_retained_original_pair_v2.json'
        write(ROOT/'jobs'/manifest,dict(jobs=retained,stop_on_failure=True,
            scope='Finish the original-pair half of an already observed action block. Do not move its paired performance point.'))
        subprocess.run([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_longbench400_pd.launch',
            'remote_family_retained_original_pair_v2','poc_flashvep.dense_ar_longbench400_pd.queue',manifest],cwd=REPO,check=True)
    for action in marker['remaining']:
        label=action['label']
        common=dict(cut=action['cut'],pair=(2,3),d_compute_throttle=True,defer_D_remote_setup=True,
            execution_family='INSTALLED_DEFAULT_COMPILE' if action.get('stock') or action.get('compiled') else 'MATCHED_NO_COMPILE',
            continuous_source_credit_limit=0 if action.get('stock') else 4,
            source_capsule=capsule_path.name,source_capsule_SHA256=marker['original_source_SHA256'],
            wall_timeout=4500,serving_timeout=4000)
        gate=ROOT/'state'/f'D_remote_family_{label}_GPU23_gate_v2.json'
        proofs=[]
        for n,dataset in ((32,'continuous_qualification32_natural.json'),(400,'longbench_balanced400_v1.json')):
            old=f'Q29_RemoteFamily_{label}_Natural{n}_v1'
            if (ROOT/'raw'/old/'dispatcher_result.json').exists():
                name=old
                assert json.loads((ROOT/'raw'/old/'dispatcher_result.json').read_text())['complete']
            else:
                name=create(f'Q29_RemoteFamily_{label}_Natural{n}_GPU23_v2',action['axis'],dataset,
                    qps=2.8,seed=2026112201,correctness=True,axis_hash_workers=8,
                    axis_capture_hidden_last_only=True,gpu_state_digest=True,axis_import_hash_after_model=True,
                    **common,**action_extra(action,dataset))
                assert execute(name),name
            coverage=audit(name);score=quality(name)
            assert coverage['passed'] and coverage['decode_full36_observation_complete'] and score['passed']
            assert all(r['status']=='PASS' for r in coverage['imported_prefix_KV'])
            proofs.append(dict(name=name,coverage=coverage,quality=score,compute_deferral=no_mixed(name)))
        write(gate,dict(passed=True,full400_passed=True,action=action,proofs=proofs,evidence='CORRECTNESS_ONLY',
            scope='Keep completed original-pair proofs. New proofs use equivalent pair2/3. Performance QPS arms move together.'))
        for qps in (1.8,2.8):
            old=f'RemoteTune_Discovery200_q{qps}_{label}_v1'
            name=create(old.removesuffix('_v1')+'_GPU23_v2',action['axis'],'balanced400_discovery.json',
                qps=qps,seed=2026112291,discovery_only=True,remote_family_gate=gate.name,
                resource_migration_from=old,scope=marker['scope'],**common,**action_extra(action,'balanced400_discovery.json'))
            assert execute(name),name
            m=summarize(name);score=quality(name)
            results.append(dict(name=name,quality=score))
            print('MIGRATED_REMOTE_DISCOVERY',label,qps,m['TTFT_p95_ms'],m['ITL_p99_ms'],m['TPOT_p95_ms'],score['passed'],flush=True)
    value=dict(complete=True,time=time.time(),candidates=original['candidates'],migrated_results=results,
        resource_handoff=MARKER,evidence='DISCOVERY_COMPLETE_WITH_EXPLICIT_RESOURCE_HANDOFF',
        scope='The operational lane0 completion includes original completed actions and new-name migrated future actions. '
            'Do not invent measurements for old unstarted performance names. Raw configs retain their actual pairs.')
    for name in retained:
        receipt=ROOT/'raw'/name/'dispatcher_result.json'
        while not receipt.exists():time.sleep(5)
        assert json.loads(receipt.read_text())['complete'],name
    value['retained_original_pair_jobs']=retained
    write(ROOT/'state/remote_family_tuning_lane0_complete_v1.json',value)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('move','resume'));args=parser.parse_args()
    move() if args.action=='move' else resume()
