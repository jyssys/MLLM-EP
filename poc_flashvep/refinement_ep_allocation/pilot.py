"""User-approved 128/32 pilot. Preserve full-run evidence and stop only its owned sessions."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .common import ROOT,REPO,HF_PYTHON,EP_PYTHON,REVISION,write,append,rows,digest,setup
from .campaign import launch
from . import resources

def freeze():
    root=ROOT/'pilot'
    root.mkdir(exist_ok=True)
    cohorts={}
    for task,n in [('gsm8k',128),('humaneval',32)]:
        source=ROOT/'manifests'/f'{task}_test.jsonl'
        chosen=rows(source)[:n]
        path=root/f'{task}_manifest.jsonl'
        payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen)
        if path.exists():
            assert path.read_text()==payload
        else:
            path.write_text(payload)
        cohorts[task]=dict(n=n,path=str(path),sha256=digest(path),ids=[r['sample_id'] for r in chosen])
    hf={r['sample_id']:r for p in (ROOT/'hf/full_gsm8k').glob('shard*.jsonl') for r in rows(p,allow_incomplete=True)}
    epfile=ROOT/'ep4/A_full_gsm8k/quality_samples.jsonl'
    ep={r['sample_id']:r for r in rows(epfile,allow_incomplete=True) if r.get('task')=='gsm8k'} if epfile.exists() else {}
    for method,records in [('REFERENCE_HF',hf),('P0_FULL_EP4',ep)]:
        saved=root/f'gsm8k_reused_{method}.jsonl'
        if not records and saved.exists():
            records.update({r['sample_id']:r for r in rows(saved)})
    wanted=set(cohorts['gsm8k']['ids'])
    assert wanted<=hf.keys() and wanted<=ep.keys(),'Pilot GSM8K requests are not all complete'
    for key in wanted:
        assert hf[key]['input_ids']==ep[key]['input_ids'],'Pilot input token mismatch'
        for record in (hf[key],ep[key]):
            assert record['revision']==REVISION and record['gen_length_cap']==2048
            assert record['block_length']==32 and record['threshold']==.95 and record['temperature']==0
    for method,records in [('REFERENCE_HF',hf),('P0_FULL_EP4',ep)]:
        path=root/f'gsm8k_reused_{method}.jsonl'
        if not path.exists():
            path.write_text(''.join(json.dumps(dict(records[k],reuse_source='FULL_RUN_COMPLETED_BEFORE_SCOPE_REDUCTION'),ensure_ascii=False)+'\n' for k in cohorts['gsm8k']['ids']))
    path=root/'scope_amendment.json'
    if not path.exists():
        write(path,dict(status='USER_APPROVED_PILOT',time=time.time(),cohorts=cohorts,
            selection='First benchmark IDs, without conditioning on answers, scores, or generation lengths.',
            checkpoint_and_generation_settings='UNCHANGED',
            supersedes='Full 1319/164 final evaluation requirement for the current pilot only.',
            limits='Not full benchmark reproduction. Cannot establish a 0.5 pp non-inferiority claim.',
            preserved_full_run_counts=dict(hf=len(hf),ep4=len(ep))))
    return cohorts

def procinfo(pid):
    proc=Path('/proc')/str(pid)
    words=(proc/'cmdline').read_bytes().decode().split('\0')
    return dict(pid=pid,uid=proc.stat().st_uid,pgid=os.getpgid(pid),sid=os.getsid(pid),command=words)

def stop_full():
    state=json.loads((ROOT/'state/campaign.json').read_text())
    follow=json.loads((ROOT/'state/followup.json').read_text())
    assert state['status']=='RUNNING' and follow['status']=='WAITING_FOR_FULL_BASELINE_QUEUE'
    controllers=[(state['pid'],'poc_flashvep.refinement_ep_allocation.campaign'),
                 (follow['pid'],'poc_flashvep.refinement_ep_allocation.baseline_followup')]
    workers=list(state['active'].values())
    inventory=[]
    for pid,module in controllers:
        record=procinfo(pid)
        assert record['uid']==os.getuid() and module in record['command']
        inventory.append(record)
    for pid in workers:
        record=procinfo(pid)
        assert record['uid']==os.getuid() and record['pgid']==pid and record['sid']==pid
        assert any(w in record['command'] for w in ('poc_flashvep.refinement_ep_allocation.hf_worker',
                                                   'poc_flashvep.refinement_ep_allocation.ep4_worker'))
        members=[]
        for proc in Path('/proc').iterdir():
            if not proc.name.isdecimal():
                continue
            try:
                member=procinfo(int(proc.name))
                if member['pgid']==pid:
                    assert member['uid']==os.getuid()
                    members.append(member)
            except (FileNotFoundError,ProcessLookupError,PermissionError):
                continue
        inventory.append(dict(record,members=members))
    write(ROOT/'pilot/prior_full_controller_state.json',dict(campaign=state,followup=follow,processes=inventory,time=time.time()))
    for active in resources.base.processes():
        owner=procinfo(active['pid'])
        assert owner['uid']==os.getuid() and owner['pgid'] in workers,'Unrelated GPU process left untouched'
    # Controllers stop first. They must not enqueue more full evaluations.
    for pid,_module in controllers:
        os.kill(pid,signal.SIGTERM)
    for pid in workers:
        try:
            os.killpg(pid,signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline=time.monotonic()+45
    while resources.base.processes() and time.monotonic()<deadline:
        time.sleep(.5)
    assert not resources.base.processes(),'A GPU remains occupied. No unrelated process was signaled.'
    write(ROOT/'state/campaign.json',dict(status='REPLACED_BY_USER_APPROVED_PILOT',previous_pid=state['pid'],time=time.time()))
    write(ROOT/'state/followup.json',dict(status='REPLACED_BY_USER_APPROVED_PILOT',previous_pid=follow['pid'],time=time.time()))
    append(ROOT/'job_events.jsonl',dict(action='USER_REQUESTED_SCOPE_REDUCTION',stopped_roots=workers,time=time.time(),
        note='Completed raw requests preserved. In-flight requests can be incomplete and remain outside the frozen pilot.'))

def run():
    setup()
    lock=(ROOT/'state/pilot.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    freeze()
    if json.loads((ROOT/'state/campaign.json').read_text())['status']=='RUNNING':
        stop_full()
    else:
        resources.base.stop(list(range(8)))
    manifest=ROOT/'pilot/humaneval_manifest.jsonl'
    jobs={}
    try:
        jobs['ep4_A']=launch('A_pilot_humaneval',[EP_PYTHON,'-m','poc_flashvep.refinement_ep_allocation.ep4_worker',
            '--manifest',str(manifest),'--gpus','0,1,2,3','--run-id','A_pilot_humaneval','--parity'],[0,1,2,3])
        for shard,gpu in enumerate(range(4,8)):
            jobs[f'hf_{gpu}']=launch(f'hf_gpu{gpu}_pilot_humaneval',[HF_PYTHON,'-m','poc_flashvep.refinement_ep_allocation.hf_worker',
                '--manifest',str(manifest),'--gpu',str(gpu),'--shard',str(shard),'--shards','4',
                '--run-id','pilot_humaneval'],[gpu])
        failed=[]
        while jobs:
            for name,child in list(jobs.items()):
                code=child.poll()
                if code is None:
                    continue
                append(ROOT/'job_events.jsonl',dict(name=name,pid=child.pid,exit_code=code,time=time.time()))
                del jobs[name]
                if code:
                    failed.append(name)
                resources.base.restore([0,1,2,3] if name=='ep4_A' else [int(name.split('_')[1])])
            write(ROOT/'state/pilot.json',dict(status='RUNNING_PILOT_BASELINE',pid=os.getpid(),
                active={n:p.pid for n,p in jobs.items()},failed=failed,time=time.time()))
            if jobs:
                time.sleep(5)
        score=subprocess.run([sys.executable,'-m','poc_flashvep.refinement_ep_allocation.postprocess',
            '--pilot','--score-humaneval'],cwd=REPO,text=True,capture_output=True)
        (ROOT/'logs/pilot_scoring.log').write_text(score.stdout+score.stderr)
        write(ROOT/'state/pilot.json',dict(status='PILOT_BASELINE_READY_REVIEW' if not failed and not score.returncode else 'PILOT_BASELINE_FAILED',
            failed=failed,scoring_returncode=score.returncode,time=time.time(),
            next='Review pilot quality. Preserve the original diffusion protocol, router and BF16. Do not claim full benchmark reproduction.'))
    finally:
        if not jobs:
            resources.base.restore(list(range(8)))

def main():
    setup()
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('start','run','freeze'))
    a=p.parse_args()
    if a.action=='freeze':
        print(json.dumps(freeze(),indent=2))
    elif a.action=='run':
        run()
    else:
        with (ROOT/'logs/pilot_supervisor.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-m','poc_flashvep.refinement_ep_allocation.pilot','run'],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print('PILOT_STARTED',child.pid)

if __name__=='__main__':
    main()
