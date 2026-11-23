"""Recover a drained guard without restarting any live or failed GPU job."""
import json
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,digest,preserve,bind


def launch_detached_guard():
    preserve();before=json.loads((ROOT/'state/guardian.json').read_text())
    assert before['status'] in ('DRAINING','RELEASED'),before
    path=ROOT/'state/guard_drain_recovery_v1.json';assert not path.exists()
    log=ROOT/'logs/detached_guardian_recovery_v1.log';assert not log.exists()
    command=[str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.bridge','guardian',
        '--compat_guard=poc_flashvep.dense_ar_output_aware_v2.bridge']
    with log.open('w') as stream:
        child=subprocess.Popen(command,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(path,dict(time=time.time(),previous_guard=before,new_guard_pid=child.pid,command=command,
        budget_SHA256=digest(ROOT/'state/budget.json'),
        observation='The guard entered DRAINING while the10-hour deadline remained valid and no STOP_GUARDIAN file existed.',
        cause='NOT_IDENTIFIED. The source also drains after SIGINT or SIGTERM. Signal origin was not recorded.',
        safety='The new guard waits for the same global flock. It cannot overlap the previous owner. '
            'Do not terminate the active GPU job. Do not bypass the ACTIVE guard check. '
            'The detached process has its own session and a normal log file,not an open tool session.'))
    return child.pid


def recovery_plan(parent_name,lane):
    runner=bind();plan_path=ROOT/'jobs'/parent_name;parent=json.loads(plan_path.read_text())
    expected=parent['lanes'][lane] if 'lanes' in parent else parent['ordered_jobs']
    missing=[];terminal=[]
    for name in expected:
        folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
        if receipt.exists():
            row=json.loads(receipt.read_text())
            terminal.append(dict(name=name,status='COMPLETE' if row['complete'] else 'FAILED_GPU_RUN_NOT_RETRIED'))
        elif folder.exists():raise AssertionError(('A live or interrupted GPU folder needs separate audit',name))
        else:missing.append(name)
    stem='guard_resume_'+plan_path.stem+'_v1';target=ROOT/'jobs'/(stem+'.json');assert not target.exists()
    lanes=[[],[],[],[]];lanes[lane]=missing
    snapshots={}
    prefixes=(plan_path.stem,'c3_credit_confirmation' if parent_name=='c3_credit_retirement_confirmation_r1.json' else plan_path.stem)
    for prefix in prefixes:
        for source in (ROOT/'state').glob(prefix+'*.json'):
            snapshots[str(source.relative_to(ROOT))]=dict(SHA256=digest(source),content=json.loads(source.read_text()))
    write(target,dict(time=time.time(),parent_plan=parent_name,parent_plan_SHA256=digest(plan_path),
        lanes=lanes,stop_on_failure=True,retained_terminal_runs=terminal,retained_preflight_state=snapshots,
        scope='Resume only immutable configurations with no raw GPU folder. No live or failed GPU run is restarted. '
            'The original preflight failures remain in this snapshot and their original state files. '
            'Same source,requests,arrival traces,output work,policies and scientific gates.'))
    return target


def revise_pending_hygiene(version=1,revision=3):
    import os
    import signal
    from pathlib import Path
    suffix='' if version==1 else '_v'+str(version)
    record=ROOT/'state'/('coordinator_hygiene_confirmation_controller'+suffix+'.json')
    old=json.loads(record.read_text());pid=old['pid'];proc=Path('/proc',str(pid))
    command=(proc/'cmdline').read_bytes()
    assert b'poc_flashvep.dense_ar_p_critical_path.hygiene_confirmation' in command and b'run' in command
    assert proc.stat().st_uid==os.getuid() and os.getsid(pid)==pid
    assert not (proc/'task'/str(pid)/'children').read_text().strip(),'Do not stop a GPU-launching controller'
    assert not (ROOT/'jobs/coordinator_hygiene_c3_confirmation_r2.json').exists()
    target=ROOT/'state'/('pending_hygiene_controller_revision_v'+str(revision)+'.json');assert not target.exists()
    write(target,dict(time=time.time(),previous=old,
        command=command.decode().split('\0'),no_children_verified=True,no_performance_plan_created=True,
        status='STOPPED_PENDING_CONTROLLER_REVISION',
        reason='Use corrected state-reader gate names. Keep all original observer/schema failures. '
            'This pending CPU controller owns no GPU job or pair lease.'))
    os.kill(pid,signal.SIGTERM)
    return pid


if __name__=='__main__':
    import sys
    if sys.argv[1]=='guard':print(launch_detached_guard(),flush=True)
    elif sys.argv[1]=='revise_hygiene':print(revise_pending_hygiene(*map(int,sys.argv[2:])),flush=True)
    else:print(recovery_plan(sys.argv[2],int(sys.argv[3])),flush=True)
