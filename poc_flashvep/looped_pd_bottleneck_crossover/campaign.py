"""Serial quiet-lane work queue with durable statuses and deadline checks."""
import argparse
import subprocess
from .common import *

def alive(pid):
    try:
        os.kill(pid,0)
        return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]!='Z'
    except (ProcessLookupError,FileNotFoundError):return False

def namespace_state(folder):
    status=folder/'status.json';result=folder/'dispatcher_result.json'
    if not status.exists():
        if result.exists() and json.loads(result.read_text())['returncode']!=0:return 'FAILED'
        raise RuntimeError('Existing namespace without terminal status: '+folder.name)
    state=json.loads(status.read_text())['status']
    if state=='RUNNING' and result.exists() and json.loads(result.read_text())['returncode']!=0:return 'FAILED'
    return state

def restore_if_quiet():
    from .resources import base as resources
    active=resources.processes()
    if active:
        write(ROOT/'state/idle_restore.json',dict(time=time.time(),status='SKIPPED_ACTIVE_GPU_PROCESSES',active=active))
        return
    resources.restore(list(range(8)))
    write(ROOT/'state/idle_restore.json',dict(time=time.time(),status='RESTORE_REQUEST_COMPLETED',
        scope='Verify actual burner processes before reporting final restoration.'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int);p.add_argument('--plan',required=True);a=p.parse_args()
    if a.wait_pid:
        while alive(a.wait_pid) and remaining()>3600:
            write(ROOT/'state/campaign.json',dict(time=time.time(),status='WAITING_EXISTING_DISPATCH',pid=a.wait_pid));time.sleep(15)
    plan=Path(a.plan)
    while remaining()>3600:
        names=json.loads(plan.read_text())['configs'];pending=[]
        skipped={json.loads(s)['name'] for s in (ROOT/'state/not_run.jsonl').read_text().splitlines()} if (ROOT/'state/not_run.jsonl').exists() else set()
        for name in names:
            cfg=json.loads((ROOT/'configs'/name).read_text());folder=ROOT/'raw'/cfg['name']
            if cfg['name'] in skipped:continue
            if folder.exists():
                state=namespace_state(folder)
                if state=='COMPLETE':continue
                if state=='FAILED':
                    append(ROOT/'state/campaign_failures.jsonl',dict(time=time.time(),name=cfg['name']));continue
                raise RuntimeError('Another active namespace must finish first: '+cfg['name'])
            pending.append(name)
        if not pending:
            write(ROOT/'state/campaign.json',dict(time=time.time(),status='PLAN_DRAINED',plan=str(plan)))
            restore_if_quiet();return
        name=pending[0];print('NEXT',name,flush=True)
        result=subprocess.run([str(PYTHON),'-m','poc_flashvep.looped_pd_bottleneck_crossover.dispatch',name,'--keep-quiet'],cwd=REPO)
        write(ROOT/'state/campaign.json',dict(time=time.time(),status='NEXT_CELL',last=name,returncode=result.returncode,pending=pending[1:]))
    budget(status='NO_NEW_LONG_JOBS')
    restore_if_quiet()

if __name__=='__main__':main()
