"""Serial quiet-lane work queue with durable statuses and deadline checks."""
import argparse
import subprocess
from .common import *

def alive(pid):
    try:
        os.kill(pid,0)
        return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]!='Z'
    except (ProcessLookupError,FileNotFoundError):return False

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
                status=folder/'status.json'
                if not status.exists():raise RuntimeError('Existing namespace without terminal status: '+cfg['name'])
                state=json.loads(status.read_text())['status']
                if state=='COMPLETE':continue
                if state=='FAILED':
                    append(ROOT/'state/campaign_failures.jsonl',dict(time=time.time(),name=cfg['name']));continue
                raise RuntimeError('Another active namespace must finish first: '+cfg['name'])
            pending.append(name)
        if not pending:
            write(ROOT/'state/campaign.json',dict(time=time.time(),status='PLAN_DRAINED',plan=str(plan)));return
        name=pending[0];print('NEXT',name,flush=True)
        result=subprocess.run([str(PYTHON),'-m','poc_flashvep.looped_pd_bottleneck_crossover.dispatch',name,'--keep-quiet'],cwd=REPO)
        write(ROOT/'state/campaign.json',dict(time=time.time(),status='NEXT_CELL',last=name,returncode=result.returncode,pending=pending[1:]))
    budget(status='NO_NEW_LONG_JOBS')

if __name__=='__main__':main()
